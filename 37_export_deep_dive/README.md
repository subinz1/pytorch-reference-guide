<div align="center">

[← Previous Module (Custom C++ Extensions)](../36_cpp_extensions/) | [🏠 Home](../README.md) | Next Module → (none)

</div>

---

# Module 37: torch.export Deep Dive

> **Prerequisites**: [Module 08 — torch.compile](../08_torch_compile/), [Module 11 — Export & Deployment](../11_export_deploy/)
> **Time**: ~3 hours
> **Files**: `export_advanced.py`, `control_flow_export.py`

---

## Table of Contents

1. [Beyond Basic Export](#1-beyond-basic-export)
2. [ExportedProgram Anatomy](#2-exportedprogram-anatomy)
3. [Graph Signature](#3-graph-signature)
4. [torch.cond — Conditional Control Flow](#4-torchcond--conditional-control-flow)
5. [torch.while_loop — Loops in Export](#5-torchwhile_loop--loops-in-export)
6. [map — Applying a Function Over a Dimension](#6-map--applying-a-function-over-a-dimension)
7. [Dynamic Shapes Advanced](#7-dynamic-shapes-advanced)
8. [draft_export](#8-draft_export)
9. [Pre-Dispatch vs Post-Dispatch IR](#9-pre-dispatch-vs-post-dispatch-ir)
10. [Custom Ops in Export](#10-custom-ops-in-export)
11. [Retraceability](#11-retraceability)
12. [Strict vs Non-Strict Export](#12-strict-vs-non-strict-export)
13. [ExportBackwardSignature](#13-exportbackwardsignature)
14. [Serialization Format](#14-serialization-format)
15. [Practical Debugging Workflow](#15-practical-debugging-workflow)
16. [Upstream Updates (July 3-6, 2026)](#16-upstream-updates-july-3-6-2026)

---

## 1. Beyond Basic Export

Module 11 covered the basics of `torch.export`: capturing a model into a graph, specifying dynamic shapes, and deploying via AOTInductor or ONNX. This module goes deeper.

Here we explore what's actually inside an `ExportedProgram`, how to handle control flow that export can't automatically trace, how to register custom ops for export, and how the two IR levels (pre-dispatch and post-dispatch) relate to each other. By the end, you'll be able to export models that most tutorials would call "unexportable."

The key insight: `torch.export` produces a **complete, self-contained graph** with no Python dependency. Unlike `torch.jit.trace` (which silently drops control flow) or `torch.jit.script` (which requires a subset of Python), `torch.export` is strict — if something can't be represented in the graph, it fails loudly. This is a feature, not a bug. The control flow primitives (`torch.cond`, `torch.while_loop`, `torch.map`) let you make branching and loops explicit so they survive export.

---

## 2. ExportedProgram Anatomy

When you call `torch.export.export(model, args)`, you get back an `ExportedProgram`. It contains everything needed to run the model without the original Python source:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        return self.linear(x) * self.scale

model = MyModel()
ep = torch.export.export(model, (torch.randn(3, 10),))
```

The `ExportedProgram` object has these key attributes:

### `graph_module`

The core FX graph that represents the computation. This is an `fx.GraphModule` whose nodes correspond to operations:

```python
print(ep.graph_module.graph)
# Shows nodes: placeholder -> linear -> mul -> output
```

Each node has an `op` (call_function, placeholder, output, etc.), a `target` (the actual function being called), and `args`/`kwargs`.

### `graph_signature`

Metadata describing what each graph input and output represents — is it a parameter, a buffer, a user input, or a gradient? This is how PyTorch knows which tensors in the flattened input list are weights vs data.

### `state_dict`

The model's parameters and buffers, keyed by their fully qualified names (`linear.weight`, `linear.bias`, `scale`).

### `range_constraints`

Constraints on symbolic integers. If you specify `Dim("batch", min=1, max=128)`, the constraint `1 <= batch <= 128` appears here. These are checked at runtime.

### `module_call_graph`

Records which submodules were called and in what order. Useful for understanding the call hierarchy in complex models.

### `constants`

Non-parameter, non-buffer constants that appear in the graph (e.g., tensors created inside `forward`). These are lifted out and stored separately.

To inspect all of these:

```python
print(f"Graph inputs:  {len(ep.graph_signature.input_specs)}")
print(f"Graph outputs: {len(ep.graph_signature.output_specs)}")
print(f"State dict:    {list(ep.state_dict.keys())}")
print(f"Constraints:   {ep.range_constraints}")
print(f"Constants:     {ep.constants}")
```

---

## 3. Graph Signature

The `GraphSignature` is critical for understanding how the exported graph maps to the original model. Every input and output has a spec:

### InputSpec

Each graph input is categorized:

| Kind | Description |
|------|-------------|
| `InputKind.PARAMETER` | Learnable parameter (e.g., `linear.weight`) |
| `InputKind.BUFFER` | Registered buffer (e.g., `scale`) |
| `InputKind.CONSTANT_TENSOR` | Lifted constant tensor |
| `InputKind.USER_INPUT` | The actual user-provided data |
| `InputKind.TOKEN` | Control flow token (for ordering side effects) |

### OutputSpec

Each graph output is categorized:

| Kind | Description |
|------|-------------|
| `OutputKind.USER_OUTPUT` | The actual return value |
| `OutputKind.LOSS_OUTPUT` | Loss value (for training export) |
| `OutputKind.BUFFER_MUTATION` | Buffer that was mutated in-place |
| `OutputKind.USER_INPUT_MUTATION` | User input that was mutated |
| `OutputKind.GRADIENT_TO_PARAMETER` | Gradient (training export) |
| `OutputKind.GRADIENT_TO_USER_INPUT` | Gradient w.r.t. user input |

The ordering matters: in the flattened graph inputs, parameters come first, then buffers, then constants, then user inputs. The graph signature tells you where each section starts and ends.

```python
for spec in ep.graph_signature.input_specs:
    print(f"  {spec.kind}: {spec.arg.name} -> {spec.target}")
```

This produces output like:

```
  InputKind.PARAMETER: p_linear_weight -> linear.weight
  InputKind.PARAMETER: p_linear_bias -> linear.bias
  InputKind.BUFFER: b_scale -> scale
  InputKind.USER_INPUT: x -> None
```

The `target` field maps back to the original `state_dict` key for parameters/buffers, or `None` for user inputs.

---

## 4. torch.cond — Conditional Control Flow

Python `if/else` statements are evaluated during tracing. Export only sees the branch that was taken for the example input. This means the other branch is silently dropped — a correctness bug.

`torch.cond` makes both branches explicit in the graph:

```python
def f(x):
    return torch.cond(
        x.sum() > 0,           # predicate (scalar bool tensor)
        lambda x: x * 2,       # true branch
        lambda x: x * -1,      # false branch
        (x,),                   # operands passed to both branches
    )
```

### Rules for torch.cond

1. **Predicate must be a scalar boolean tensor** — not a Python bool. Use tensor comparisons: `x.sum() > 0`, `x.shape[0] > 5` won't work (that's a Python int comparison).

2. **Both branches must return the same structure** — same number of tensors, same shapes, same dtypes. Export traces both branches and checks they match.

3. **Branches can't have side effects** — no in-place ops on captured state, no mutation of external variables. The branches are pure functions of the operands.

4. **Operands are explicit** — everything the branches need must be passed via the `operands` tuple. Closures over external tensors are allowed but must follow the same rules.

### Multiple Outputs

Branches can return tuples:

```python
def true_fn(x, y):
    return x + 1, y * 2

def false_fn(x, y):
    return x - 1, y * 0.5

result_a, result_b = torch.cond(pred, true_fn, false_fn, (x, y))
```

### Nesting

`torch.cond` calls can be nested — one branch can itself contain another `torch.cond`:

```python
def outer_true(x):
    return torch.cond(x[0] > 0, lambda x: x + 10, lambda x: x + 20, (x,))

def outer_false(x):
    return x - 1

result = torch.cond(x.sum() > 0, outer_true, outer_false, (x,))
```

### Graph Representation

In the exported graph, `torch.cond` appears as a `higher_order_op` node with two subgraph attributes — one for each branch. Both subgraphs are fully traced and available for inspection, optimization, and code generation.

---

## 5. torch.while_loop — Loops in Export

For data-dependent loops (where the number of iterations depends on runtime values), `torch.while_loop` provides exportable iteration:

```python
def cond_fn(x, count):
    return count < 10  # loop while this is True

def body_fn(x, count):
    return x + 1, count + 1  # return updated carried inputs

init_x = torch.zeros(5)
init_count = torch.tensor(0)

result_x, result_count = torch.while_loop(cond_fn, body_fn, (init_x, init_count))
```

### Rules

1. **cond_fn** takes the carried inputs and returns a scalar boolean tensor.
2. **body_fn** takes the carried inputs and returns updated values with the same structure, shapes, and dtypes.
3. **Carried inputs** are the loop state — they're passed to both `cond_fn` and `body_fn` and updated each iteration.
4. No dynamic shape changes across iterations — the shapes of carried inputs are fixed.

### Comparison with Python Loops

| Feature | Python `while` | `torch.while_loop` |
|---------|---------------|-------------------|
| Export | Only last iteration traced | Both branches traced |
| Iteration count | Can be dynamic | Can be dynamic |
| In graph | Unrolled (if static) or fails | Single loop node |
| Side effects | Allowed | Not allowed |

---

## 6. map — Applying a Function Over a Dimension

`torch.map` applies a function to each element along the first dimension of input tensors:

```python
def double(x):
    return x * 2

xs = torch.randn(5, 3)  # 5 elements, each of shape (3,)
result = torch.map(double, xs)  # applies double to each row
# result.shape == (5, 3)
```

This is useful when you want to express per-element operations that are more complex than what broadcasting handles. In the exported graph, `torch.map` appears as a higher-order op with a subgraph for the mapped function.

### Rules

1. The function is applied to slices along dimension 0.
2. The function must return tensors with consistent shapes.
3. Multiple input tensors can be passed — they must have the same size along dimension 0.

---

## 7. Dynamic Shapes Advanced

Module 11 introduced basic dynamic shapes. Here we cover the full `Dim` API and advanced constraint patterns.

### The Dim API

```python
from torch.export import Dim

batch = Dim("batch", min=1, max=128)
seq = Dim("seq", min=1, max=2048)
```

`Dim` creates a symbolic integer with optional bounds. Use it to tell export which dimensions can vary at runtime:

```python
ep = torch.export.export(
    model,
    (torch.randn(4, 16),),
    dynamic_shapes={"x": {0: batch, 1: seq}},
)
```

### Shared Dims Across Inputs

When two inputs must have the same dynamic dimension, use the same `Dim` object:

```python
batch = Dim("batch", min=1, max=64)
ep = torch.export.export(
    model,
    (torch.randn(4, 10), torch.randn(4, 10)),
    dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
)
```

This tells export that `x.shape[0] == y.shape[0]` at all times.

### Dim.AUTO

When you don't want to manually specify every dimension, `Dim.AUTO` infers dynamism automatically:

```python
ep = torch.export.export(
    model,
    (torch.randn(4, 10),),
    dynamic_shapes={"x": {0: Dim.AUTO}},
)
```

`Dim.AUTO` examines the graph and infers appropriate constraints. It's convenient for exploratory work but may produce overly conservative or overly permissive constraints.

### The dims() Helper

For models with many inputs, `dims()` creates multiple `Dim` objects at once:

```python
from torch.export import dims

batch, seq, hidden = dims("batch", "seq", "hidden")
```

### Runtime Assertions with torch._check

Add runtime constraints that export verifies symbolically:

```python
def forward(self, x):
    torch._check(x.shape[0] > 0)
    torch._check(x.shape[1] % 2 == 0)
    return self.linear(x)
```

These become range constraints in the exported program and are checked when the model is loaded and run with new inputs.

### ShapesCollection

For complex models with many inputs, `ShapesCollection` provides a cleaner API:

```python
from torch.export import ShapesCollection

shapes = ShapesCollection()
shapes[model.forward]["x"] = {0: batch}
shapes[model.forward]["y"] = {0: batch, 1: seq}
```

---

## 8. draft_export

When `torch.export.export()` fails, the error message can be cryptic. `draft_export` provides a more forgiving mode that returns a report explaining what went wrong:

```python
from torch.export import draft_export

ep, report = draft_export(model, (example_input,))
```

Instead of raising on the first issue, `draft_export` continues tracing and collects all problems. The report includes:

- **Missing fake implementations**: Custom ops without `register_fake`
- **Data-dependent control flow**: Python `if/else` that depends on tensor values
- **Unsupported Python constructs**: Features that can't be represented in the graph
- **Dynamic shape issues**: Constraints that couldn't be satisfied

The returned `ExportedProgram` may be incomplete or incorrect — it's a debugging aid, not a production artifact. The workflow is:

1. Try `export()` — if it succeeds, you're done
2. If it fails, run `draft_export()` to get the full picture
3. Fix issues one by one (add `torch.cond`, register fakes, add constraints)
4. Try `export()` again

```python
ep, report = draft_export(model, args)
if report:
    for issue in report:
        print(f"Issue: {issue}")
```

---

## 9. Pre-Dispatch vs Post-Dispatch IR

Export produces graphs at two levels of abstraction:

### Post-Dispatch (Default)

The default `export()` decomposes high-level ops into lower-level primitives. For example, `nn.Linear` becomes `aten.mm` + `aten.add`, and `F.relu` becomes `aten.clamp_min`:

```python
ep = torch.export.export(model, args)
# Graph contains: aten.mm, aten.add, aten.clamp_min, etc.
```

This is closer to what hardware backends need. It's the right choice for deployment, AOTInductor, and ONNX export.

### Pre-Dispatch

With `pre_dispatch=True`, export captures higher-level ops that are closer to the user's code:

```python
ep = torch.export.export(model, args, pre_dispatch=True)
# Graph contains: aten.linear, aten.relu, etc.
```

Pre-dispatch preserves composite ops — you see `linear` instead of `mm + add`. This is useful for:

- **Graph analysis**: Understanding what the model does at a high level
- **Custom transformations**: Rewriting ops before decomposition
- **Debugging**: Matching graph nodes back to source code

### Converting Between IRs

You can convert from pre-dispatch to post-dispatch using `run_decompositions()`:

```python
ep_pre = torch.export.export(model, args, pre_dispatch=True)
ep_post = ep_pre.run_decompositions()
```

You cannot go in the other direction — decomposition is one-way. The typical workflow is to export at pre-dispatch for analysis, then decompose for deployment.

### Choosing the Right IR

| Use Case | IR Level | Why |
|----------|----------|-----|
| AOTInductor | Post-dispatch | Backend needs decomposed ops |
| ONNX export | Post-dispatch | Maps directly to ONNX ops |
| Graph analysis | Pre-dispatch | Higher-level, easier to read |
| Custom passes | Pre-dispatch | Operate on meaningful ops |
| Training export | Pre-dispatch | Preserve autograd-relevant ops |

---

## 10. Custom Ops in Export

Custom ops (defined via `torch.library`) need a **fake implementation** (also called a Meta implementation) for export to work. Export doesn't run the real computation — it traces symbolically, so it needs to know the output shapes and dtypes without executing the kernel.

### The Pattern

```python
# Step 1: Define the op
@torch.library.custom_op("mylib::relu_squared", mutates_args=())
def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2

# Step 2: Register the fake implementation
@relu_squared.register_fake
def relu_squared_fake(x):
    return torch.empty_like(x)
```

The fake implementation:
- Receives `FakeTensor` inputs (tensors with shapes/dtypes but no data)
- Must return tensors with the correct shapes and dtypes
- Should NOT do actual computation — just allocate empty tensors with the right metadata

### Without register_fake

If you skip `register_fake`, export fails:

```python
@torch.library.custom_op("mylib::bad_op", mutates_args=())
def bad_op(x: torch.Tensor) -> torch.Tensor:
    return x * 2

# This will fail:
# torch.export.export(model_using_bad_op, args)
# RuntimeError: mylib::bad_op does not have a fake impl
```

### Complex Output Shapes

When the output shape depends on input values (not just shapes), you need `torch.library.FakeTensorMode`:

```python
@torch.library.custom_op("mylib::nonzero_count", mutates_args=())
def nonzero_count(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(x.nonzero().shape[0])

@nonzero_count.register_fake
def nonzero_count_fake(x):
    ctx = torch.library.get_ctx()
    # Data-dependent output: we don't know the count at trace time
    u = ctx.new_dynamic_size()
    return torch.empty(u, dtype=torch.long, device=x.device)
```

---

## 11. Retraceability

An exported program can be re-exported (re-traced). This enables several workflows:

### Adding Decompositions

Export a model, then re-export with additional decompositions to break down composite ops:

```python
ep1 = torch.export.export(model, args)
# ep1 has high-level ops

ep2 = torch.export.export(ep1.module(), args)
# ep2 may have different decompositions
```

### Changing Dynamic Shape Constraints

Re-export with different dynamic shapes to tighten or loosen constraints:

```python
batch = Dim("batch", min=1, max=64)
ep1 = torch.export.export(model, args, dynamic_shapes={"x": {0: batch}})

# Later, tighten the constraint
batch_tight = Dim("batch", min=1, max=32)
ep2 = torch.export.export(ep1.module(), args, dynamic_shapes={"x": {0: batch_tight}})
```

### Composing Exported Programs

Re-tracing lets you compose multiple exported programs into a pipeline:

```python
class Pipeline(nn.Module):
    def __init__(self, encoder_ep, decoder_ep):
        super().__init__()
        self.encoder = encoder_ep.module()
        self.decoder = decoder_ep.module()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

pipeline_ep = torch.export.export(Pipeline(ep_enc, ep_dec), args)
```

### Caveats

- Re-tracing may produce different graphs if the model has Python-level logic
- Constraints from the original export are not automatically carried forward
- Custom ops need their fake implementations registered for the re-trace too

---

## 12. Strict vs Non-Strict Export

### Strict Mode (Default)

`strict=True` uses full symbolic tracing via Dynamo. Every Python operation is symbolically evaluated:

```python
ep = torch.export.export(model, args, strict=True)
```

Strict mode catches issues early:
- Data-dependent control flow → error
- Graph breaks → error
- Unsupported Python → error

This produces the most reliable exported programs.

### Non-Strict Mode

`strict=False` is more permissive. It allows some Python constructs that strict mode rejects:

```python
ep = torch.export.export(model, args, strict=False)
```

Non-strict mode:
- Uses a less aggressive tracing approach
- May allow some Python control flow (if it doesn't depend on tensor data)
- Can handle some patterns that cause graph breaks in strict mode
- May miss some issues that strict mode catches

### When to Use Each

| Scenario | Mode |
|----------|------|
| Production deployment | `strict=True` (catches all issues) |
| Iterative development | `strict=False` (get something working first) |
| Complex Python logic | `strict=False` (then migrate to strict) |
| Custom frameworks | `strict=False` (framework code may not trace) |
| Maximum reliability | `strict=True` (gold standard) |

The recommended workflow is to start with `strict=False` to get a working export, then switch to `strict=True` and fix any issues. Strict mode produces more optimizable graphs.

---

## 13. ExportBackwardSignature

For training-time export (experimental), `torch.export` can capture both forward and backward:

```python
# Experimental API
ep = torch.export.export_for_training(model, args)
```

The backward signature specifies:
- Which parameters have gradients in the output
- The mapping from output gradients back to parameter gradients
- Loss outputs that drive the backward pass

This is used by frameworks that need to export the training loop itself, not just inference. The API is still evolving — check the PyTorch docs for the latest status.

Key considerations:
- Autograd graphs are more complex than forward-only graphs
- In-place operations and aliasing add complications
- Not all models support training export yet

---

## 14. Serialization Format

### PT2 Archive Format

`torch.export.save()` produces a **PT2 archive** — a zip file containing:

```
model.pt2
├── data/                    # Serialized tensor data
│   ├── 0                    # Weight tensor 0
│   ├── 1                    # Weight tensor 1
│   └── ...
├── constants/               # Non-parameter constants
├── export_graph.json        # The FX graph (serialized)
├── version                  # Format version
└── extra/                   # Sample inputs, metadata
    └── sample_inputs.json
```

### Save and Load

```python
# Save
torch.export.save(ep, "model.pt2")

# Load
ep_loaded = torch.export.load("model.pt2")

# Run
result = ep_loaded.module()(input_tensor)
```

### Backward Compatibility

The serialization format is versioned. PyTorch maintains backward compatibility:
- Newer PyTorch can load older archives
- Older PyTorch may not load newer archives (forward compatibility is not guaranteed)
- The graph serialization uses a stable schema based on the ATen operator set

### Comparison with Other Formats

| Format | Use Case | Preserves Graph | Portable |
|--------|----------|----------------|----------|
| `torch.save` (pickle) | Checkpointing | No | No (needs source) |
| `torch.export.save` (PT2) | Deployment | Yes | Yes |
| `torch.package` | Hermetic archive | Source code | Yes |
| ONNX | Cross-framework | Yes | Yes |

---

## 15. Practical Debugging Workflow

When exporting a complex model, issues are common. Here's a systematic approach:

### Step 1: Try Export

```python
try:
    ep = torch.export.export(model, args)
    print("Export succeeded!")
except Exception as e:
    print(f"Export failed: {e}")
```

### Step 2: Use draft_export

```python
ep, report = torch.export.draft_export(model, args)
for issue in report:
    print(f"Issue: {issue}")
```

### Step 3: Fix Issues One by One

**Data-dependent control flow** → Replace with `torch.cond`:

```python
# Before (fails)
def forward(self, x):
    if x.sum() > 0:  # data-dependent
        return x * 2
    return x * -1

# After (works)
def forward(self, x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x * -1,
        (x,),
    )
```

**Missing fake implementation** → Add `register_fake`:

```python
@my_custom_op.register_fake
def my_custom_op_fake(x):
    return torch.empty_like(x)
```

**Dynamic shape constraint violation** → Add explicit constraints:

```python
batch = Dim("batch", min=1, max=256)
ep = torch.export.export(
    model, args,
    dynamic_shapes={"x": {0: batch}},
)
```

**Unsupported Python construct** → Refactor or use `strict=False`:

```python
# Try non-strict first
ep = torch.export.export(model, args, strict=False)
# Then work toward strict=True
```

### Step 4: Validate

```python
# Test with original inputs
original_result = model(*args)
exported_result = ep.module()(*args)
assert torch.allclose(original_result, exported_result)

# Test with different shapes (if dynamic)
new_args = (torch.randn(8, 10),)  # different batch size
exported_result = ep.module()(*new_args)
```

### Step 5: Save

```python
torch.export.save(ep, "model.pt2")
ep_loaded = torch.export.load("model.pt2")
```

---

## 16. Upstream Updates (July 3-6, 2026)

Recent PyTorch changes relevant to export and compilation:

### Fix Strict Export of Unregistered Parameters (#185728)

Previously, `strict=True` export could fail when a model referenced parameters that weren't registered via `register_parameter`. This fix handles the case where tensors used as parameters are regular attributes (not `nn.Parameter`), ensuring they're correctly captured as constants rather than causing a tracing failure.

### Inductor FFT f16/bf16 Support (#180766)

TorchInductor now supports FFT operations in float16 and bfloat16. Previously, FFT ops were silently upcasted to float32, causing unexpected memory usage and performance degradation. This is relevant to export because the post-dispatch IR may contain decomposed FFT ops that Inductor needs to handle.

### max_autotune Under CUDA Graph (#179246)

The `max_autotune` mode in Inductor now works correctly when CUDA Graphs are enabled. This fixes a class of bugs where autotuning would select a kernel configuration during tracing that was incompatible with CUDA Graph capture. For exported models deployed via AOTInductor, this means more reliable performance tuning.

### Inductor TF32 Advisory Suppression (#185541)

Inductor no longer emits spurious TF32 advisory warnings during compilation. Previously, every compilation would warn about TF32 matmul precision even when the user had explicitly set the precision policy. This reduces noise in export/compile logs.

### ShardedTensor Device-Agnostic Transfers (#187939)

`ShardedTensor` now supports device-agnostic transfers, enabling export of models that use tensor parallelism. Previously, exporting a model with sharded parameters required all shards to be on the same device type. This unblocks export workflows for distributed models.

---

## Key Takeaways

1. **`ExportedProgram` is fully self-contained** — graph, weights, constraints, and metadata in one object
2. **Graph signature** tells you exactly what each input/output represents (parameter, buffer, user data)
3. **`torch.cond` and `torch.while_loop`** make control flow explicit — the graph captures both branches
4. **`Dim` API** with shared dims and `Dim.AUTO` gives fine-grained control over dynamic shapes
5. **`draft_export`** is your best debugging friend — it reports all issues instead of failing on the first
6. **Pre-dispatch IR** preserves high-level ops; **post-dispatch IR** decomposes for backends
7. **Custom ops need `register_fake`** — without it, export can't trace through your op
8. **Strict mode is the gold standard** — use non-strict for development, strict for production
9. **PT2 archive** is the serialization format — versioned, portable, self-contained

---

### Further Resources

- [torch.export Documentation](https://pytorch.org/docs/stable/export.html) — official API reference
- [Export Tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html) — step-by-step walkthrough
- [Module 08 — torch.compile](../08_torch_compile/) — the compilation pipeline that export feeds into
- [Module 11 — Export & Deployment](../11_export_deploy/) — basics of export and deployment paths
- [Module 35 — The Dispatcher](../35_dispatcher/) — how ops are dispatched (relevant to custom ops)

---

<div align="center">

[← Previous Module (Custom C++ Extensions)](../36_cpp_extensions/) | [🏠 Home](../README.md) | Next Module → (none)

**Notebook**: [`37_export_deep_dive.ipynb`](../notebooks/37_export_deep_dive.ipynb)

</div>
