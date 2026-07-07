<div align="center">

[← Previous Module (torch.export Deep Dive)](../37_export_deep_dive/) | [🏠 Home](../README.md) | Next Module → (none)

</div>

---

# Module 38: Compiled Autograd & AOTAutograd

> **Prerequisites**: [Module 03 — Autograd](../03_autograd/), [Module 04 — Neural Networks](../04_neural_networks/), [Module 08 — torch.compile](../08_torch_compile/), [Module 35 — The Dispatcher](../35_dispatcher/)
> **Time**: ~3 hours
> **Files**: `aot_autograd_explained.py`, `compiled_backward.py`

---

## Table of Contents

1. [The Problem: Eager Backward is Slow](#1-the-problem-eager-backward-is-slow)
2. [AOTAutograd — Ahead-of-Time Autograd](#2-aotautograd--ahead-of-time-autograd)
3. [How AOTAutograd Works](#3-how-aotautograd-works)
4. [The Joint Graph](#4-the-joint-graph)
5. [min_cut_rematerialization_partition](#5-min_cut_rematerialization_partition)
6. [Compiled Autograd](#6-compiled-autograd)
7. [Viewing the Forward and Backward Graphs](#7-viewing-the-forward-and-backward-graphs)
8. [What AOTAutograd Enables](#8-what-aotautograd-enables)
9. [Activation Memory in AOTAutograd](#9-activation-memory-in-aotautograd)
10. [AOTAutograd vs Standard Autograd](#10-aotautograd-vs-standard-autograd)
11. [Functorch and AOTAutograd](#11-functorch-and-aotautograd)
12. [Debugging AOTAutograd](#12-debugging-aotautograd)
13. [Upstream Updates (July 5-7, 2026)](#13-upstream-updates-july-5-7-2026)

---

## 1. The Problem: Eager Backward is Slow

In eager mode, PyTorch's backward pass executes operations one-by-one through the C++ autograd engine. Each op launches independently: a kernel for the matmul gradient, another for the activation gradient, another for the bias gradient, and so on. Every launch carries overhead — kernel dispatch, memory allocation, synchronization.

`torch.compile` solves this for the **forward** pass. Dynamo captures the forward graph, Inductor fuses and optimizes it, and the result runs as an efficient kernel sequence. But by default, the backward pass is still eager. The autograd engine builds the backward graph at runtime by walking the chain of `grad_fn` objects, and then executes each node sequentially.

This means a compiled model is only half-optimized:

```
Forward:   [Compiled — fused, optimized, fast]
Backward:  [Eager — one op at a time, dispatch overhead, no fusion]
```

For training workloads, the backward pass typically takes 2-3x longer than the forward (more ops, more memory traffic). Leaving it unoptimized is a major missed opportunity. AOTAutograd and Compiled Autograd fix this by bringing the backward pass into the compilation pipeline.

The fundamental tension:
- Standard autograd must be **general** — it handles arbitrary dynamic graphs, in-place ops, hooks, and complex control flow
- Compilation wants **static** graphs — known shapes, known ops, no Python callbacks
- AOTAutograd bridges this by trading generality for performance: trace the backward at compile time, producing a static graph that Inductor can optimize

---

## 2. AOTAutograd — Ahead-of-Time Autograd

AOTAutograd's core idea is simple: instead of building the backward graph at runtime, trace **both** forward and backward at compile time. This produces two FX graphs — a forward graph and a backward graph — and both get independently optimized by the backend (typically Inductor).

```
User Model → [AOTAutograd] → Forward Graph (FX) + Backward Graph (FX)
                                      ↓                    ↓
                                [Inductor]           [Inductor]
                                      ↓                    ↓
                               Optimized Fwd        Optimized Bwd
```

This is what happens under the hood when you call `torch.compile` on a model that requires gradients. Dynamo captures the forward operations, then hands them to AOTAutograd, which:

1. Traces the forward to get an FX graph
2. Runs `torch.autograd.grad()` on the traced forward to produce the backward
3. Combines them into a **joint graph**
4. Partitions the joint graph into separate forward and backward graphs
5. Passes each graph to the backend compiler

The result: both forward and backward run at compiled speed, with cross-op fusion, memory planning, and kernel optimization.

AOTAutograd is not a user-facing API in most workflows — it's a component inside the `torch.compile` pipeline. But understanding it is essential for debugging compilation issues, understanding memory behavior, and using advanced features like custom partitioning.

---

## 3. How AOTAutograd Works

Here's the step-by-step process AOTAutograd follows:

### Step 1: Functionalization

The input model may contain mutations (in-place ops like `x.add_(1)`) and views (ops like `x.view(-1)` that share memory). These are problematic for tracing because they create implicit dependencies.

Functionalization rewrites the model to eliminate mutations and views:
- `x.add_(1)` becomes `x = x + 1` (out-of-place)
- `x.view(-1)` becomes `x.reshape(-1)` with explicit copy semantics

This produces a **pure functional** model — no side effects, no aliasing. Every operation takes inputs and produces new outputs.

### Step 2: Trace Forward with make_fx

Using functorch's `make_fx`, AOTAutograd traces the functionalized forward pass. This produces an FX graph where every node is an ATen operation:

```python
# Conceptually:
from torch.fx.experimental.proxy_tensor import make_fx

fx_forward = make_fx(functionalized_model)(*example_inputs)
```

The trace uses `FakeTensor` mode — no actual computation happens. Instead, tensor metadata (shape, dtype, device) flows through the graph, recording every operation.

### Step 3: Derive Backward with torch.autograd.grad

With the traced forward graph, AOTAutograd calls `torch.autograd.grad()` on it to produce the backward operations. This is possible because the forward graph is itself a differentiable program — every ATen op has a registered derivative formula.

```python
# Conceptually:
outputs = fx_forward(*inputs)
grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs)
```

This produces the backward ops as additional nodes in the graph.

### Step 4: Build the Joint Graph

The forward ops and backward ops are combined into a single **joint graph**. This graph takes the original inputs and grad_outputs, and produces both the forward outputs and the gradients:

```
Joint Graph:
  Inputs: [x, weight, bias, grad_output]
  Forward ops: linear, relu, ...
  Backward ops: relu_backward, linear_backward, ...
  Outputs: [forward_output, grad_x, grad_weight, grad_bias]
```

Having everything in one graph is critical — it allows the partitioner to reason about which forward activations are needed by which backward ops.

### Step 5: Partition into Forward and Backward

The joint graph is split into two separate graphs. The key decision: which intermediate tensors from the forward need to be **saved** for the backward?

The partitioner inserts "save" nodes at the boundary — the forward graph's extra outputs become the backward graph's extra inputs. These are the **saved tensors** (equivalent to what `ctx.save_for_backward()` does in a manual `autograd.Function`).

### Step 6: Compile Each Graph

Both graphs are independently passed to the backend compiler (Inductor). Each gets the full optimization treatment: operator fusion, memory planning, kernel generation, and code generation.

---

## 4. The Joint Graph

Before partitioning, there's one unified graph containing both forward and backward operations. Understanding this graph is key to understanding AOTAutograd's behavior.

For a simple linear model `y = relu(Wx + b)`:

```
Joint Graph:
  %x        : input
  %weight   : parameter
  %bias     : parameter
  %grad_out : gradient of loss w.r.t. output

  # Forward
  %mm       = aten.mm(%x, %weight.t())
  %add      = aten.add(%mm, %bias)
  %relu     = aten.relu(%add)

  # Backward
  %relu_bwd = aten.threshold_backward(%grad_out, %relu, 0)
  %grad_b   = aten.sum(%relu_bwd, dim=0)
  %grad_w   = aten.mm(%relu_bwd.t(), %x)
  %grad_x   = aten.mm(%relu_bwd, %weight)

  return (%relu, %grad_x, %grad_w, %grad_b)
```

Notice that the backward ops reference forward tensors:
- `threshold_backward` needs `%relu` (to know which elements were zeroed)
- `grad_w` computation needs `%x` (the original input)
- `grad_x` computation needs `%weight`

The partitioner must decide: should `%relu` be saved from forward, or recomputed during backward? Should `%x` be saved? These decisions directly control memory usage.

---

## 5. min_cut_rematerialization_partition

The default partitioner in AOTAutograd uses a **min-cut algorithm** to decide which activations to save vs recompute. This is the same concept as activation checkpointing, but applied automatically at the operator level.

### The Tradeoff

Every forward activation used by the backward pass presents a choice:
- **Save it**: use memory to store it from forward until backward needs it
- **Recompute it**: don't save it, but recompute it during backward (uses extra FLOPs)

The min-cut partitioner formulates this as a graph cut problem:
- Nodes have **costs** (memory for saving, FLOPs for recomputing)
- The algorithm finds the cut that **minimizes total memory** while respecting a compute budget

### What Gets Saved vs Recomputed

The partitioner uses heuristics about operation costs:

**Typically saved** (expensive to recompute):
- Matrix multiplications (`aten.mm`, `aten.bmm`)
- Convolutions (`aten.convolution`)
- Attention scores
- Any op with high FLOP count

**Typically recomputed** (cheap to recompute):
- Element-wise ops: `relu`, `add`, `mul`, `sigmoid`
- Reductions: `sum`, `mean`
- Type conversions: `to`, `float`
- Shape ops: `view`, `reshape`, `transpose`

### Example

For a model with `y = relu(linear(x))`:

Without rematerialization (save everything):
```
Forward saves: [mm_result, add_result, relu_result, x, weight]
Memory: 5 tensors
```

With min-cut rematerialization:
```
Forward saves: [mm_result, x, weight]  # relu/add are recomputed
Memory: 3 tensors
Backward recomputes: add = mm_result + bias; relu = clamp(add, 0)
```

The relu and add are cheap to recompute (element-wise), so the partitioner drops them from saved tensors. The matmul result is expensive to recompute, so it's saved.

### Controlling the Partitioner

You can influence partitioner behavior:

```python
# Force specific ops to be saved (not recomputed)
torch._functorch.config.ban_recompute_ops = ["aten.mm"]

# Force specific ops to be recomputed (not saved)
torch._functorch.config.force_recompute_ops = ["aten.relu"]
```

For debugging:
```python
torch._functorch.config.debug_partitioner = True
```

This prints which ops are saved, which are recomputed, and why.

---

## 6. Compiled Autograd

Compiled Autograd goes a step further than AOTAutograd. While AOTAutograd compiles the forward and backward **graphs**, the autograd engine itself is still in C++ and dispatches backward ops one-by-one. Compiled Autograd compiles the **engine's execution** — the entire backward pass becomes one compiled unit.

### Enabling Compiled Autograd

```python
torch._dynamo.config.compiled_autograd = True
model = torch.compile(model)

# Both forward and backward are compiled
loss = model(x).sum()
loss.backward()  # The backward is ALSO compiled
```

### What Changes

Without Compiled Autograd:
```
loss.backward()
  → C++ autograd engine walks grad_fn chain
  → Dispatches MulBackward0 → eager kernel
  → Dispatches AddmmBackward0 → eager kernel
  → Dispatches ReluBackward0 → eager kernel
  → ... (one dispatch per op)
```

With Compiled Autograd:
```
loss.backward()
  → Dynamo captures the entire backward execution
  → Produces one FX graph for the full backward
  → Inductor compiles it into fused kernels
  → Runs as optimized kernel sequence
```

The key difference from AOTAutograd: AOTAutograd traces the backward at the op level using `torch.autograd.grad()`. Compiled Autograd captures the actual autograd engine's execution, including any hooks, accumulation logic, and multi-output handling.

### When to Use Compiled Autograd

Compiled Autograd is most beneficial when:
- The backward pass has many small operations that can be fused
- You're training on GPU and kernel launch overhead is significant
- The model structure is static (same operations every iteration)

It's less beneficial when:
- The model uses complex autograd hooks
- The backward graph changes dynamically
- You're on CPU where kernel launch overhead is minimal

### Interaction with torch.compile

Compiled Autograd works in conjunction with `torch.compile`. When both are enabled:

1. `torch.compile` captures the forward via Dynamo
2. AOTAutograd traces forward+backward and partitions them
3. Compiled Autograd captures the autograd engine's backward execution
4. Inductor compiles everything

The result is that both the forward and backward passes run as optimized, fused kernel sequences.

---

## 7. Viewing the Forward and Backward Graphs

For debugging and understanding, you can inspect the graphs that AOTAutograd produces using the low-level `aot_function` API:

```python
from torch._functorch.aot_autograd import aot_function

def inspect_compiler(gm, example_inputs):
    print("=" * 60)
    print("Graph:")
    gm.graph.print_tabular()
    print(f"Number of nodes: {len(list(gm.graph.nodes))}")
    return gm  # return the graph module as-is (no optimization)

def my_fn(x, weight):
    return torch.relu(x @ weight)

compiled = aot_function(
    my_fn,
    fw_compiler=inspect_compiler,  # called with forward graph
    bw_compiler=inspect_compiler,  # called with backward graph
)

x = torch.randn(4, 8, requires_grad=True)
w = torch.randn(8, 4, requires_grad=True)

out = compiled(x, w)
out.sum().backward()
```

This prints both the forward and backward FX graphs, showing every operation and the data flow between them.

### Using TORCH_LOGS

For `torch.compile`, use environment variables:

```bash
# See AOTAutograd forward/backward graphs
TORCH_LOGS="aot" python train.py

# See generated Inductor code
TORCH_LOGS="output_code" python train.py

# See everything
TORCH_LOGS="aot,output_code,graph_breaks" python train.py
```

### Programmatic Logging

```python
import logging
torch._logging.set_logs(aot=logging.DEBUG)
```

This produces verbose output showing the joint graph, the partitioning decisions, and the final forward/backward graphs.

---

## 8. What AOTAutograd Enables

By compiling the backward pass alongside the forward, AOTAutograd unlocks optimizations that are impossible in eager mode:

### Operator Fusion Across Forward/Backward Boundary

In eager mode, the forward and backward are separate execution phases. The compiler can't fuse a forward activation with its corresponding backward derivative. With AOTAutograd, both are visible in a single compilation unit, enabling cross-boundary fusion.

### Dead Code Elimination in Backward

If a gradient is unused (e.g., a parameter's `requires_grad=False`), AOTAutograd can eliminate the entire backward subgraph for that parameter. In eager mode, the autograd engine would still compute it.

### Constant Folding in Backward

Operations that depend only on constants (like weight shapes) can be folded at compile time. The backward graph is simplified before code generation.

### Memory Planning

Inductor can plan memory allocation for the entire backward pass upfront. In eager mode, each backward op allocates its output independently. With compilation, Inductor can reuse memory buffers across non-overlapping operations.

### Kernel Fusion

Small backward operations (element-wise gradients, accumulations) are fused into larger kernels. Instead of launching 20 separate kernels for 20 backward ops, Inductor might generate 3-4 fused kernels that do the same work with much less launch overhead.

---

## 9. Activation Memory in AOTAutograd

The partitioner controls what "saved tensors" are passed from forward to backward. This directly determines training memory consumption — fewer saved tensors means less memory, but potentially more recomputation.

### Viewing Saved Tensors

```python
from torch._functorch.aot_autograd import aot_function

saved_tensors_count = []

def counting_compiler(gm, example_inputs):
    # Count extra outputs in forward = saved tensors
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    n_outputs = len(output_node.args[0])
    saved_tensors_count.append(n_outputs)
    return gm

compiled = aot_function(fn, fw_compiler=counting_compiler, bw_compiler=lambda gm, _: gm)
```

### Memory Impact

For a transformer layer with hidden_size=1024, batch_size=32, seq_len=512:

| Activation | Size | Saved? (min-cut) |
|-----------|------|-------------------|
| QKV projection output | 48 MB | Yes (matmul) |
| Attention scores | 32 MB | Yes (matmul) |
| Post-softmax attention | 32 MB | Yes (expensive) |
| ReLU mask | 2 MB | No (recomputed) |
| LayerNorm intermediate | 4 MB | No (recomputed) |
| Residual add result | 8 MB | No (recomputed) |

The min-cut partitioner saves roughly 112 MB instead of 126 MB — a 11% reduction for this layer. Across a 24-layer model, that's over 300 MB saved.

### Interaction with Activation Checkpointing

AOTAutograd's rematerialization is **complementary** to user-level activation checkpointing (`torch.utils.checkpoint`). Checkpointing operates at the module level (recompute an entire layer), while AOTAutograd's min-cut operates at the operator level (recompute individual cheap ops).

You can use both:
```python
# Module-level: recompute entire transformer layers
model = checkpoint_wrapper(model)

# Op-level: within each layer, AOTAutograd's min-cut
# further reduces saved tensors
model = torch.compile(model)
```

---

## 10. AOTAutograd vs Standard Autograd

| Feature | Standard Autograd | AOTAutograd |
|---------|------------------|-------------|
| Graph built | Runtime (during forward) | Compile time (ahead-of-time) |
| Backward optimized | No (eager dispatch) | Yes (Inductor compiles backward) |
| Memory planning | Manual (user calls checkpoint) | Automatic (min-cut partitioner) |
| Kernel fusion | None (one kernel per op) | Yes (Inductor fuses backward ops) |
| Works with compile | Forward only | Forward + backward |
| Dynamic graphs | Fully supported | Requires recompilation on change |
| Autograd hooks | Fully supported | Limited support |
| In-place ops | Fully supported | Functionalized (no in-place) |
| Debugging | Easy (Python stack traces) | Harder (compiled code) |
| Overhead | None | Compilation cost (amortized) |

### When Standard Autograd is Better

- **Debugging**: When you need Python-level stack traces and step-through debugging
- **Dynamic models**: Models where the computation graph changes every iteration (e.g., tree-RNNs)
- **Complex hooks**: Models that rely heavily on autograd hooks for gradient manipulation
- **One-off computations**: When compilation cost isn't amortized (few iterations)

### When AOTAutograd is Better

- **Training throughput**: When you need maximum training speed
- **Large models**: Where kernel launch overhead is a bottleneck
- **Static models**: Where the graph doesn't change between iterations (transformers, CNNs)
- **Memory-constrained**: Where automatic rematerialization helps fit larger batches

---

## 11. Functorch and AOTAutograd

AOTAutograd is built on top of functorch's primitives. Understanding this connection clarifies how the system works.

### make_fx

`make_fx` is functorch's functional tracing tool. It runs a function with `FakeTensor` inputs and records every ATen operation into an FX graph:

```python
from torch.fx.experimental.proxy_tensor import make_fx

def fn(x):
    return torch.relu(x) + 1

gm = make_fx(fn)(torch.randn(4))
print(gm.graph)
```

AOTAutograd uses `make_fx` to trace the forward pass.

### torch.autograd.grad on Traced Graphs

The traced forward graph is itself differentiable — every ATen op has registered derivatives. AOTAutograd calls `torch.autograd.grad()` on the traced graph to produce backward operations:

```python
def joint_fn(primals, tangents):
    # Forward
    output = traced_forward(*primals)
    # Backward (using autograd on the trace)
    grads = torch.autograd.grad(output, primals, tangents)
    return output, grads

joint_graph = make_fx(joint_fn)(primals, tangents)
```

This is the joint graph — forward and backward combined.

### grad and vmap

Functorch's `grad` transform is closely related. While `torch.autograd.grad` computes gradients imperatively, functorch's `grad` wraps a function to return its gradient:

```python
from torch.func import grad

def loss_fn(x):
    return (x ** 2).sum()

grad_fn = grad(loss_fn)
gradient = grad_fn(torch.randn(4))
```

AOTAutograd uses the imperative `torch.autograd.grad` rather than functorch's `grad` transform, but the mathematical operation is the same.

---

## 12. Debugging AOTAutograd

### Environment Variables

```bash
# See forward and backward graphs
TORCH_LOGS="aot" python script.py

# See generated Inductor code for both graphs
TORCH_LOGS="output_code" python script.py

# See partitioner decisions
TORCH_LOGS="aot" python script.py
# Look for lines containing "partition" in the output

# Combined: full picture
TORCH_LOGS="aot,output_code,graph_breaks" python script.py
```

### Partitioner Debugging

```python
torch._functorch.config.debug_partitioner = True
```

This prints which tensors the partitioner decided to save vs recompute, and the cost estimates that drove those decisions.

### Common Issues

**Graph breaks in backward**: If Dynamo encounters an unsupported operation during backward tracing, it inserts a graph break. This fragments the backward into multiple compiled regions with eager transitions between them.

```python
# Check for graph breaks
torch._dynamo.config.verbose = True
# Look for "graph break" in output
```

**Shape mismatch errors**: These occur when the backward graph expects a different shape than what the forward produces. Usually caused by dynamic shapes or data-dependent operations.

```python
# Use TORCH_LOGS to see the shapes at each node
TORCH_LOGS="aot,dynamic" python script.py
```

**Functionalization errors**: Some in-place operations can't be functionalized. The error message will mention `functionalize` or `FunctionalTensorWrapper`.

**Recompilation storms**: If the model's shapes change frequently, AOTAutograd recompiles the forward and backward graphs each time. Use `torch._dynamo.config.cache_size_limit` and dynamic shapes to mitigate.

### Comparing Eager vs Compiled Results

```python
model_eager = MyModel()
model_compiled = torch.compile(MyModel())

# Same weights
model_compiled.load_state_dict(model_eager.state_dict())

x = torch.randn(4, 8, requires_grad=True)
x_copy = x.clone().detach().requires_grad_(True)

# Forward
y_eager = model_eager(x)
y_compiled = model_compiled(x_copy)

# Backward
y_eager.sum().backward()
y_compiled.sum().backward()

# Compare gradients
print(torch.allclose(x.grad, x_copy.grad))  # Should be True
```

---

## 13. Upstream Updates (July 5-7, 2026)

Recent changes relevant to compiled autograd and AOTAutograd:

### Inductor Accumulator addmm Preservation (#184296)

The Inductor backend now preserves accumulator precision for `addmm` operations during backward pass compilation. Previously, intermediate accumulations could lose precision when Inductor fused multiple matmul backward ops. This fix ensures that gradient accumulation in compiled backward passes matches eager autograd numerics, particularly important for large-scale training where small numerical differences compound across many steps.

### standalone_compile Fake Mode Fix (#185638)

Fixed an issue where `standalone_compile` would fail when entering fake mode for AOTAutograd tracing. The bug manifested as shape inference errors during joint graph construction — fake tensors would lose their symbolic shape information when passed through certain decomposition rules. This fix ensures that standalone compilation (used for ahead-of-time compilation workflows) correctly maintains shape metadata throughout the AOTAutograd pipeline.

### CPU Outer-Loop Buffer Reuse (#185855)

Inductor's CPU backend now supports buffer reuse across outer loop iterations in compiled backward graphs. When the backward pass contains reduction operations that produce intermediate buffers, those buffers are now recycled rather than reallocated each iteration. This reduces memory allocation pressure during compiled backward passes on CPU, particularly for models with many small reduction ops in their gradient computation.

### Avoid CUDA Init in CPU Compile (#186403)

AOTAutograd and Inductor no longer trigger CUDA initialization when compiling models on CPU. Previously, importing certain compilation modules would call `torch.cuda.is_available()` or `torch.cuda.device_count()`, which initializes the CUDA runtime. This was wasteful for CPU-only workloads and caused failures in environments without GPU drivers. The fix lazily gates CUDA-specific code paths.

### Stateless RNG Clone Fix (#188495)

Fixed a bug in AOTAutograd's functionalization pass where stateless RNG operations (used by dropout and similar stochastic layers) were not correctly cloned during joint graph construction. The symptom was that compiled training would produce different random masks in forward vs backward, leading to incorrect gradients for models with dropout. The fix ensures that the RNG state is properly snapshotted at the partition boundary.

### NativeRT Warp Size Query for Triton (#188881)

The NativeRT inference engine now correctly queries the GPU's warp size when running Triton-generated kernels from AOTInductor. This is relevant to AOTAutograd because compiled backward passes may be deployed via AOTInductor for inference-time gradient computation (e.g., in differentiable rendering or physics simulation). Previously, NativeRT assumed a warp size of 32, which fails on non-NVIDIA hardware.

---

## Key Takeaways

1. **Eager backward is the bottleneck** — `torch.compile` alone only optimizes the forward; the backward is still eager dispatch with per-op overhead
2. **AOTAutograd traces both passes at compile time** — it uses `make_fx` + `torch.autograd.grad` to produce forward and backward FX graphs that Inductor can optimize
3. **The joint graph is central** — forward and backward ops start in one graph, then the partitioner splits them, deciding what to save vs recompute
4. **min-cut partitioner automates checkpointing** — instead of manually wrapping layers in `checkpoint()`, the partitioner does operator-level save/recompute decisions based on cost
5. **Compiled Autograd goes further** — it compiles the autograd engine's execution itself, capturing hooks, accumulation, and multi-output handling
6. **Use `aot_function` to inspect graphs** — pass custom `fw_compiler` and `bw_compiler` callbacks to see exactly what AOTAutograd produces
7. **Debugging uses TORCH_LOGS** — `TORCH_LOGS="aot"` shows graphs, `debug_partitioner=True` shows save/recompute decisions
8. **Both forward and backward run at Inductor speed** — operator fusion, memory planning, and kernel optimization apply to the backward pass too
9. **Complementary to activation checkpointing** — AOTAutograd's min-cut is op-level; user checkpointing is module-level; they compose

---

### Further Resources

- [AOTAutograd Documentation](https://pytorch.org/docs/stable/torch.compiler_aot_autograd.html) — official API reference
- [Compiled Autograd Tutorial](https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html) — step-by-step walkthrough
- [Module 03 — Autograd](../03_autograd/) — foundational autograd concepts
- [Module 08 — torch.compile](../08_torch_compile/) — the compilation pipeline
- [Module 16 — Activation Checkpointing](../16_activation_checkpointing/) — manual memory optimization
- [Module 35 — The Dispatcher](../35_dispatcher/) — how ops are dispatched

---

<div align="center">

[← Previous Module (torch.export Deep Dive)](../37_export_deep_dive/) | [🏠 Home](../README.md) | Next Module → (none)

**Notebook**: [`38_compiled_autograd.ipynb`](../notebooks/38_compiled_autograd.ipynb)

</div>
