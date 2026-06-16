<div align="center">

[← Previous Module](../22_llm_recipes/) | [🏠 Home](../README.md) | Next Module →

</div>

---

> **Module 23** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 04 — Neural Networks](../04_neural_networks/), [Module 08 — torch.compile](../08_torch_compile/)
> **Time to complete:** ~3 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — torch.fx theory, IR, passes, patterns |
| [`fx_basics.py`](fx_basics.py) | Symbolic tracing, graph inspection, ShapeProp |
| [`graph_passes.py`](graph_passes.py) | Graph transformations, pattern matching, Interpreter, Transformer |

---

# torch.fx — Graph-Level Model Transformation

## Table of Contents

1. [What is torch.fx?](#1-what-is-torchfx)
2. [Symbolic Tracing](#2-symbolic-tracing)
3. [The FX Graph IR](#3-the-fx-graph-ir)
4. [Graph Inspection](#4-graph-inspection)
5. [Graph Transformation — Adding, Removing, Replacing Nodes](#5-graph-transformation--adding-removing-replacing-nodes)
6. [Pattern Matching and Replacement](#6-pattern-matching-and-replacement)
7. [Practical Pass Examples](#7-practical-pass-examples)
8. [ShapeProp](#8-shapeprop)
9. [graph.lint()](#9-graphlint)
10. [torch.fx.Interpreter](#10-torchfxinterpreter)
11. [torch.fx.Transformer](#11-torchfxtransformer)
12. [FX in the Compilation Stack](#12-fx-in-the-compilation-stack)
13. [Upstream Updates (June 15–16, 2026)](#13-upstream-updates-june-15-16-2026)
14. [Summary](#14-summary)

---

## 1. What is torch.fx?

`torch.fx` is a **Python-to-Python transformation framework** for PyTorch. It lets you:

1. **Capture** a PyTorch module's forward logic into a graph-based intermediate representation (IR)
2. **Inspect** and **modify** that graph programmatically
3. **Generate** a new, executable PyTorch module from the modified graph

```
┌─────────────┐    symbolic_trace    ┌──────────────┐    transform    ┌──────────────┐
│  nn.Module   │ ──────────────────▶ │  FX Graph IR │ ─────────────▶ │  FX Graph IR │
│  (Python)    │                     │  (nodes)     │                │  (modified)  │
└─────────────┘                     └──────────────┘                └──────┬───────┘
                                                                          │
                                                                   graph.recompile()
                                                                          │
                                                                   ┌──────▼───────┐
                                                                   │ GraphModule  │
                                                                   │ (executable) │
                                                                   └──────────────┘
```

**Where FX is used:**
- **torch.compile** — Dynamo captures FX graphs, AOTAutograd transforms them, Inductor lowers them
- **Quantization** — FX graph mode quantization inserts observers and rewrites ops
- **Distributed** — Tensor Parallel, Pipeline Parallel, and FSDP use FX passes for graph partitioning
- **Custom optimizations** — fuse ops, eliminate dead code, add profiling

The key advantage over tracing with `torch.jit.trace` is that FX operates at the **Python level** — the graph is pure Python, the transformations are pure Python, and the output is a standard `nn.Module` subclass.

---

## 2. Symbolic Tracing

### Basic Tracing

```python
import torch
import torch.nn as nn
import torch.fx

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = MyModel()
traced = torch.fx.symbolic_trace(model)
```

`symbolic_trace` executes `forward()` with **Proxy** values instead of real tensors. Proxies record every operation, building a graph that represents the computation.

The returned `traced` is a `GraphModule` — a subclass of `nn.Module` with:
- `traced.graph` — the `Graph` object (node-level IR)
- `traced.code` — auto-generated Python source code
- `traced.forward` — the compiled forward method

### What Gets Captured

```python
print(traced.code)
# def forward(self, x):
#     linear1 = self.linear1(x);  x = None
#     relu = torch.relu(linear1);  linear1 = None
#     linear2 = self.linear2(relu);  relu = None
#     return linear2
```

### Limitations of Symbolic Tracing

Symbolic tracing has fundamental limitations because it traces with proxy values, not real data:

| Limitation | Example | Why It Fails |
|-----------|---------|-------------|
| Data-dependent control flow | `if x.sum() > 0:` | Proxy doesn't have a real value to branch on |
| Dynamic shapes | `x[:, :n]` where `n` is runtime-determined | Proxy doesn't have concrete shapes |
| Non-torch Python ops | `print(x.shape)`, list comprehensions over tensors | Proxies don't support arbitrary Python |
| Non-`forward` methods | `self.helper(x)` not called from forward | Only forward is traced |

**Workarounds:**

```python
# 1. Use torch.fx.wrap to mark functions as "leaf" (not traced into)
@torch.fx.wrap
def my_custom_op(x):
    if x.sum() > 0:
        return x * 2
    return x

# 2. Use concrete_args to fix certain inputs
traced = torch.fx.symbolic_trace(model, concrete_args={"training": False})
```

For models with dynamic control flow, `torch.compile` (Dynamo) is preferred — it handles breaks and recompiles automatically.

---

## 3. The FX Graph IR

An FX `Graph` is a DAG (directed acyclic graph) of `Node` objects. Each node represents one operation.

### Node Operations

| `node.op` | Meaning | `node.target` | Example |
|-----------|---------|---------------|---------|
| `placeholder` | Function input | Parameter name | `x` |
| `get_attr` | Access self.attr | Attribute path (string) | `self.weight` |
| `call_function` | Free function call | The function itself | `torch.relu`, `operator.add` |
| `call_method` | Method on a value | Method name (string) | `.view()`, `.relu()` |
| `call_module` | Call a submodule | Module path (string) | `self.linear1` |
| `output` | Return value | `"output"` | The return node |

### Node Anatomy

Every node has:
- **`op`** — one of the six operations above
- **`name`** — unique identifier (e.g., `"relu"`, `"linear1"`)
- **`target`** — what to call (function, method name, module path, or attribute path)
- **`args`** — positional arguments (tuple of Nodes or constants)
- **`kwargs`** — keyword arguments (dict)
- **`users`** — dict of downstream nodes that consume this node's output

```python
graph = traced.graph
for node in graph.nodes:
    print(f"op={node.op:15s} name={node.name:15s} target={node.target}")
    print(f"  args={node.args}  kwargs={node.kwargs}")
    print(f"  users={list(node.users.keys())}")
```

### The Graph as a Linked List

FX nodes form a doubly-linked list in execution order. You can iterate forward (`graph.nodes`) or navigate with `node.prev` and `node.next`.

```
placeholder:x → call_module:linear1 → call_function:relu → call_module:linear2 → output
```

### Users and Dependencies

```python
for node in graph.nodes:
    if node.op == "call_function" and node.target == torch.relu:
        # Who uses the output of relu?
        for user in node.users:
            print(f"relu output used by: {user.name} ({user.op})")
        # What are relu's inputs?
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                print(f"relu input from: {arg.name} ({arg.op})")
```

---

## 4. Graph Inspection

### Print Tabular

```python
traced.graph.print_tabular()
```

Output:
```
opcode         name     target                   args          kwargs
-------------  -------  -----------------------  ------------  --------
placeholder    x        x                        ()            {}
call_module    linear1  linear1                  (x,)          {}
call_function  relu     <built-in function relu>  (linear1,)    {}
call_module    linear2  linear2                  (relu,)       {}
output         output   output                   (linear2,)    {}
```

### Counting Operations

```python
from collections import Counter

op_counts = Counter()
for node in graph.nodes:
    if node.op == "call_function":
        op_counts[node.target.__name__] += 1
    elif node.op == "call_module":
        module = traced.get_submodule(node.target)
        op_counts[type(module).__name__] += 1
    elif node.op == "call_method":
        op_counts[node.target] += 1

print(op_counts)  # Counter({'Linear': 2, 'relu': 1})
```

### Accessing the Generated Code

```python
# Human-readable Python
print(traced.code)

# The graph itself
print(traced.graph)

# Serializable format
print(traced.graph.python_code(root_module="self"))
```

### Finding Specific Patterns

```python
def find_linear_chains(graph_module):
    """Find consecutive Linear → Linear without activation."""
    chains = []
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        mod = graph_module.get_submodule(node.target)
        if not isinstance(mod, nn.Linear):
            continue
        for user in node.users:
            if user.op == "call_module":
                user_mod = graph_module.get_submodule(user.target)
                if isinstance(user_mod, nn.Linear):
                    chains.append((node, user))
    return chains
```

---

## 5. Graph Transformation — Adding, Removing, Replacing Nodes

### Context Managers for Insertion

FX provides context managers that control where new nodes are inserted:

```python
graph = traced.graph

# Insert before a specific node
with graph.inserting_before(target_node):
    new_node = graph.call_function(torch.neg, args=(some_node,))

# Insert after a specific node
with graph.inserting_after(target_node):
    new_node = graph.call_function(torch.abs, args=(some_node,))
```

### Replacing a Node's Function

```python
for node in graph.nodes:
    if node.op == "call_function" and node.target == torch.relu:
        node.target = torch.nn.functional.gelu
```

After any modification, recompile:
```python
graph.lint()           # validate correctness
traced.recompile()     # regenerate code from graph
```

### Erasing Nodes

```python
# Must replace all uses first
node_to_remove.replace_all_uses_with(replacement_node)
graph.erase_node(node_to_remove)
```

The node can only be erased if it has **zero users**. Call `replace_all_uses_with` first.

### Inserting a New Submodule Call

```python
# Add a new module to the GraphModule
traced.add_module("new_bn", nn.BatchNorm1d(20))

# Insert a call_module node
with graph.inserting_after(linear1_node):
    bn_node = graph.call_module("new_bn", args=(linear1_node,))
    # Rewire: everything that used linear1's output now uses bn's output
    linear1_node.replace_all_uses_with(bn_node)
    # But bn itself still needs linear1 as input
    bn_node.args = (linear1_node,)

graph.lint()
traced.recompile()
```

---

## 6. Pattern Matching and Replacement

### `replace_pattern`

`torch.fx.subgraph_rewriter.replace_pattern` finds subgraph patterns and replaces them:

```python
from torch.fx import subgraph_rewriter

# Define the pattern to match
def pattern(x):
    x = torch.add(x, x)
    x = torch.relu(x)
    return x

# Define the replacement
def replacement(x):
    return torch.nn.functional.gelu(torch.mul(x, 2))

# Apply
replaced = subgraph_rewriter.replace_pattern(traced, pattern, replacement)
print(f"Replaced {len(replaced)} matches")
```

### How Pattern Matching Works

1. The pattern function is symbolically traced into a small graph
2. FX searches the target graph for subgraphs that are **structurally isomorphic**
3. Matched subgraphs are spliced out and replaced with the replacement graph
4. Input/output edges are rewired automatically

### Limitations

- Pattern matching is **structural**, not **semantic** — `torch.add(x, y)` won't match `x + y` (which becomes `operator.add`)
- The pattern must be traceable by `symbolic_trace`
- Wildcards aren't directly supported — every node in the pattern must match

---

## 7. Practical Pass Examples

### Pass 1: Replace ReLU with GELU

```python
def replace_relu_with_gelu(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        # Handle call_function: torch.relu or F.relu
        if node.op == "call_function" and node.target in (
            torch.relu, torch.nn.functional.relu
        ):
            node.target = torch.nn.functional.gelu
        # Handle call_module: nn.ReLU instances
        elif node.op == "call_module":
            mod = gm.get_submodule(node.target)
            if isinstance(mod, nn.ReLU):
                # Replace the module itself
                parent_name, _, attr_name = node.target.rpartition(".")
                parent = gm.get_submodule(parent_name) if parent_name else gm
                setattr(parent, attr_name, nn.GELU())
    gm.graph.lint()
    gm.recompile()
    return gm
```

### Pass 2: Add Timing Instrumentation

```python
import time

def add_timing(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op in ("call_function", "call_module", "call_method"):
            with graph.inserting_before(node):
                start = graph.call_function(time.perf_counter, args=())
            with graph.inserting_after(node):
                end = graph.call_function(time.perf_counter, args=())
                graph.call_function(
                    print,
                    args=(f"{node.name}: ",),
                )
    graph.lint()
    gm.recompile()
    return gm
```

### Pass 3: Fuse Consecutive Linear Layers

When two `nn.Linear` layers have no activation between them, they can be algebraically fused: `W2(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)`.

```python
def fuse_linear_layers(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_module":
            continue
        mod1 = gm.get_submodule(node.target)
        if not isinstance(mod1, nn.Linear):
            continue
        # Check single user, also a Linear
        users = list(node.users.keys())
        if len(users) != 1 or users[0].op != "call_module":
            continue
        next_node = users[0]
        mod2 = gm.get_submodule(next_node.target)
        if not isinstance(mod2, nn.Linear):
            continue
        # Fuse: W_fused = W2 @ W1, b_fused = W2 @ b1 + b2
        with torch.no_grad():
            W_fused = mod2.weight @ mod1.weight
            b_fused = mod2.weight @ mod1.bias + mod2.bias if mod1.bias is not None else mod2.bias
        fused = nn.Linear(mod1.in_features, mod2.out_features)
        fused.weight = nn.Parameter(W_fused)
        fused.bias = nn.Parameter(b_fused)
        # Replace in graph
        gm.add_module(f"fused_{node.name}_{next_node.name}", fused)
        with graph.inserting_before(node):
            fused_node = graph.call_module(
                f"fused_{node.name}_{next_node.name}",
                args=node.args,
            )
        next_node.replace_all_uses_with(fused_node)
        graph.erase_node(next_node)
        graph.erase_node(node)
    graph.lint()
    gm.recompile()
    return gm
```

### Pass 4: Dead Code Elimination

```python
def eliminate_dead_code(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm
```

FX has built-in dead code elimination. A node is "dead" if it has no users and no side effects. The `eliminate_dead_code()` method removes all such nodes.

### Pass 5: Constant Folding

If a subgraph depends only on constants (parameters, no placeholders), it can be evaluated once and replaced with the result:

```python
def constant_fold(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        # Check if all args are constants or get_attr
        if all(
            not isinstance(a, torch.fx.Node) or a.op == "get_attr"
            for a in node.args
        ):
            # Evaluate the node with real values
            interp = torch.fx.Interpreter(gm)
            # ... fold constant into a get_attr node
            pass
    graph.lint()
    gm.recompile()
    return gm
```

In practice, `torch._inductor.constant_folding` provides a production-grade implementation.

---

## 8. ShapeProp

`ShapeProp` propagates tensor metadata (shape, dtype, device) through the graph by running the graph with real inputs and recording the output metadata at each node.

```python
from torch.fx.passes.shape_prop import ShapeProp

model = MyModel()
traced = torch.fx.symbolic_trace(model)

# Run shape propagation with a sample input
sample = torch.randn(4, 10)
ShapeProp(traced).propagate(sample)

# Now every node has shape metadata
for node in traced.graph.nodes:
    if "tensor_meta" in node.meta:
        meta = node.meta["tensor_meta"]
        print(f"{node.name:15s} shape={meta.shape} dtype={meta.dtype}")
```

Output:
```
x               shape=torch.Size([4, 10]) dtype=torch.float32
linear1         shape=torch.Size([4, 20]) dtype=torch.float32
relu            shape=torch.Size([4, 20]) dtype=torch.float32
linear2         shape=torch.Size([4, 5])  dtype=torch.float32
```

This is essential for optimization passes that need to know tensor dimensions — e.g., deciding whether to fuse operations based on their sizes.

---

## 9. graph.lint()

`graph.lint()` validates the graph's structural integrity:

```python
traced.graph.lint()
```

It checks:
- Every node's `args` and `kwargs` reference valid nodes in the same graph
- There is exactly one `output` node
- Placeholder nodes come before all other nodes
- No cycles exist
- All `call_module` targets exist on the root module
- Node users are consistent (if A uses B, then A is in B.users)

Always call `graph.lint()` after any graph transformation. It catches bugs early — an invalid graph will produce cryptic errors at execution time.

---

## 10. torch.fx.Interpreter

The `Interpreter` executes a `GraphModule` node-by-node, giving you hooks to customize behavior at each step.

```python
class ProfilingInterpreter(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.profiling_results = {}

    def run_node(self, node):
        start = time.perf_counter()
        result = super().run_node(node)
        elapsed = time.perf_counter() - start
        self.profiling_results[node.name] = elapsed
        return result

interp = ProfilingInterpreter(traced)
output = interp.run(torch.randn(4, 10))

for name, t in sorted(interp.profiling_results.items(), key=lambda x: -x[1]):
    print(f"{name:20s} {t*1000:.3f} ms")
```

### Interpreter Methods You Can Override

| Method | Called When | Use Case |
|--------|-----------|----------|
| `run_node(node)` | Every node | Profiling, logging, error handling |
| `call_function(target, args, kwargs)` | `call_function` nodes | Mock functions, replace ops |
| `call_method(target, args, kwargs)` | `call_method` nodes | Intercept method calls |
| `call_module(target, args, kwargs)` | `call_module` nodes | Swap modules, add hooks |
| `placeholder(target, args, kwargs)` | Input nodes | Modify inputs |
| `get_attr(target, args, kwargs)` | Attribute access | Intercept param loads |
| `output(target, args, kwargs)` | Return node | Post-process outputs |

### Shape Inference Interpreter

```python
class ShapeInterpreter(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.node_shapes = {}

    def run_node(self, node):
        result = super().run_node(node)
        if isinstance(result, torch.Tensor):
            self.node_shapes[node.name] = result.shape
        return result
```

---

## 11. torch.fx.Transformer

`Transformer` is a higher-level API for node-by-node graph rewriting. You subclass it and override methods per op type. It creates a **new** graph (instead of modifying in-place).

```python
class ReLUToGELU(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == torch.relu:
            target = torch.nn.functional.gelu
        return super().call_function(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        mod = self.fetch_attr(target)
        if isinstance(mod, nn.ReLU):
            return super().call_function(
                torch.nn.functional.gelu, args, kwargs
            )
        return super().call_module(target, args, kwargs)

transformed = ReLUToGELU(traced).transform()
```

### Transformer vs. Manual Graph Manipulation

| Aspect | `Transformer` | Manual (`graph.nodes` iteration) |
|--------|--------------|----------------------------------|
| Creates new graph | Yes | No (in-place) |
| Node remapping | Automatic | Manual |
| Easier for per-node transforms | Yes | No |
| Better for structural changes | No | Yes (inserting/removing) |
| Risk of dangling references | Low | Higher |

Use `Transformer` when your pass maps each node to zero or more nodes. Use manual manipulation when you need to analyze graph structure (chains, patterns) before deciding what to change.

---

## 12. FX in the Compilation Stack

### torch.compile Pipeline

```
           torch.compile(model)
                  │
         ┌────────▼────────┐
         │  TorchDynamo    │  Captures Python bytecode → FX Graph
         │  (Python → FX)  │  Handles control flow via graph breaks
         └────────┬────────┘
                  │  FX Graph (ATen-level ops)
         ┌────────▼────────┐
         │  AOTAutograd    │  Joint forward+backward graph
         │  (FX → FX)     │  Partitions into fwd/bwd graphs
         └────────┬────────┘
                  │  FX Graph (decomposed ATen ops)
         ┌────────▼────────┐
         │  Inductor       │  Lowers FX Graph → Triton/C++ code
         │  (FX → code)   │  Fusion, scheduling, code generation
         └─────────────────┘
```

### Dynamo Produces FX Graphs

Unlike `symbolic_trace`, Dynamo operates at the **bytecode level**. It:
- Handles control flow by inserting **graph breaks**
- Supports dynamic shapes
- Captures the actual operations executed (not just proxy-traced forward)

```python
def dynamo_backend(gm: torch.fx.GraphModule, example_inputs):
    """Custom backend receives an FX GraphModule."""
    print("Received FX graph:")
    gm.graph.print_tabular()
    return gm  # return as-is for debugging

model = torch.compile(MyModel(), backend=dynamo_backend)
model(torch.randn(4, 10))
```

### Inductor FX Passes

Inductor applies many FX passes before code generation. They live in `torch/_inductor/fx_passes/`:

| Pass | What It Does |
|------|-------------|
| `decompositions.py` | Break complex ops into primitives |
| `fuse_attention.py` | Pattern-match and fuse attention |
| `group_batch_fusion.py` | Batch small ops together |
| `joint_graph.py` | Optimizations on the joint fwd+bwd graph |
| `post_grad.py` | Post-autograd optimizations |
| `pre_grad.py` | Pre-autograd optimizations |

### Writing a Custom Inductor Pass

```python
from torch._inductor import config

def my_custom_pass(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        # Your optimization here
        pass
    gm.graph.lint()
    gm.recompile()
    return gm

# Register as a post-grad pass
config.post_grad_custom_post_pass = my_custom_pass
```

---

## 13. Upstream Updates (June 15-16, 2026)

Recent changes in the PyTorch repository that touch FX, Dynamo, Inductor, and related infrastructure:

### TokenSwitch for Distributed Token Routing (#178712)
New `TokenSwitch` primitive for distributed expert-parallel token routing. Uses FX graph representation for expressing token dispatch and combine patterns across devices.

### Dynamo O(N^2) Decomposition Fix (#177927)
Fixed a performance regression where Dynamo's decomposition pass exhibited O(N^2) behavior on large graphs. The fix avoids redundant node iteration during decomposition table lookup.

### Dynamo Native itertools Variables (#186973, #186974)
Replaced polyfill implementations of `itertools.product`, `itertools.chain`, and related functions with native Dynamo variable tracking. This eliminates graph breaks when models use `itertools` in traced code and avoids unnecessary Python overhead.

### Inductor NVGEMM Disk Cache (#187013)
Added persistent disk caching for NVGEMM (NVIDIA GEMM library) autotuning results. Previously, autotuning was repeated on every process restart. The disk cache persists winning kernel configurations across runs, significantly reducing warm-up time for workloads heavy in matrix multiplications.

### DTensor single_dim_strategy for Reduction Ops (#179201)
Extended DTensor's `single_dim_strategy` to handle reduction operations. This enables more efficient sharding strategies when reductions operate on a single dimension, improving distributed training throughput for models with dimension-specific reductions.

### MPS Metal Kernel Migrations
Several Metal kernel migrations for Apple Silicon:
- `index_add` — moved from MPSGraph to native Metal shader for better performance
- `logical_not` — native Metal implementation replacing MPSGraph path
- Faster reduction kernels — optimized Metal shaders for sum/mean/max operations

### Dynamo `nb_inv` Slot Support (#185641)
Added support for the `__invert__` / `nb_inv` numeric slot in Dynamo's variable tracker. Models using bitwise inversion (`~x`) on custom types no longer cause graph breaks.

---

## 14. Summary

### FX Concepts at a Glance

```
┌─────────────────────────────────────────────────┐
│                 torch.fx                         │
│                                                  │
│  symbolic_trace ──▶ Graph (Nodes) ──▶ GraphModule│
│                                                  │
│  Node ops:                                       │
│    placeholder, get_attr, call_function,         │
│    call_method, call_module, output              │
│                                                  │
│  Transform APIs:                                 │
│    ├── Manual: inserting_before/after, erase     │
│    ├── replace_pattern (subgraph rewriter)       │
│    ├── Interpreter (execute with hooks)          │
│    └── Transformer (node-level rewrite)          │
│                                                  │
│  Validation: graph.lint(), ShapeProp             │
│                                                  │
│  In torch.compile:                               │
│    Dynamo → AOTAutograd → Inductor               │
│    (all use FX Graphs internally)                │
└─────────────────────────────────────────────────┘
```

### When to Use Each API

| Goal | API |
|------|-----|
| Inspect model structure | `symbolic_trace` + iterate `graph.nodes` |
| Simple op replacement | Manual node iteration, change `node.target` |
| Structural transforms (fuse, split) | Manual `inserting_before/after` + `erase_node` |
| Pattern-based replacement | `replace_pattern` |
| Per-node behavior (profiling, logging) | `Interpreter` subclass |
| Clean per-node transforms | `Transformer` subclass |
| Production optimization passes | Custom Inductor passes |

### Key Rules

1. **Always call `graph.lint()`** after modifying a graph
2. **Always call `gm.recompile()`** after modifying the graph of a `GraphModule`
3. **Erase nodes bottom-up** — a node can only be erased when it has zero users
4. **`replace_all_uses_with`** before erasing a node with users
5. **`symbolic_trace` ≠ `torch.compile`** — symbolic trace is simpler but less powerful; torch.compile (Dynamo) handles control flow and dynamic shapes

---

## Further Reading

- [torch.fx Official Docs](https://pytorch.org/docs/stable/fx.html) — API reference
- [torch.fx Technical Overview](https://pytorch.org/docs/stable/fx.html#torch-fx-technical-overview) — design philosophy
- [FX Graph Mode Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html) — quantization with FX
- [Building a Custom Backend](https://pytorch.org/docs/stable/torch.compiler_custom_backends.html) — receive FX graphs from torch.compile
- [Inductor Deep Dive](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-aware-code-generation/747) — how Inductor uses FX

---

<div align="center">

[← Previous Module](../22_llm_recipes/) | [🏠 Home](../README.md) | Next Module →

**Notebook**: [`23_fx_transforms.ipynb`](../notebooks/23_fx_transforms.ipynb)

</div>
