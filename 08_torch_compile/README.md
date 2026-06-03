# Module 08: torch.compile — The Complete Guide

## Overview

`torch.compile` is PyTorch's JIT compiler that makes your models run faster
without changing your code. Introduced in PyTorch 2.0, it captures your Python
model into an optimized graph and generates efficient machine code.

---

## 1. What is torch.compile?

### The Problem

Standard PyTorch executes operations one-by-one ("eager mode"). Each operation:
1. Dispatches from Python to C++
2. Launches a kernel
3. Reads/writes memory
4. Returns to Python

This means:
- Python overhead on every operation
- No opportunity to fuse adjacent operations
- Suboptimal memory access patterns

### The Solution

`torch.compile` captures a sequence of operations, optimizes them as a group,
and generates fused kernels that minimize memory traffic and Python overhead.

### Expected Speedup

- Typical: 20-50% faster on GPU, 10-30% on CPU
- Best case: 2-3x for memory-bound models (transformers)
- Worst case: No speedup if the model has many graph breaks

---

## 2. How It Works — The 3 Stages

### Stage 1: TorchDynamo (Graph Capture)

TorchDynamo intercepts Python bytecode execution and traces the operations
into an FX graph (a simple intermediate representation).

**How it works (simplified):**
1. Your function runs normally
2. Dynamo watches which torch operations are called
3. It records them into a graph
4. For subsequent calls with the same "shape signature," it replays the graph

**Guards:** Dynamo records assumptions about inputs (dtype, device, shape).
If an assumption is violated on a later call, the graph is invalidated and
Dynamo retraces (recompiles) the function.

**Graph breaks:** When Dynamo encounters something it can't trace (like `print()`
with a tensor, or data-dependent control flow), it "breaks" the graph — splits
execution into: compiled-graph-1 → Python → compiled-graph-2.

### Stage 2: AOTAutograd (Ahead-of-Time Autograd)

After capturing the forward graph, AOTAutograd:
1. Traces the backward pass as well (ahead of time)
2. Partitions into a forward graph and backward graph
3. Both are passed to the backend for optimization

This is important because the backward pass can also be optimized and fused.

### Stage 3: TorchInductor (Code Generation)

Inductor takes the graph and generates optimized code:
- **For GPU**: Generates Triton kernels (a Python-based GPU programming language)
- **For CPU**: Generates C++/OpenMP code

Key optimizations:
- **Operator fusion**: Combine multiple ops into one kernel (e.g., linear + relu)
- **Memory planning**: Reuse memory buffers, minimize allocations
- **Layout optimization**: Choose best memory layout for the hardware

---

## 3. Basic Usage

### Three ways to use torch.compile:

```python
# Method 1: Compile a model
compiled_model = torch.compile(model)
output = compiled_model(input)

# Method 2: Decorator on a function
@torch.compile
def my_function(x, y):
    return torch.matmul(x, y) + x

# Method 3: Compile a specific function
compiled_fn = torch.compile(my_function)
```

### First Call is Slow

The first call triggers compilation (tracing + code generation). Subsequent
calls with the same input shapes are fast:

```python
compiled_model = torch.compile(model)

# First call: SLOW (compiles)
output = compiled_model(input_batch_1)

# Second call: FAST (uses compiled code)
output = compiled_model(input_batch_2)
```

---

## 4. Compilation Modes

```python
torch.compile(model, mode="default")         # Balanced
torch.compile(model, mode="reduce-overhead") # Minimize framework overhead
torch.compile(model, mode="max-autotune")    # Maximum optimization effort
```

### "default"
- Balanced between compilation time and runtime performance
- Uses a reasonable set of optimizations
- Good starting point

### "reduce-overhead"
- Uses CUDA graphs to eliminate kernel launch overhead
- Best for models with many small kernels
- May increase memory usage
- GPU only

### "max-autotune"
- Tries many kernel implementations and picks the fastest
- Much longer compilation time
- Best runtime performance
- Good for production deployment after development

### Comparison

| Mode             | Compile Time | Runtime Speed | Memory  | Best For       |
|------------------|--------------|---------------|---------|----------------|
| default          | Fast         | Good          | Normal  | Development    |
| reduce-overhead  | Medium       | Better        | Higher  | Small ops (GPU)|
| max-autotune     | Slow         | Best          | Normal  | Production     |

---

## 5. Graph Breaks

### What Causes Graph Breaks

A graph break splits the compiled region into multiple segments, with Python
execution in between. This reduces optimization opportunities.

Common causes:
1. **print() with tensor values** — requires Python execution
2. **Unsupported Python operations** — certain builtins
3. **Data-dependent control flow** — `if tensor.item() > 0:`
4. **Calling non-compilable functions** — some third-party code
5. **In-place operations on views** (in some cases)
6. **Python side-effects** — logging, writing to files

### How to Find Graph Breaks

```python
# Method 1: explain()
explanation = torch._dynamo.explain(model)(sample_input)
print(explanation)

# Method 2: fullgraph=True raises an error on any break
compiled = torch.compile(model, fullgraph=True)
try:
    compiled(input)
except Exception as e:
    print(f"Graph break: {e}")

# Method 3: Logging
import logging
torch._logging.set_logs(graph_breaks=True)
```

### How to Fix Graph Breaks

```python
# BAD: print causes graph break
def forward(self, x):
    x = self.linear(x)
    print(f"Shape: {x.shape}")  # GRAPH BREAK!
    return self.relu(x)

# GOOD: remove print or use torch._dynamo.config.suppress_errors
def forward(self, x):
    x = self.linear(x)
    return self.relu(x)

# BAD: data-dependent control flow
def forward(self, x):
    if x.sum() > 0:  # GRAPH BREAK! (value depends on data)
        return x * 2
    return x

# GOOD: use torch.where for data-dependent logic
def forward(self, x):
    return torch.where(x.sum() > 0, x * 2, x)
```

---

## 6. fullgraph=True

Forces compilation of the entire function as a single graph. If there would
be any graph break, compilation fails with an error instead of silently
degrading.

```python
@torch.compile(fullgraph=True)
def my_fn(x):
    # This MUST be fully traceable — no graph breaks allowed
    return x.sin() + x.cos()
```

Use this when:
- You want maximum performance (no graph breaks = fully optimized)
- You want to catch non-compilable code early
- In production code where you've already fixed all breaks

---

## 7. Dynamic Shapes

### The Problem: Recompilation

By default, torch.compile captures the exact shapes of inputs. If shapes
change, it must recompile:

```python
compiled_fn = torch.compile(fn)
compiled_fn(torch.randn(32, 64))   # Compiles for shape [32, 64]
compiled_fn(torch.randn(16, 64))   # Recompiles for shape [16, 64]!
compiled_fn(torch.randn(8, 64))    # Recompiles again!
```

### The Solution: dynamic=True

```python
compiled_fn = torch.compile(fn, dynamic=True)
compiled_fn(torch.randn(32, 64))   # Compiles with symbolic shapes
compiled_fn(torch.randn(16, 64))   # Reuses compiled code!
compiled_fn(torch.randn(8, 64))    # Reuses again!
```

With `dynamic=True`, Dynamo uses symbolic shapes (e.g., `s0` instead of `32`),
and the generated code handles any batch size.

### Automatic Dynamic Shapes

PyTorch can also automatically detect that a dimension varies and switch to
dynamic shapes after seeing multiple sizes:

```python
# After 2 recompilations on the same dimension, Dynamo marks it dynamic
compiled_fn(torch.randn(32, 64))   # Compile for [32, 64]
compiled_fn(torch.randn(16, 64))   # Recompile, mark dim 0 as dynamic
compiled_fn(torch.randn(8, 64))    # Uses dynamic code — no recompile!
```

### mark_dynamic

For fine-grained control:

```python
x = torch.randn(32, 64)
torch._dynamo.mark_dynamic(x, 0)  # Mark dimension 0 as dynamic
compiled_fn(x)  # Compiles with dynamic dim 0, static dim 1 (64)
```

---

## 8. Compilation Cache

### How Caching Works

Compiled code is cached so you don't recompile every time you restart:

1. **In-memory cache**: Within a process, same function + same guards = reuse
2. **Persistent cache**: Across process restarts (PyTorch 2.1+), compiled
   artifacts are stored on disk

### Persistent Cache

```python
# Enable persistent cache (stored in ~/.cache/torch/inductor/)
import torch._inductor.config
torch._inductor.config.fx_graph_cache = True
```

This means:
- First run: full compilation (slow)
- Second run: loads from cache (fast startup)

---

## 9. Compiler Stances

Stances control how the compiler handles recompilation:

```python
# Don't compile at all — run in eager mode
torch.compiler.set_stance("force_eager")

# Warn (log) on recompilation instead of silently recompiling
torch.compiler.set_stance("eager_on_recompile")

# Error if recompilation would occur
torch.compiler.set_stance("fail_on_recompile")

# Default behavior
torch.compiler.set_stance("default")
```

Use `fail_on_recompile` in production to catch unexpected dynamic behavior
that would hurt performance.

---

## 10. Debugging torch.compile

### torch._dynamo.explain()

Shows what happened during compilation:

```python
explanation = torch._dynamo.explain(compiled_fn)(input)
print(explanation)
# Shows: number of graphs, graph breaks, break reasons
```

### TORCH_LOGS Environment Variable

```bash
# See graph breaks
TORCH_LOGS="graph_breaks" python script.py

# See what Dynamo captured
TORCH_LOGS="dynamo" python script.py

# See generated code
TORCH_LOGS="output_code" python script.py

# See recompilation reasons
TORCH_LOGS="recompiles" python script.py
```

### Common Errors

1. **"torch._dynamo.exc.Unsupported"** — An operation can't be traced.
   Fix: rewrite using supported operations.

2. **Recompilation spam** — Model keeps recompiling.
   Fix: Use `dynamic=True` or ensure consistent input shapes.

3. **Incorrect results** — Compiled code gives different outputs.
   Fix: Report as a bug. Workaround: mark the function with
   `torch._dynamo.disable()`.

---

## 11. torch._dynamo.reset()

Clears all compiled graphs and cached state:

```python
torch._dynamo.reset()
```

Useful when:
- Testing different compilation settings
- Debugging compilation issues
- Benchmarking (to force recompilation)

---

## 12. Custom Backends

You can write your own backend that receives the FX graph:

```python
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Custom backend that receives the graph and returns a callable.

    Args:
        gm: The captured FX graph module
        example_inputs: Example inputs used during tracing

    Returns:
        A callable that takes the same inputs and produces outputs
    """
    # Inspect the graph
    print(f"Graph has {len(list(gm.graph.nodes))} nodes")

    # You can transform the graph here, or just return it as-is
    # (gm is already callable)
    return gm

compiled_fn = torch.compile(fn, backend=my_backend)
```

This is useful for:
- Profiling what operations are captured
- Custom optimizations
- Research on graph transformations

---

## 13. Compiled Autograd

By default, `torch.compile` only compiles the forward pass. The backward pass
still runs in eager mode. Compiled Autograd compiles the backward too:

```python
with torch._dynamo.compiled_autograd.enable(torch.compile(backend="inductor")):
    loss.backward()
```

Benefits:
- Backward pass also gets operator fusion
- End-to-end compilation of training step

---

## 14. Performance Tips

### When torch.compile Helps Most

- Transformer models (lots of small ops to fuse)
- Memory-bound operations (fusion reduces memory traffic)
- Models with many element-wise operations in sequence
- Standard architectures using torch.nn modules

### When It Doesn't Help Much

- Already compute-bound (large matmuls with batch dim)
- Heavy graph breaks (too much falls back to Python)
- Very small models (compilation overhead > runtime savings)
- Highly dynamic models (frequent recompilation)

### Best Practices

1. **Profile first**: Know where time is spent before compiling
2. **Start simple**: `torch.compile(model)` with defaults
3. **Check for graph breaks**: Use `explain()` or `fullgraph=True`
4. **Use dynamic=True** if batch sizes vary
5. **Use max-autotune** for production deployments
6. **Cache compiled code** for fast startup

---

## 15. FX Graph Basics

The intermediate representation (IR) used by torch.compile is an FX graph:

```python
import torch.fx

def fn(x, y):
    z = x + y
    return z.relu()

# Trace into FX graph
traced = torch.fx.symbolic_trace(fn)
print(traced.graph)
```

Output:
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=1] = placeholder[target=y]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y))
    %relu : [num_users=1] = call_method[target=relu](args = (%add,))
    return relu
```

Node types:
- `placeholder` — function inputs
- `call_function` — calls to functions (like torch.add)
- `call_method` — method calls on tensors (.relu(), .view(), etc.)
- `call_module` — calls to nn.Module submodules
- `output` — return value

Understanding FX graphs helps when:
- Writing custom backends
- Debugging compilation issues
- Understanding what optimizations are applied

---

## Summary

| Feature           | What It Does                              | When to Use              |
|-------------------|-------------------------------------------|--------------------------|
| torch.compile     | Compiles model for speed                  | Always (production)      |
| fullgraph=True    | Errors on graph breaks                    | Ensuring no breaks       |
| dynamic=True      | Handles varying shapes                    | Variable batch sizes     |
| max-autotune      | Maximum optimization                      | Deployment               |
| reduce-overhead   | Minimizes launch overhead                 | Many small GPU ops       |
| explain()         | Shows compilation info                    | Debugging                |
| custom backend    | Custom graph processing                   | Research/profiling       |

### Quick Start

```python
# Step 1: Just compile it
model = torch.compile(model)

# Step 2: Check for issues
explanation = torch._dynamo.explain(model)(sample_input)

# Step 3: Optimize
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```
