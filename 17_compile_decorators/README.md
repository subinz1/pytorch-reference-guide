<div align="center">

[← Previous Module](../16_activation_checkpointing/) | [🏠 Home](../README.md) | [Next Module →](../18_torch_package/)

</div>

---

> **Module 17** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 08 — torch.compile](../08_torch_compile/)
> **Time to complete:** ~1 hour

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `compile_control.py` | torch.compile decorators & control APIs — fine-grained control over what gets compiled and how |

---

# Module 17: torch.compile Decorators & Control APIs

*Day 3 of the incremental learning series*

---

## Beyond torch.compile() — Fine-Grained Compilation Control

Module 08 taught you the basics of `torch.compile`. This module dives into the **decorator and control APIs** that give you precise control over what gets compiled, how shapes are handled, and how to debug compilation issues.

---

## Table of Contents

1. [Compiler Stances — Global Compilation Behavior](#1-compiler-stances)
2. [torch.compiler.disable — Skip Compilation](#2-disable)
3. [allow_in_graph / disallow_in_graph — Control Tracing](#3-allow-and-disallow-in-graph)
4. [substitute_in_graph — Replace Functions for Tracing](#4-substitute-in-graph)
5. [mark_dynamic / mark_static — Shape Control](#5-mark-dynamic--mark-static)
6. [graph_break / error_on_graph_break — Break Control](#6-graph-break-control)
7. [assume_constant_result — Constant Folding](#7-assume-constant-result)
8. [comptime — Compile-Time Debugging](#8-comptime)
9. [torch._dynamo.explain — Understanding Compilation](#9-explain)
10. [TORCH_LOGS — Logging and Debugging](#10-torch-logs)
11. [What's New: Recent Upstream Changes](#11-whats-new)

---

## 1. Compiler Stances

Stances control the **global compilation behavior** — useful for debugging and gradual adoption:

```python
import torch

torch.compiler.set_stance("default")             # Normal compilation
torch.compiler.set_stance("force_eager")          # Skip all compilation
torch.compiler.set_stance("eager_on_recompile")   # Compile once, eager on recompile
torch.compiler.set_stance("fail_on_recompile")    # Error on recompilation
torch.compiler.set_stance("eager_then_compile")   # Eager first, compile on second call

# Context manager form
with torch.compiler.set_stance("force_eager"):
    output = compiled_model(x)  # Runs eagerly despite being compiled
```

| Stance | Behavior | Use Case |
|--------|----------|----------|
| `default` | Normal compilation | Production |
| `force_eager` | Skip all compilation | Debugging, profiling eager |
| `eager_on_recompile` | Compile once, eager on recompile | Avoid compile-time storms |
| `fail_on_recompile` | Error on recompilation | CI, catch shape issues |
| `eager_then_compile` | Eager first call, compile on second | Warmup tolerance |

---

## 2. Disable — Skip Compilation for Specific Functions

```python
@torch.compiler.disable
def preprocessing(x):
    """Uses Python features that don't compile well."""
    results = []
    for i in range(x.shape[0]):
        if x[i].item() > 0:
            results.append(x[i] * 2)
        else:
            results.append(x[i])
    return torch.stack(results)

@torch.compile
def model_fn(x):
    x = preprocessing(x)     # Runs eagerly (disabled)
    return x.relu().sum()     # This part is compiled
```

**Non-recursive disable** (only disables the decorated function, not functions it calls):

```python
@torch.compiler.disable(recursive=False)
def outer(x):
    return inner(x)  # inner() will still be compiled
```

---

## 3. allow_in_graph / disallow_in_graph

```python
@torch.compiler.allow_in_graph   # Opaque node in graph
@torch._dynamo.disallow_in_graph  # Forces graph break
@torch._dynamo.forbid_in_graph    # Raises error during tracing
```

---

## 4. substitute_in_graph — Replace Functions for Tracing

```python
def original_fn(x):
    result = x.tolist()  # Not traceable!
    return torch.tensor(result)

def traceable_fn(x):
    return x.clone()

torch.compiler.substitute_in_graph(original_fn, traceable_fn)
```

---

## 5. mark_dynamic / mark_static — Shape Control

```python
x = torch.randn(batch_size, 512)
torch._dynamo.mark_dynamic(x, 0)  # Dim 0 is dynamic — no recompile for different batch sizes
torch._dynamo.mark_static(x, 1)   # Dim 1 is always 512
torch._dynamo.mark_static_address(weight)  # Data pointer won't change
```

---

## 6. Graph Break Control

```python
torch._dynamo.graph_break()       # Force a graph break
torch.compile(fullgraph=True)     # Error on any break

@torch._dynamo.error_on_graph_break
def must_be_one_graph(x):
    return x + 1
```

---

## 7. assume_constant_result

```python
@torch._dynamo.assume_constant_result
def get_config():
    return load_config()["lr"]  # Folded to constant at compile time
```

---

## 8. comptime — Compile-Time Debugging

```python
from torch._dynamo.comptime import comptime

@torch.compile
def fn(x):
    comptime.breakpoint()  # pdb during COMPILATION
    return x + 1
# In pdb: ctx.print_locals(), ctx.print_graph(), ctx.print_bt()
```

---

## 9. torch._dynamo.explain

```python
explanation = torch._dynamo.explain(my_fn)(torch.randn(10))
print(explanation)  # Shows graphs, breaks, guards
```

---

## 10. TORCH_LOGS — Logging and Debugging

```bash
TORCH_LOGS="graph_breaks" python train.py       # Graph break reasons
TORCH_LOGS="guards,recompiles" python train.py  # Guard failures
TORCH_LOGS="graph_code" python train.py         # Captured FX graph
TORCH_LOGS="output_code" python train.py        # Generated Triton/C++
TORCH_LOGS="+dynamo" python train.py            # Full debug
TORCH_TRACE=/tmp/trace python train.py          # Structured tracing
```

---

## 11. What's New: Recent Upstream Changes (June 4-8, 2026)

194 commits landed on PyTorch main in the last 4 days:

### QuACK GEMM Kernels Vendored
PyTorch now vendors the QuACK library from Dao-AILab — high-performance CuTeDSL GEMM epilogue adapters with fused RMSNorm. Located at `torch/_vendor/quack/`.

### CUPTI Monitor — Continuous GPU Profiling
New `torch.profiler._cupti_monitor` for continuous CUPTI activity monitoring across the entire program (not just a profiling window).

### Optimized _foreach_mm — Grouped GEMMs
New Python override dispatching to nvmath cublasLt grouped GEMM (bf16) or CUTLASS. At `torch/_native/ops/foreach_mm/`.

### AArch64 torch.compile
Armv9-A target support — compiled models now work on ARM servers and edge devices.

### DTensor Autogen Ops
Auto-generated sharding strategies expanding DTensor op coverage. At `torch/distributed/tensor/_ops/autogen.py`.

### NCCL Symmetric Memory Registration
External NCCL comm registration for symmetric memory at `torch/distributed/_symmetric_memory/_nccl.py`.

### Inductor Heuristics Module
Refactored Triton template heuristics into `torch/_inductor/heuristics/`.

---

## Quick Reference

| API | What It Does |
|-----|-------------|
| `set_stance()` | Global compilation behavior |
| `@disable` | Skip compilation for a function |
| `@allow_in_graph` | Treat as opaque graph node |
| `substitute_in_graph()` | Replace with traceable version |
| `mark_dynamic()` | Declare dynamic dimension |
| `mark_static()` | Declare static dimension |
| `fullgraph=True` | Error on any graph break |
| `graph_break()` | Force a graph break |
| `explain()` | Get compilation report |
| `comptime.breakpoint()` | Debug during compilation |
| `CompileCounter` | Count compilations in tests |
| `EagerAndRecordGraphs` | Inspect captured FX graphs |

---

<div align="center">

[← Previous Module](../16_activation_checkpointing/) | [🏠 Home](../README.md) | [Next Module →](../18_torch_package/)

**No dedicated notebook** — covered in [Module 08 notebook](../notebooks/08_torch_compile_masterclass.ipynb)

</div>
