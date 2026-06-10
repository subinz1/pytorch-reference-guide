<div align="center">

[← Previous Module](../18_torch_package/) | [🏠 Home](../README.md) | [Next Module →](#)

</div>

---

> **Module 19** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 02 — Tensors](../02_tensors/), [Module 03 — Autograd](../03_autograd/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `torch_function_examples.py` | __torch_function__ and __torch_dispatch__ — overriding how PyTorch operations work on custom types |

---

# Module 19: `__torch_function__` & `__torch_dispatch__` — Tensor Subclassing

*Day 5 of the incremental learning series*

---

## Why This Matters

Every time you call `torch.add(x, y)` or `x + y`, PyTorch checks: *does this tensor have a custom dispatch protocol?* Two protocols exist:

- **`__torch_function__`** — Python-level override. Intercepts **any** PyTorch function call. Like NumPy's `__array_function__`.
- **`__torch_dispatch__`** — Lower-level override. Intercepts at the **ATen operator** level (after decompositions). More powerful, used by DTensor, FakeTensor, and torch.compile internals.

These are the extension points that power:
- **DTensor** (distributed tensor) — sharding logic via `__torch_dispatch__`
- **FakeTensor** (torch.compile) — shape-only tensors via `__torch_dispatch__`
- **Logging/profiling** — intercept all operations without modifying model code
- **Custom tensor types** — sparse, quantized, masked tensors
- **Unit conversion** — tensors that carry physical units

---

## Table of Contents

1. [`__torch_function__` — Python-Level Override](#1-torch_function)
2. [TorchFunctionMode — Override Without Subclassing](#2-torchfunctionmode)
3. [`__torch_dispatch__` — ATen-Level Override](#3-torch_dispatch)
4. [TorchDispatchMode — Mode-Based Dispatch](#4-torchdispatchmode)
5. [Practical Examples](#5-practical-examples)
6. [When to Use Which Protocol](#6-when-to-use-which)
7. [Upstream Updates (June 9-10, 2026)](#7-upstream-updates)

---

## 1. `__torch_function__` — Python-Level Override

When you define `__torch_function__` on a class, PyTorch calls it instead of the normal implementation for **any** torch function that receives your object as an argument.

```python
import torch

class ScaledTensor:
    """A tensor wrapper that tracks a scaling factor."""

    def __init__(self, data, scale=1.0):
        self.data = data
        self.scale = scale

    def __repr__(self):
        return f"ScaledTensor(data={self.data}, scale={self.scale})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Called for any torch.* function involving this type."""
        if kwargs is None:
            kwargs = {}

        # Extract ScaledTensors from args, replace with raw data
        new_args = []
        scale = 1.0
        for a in args:
            if isinstance(a, ScaledTensor):
                new_args.append(a.data)
                scale = a.scale
            else:
                new_args.append(a)

        # Call the original function on raw tensors
        result = func(*new_args, **kwargs)

        # Wrap the result back
        if isinstance(result, torch.Tensor):
            return ScaledTensor(result, scale)
        return result

# Usage
x = ScaledTensor(torch.tensor([1.0, 2.0, 3.0]), scale=0.5)
y = ScaledTensor(torch.tensor([4.0, 5.0, 6.0]), scale=0.5)

z = torch.add(x, y)       # Calls ScaledTensor.__torch_function__!
print(z)                    # ScaledTensor(data=tensor([5., 7., 9.]), scale=0.5)

w = torch.mul(x, 2)        # Also intercepted
print(w)                    # ScaledTensor(data=tensor([2., 4., 6.]), scale=0.5)
```

### How the Protocol Works

1. PyTorch checks if **any argument** has `__torch_function__`
2. If yes, it calls `__torch_function__(func, types, args, kwargs)` where:
   - `func` — the original function (e.g., `torch.add`)
   - `types` — tuple of types that implement `__torch_function__`
   - `args` — positional arguments
   - `kwargs` — keyword arguments
3. Your implementation decides what to do and returns the result

---

## 2. TorchFunctionMode — Override Without Subclassing

Modes let you override **all** torch operations within a context manager — no tensor subclass needed:

```python
from torch.overrides import TorchFunctionMode

class LoggingMode(TorchFunctionMode):
    """Logs every torch operation."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        print(f"  Called: {func.__module__}.{func.__name__}")
        return func(*args, **kwargs)

# All torch ops inside the context are logged
with LoggingMode():
    x = torch.randn(3, 4)      # Logged
    y = x + 1                   # Logged
    z = torch.relu(y)           # Logged
    w = z.mean()                # Logged
```

### Use Cases for Modes
- **Logging/debugging** — see every operation a model performs
- **Profiling** — count operations, measure shapes
- **Mocking** — override factory functions (torch.randn, torch.zeros)
- **Validation** — check all inputs are on the correct device

---

## 3. `__torch_dispatch__` — ATen-Level Override

`__torch_dispatch__` intercepts at a **lower level** — after Python function dispatch, at the ATen operator level. This is where the real computation happens.

```python
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

class LoggingTensor(torch.Tensor):
    """A tensor subclass that logs all ATen operations."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Called for every ATen operator."""
        if kwargs is None:
            kwargs = {}

        # Unwrap LoggingTensors to plain tensors
        def unwrap(t):
            return t.elem if isinstance(t, LoggingTensor) else t

        print(f"  dispatch: {func.__name__}")

        # Call the actual ATen op
        result = func(*args, **kwargs)
        return result

x = LoggingTensor(torch.randn(3, 4))
y = x + 1       # Dispatches through __torch_dispatch__
z = y.relu()     # Also dispatched
```

### Key Differences from `__torch_function__`

| Feature | `__torch_function__` | `__torch_dispatch__` |
|---------|---------------------|---------------------|
| Level | Python API | ATen operators |
| Input | torch.add, torch.nn.functional.relu | aten.add.Tensor, aten.relu.default |
| Decomposition | Before | After (sees primitive ops) |
| Used by | Custom wrappers, logging | DTensor, FakeTensor, torch.compile |
| Subclass required | No (can use any class) | Yes (must subclass torch.Tensor) |

---

## 4. TorchDispatchMode — Mode-Based Dispatch

Like `TorchFunctionMode`, but at the ATen operator level:

```python
from torch.utils._python_dispatch import TorchDispatchMode

class CountOps(TorchDispatchMode):
    """Count all ATen operations in a scope."""

    def __init__(self):
        super().__init__()
        self.ops = {}

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        name = str(func.name())
        self.ops[name] = self.ops.get(name, 0) + 1
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

# Count ops in a forward pass
counter = CountOps()
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 5),
)

with counter:
    output = model(torch.randn(4, 10))

print("Operations performed:")
for op, count in sorted(counter.ops.items()):
    print(f"  {op}: {count}x")
```

---

## 5. Practical Examples

### Example 1: Device-Checking Mode

```python
class DeviceCheckMode(TorchFunctionMode):
    """Error if any tensor is on the wrong device."""

    def __init__(self, expected_device):
        self.expected_device = torch.device(expected_device)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        for a in args:
            if isinstance(a, torch.Tensor) and a.device != self.expected_device:
                raise RuntimeError(
                    f"{func.__name__}: tensor on {a.device}, "
                    f"expected {self.expected_device}"
                )
        return func(*args, **kwargs)

# This catches CPU/GPU mismatches early
with DeviceCheckMode("cpu"):
    x = torch.randn(3, 4)  # OK
    y = x + 1               # OK
```

### Example 2: Shape Logging Mode

```python
class ShapeTracer(TorchDispatchMode):
    """Track input/output shapes of all ops."""

    def __init__(self):
        super().__init__()
        self.traces = []

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)

        in_shapes = [a.shape for a in args if isinstance(a, torch.Tensor)]
        out_shape = result.shape if isinstance(result, torch.Tensor) else "N/A"
        self.traces.append((func.name(), in_shapes, out_shape))
        return result
```

### Example 3: Tensor with Units (Physics)

```python
class UnitTensor:
    """Tensor that tracks physical units (e.g., meters, seconds)."""

    def __init__(self, data, unit=""):
        self.data = data
        self.unit = unit

    def __repr__(self):
        return f"{self.data} [{self.unit}]"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        tensors = [a for a in args if isinstance(a, UnitTensor)]
        raw_args = [a.data if isinstance(a, UnitTensor) else a for a in args]
        result = func(*raw_args, **kwargs)

        if func == torch.mul and len(tensors) == 2:
            unit = f"{tensors[0].unit}*{tensors[1].unit}"
        elif func == torch.div and len(tensors) == 2:
            unit = f"{tensors[0].unit}/{tensors[1].unit}"
        else:
            unit = tensors[0].unit if tensors else ""

        if isinstance(result, torch.Tensor):
            return UnitTensor(result, unit)
        return result

distance = UnitTensor(torch.tensor(100.0), "m")
time = UnitTensor(torch.tensor(9.58), "s")
speed = torch.div(distance, time)
print(f"Speed: {speed}")  # 10.44 [m/s]
```

---

## 6. When to Use Which Protocol

| Scenario | Use |
|----------|-----|
| Log/trace all torch function calls | `TorchFunctionMode` |
| Custom tensor wrapper (non-subclass) | `__torch_function__` |
| Override ATen ops for a tensor subclass | `__torch_dispatch__` |
| Count/profile ops at ATen level | `TorchDispatchMode` |
| Build a new tensor type (like DTensor) | `__torch_dispatch__` |
| Intercept factory functions (torch.randn) | `TorchFunctionMode` |
| Works with torch.compile | `__torch_dispatch__` (preferred) |

### The Dispatch Stack

```
User code: torch.nn.functional.relu(x)
     |
     v
__torch_function__     <- Python-level, sees relu
     |
     v
Decompositions         <- relu -> clamp(x, min=0)
     |
     v
__torch_dispatch__     <- ATen-level, sees aten.clamp.default
     |
     v
C++ dispatcher         <- Routes to CPU/CUDA/etc. kernel
```

---

## 7. Upstream Updates (June 9-10, 2026)

Recent PyTorch main commits:

- **FSDP2 separate reduce-scatter group** — Opt-in all-gather/reduce-scatter overlap via `set_separate_reduce_scatter_group` (`#186335`)
- **Activation offloading pinned memory pool** — Dedicated pinned memory pool for activation offloading ops (`#186162`)
- **Activation offloading stride preservation** — Preserves original tensor strides across offload/reload (`#186396`)
- **BERT SDPA pattern on CUDA** — Enables BERT attention pattern for SDPA on CUDA (`#184417`)
- **DTensor group_norm fix** — Fixes crash when weight=None in group_norm under DTensor (`#184819`)
- **Pipeline parallel backward fix** — Fixes None gradient handling in pipeline backward send/recv (`#182182`)
- **Torch.cuda.stream round-trip** — Dynamo now correctly handles `torch.cuda.stream` context managers across graph breaks (`#184487`)
- **TORCH_TRACE fork-safety** — Structured tracing logs now preserved across forks (`#184772`)
- **Open registration profiler** — Activity profiler support for custom backend devices via open registration (`new test files`)

---

## Quick Reference

```python
# __torch_function__ -- Python-level override (any class)
class MyType:
    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        ...

# TorchFunctionMode -- scope-based override (no subclass needed)
class MyMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        ...

# __torch_dispatch__ -- ATen-level override (tensor subclass)
class MyTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        ...

# TorchDispatchMode -- scope-based ATen override
class MyDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ...
```

---

## Further Reading

- Source: `torch/overrides.py` (torch_function), `torch/utils/_python_dispatch.py` (torch_dispatch)
- [Extending PyTorch docs](https://pytorch.org/docs/stable/notes/extending.html)
- [`__torch_function__` protocol](https://pytorch.org/docs/stable/notes/extending.html#extending-torch)

---

<div align="center">

[← Previous Module](../18_torch_package/) | [🏠 Home](../README.md) | [Next Module →](#)

**No dedicated notebook** — see examples in `torch_function_examples.py`

</div>
