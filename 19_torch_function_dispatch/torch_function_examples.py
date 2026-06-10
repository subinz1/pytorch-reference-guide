"""
__torch_function__ & __torch_dispatch__ — Tensor Subclassing
==============================================================
Override how PyTorch operations work on custom types.
All examples run on CPU.
"""

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

print("=" * 65)
print("1. __torch_function__ — Custom Tensor Wrapper")
print("=" * 65)

class ScaledTensor:
    """A wrapper that tracks a scaling factor alongside tensor data."""

    def __init__(self, data, scale=1.0):
        self.data = data
        self.scale = scale

    def __repr__(self):
        return f"ScaledTensor(shape={list(self.data.shape)}, scale={self.scale})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Unwrap ScaledTensors to raw data
        new_args = []
        scale = 1.0
        for a in args:
            if isinstance(a, ScaledTensor):
                new_args.append(a.data)
                scale = a.scale
            else:
                new_args.append(a)

        result = func(*new_args, **kwargs)

        if isinstance(result, torch.Tensor):
            return ScaledTensor(result, scale)
        return result

x = ScaledTensor(torch.tensor([1.0, 2.0, 3.0]), scale=0.5)
y = ScaledTensor(torch.tensor([4.0, 5.0, 6.0]), scale=0.5)

z = torch.add(x, y)
print(f"torch.add(x, y) = {z}")
print(f"  Result data: {z.data}")
print(f"  Scale preserved: {z.scale}")

w = torch.mul(x, 3)
print(f"torch.mul(x, 3) = {w}")
print(f"  Result data: {w.data}")

print("\n" + "=" * 65)
print("2. TorchFunctionMode — Logging All Operations")
print("=" * 65)

class LoggingMode(TorchFunctionMode):
    """Logs every torch operation without modifying behavior."""

    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else str(func)
        self.log.append(name)
        return func(*args, **kwargs)

logger = LoggingMode()
with logger:
    x = torch.randn(3, 4)
    y = x + 1
    z = torch.relu(y)
    w = z.mean()

print(f"Operations logged ({len(logger.log)}):")
for op in logger.log[:10]:
    print(f"  {op}")
if len(logger.log) > 10:
    print(f"  ... and {len(logger.log) - 10} more")

print("\n" + "=" * 65)
print("3. TorchFunctionMode — Operation Counter")
print("=" * 65)

class OpCounter(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.counts = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **kwargs)

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

counter = OpCounter()
with counter:
    output = model(torch.randn(8, 20))

print(f"Operations in forward pass:")
for name, count in sorted(counter.counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:30s}: {count}x")

print("\n" + "=" * 65)
print("4. __torch_dispatch__ — ATen-Level Override")
print("=" * 65)

from torch.utils._python_dispatch import TorchDispatchMode

class ATenCounter(TorchDispatchMode):
    """Count ATen-level operations (lower-level than torch_function)."""

    def __init__(self):
        super().__init__()
        self.ops = {}

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = str(func.name())
        self.ops[name] = self.ops.get(name, 0) + 1
        return func(*args, **kwargs)

aten_counter = ATenCounter()
with aten_counter:
    x = torch.randn(4, 20)
    output = model(x)
    loss = output.sum()
    loss.backward()

print(f"ATen operations in forward+backward ({len(aten_counter.ops)} unique ops):")
for name, count in sorted(aten_counter.ops.items(), key=lambda x: -x[1])[:12]:
    print(f"  {name:45s}: {count}x")

print("\n" + "=" * 65)
print("5. COMPARING torch_function vs torch_dispatch")
print("=" * 65)

# torch_function sees high-level ops
tf_counter = OpCounter()
with tf_counter:
    y = torch.nn.functional.relu(torch.randn(5))

# torch_dispatch sees ATen ops
td_counter = ATenCounter()
with td_counter:
    y = torch.nn.functional.relu(torch.randn(5))

print("F.relu as seen by __torch_function__:")
for name in tf_counter.counts:
    print(f"  {name}")

print("\nF.relu as seen by __torch_dispatch__:")
for name in td_counter.ops:
    print(f"  {name}")

print("""
Key insight:
  __torch_function__ sees: F.relu (high-level Python API)
  __torch_dispatch__ sees: aten.relu.default (low-level ATen op)

  For composite ops (e.g., F.layer_norm), torch_dispatch sees
  the decomposed primitive ops (aten.mean, aten.sub, aten.mul, etc.)
""")

print("=" * 65)
print("6. PRACTICAL: Shape Tracer Mode")
print("=" * 65)

class ShapeTracer(TorchDispatchMode):
    """Track shapes through a model."""

    def __init__(self):
        super().__init__()
        self.traces = []

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)

        in_shapes = tuple(a.shape for a in args if isinstance(a, torch.Tensor))
        out_shape = result.shape if isinstance(result, torch.Tensor) else None

        if in_shapes and out_shape is not None:
            self.traces.append({
                'op': func.name(),
                'in': in_shapes,
                'out': out_shape
            })
        return result

tracer = ShapeTracer()
model_test = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

with tracer:
    model_test(torch.randn(32, 784))

print(f"Shape trace ({len(tracer.traces)} ops with tensor I/O):")
for t in tracer.traces[:8]:
    in_str = ", ".join(str(tuple(s)) for s in t['in'])
    print(f"  {t['op']:35s} [{in_str}] -> {tuple(t['out'])}")
if len(tracer.traces) > 8:
    print(f"  ... and {len(tracer.traces) - 8} more")

print("\n" + "=" * 65)
print("7. PRACTICAL: Tensor with Physical Units")
print("=" * 65)

class UnitTensor:
    """Tensor that carries physical units."""

    def __init__(self, data, unit=""):
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        self.unit = unit

    def __repr__(self):
        return f"{self.data.item():.4f} [{self.unit}]"

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
        elif func in (torch.add, torch.sub) and len(tensors) == 2:
            if tensors[0].unit != tensors[1].unit:
                raise ValueError(f"Cannot {func.__name__} {tensors[0].unit} and {tensors[1].unit}")
            unit = tensors[0].unit
        else:
            unit = tensors[0].unit if tensors else ""

        if isinstance(result, torch.Tensor):
            return UnitTensor(result, unit)
        return result

distance = UnitTensor(100.0, "m")
time = UnitTensor(9.58, "s")
speed = torch.div(distance, time)
print(f"100m sprint: {speed}")

mass = UnitTensor(70.0, "kg")
force = torch.mul(mass, UnitTensor(9.81, "m/s^2"))
print(f"Gravity force: {force}")

d1 = UnitTensor(10.0, "m")
d2 = UnitTensor(20.0, "m")
total = torch.add(d1, d2)
print(f"Total distance: {total}")

try:
    bad = torch.add(distance, time)
except ValueError as e:
    print(f"Unit mismatch caught: {e}")

print("\n" + "=" * 65)
print("8. DISPATCH STACK VISUALIZATION")
print("=" * 65)

print("""
The full dispatch stack for `x + y` where x is a custom tensor subclass:

  User code:  x + y
       │
       ▼
  Python:     torch.Tensor.__add__(x, y)
       │
       ▼
  __torch_function__  ← Intercepts here (Python API level)
       │                 Sees: torch.Tensor.__add__
       ▼
  Decompositions      ← Composite ops broken into primitives
       │
       ▼
  __torch_dispatch__  ← Intercepts here (ATen op level)
       │                 Sees: aten.add.Tensor
       ▼
  C++ Dispatcher      ← Routes to correct kernel
       │                 (CPU, CUDA, Autograd, etc.)
       ▼
  Kernel              ← Actual computation
""")

print("=" * 65)
print("SUMMARY")
print("=" * 65)

print("""
Protocol                 Level      Subclass?   Best For
─────────────────────────────────────────────────────────
__torch_function__       Python     No          Wrappers, units, logging
TorchFunctionMode        Python     No          Scoped logging/mocking
__torch_dispatch__       ATen       Yes         Tensor subclasses (DTensor)
TorchDispatchMode        ATen       No          Op counting, shape tracing
""")

print("Done!")
