<div align="center">

[← Previous Module](../13_advanced/) | [🏠 Home](../README.md) | [Next Module →](../15_practical_utilities/)

</div>

---

> **Module 14** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 03 — Autograd](../03_autograd/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `test_example.py` | Example test file using PyTorch's TestCase |
| `reproducibility.py` | Reproducibility in PyTorch — complete setup for reproducible experiments |
| `benchmarking.py` | Benchmarking with torch.utils.benchmark |

---

# Module 14: Testing and Reproducibility

Writing tests and ensuring reproducibility are essential skills that separate
hobby projects from production-quality deep learning code. This module covers
PyTorch's testing framework, reproducibility techniques, and benchmarking.

---

## PyTorch's Testing Framework

PyTorch has its own testing infrastructure built on top of Python's `unittest`.
The key class is `TestCase` from `torch.testing._internal.common_utils`, which
provides tensor-aware assertions and convenient utilities.

### Basic Test Structure

```python
from torch.testing._internal.common_utils import run_tests, TestCase
import torch

class TestMyFeature(TestCase):
    def test_addition(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = a + b
        expected = torch.tensor([5.0, 7.0, 9.0])
        self.assertEqual(result, expected)

    def test_shape(self):
        x = torch.randn(3, 4, 5)
        self.assertEqual(x.shape, (3, 4, 5))

    def test_dtype(self):
        x = torch.zeros(5, dtype=torch.float32)
        self.assertEqual(x.dtype, torch.float32)

if __name__ == "__main__":
    run_tests()
```

### assertEqual for Tensors

PyTorch's `assertEqual` is smarter than the standard library version. For
tensors, it checks:
- Shape equality
- Dtype equality
- Value equality (with configurable tolerance for floating point)

```python
# Exact equality (for integer tensors)
self.assertEqual(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))

# Approximate equality (for float tensors) — uses default tolerances
self.assertEqual(
    torch.tensor([1.0, 2.0]),
    torch.tensor([1.0 + 1e-7, 2.0 - 1e-7]),
)

# Custom tolerances
self.assertEqual(a, b, atol=1e-4, rtol=1e-4)
```

### Useful Assertions

```python
# Check that a function raises a specific exception
with self.assertRaises(RuntimeError):
    torch.tensor([1, 2]) + torch.tensor([1, 2, 3])

# Check error message content
with self.assertRaisesRegex(RuntimeError, "size mismatch"):
    bad_operation()

# Boolean checks
self.assertTrue(torch.all(x > 0))
self.assertFalse(torch.any(torch.isnan(x)))
```

### Parametrized Tests

Use the `@parametrize` decorator to run a test with multiple inputs:

```python
from torch.testing._internal.common_utils import parametrize

class TestOps(TestCase):
    @parametrize("dtype", [torch.float32, torch.float64])
    @parametrize("size", [(2, 3), (4, 5)])
    def test_zeros(self, dtype, size):
        x = torch.zeros(size, dtype=dtype)
        self.assertEqual(x.sum().item(), 0.0)
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(x.shape, size)
```

This generates 4 test cases (2 dtypes x 2 sizes), each with a descriptive name.

### Device-Generic Tests

For testing across CPU and (optionally) GPU:

```python
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)

class TestMyOp(TestCase):
    @dtypes(torch.float32, torch.float64)
    def test_my_op(self, device, dtype):
        x = torch.randn(10, device=device, dtype=dtype)
        result = my_op(x)
        self.assertEqual(result.device.type, device)
        self.assertEqual(result.dtype, dtype)

instantiate_device_type_tests(TestMyOp, globals())
```

This creates separate test classes for CPU and CUDA (if available), testing
each dtype on each device.

---

## OpInfo Framework

PyTorch uses the OpInfo framework for systematic operator testing. Each
operator has an `OpInfo` entry that describes:
- The operator function
- Valid input dtypes
- Sample inputs (for testing)
- Reference implementations (for correctness checking)
- Gradient test configurations

While you probably won't need to write OpInfo entries unless contributing to
PyTorch core, understanding the concept helps you write better tests:

```python
# The idea: describe an op's properties declaratively, then auto-generate tests
# OpInfo("torch.add",
#     dtypes=floating_types_and(torch.half),
#     sample_inputs_func=sample_inputs_add,
#     supports_out=True,
# )
```

---

## Reproducibility

Non-deterministic behavior makes debugging nearly impossible. Here's how to
control randomness in PyTorch.

### Setting All Seeds

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Also seeds all CUDA devices:
    torch.cuda.manual_seed_all(seed)
```

### Deterministic Mode

Even with fixed seeds, some operations have non-deterministic GPU
implementations for performance. To enforce full determinism:

```python
torch.use_deterministic_algorithms(True)
# Also set the CUBLAS workspace config for full CUDA determinism:
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Warning: deterministic mode may be slower and some operations will raise
errors if no deterministic implementation exists.

### torch.backends Settings

```python
# CuDNN: control convolution algorithm selection
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# benchmark=True auto-selects the fastest algorithm, but this selection is
# non-deterministic. Set to False for reproducibility.
```

### DataLoader Reproducibility

```python
def seed_worker(worker_id):
    """Ensure each DataLoader worker has a different but reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

See `reproducibility.py` for a complete setup.

---

## Benchmarking

Reliable benchmarking requires attention to detail: warmup, synchronization,
statistical rigor.

### torch.utils.benchmark.Timer

The recommended way to benchmark PyTorch code:

```python
from torch.utils.benchmark import Timer

t = Timer(
    stmt="torch.mm(A, B)",
    globals={"A": torch.randn(256, 256), "B": torch.randn(256, 256), "torch": torch},
    label="Matrix Multiplication",
    sub_label="256x256",
)

# blocked_autorange: automatically determines the number of runs
result = t.blocked_autorange(min_run_time=1.0)
print(result)  # prints median, IQR, and other statistics
```

Why use Timer instead of `time.time()`?
1. **Warmup**: Automatically warms up JIT compilation, CUDA initialization
2. **Statistics**: Reports median and IQR, not just mean
3. **Synchronization**: Handles CUDA synchronization correctly
4. **Isolation**: Minimizes interference from other processes

### Comparing Implementations

```python
from torch.utils.benchmark import Compare

results = []
for size in [64, 128, 256, 512]:
    A = torch.randn(size, size)
    B = torch.randn(size, size)

    for label, stmt in [("mm", "torch.mm(A, B)"), ("@", "A @ B")]:
        t = Timer(
            stmt=stmt,
            globals={"A": A, "B": B, "torch": torch},
            label="matmul",
            sub_label=label,
            description=f"{size}x{size}",
        )
        results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(results)
compare.print()
```

### Common Benchmarking Mistakes

1. **No warmup**: First runs include JIT compilation and caching overhead
2. **No CUDA synchronization**: CUDA ops are asynchronous — time.time() measures
   only the kernel *launch*, not execution
3. **Too few runs**: A single measurement is noisy
4. **Benchmarking in training mode**: BatchNorm and Dropout add overhead

See `benchmarking.py` for complete examples.

---

## Common Testing Patterns

### Testing Numerical Correctness

```python
def test_softmax_correctness(self):
    x = torch.randn(5, 10)
    result = F.softmax(x, dim=-1)

    # Property: sums to 1 along the softmax dimension
    self.assertTrue(torch.allclose(result.sum(dim=-1), torch.ones(5), atol=1e-6))

    # Property: all values are in [0, 1]
    self.assertTrue((result >= 0).all())
    self.assertTrue((result <= 1).all())
```

### Testing Gradient Correctness

```python
from torch.autograd import gradcheck

def test_custom_op_gradient(self):
    func = my_custom_op
    # Use float64 for numerical gradient checking (better precision)
    input = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    self.assertTrue(gradcheck(func, input, eps=1e-6, atol=1e-4))
```

### Testing with Approximate Equality

```python
# torch.testing.assert_close: the modern way to check approximate equality
torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

# For very loose checks (e.g., stochastic operations)
self.assertTrue(torch.allclose(result, expected, atol=0.1, rtol=0.1))
```

### Testing Model Invariants

```python
def test_model_deterministic_eval(self):
    """In eval mode, same input should produce same output."""
    model = MyModel()
    model.eval()
    x = torch.randn(4, 10)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    self.assertEqual(out1, out2)

def test_model_output_shape(self):
    """Output shape should be (batch_size, num_classes)."""
    model = MyModel(num_classes=10)
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, 10))
```

---

## Summary

| Topic | Key Tool | When to Use |
|-------|----------|-------------|
| Basic testing | `TestCase`, `assertEqual` | Every project |
| Parametrized tests | `@parametrize` | Testing across configs |
| Device tests | `instantiate_device_type_tests` | Cross-device testing |
| Reproducibility | `set_seed()`, deterministic mode | Debugging, CI |
| Benchmarking | `torch.utils.benchmark.Timer` | Performance comparison |
| Gradient checks | `torch.autograd.gradcheck` | Custom autograd |

## Files in This Module

- `test_example.py` — Complete test file using PyTorch's TestCase
- `reproducibility.py` — Full reproducibility setup and verification
- `benchmarking.py` — Benchmarking with torch.utils.benchmark

---

<div align="center">

[← Previous Module](../13_advanced/) | [🏠 Home](../README.md) | [Next Module →](../15_practical_utilities/)

**[📓 Open Notebook](../notebooks/13_testing_and_reproducibility.ipynb)** — Interactive version of this module

</div>
