# Testing Tensor Code

## Test values, shapes, devices, and gradients

Numerical tests should explain what tolerance is appropriate and should avoid
accidentally comparing tensors on incompatible devices or dtypes.

```python
import torch

actual = torch.tensor([1.0, 2.0001])
expected = torch.tensor([1.0, 2.0])

torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
assert actual.shape == expected.shape
assert actual.dtype == expected.dtype
```

## Checklist

- Use `torch.testing.assert_close` instead of ad hoc absolute-error checks.
- Include edge shapes: batch size one, empty-compatible axes, and odd dimensions.
- Test CPU first, then instantiate device-specific coverage where supported.
- Use `gradcheck` for differentiable custom operations with double precision.
- Seed randomized tests and include a failure-reproduction hint.

Test observable contracts rather than implementation details. A test that
survives a correct refactor is more valuable than one that mirrors the code.

