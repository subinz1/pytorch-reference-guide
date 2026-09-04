# Module 48: Custom Autograd Functions

## Overview

`torch.autograd.Function` lets you define **custom forward and backward** behavior
when built-in ops are not enough: fused kernels, numerically stable formulas,
or ops that need a hand-written gradient. This module goes beyond the intro in
[Module 03](../03_autograd/) and focuses on production patterns: `ctx` usage,
`@staticmethod`, `gradcheck`, and preparing for higher-order gradients.

## Key Concepts

### Subclassing `torch.autograd.Function`

Implement two static methods:

- **`forward(ctx, *args)`** — compute the output; save tensors needed for backward
  with `ctx.save_for_backward(...)` (preferred) or set attributes on `ctx`.
- **`backward(ctx, *grad_outputs)`** — return gradients for each forward input
  that requires grad (use `None` for non-tensor / non-differentiable args).

Call the function with `MyOp.apply(x, y)` — never instantiate the class yourself.

### What to Save on `ctx`

- Prefer `ctx.save_for_backward(*tensors)` so autograd can free memory when safe.
- Store non-tensor metadata as plain `ctx` attributes (`ctx.dim`, `ctx.eps`).
- Avoid saving huge intermediates if you can recompute them cheaply in backward.

### Correctness: `gradcheck` / `gradgradcheck`

Always verify analytical gradients against finite differences:

```python
from torch.autograd import gradcheck
assert gradcheck(MyOp.apply, (x.double(),), eps=1e-6, atol=1e-4)
```

Use `gradgradcheck` when you need second-order correctness (see `double_backward.py`).

### Common Patterns

| Pattern | When |
|---------|------|
| Elementwise fused op | Custom CUDA/Triton forward + matching backward |
| Numerically stable log-sum-exp | Forward uses max-trick; backward uses softmax |
| Straight-through estimator (STE) | Forward discrete; backward identity |
| Non-differentiable arg | Return `None` in that backward slot |

## Examples

```python
import torch
from torch.autograd import Function

class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x

x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)
y.backward()
print(x.grad)  # 6.0
```

## When to Use

- You need a **custom gradient** (STE, clipped grads, surrogate losses).
- You fuse ops for performance and must teach autograd the fused backward.
- Built-in ops lack the formula you need, or you want a stable formulation.
- Prefer functorch / existing ops when they already cover the math — less code, fewer bugs.

## Files in This Module

| File | Description |
|------|-------------|
| `autograd_function_basics.py` | Forward/backward patterns, STE, gradcheck |
| `double_backward.py` | Higher-order grads and `create_graph` |

## References

- [Extending PyTorch — Custom Functions](https://pytorch.org/docs/stable/notes/extending.html)
- [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#function)
- [gradcheck](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html)
- Module 03: [Autograd](../03_autograd/) · Module 38: [Compiled Autograd](../38_compiled_autograd/)
