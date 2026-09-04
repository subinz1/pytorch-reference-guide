"""
Module 48: Custom Autograd Function Basics
==========================================
Patterns for torch.autograd.Function: forward/backward, ctx.save_for_backward,
straight-through estimators, and gradcheck verification.

Run: python autograd_function_basics.py
"""

import torch
from torch.autograd import Function, gradcheck


# ---------------------------------------------------------------------------
# PART 1: Basic custom op — square
# ---------------------------------------------------------------------------
class Square(Function):
    """y = x^2 with analytical gradient dy/dx = 2x."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x


# ---------------------------------------------------------------------------
# PART 2: Numerically stable log-sum-exp
# ---------------------------------------------------------------------------
class LogSumExp(Function):
    """log(sum(exp(x))) along the last dim, max-trick for stability."""

    @staticmethod
    def forward(ctx, x):
        # max over last dim, keep dims for broadcasting
        x_max = x.max(dim=-1, keepdim=True).values
        shifted = x - x_max
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(exp_shifted, sum_exp)
        return (sum_exp.log() + x_max).squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        exp_shifted, sum_exp = ctx.saved_tensors
        # Softmax of the shifted logits
        softmax = exp_shifted / sum_exp
        # Expand grad_output to match softmax shape
        return grad_output.unsqueeze(-1) * softmax


# ---------------------------------------------------------------------------
# PART 3: Straight-through estimator (binary threshold)
# ---------------------------------------------------------------------------
class BinarySTE(Function):
    """Forward: hard threshold; backward: identity (STE)."""

    @staticmethod
    def forward(ctx, x, threshold=0.0):
        ctx.threshold = threshold
        return (x > threshold).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient flows as if forward were identity; threshold is non-diff
        return grad_output, None


# ---------------------------------------------------------------------------
# PART 4: Multi-input op with a non-differentiable arg
# ---------------------------------------------------------------------------
class ScaleAdd(Function):
    """y = a * x + b; scale `a` may be a Python float (no grad)."""

    @staticmethod
    def forward(ctx, x, b, a):
        ctx.save_for_backward(x, b)
        ctx.a = a
        return a * x + b

    @staticmethod
    def backward(ctx, grad_output):
        x, b = ctx.saved_tensors
        a = ctx.a
        grad_x = grad_output * a if x.requires_grad else None
        grad_b = grad_output if b.requires_grad else None
        # No gradient for Python float `a`
        return grad_x, grad_b, None


def demo_square():
    print("=" * 70)
    print("PART 1: Square.apply")
    print("=" * 70)
    x = torch.tensor(3.0, requires_grad=True)
    y = Square.apply(x)
    y.backward()
    print(f"x={x.item()}, y={y.item()}, x.grad={x.grad.item()} (expect 6.0)")


def demo_logsumexp():
    print("\n" + "=" * 70)
    print("PART 2: LogSumExp vs torch.logsumexp")
    print("=" * 70)
    x = torch.randn(4, 8, requires_grad=True)
    y_custom = LogSumExp.apply(x)
    y_ref = torch.logsumexp(x, dim=-1)
    print(f"max |diff| vs torch.logsumexp: {(y_custom - y_ref).abs().max().item():.2e}")

    y_custom.sum().backward()
    print(f"grad norm: {x.grad.norm().item():.4f}")


def demo_ste():
    print("\n" + "=" * 70)
    print("PART 3: Straight-through estimator")
    print("=" * 70)
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = BinarySTE.apply(x)
    print(f"forward (binary): {y.tolist()}")
    y.sum().backward()
    print(f"STE grads (identity): {x.grad.tolist()}")


def demo_scale_add():
    print("\n" + "=" * 70)
    print("PART 4: ScaleAdd with non-diff float arg")
    print("=" * 70)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([0.5, -0.5], requires_grad=True)
    y = ScaleAdd.apply(x, b, 3.0)
    y.sum().backward()
    print(f"y={y.tolist()}, x.grad={x.grad.tolist()}, b.grad={b.grad.tolist()}")


def run_gradchecks():
    print("\n" + "=" * 70)
    print("PART 5: gradcheck")
    print("=" * 70)
    x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    ok_sq = gradcheck(Square.apply, (x,), eps=1e-6, atol=1e-4)
    ok_lse = gradcheck(LogSumExp.apply, (x,), eps=1e-6, atol=1e-4)
    print(f"Square gradcheck: {ok_sq}")
    print(f"LogSumExp gradcheck: {ok_lse}")


if __name__ == "__main__":
    demo_square()
    demo_logsumexp()
    demo_ste()
    demo_scale_add()
    run_gradchecks()
    print("\nAll demos finished.")
