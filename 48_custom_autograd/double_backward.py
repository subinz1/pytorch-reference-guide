"""
Module 48: Double Backward / Higher-Order Gradients
===================================================
Shows how custom Functions support second-order derivatives via create_graph,
and how to verify with gradgradcheck. Also covers gotchas with in-place ops
and non-twice-differentiable STE patterns.

Run: python double_backward.py
"""

import torch
from torch.autograd import Function, grad, gradcheck, gradgradcheck


# ---------------------------------------------------------------------------
# Twice-differentiable softplus-like op: softplus(x) = log(1 + exp(x))
# ---------------------------------------------------------------------------
class SoftPlus(Function):
    """Numerically stable softplus with a twice-differentiable backward."""

    @staticmethod
    def forward(ctx, x):
        # softplus(x) = log1p(exp(x)) with clamp for large |x|
        ctx.save_for_backward(x)
        return torch.where(x > 20, x, torch.log1p(torch.exp(torch.clamp(x, max=20))))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # d/dx softplus = sigmoid(x); keep graph if create_graph=True
        sigmoid = torch.sigmoid(x)
        return grad_output * sigmoid


# ---------------------------------------------------------------------------
# Op that is ONLY once differentiable (bad for Hessian / double backward)
# ---------------------------------------------------------------------------
class AbsOnce(Function):
    """|x| with a subgradient at 0 — fine for first order, weak for second."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.abs()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x.sign()


def first_order_vs_create_graph():
    print("=" * 70)
    print("PART 1: create_graph=False vs True")
    print("=" * 70)
    x = torch.tensor(2.0, requires_grad=True)
    y = SoftPlus.apply(x)

    # First-order only — graph freed after backward
    (g,) = grad(y, x, create_graph=False)
    print(f"softplus'(2) ≈ {g.item():.6f} (sigmoid(2)≈{torch.sigmoid(torch.tensor(2.0)).item():.6f})")
    print(f"g.requires_grad after create_graph=False: {g.requires_grad}")

    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = SoftPlus.apply(x2)
    (g2,) = grad(y2, x2, create_graph=True)
    print(f"g2.requires_grad after create_graph=True: {g2.requires_grad}")

    # Second derivative: d/dx sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    (g2_grad,) = grad(g2, x2)
    s = torch.sigmoid(torch.tensor(2.0))
    expected = (s * (1 - s)).item()
    print(f"softplus''(2) ≈ {g2_grad.item():.6f} (expect ≈ {expected:.6f})")


def hessian_diagonal_demo():
    print("\n" + "=" * 70)
    print("PART 2: Diagonal of Hessian via double backward")
    print("=" * 70)
    x = torch.randn(5, dtype=torch.double, requires_grad=True)
    # f(x) = sum softplus(x_i)  →  Hessian is diagonal with softplus''(x_i)
    y = SoftPlus.apply(x).sum()
    (grads,) = grad(y, x, create_graph=True)
    hess_diag = []
    for i in range(x.numel()):
        (h_i,) = grad(grads[i], x, retain_graph=True)
        hess_diag.append(h_i[i].item())
    print(f"x:        {x.detach().numpy().round(4)}")
    print(f"H_ii:     {[round(v, 6) for v in hess_diag]}")
    s = torch.sigmoid(x.detach())
    expected = (s * (1 - s)).tolist()
    print(f"expected: {[round(v, 6) for v in expected]}")


def gradgradcheck_softplus():
    print("\n" + "=" * 70)
    print("PART 3: gradcheck + gradgradcheck")
    print("=" * 70)
    x = torch.randn(4, dtype=torch.double, requires_grad=True)
    ok1 = gradcheck(SoftPlus.apply, (x,), eps=1e-6, atol=1e-4)
    ok2 = gradgradcheck(SoftPlus.apply, (x,), eps=1e-6, atol=1e-4)
    print(f"SoftPlus gradcheck:     {ok1}")
    print(f"SoftPlus gradgradcheck: {ok2}")


def once_differentiable_pitfall():
    print("\n" + "=" * 70)
    print("PART 4: Once-differentiable pitfall (AbsOnce)")
    print("=" * 70)
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = AbsOnce.apply(x).sum()
    (g,) = grad(y, x, create_graph=True)
    print(f"first grads (sign): {g.tolist()}")
    # Second derivatives of sign are zero almost everywhere —
    # gradgradcheck may fail near kink or report zeros.
    try:
        (h,) = grad(g.sum(), x)
        print(f"second grads (≈0 a.e.): {h.tolist()}")
    except RuntimeError as e:
        print(f"double backward failed: {e}")

    print("Tip: for Hessians, prefer smooth surrogates (softplus, softsign).")


def nested_autograd_functional():
    print("\n" + "=" * 70)
    print("PART 5: torch.autograd.functional.hessian on SoftPlus sum")
    print("=" * 70)
    from torch.autograd.functional import hessian

    def f(v):
        return SoftPlus.apply(v).sum()

    v = torch.randn(3, dtype=torch.double)
    H = hessian(f, v)
    print(f"Hessian shape: {tuple(H.shape)}")
    print(f"Off-diagonal max |H_ij|: {(H - torch.diag(H.diag())).abs().max().item():.2e}")
    print(f"Diagonal: {H.diag().detach().numpy().round(6)}")


if __name__ == "__main__":
    first_order_vs_create_graph()
    hessian_diagonal_demo()
    gradgradcheck_softplus()
    once_differentiable_pitfall()
    nested_autograd_functional()
    print("\nAll double-backward demos finished.")
