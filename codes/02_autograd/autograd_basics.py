"""
Autograd Basics — Automatic Differentiation in PyTorch
=======================================================
Covers: gradient tracking, backward pass, gradient control, custom functions.
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("1. BASIC GRADIENT COMPUTATION")
print("=" * 60)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y + 3
loss = z.sum()

print(f"x = {x}")
print(f"y = x * 2 = {y}")
print(f"z = y + 3 = {z}")
print(f"loss = z.sum() = {loss}")

# Backward pass
loss.backward()
print(f"\nx.grad = {x.grad}")  # d(loss)/dx = 2 for each element

# grad_fn chain
print(f"\ny.grad_fn: {y.grad_fn}")
print(f"z.grad_fn: {z.grad_fn}")
print(f"loss.grad_fn: {loss.grad_fn}")

print("\n" + "=" * 60)
print("2. GRADIENT ACCUMULATION")
print("=" * 60)

x = torch.tensor([1.0, 2.0], requires_grad=True)

# Gradients accumulate by default
y1 = (x ** 2).sum()
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

y2 = (x ** 3).sum()
y2.backward()
print(f"After second backward (accumulated): x.grad = {x.grad}")

# Zero gradients before next use
x.grad.zero_()
y3 = (x ** 2).sum()
y3.backward()
print(f"After zeroing + backward: x.grad = {x.grad}")

print("\n" + "=" * 60)
print("3. DETACH & NO_GRAD")
print("=" * 60)

x = torch.randn(3, requires_grad=True)

# detach — creates a tensor that doesn't track gradients
y = x.detach()
print(f"detached requires_grad: {y.requires_grad}")

# torch.no_grad() — context manager, no graph built
with torch.no_grad():
    z = x * 2
    print(f"no_grad requires_grad: {z.requires_grad}")

# torch.inference_mode() — even faster, disallows grad-requiring ops
with torch.inference_mode():
    w = x * 3
    print(f"inference_mode requires_grad: {w.requires_grad}")

print("\n" + "=" * 60)
print("4. HIGHER-ORDER GRADIENTS")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x^3, dy/dx = 3x^2, d2y/dx2 = 6x

# First derivative
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"x = {x}")
print(f"y = x^3 = {y}")
print(f"dy/dx = 3x^2 = {grad1}")  # 3 * 4 = 12

# Second derivative
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d2y/dx2 = 6x = {grad2}")  # 6 * 2 = 12

print("\n" + "=" * 60)
print("5. JACOBIAN & HESSIAN")
print("=" * 60)

from torch.autograd.functional import jacobian, hessian

def f(x):
    return torch.stack([x[0] ** 2 + x[1], x[1] * x[2]])

x = torch.tensor([1.0, 2.0, 3.0])
J = jacobian(f, x)
print(f"f: R^3 -> R^2")
print(f"Jacobian (2x3):\n{J}")

def g(x):
    return (x ** 3).sum()

H = hessian(g, x)
print(f"\ng: R^3 -> R (sum of cubes)")
print(f"Hessian (3x3):\n{H}")

print("\n" + "=" * 60)
print("6. CUSTOM AUTOGRAD FUNCTION")
print("=" * 60)

class MySiLU(torch.autograd.Function):
    """SiLU / Swish activation: f(x) = x * sigmoid(x)"""

    @staticmethod
    def forward(ctx, x):
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid_x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_x = ctx.saved_tensors
        # f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        grad = sigmoid_x * (1 + x * (1 - sigmoid_x))
        return grad_output * grad

# Test
x = torch.randn(5, requires_grad=True)
y = MySiLU.apply(x)
y.sum().backward()
print(f"Input:    {x.data}")
print(f"Output:   {y.data}")
print(f"Gradient: {x.grad}")

# Compare with built-in
x2 = x.detach().clone().requires_grad_(True)
y2 = F.silu(x2)
y2.sum().backward()
print(f"\nBuilt-in SiLU grad matches: {torch.allclose(x.grad, x2.grad)}")

# Gradient check (numerical verification)
x_check = torch.randn(5, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(MySiLU.apply, (x_check,), eps=1e-6)
print("Gradient check passed!")

print("\nDone!")
