"""
Module 03: Gradient Basics
===========================
Core autograd concepts: requires_grad, backward(), gradient zeroing,
no_grad, inference_mode, and the standard training loop pattern.

Run: python gradient_basics.py
"""

import torch
import torch.nn as nn

print("=" * 70)
print("PART 1: BASIC GRADIENT COMPUTATION")
print("=" * 70)

# requires_grad=True tells PyTorch to track operations on this tensor
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

print(f"x = {x.item()}")
print(f"y = x^2 + 2x + 1 = {y.item()}")
print(f"y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")

# Compute dy/dx
y.backward()
print(f"\ndy/dx = 2x + 2 = {x.grad.item()}")
print(f"Expected: 2*{x.item()} + 2 = {2 * x.item() + 2}")


print("\n" + "=" * 70)
print("PART 2: VECTOR GRADIENTS")
print("=" * 70)

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# f(x) = sum(x^2) — a scalar-valued function of a vector
f = (x ** 2).sum()

print(f"x = {x.tolist()}")
print(f"f = sum(x^2) = {f.item()}")

f.backward()
print(f"df/dx = 2x = {x.grad.tolist()}")
print(f"Expected: {(2 * x).tolist()}")


print("\n" + "=" * 70)
print("PART 3: GRADIENT OF NON-SCALAR OUTPUTS")
print("=" * 70)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y is a vector, not a scalar

print(f"x = {x.tolist()}")
print(f"y = x^2 = {y.tolist()}")

# For non-scalar outputs, we must provide a gradient argument.
# This computes the vector-Jacobian product: v^T @ J
# where J is the Jacobian and v is the gradient argument.
v = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient=v)
print(f"\nWith gradient=[1,1,1] (equivalent to (x^2).sum().backward()):")
print(f"  x.grad = {x.grad.tolist()}")

# Reset and try with different weights
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
v = torch.tensor([1.0, 0.0, 0.0])  # Only care about first element
y.backward(gradient=v)
print(f"\nWith gradient=[1,0,0] (only derivative w.r.t. first element):")
print(f"  x.grad = {x.grad.tolist()}")


print("\n" + "=" * 70)
print("PART 4: GRADIENT ACCUMULATION — THE BIGGEST GOTCHA")
print("=" * 70)

x = torch.tensor(1.0, requires_grad=True)

print("Without zeroing gradients:")
for i in range(5):
    y = x * (i + 1)
    y.backward()
    print(f"  Step {i}: loss factor={(i+1)}, x.grad={x.grad.item()}")

print("\nGradients accumulated! Expected: 1, 2, 3, 4, 5")
print("Got: 1, 3, 6, 10, 15 (cumulative sums)")

print("\nWith zeroing gradients:")
x = torch.tensor(1.0, requires_grad=True)
for i in range(5):
    y = x * (i + 1)
    y.backward()
    print(f"  Step {i}: loss factor={(i+1)}, x.grad={x.grad.item()}")
    x.grad.zero_()  # Zero the gradient after using it

print("\nNow each gradient is independent!")


print("\n" + "=" * 70)
print("PART 5: torch.no_grad() CONTEXT")
print("=" * 70)

x = torch.tensor(2.0, requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}")

# Without no_grad: operations are tracked
y = x * 3
print(f"\nWith grad tracking:")
print(f"  y = x * 3, y.requires_grad={y.requires_grad}, y.grad_fn={y.grad_fn}")

# With no_grad: operations are NOT tracked
with torch.no_grad():
    z = x * 3
    print(f"\nWith torch.no_grad():")
    print(f"  z = x * 3, z.requires_grad={z.requires_grad}, z.grad_fn={z.grad_fn}")

# Use case: manual parameter update
print("\nManual parameter update pattern:")
w = torch.tensor(5.0, requires_grad=True)
loss = (w - 3) ** 2
loss.backward()
print(f"  Before update: w={w.item()}, grad={w.grad.item()}")

with torch.no_grad():
    w -= 0.1 * w.grad

print(f"  After update:  w={w.item()}")
print(f"  (Without no_grad, this would create a new tensor and break leaf status)")


print("\n" + "=" * 70)
print("PART 6: torch.inference_mode()")
print("=" * 70)

x = torch.tensor(1.0, requires_grad=True)

with torch.inference_mode():
    y = x * 2
    print(f"In inference_mode:")
    print(f"  y.requires_grad: {y.requires_grad}")

    # inference_mode tensors cannot participate in autograd at all
    print(f"  y is inference tensor: {y.is_inference()}")

print("\nKey differences from no_grad:")
print("  - inference_mode is faster (skips more autograd bookkeeping)")
print("  - inference tensors cannot be used in autograd later")
print("  - Use for production inference; use no_grad for validation during training")


print("\n" + "=" * 70)
print("PART 7: detach()")
print("=" * 70)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach()

print(f"y = x^2")
print(f"y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")
print(f"\nz = y.detach()")
print(f"z.requires_grad: {z.requires_grad}")
print(f"z.grad_fn: {z.grad_fn}")
print(f"z shares data with y: {z.data_ptr() == y.data_ptr()}")

# Common use case: getting a value without gradient tracking
print(f"\nGetting scalar value:")
print(f"  loss.item() for scalars: {y.item()}")
print(f"  tensor.detach().numpy() for arrays: not shown (would need CPU tensor)")

# GAN-style gradient stopping
print("\nGAN-style usage:")
print("  fake = generator(noise)")
print("  d_loss = discriminator(fake.detach())  # stops gradient to generator")


print("\n" + "=" * 70)
print("PART 8: MULTI-VARIABLE GRADIENTS")
print("=" * 70)

# f(x, y) = x^2*y + y^3
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

f = x ** 2 * y + y ** 3  # f = x²y + y³

f.backward()
print(f"f(x,y) = x^2*y + y^3")
print(f"x={x.item()}, y={y.item()}")
print(f"f = {f.item()}")
print(f"\ndf/dx = 2xy = {x.grad.item()} (expected: {2 * 2.0 * 3.0})")
print(f"df/dy = x^2 + 3y^2 = {y.grad.item()} (expected: {4.0 + 27.0})")


print("\n" + "=" * 70)
print("PART 9: THE STANDARD TRAINING LOOP")
print("=" * 70)

# Create a simple linear regression problem
torch.manual_seed(42)
X = torch.randn(100, 3)  # 100 samples, 3 features
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = torch.tensor(1.0)
Y = X @ true_w + true_b + 0.1 * torch.randn(100)

# Model parameters
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

print(f"True weights: {true_w.tolist()}, bias: {true_b.item()}")
print(f"Initial weights: {w.tolist()}")
print(f"Initial bias: {b.item()}")

print(f"\n{'Epoch':>5} {'Loss':>10} {'w':>25} {'b':>8}")
print("-" * 55)

for epoch in range(200):
    # Forward pass
    y_pred = X @ w + b
    loss = ((y_pred - Y) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update parameters (no_grad prevents autograd tracking)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 40 == 0:
        w_str = f"[{w[0].item():.3f}, {w[1].item():.3f}, {w[2].item():.3f}]"
        print(f"{epoch:5d} {loss.item():10.6f} {w_str:>25} {b.item():8.4f}")

print(f"\nFinal weights: [{w[0].item():.3f}, {w[1].item():.3f}, {w[2].item():.3f}]")
print(f"True weights:  {true_w.tolist()}")
print(f"Final bias: {b.item():.3f}, True bias: {true_b.item()}")


print("\n" + "=" * 70)
print("PART 10: USING torch.optim (THE BETTER WAY)")
print("=" * 70)

# Same problem, but using an optimizer
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.1)

print(f"{'Epoch':>5} {'Loss':>10}")
print("-" * 18)

for epoch in range(200):
    optimizer.zero_grad()           # 1. Zero gradients

    y_pred = X @ w + b             # 2. Forward
    loss = ((y_pred - Y) ** 2).mean()  # 3. Loss

    loss.backward()                 # 4. Backward
    optimizer.step()                # 5. Update

    if epoch % 40 == 0:
        print(f"{epoch:5d} {loss.item():10.6f}")

print(f"\nFinal weights: [{w[0].item():.3f}, {w[1].item():.3f}, {w[2].item():.3f}]")
print(f"Final bias: {b.item():.3f}")
print("\nThe 5-step loop: zero_grad → forward → loss → backward → step")
print("This pattern is the heartbeat of all PyTorch training.")

print("\n" + "=" * 70)
print("Gradient basics demonstration complete!")
print("=" * 70)
