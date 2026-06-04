"""
Weight Parametrization — Enforcing Constraints on Parameters
=============================================================
Learn how to use torch.nn.utils.parametrize to enforce constraints
like symmetry, orthogonality, and positivity on model weights.
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

print("=" * 65)
print("1. BASIC PARAMETRIZATION — Making a Weight Symmetric")
print("=" * 65)

# A parametrization is just an nn.Module whose forward() transforms
# the unconstrained parameter into a constrained one.

class Symmetric(nn.Module):
    """Transforms any matrix into a symmetric matrix.
    Takes the upper triangle and mirrors it to the lower triangle."""
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

# Create a regular linear layer
linear = nn.Linear(4, 4)
print(f"Before parametrization:")
print(f"  weight is symmetric: {torch.allclose(linear.weight, linear.weight.T)}")

# Register the parametrization
P.register_parametrization(linear, "weight", Symmetric())

print(f"\nAfter parametrization:")
print(f"  weight is symmetric: {torch.allclose(linear.weight, linear.weight.T)}")
print(f"  weight:\n{linear.weight.data.round(decimals=3)}")

# The original unconstrained parameter lives here:
print(f"\n  Original (unconstrained) stored at:")
print(f"    linear.parametrizations.weight.original.shape = "
      f"{linear.parametrizations.weight.original.shape}")

# The optimizer updates the ORIGINAL, and the parametrization is applied automatically
print(f"\n  Parameters for optimizer: {[n for n, _ in linear.named_parameters()]}")

print("\n" + "=" * 65)
print("2. CUSTOM PARAMETRIZATION — Positive Weights")
print("=" * 65)

class Positive(nn.Module):
    """Ensures all weight values are strictly positive via softplus."""
    def forward(self, X):
        return torch.nn.functional.softplus(X)

linear_pos = nn.Linear(3, 3)
P.register_parametrization(linear_pos, "weight", Positive())

print(f"All weights positive: {(linear_pos.weight > 0).all().item()}")
print(f"Weight:\n{linear_pos.weight.data.round(decimals=4)}")

# This is useful for things like variance parameters, distance matrices, etc.

print("\n" + "=" * 65)
print("3. BUILT-IN PARAMETRIZATIONS")
print("=" * 65)

from torch.nn.utils import parametrizations

# --- Orthogonal parametrization ---
# Forces W^T W = I (or W W^T = I for tall matrices)
# Crucial for RNNs to prevent vanishing/exploding gradients

linear_orth = nn.Linear(5, 5)
parametrizations.orthogonal(linear_orth, "weight")

W = linear_orth.weight
eye = torch.eye(5)
print("Orthogonal parametrization:")
print(f"  W^T W ≈ I: {torch.allclose(W.T @ W, eye, atol=1e-5)}")

# --- Spectral norm parametrization ---
# Constrains the largest singular value to 1
# Key for GAN stability (Discriminator)

conv = nn.Conv2d(3, 16, 3, padding=1)
parametrizations.spectral_norm(conv, "weight")

# Check: compute SVD and verify largest singular value ≈ 1
W_flat = conv.weight.reshape(conv.weight.shape[0], -1)
sigma_max = torch.linalg.svdvals(W_flat)[0]
print(f"\nSpectral norm parametrization:")
print(f"  Largest singular value: {sigma_max:.4f} (should be ≈ 1.0)")

# --- Weight norm parametrization ---
# Decouples magnitude (g) and direction (v/||v||)

linear_wn = nn.Linear(10, 5)
parametrizations.weight_norm(linear_wn, "weight")
print(f"\nWeight norm parametrization:")
print(f"  Has weight_g and weight_v in parametrizations")

print("\n" + "=" * 65)
print("4. STACKING MULTIPLE PARAMETRIZATIONS")
print("=" * 65)

# You can stack multiple parametrizations on the same parameter!
# They are applied in the order they were registered.

linear_stack = nn.Linear(4, 4)

class MakeSymmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

class ScaleDown(nn.Module):
    """Scale weights to have Frobenius norm = 1."""
    def forward(self, X):
        return X / X.norm()

P.register_parametrization(linear_stack, "weight", MakeSymmetric())
P.register_parametrization(linear_stack, "weight", ScaleDown())

W = linear_stack.weight
print(f"Symmetric: {torch.allclose(W, W.T)}")
print(f"Frobenius norm: {W.norm():.4f} (should be ≈ 1.0)")

print("\n" + "=" * 65)
print("5. CACHING FOR EFFICIENCY")
print("=" * 65)

# If you access a parametrized weight multiple times in forward(),
# the parametrization runs each time. Use caching to avoid this:

class ExpensiveParametrization(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
    def forward(self, X):
        self.call_count += 1
        return torch.matrix_exp(X - X.T)  # Expensive!

linear_exp = nn.Linear(4, 4)
param = ExpensiveParametrization()
P.register_parametrization(linear_exp, "weight", param)

# Without caching: each access recomputes
_ = linear_exp.weight
_ = linear_exp.weight
_ = linear_exp.weight
print(f"Without caching: {param.call_count} calls for 3 accesses")

param.call_count = 0
with P.cached():
    _ = linear_exp.weight
    _ = linear_exp.weight
    _ = linear_exp.weight
print(f"With caching:    {param.call_count} call for 3 accesses")

print("\n" + "=" * 65)
print("6. REMOVING PARAMETRIZATIONS")
print("=" * 65)

linear_rm = nn.Linear(3, 3)
P.register_parametrization(linear_rm, "weight", Symmetric())

print(f"Is parametrized: {P.is_parametrized(linear_rm)}")
print(f"Is parametrized (weight): {P.is_parametrized(linear_rm, 'weight')}")

# Remove, keeping the constrained (symmetric) value
P.remove_parametrizations(linear_rm, "weight", leave_parametrized=True)
print(f"\nAfter removal:")
print(f"  Is parametrized: {P.is_parametrized(linear_rm)}")
print(f"  Weight still symmetric: {torch.allclose(linear_rm.weight, linear_rm.weight.T)}")

print("\n" + "=" * 65)
print("7. TRAINING WITH PARAMETRIZED WEIGHTS")
print("=" * 65)

# Parametrizations are fully compatible with autograd and optimizers.

model = nn.Linear(10, 5)
P.register_parametrization(model, "weight", Symmetric())

# Note: For non-square, Symmetric doesn't make sense. Let's use Positive instead.
model2 = nn.Linear(10, 5)
P.register_parametrization(model2, "weight", Positive())

optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# Dummy training step
x = torch.randn(8, 10)
target = torch.randn(8, 5)

for step in range(5):
    optimizer.zero_grad()
    output = model2(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    all_positive = (model2.weight > 0).all().item()
    print(f"  Step {step+1}: loss={loss.item():.4f}, all weights positive: {all_positive}")

print("\nDone!")
