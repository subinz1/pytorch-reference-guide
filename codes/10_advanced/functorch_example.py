"""
Functorch — vmap, grad, Jacobian, Hessian
===========================================
Covers: vectorized map, functional transforms, per-sample gradients.
"""

import torch
import torch.nn as nn
from torch.func import vmap, grad, jacrev, jacfwd, hessian

print("=" * 60)
print("1. vmap — VECTORIZED MAP")
print("=" * 60)

# Process single sample
def process_single(x):
    return x @ torch.randn(5, 3) + torch.randn(3)

# Manually batch
x_batch = torch.randn(8, 5)

# With vmap — automatic batching!
batched_fn = vmap(process_single)
result = batched_fn(x_batch)
print(f"Input batch:  {x_batch.shape}")
print(f"Output batch: {result.shape}")

print("\n" + "=" * 60)
print("2. grad — FUNCTIONAL GRADIENT")
print("=" * 60)

def scalar_fn(x):
    return (x ** 2).sum()

grad_fn = grad(scalar_fn)
x = torch.tensor([1.0, 2.0, 3.0])
g = grad_fn(x)
print(f"x = {x}")
print(f"grad(sum(x^2)) = 2x = {g}")

# Composable: grad of grad
grad2_fn = grad(grad(lambda x: x ** 3))
print(f"\nd2/dx2 (x^3) at x=2.0: {grad2_fn(torch.tensor(2.0))}")  # 6*2 = 12

print("\n" + "=" * 60)
print("3. PER-SAMPLE GRADIENTS")
print("=" * 60)

model = nn.Linear(10, 5)
loss_fn = nn.MSELoss(reduction='sum')

def compute_loss(params, buffers, x, y):
    pred = torch.func.functional_call(model, (params, buffers), (x,))
    return loss_fn(pred, y)

params = dict(model.named_parameters())
buffers = dict(model.named_buffers())

# Per-sample gradient computation
x_batch = torch.randn(8, 10)
y_batch = torch.randn(8, 5)

# grad over the sample dimension
ft_compute_grad = grad(compute_loss)
ft_per_sample = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
per_sample_grads = ft_per_sample(params, buffers, x_batch, y_batch)

for name, g in per_sample_grads.items():
    print(f"  {name}: per-sample grad shape = {g.shape}")

print("\n" + "=" * 60)
print("4. JACOBIAN")
print("=" * 60)

def f(x):
    return torch.stack([
        x[0] ** 2 + x[1],
        x[1] * x[2],
        torch.sin(x[0])
    ])

x = torch.tensor([1.0, 2.0, 3.0])

# Reverse-mode Jacobian (efficient when output < input)
J_rev = jacrev(f)(x)
print(f"Jacobian (reverse mode):\n{J_rev}")

# Forward-mode Jacobian (efficient when input < output)
J_fwd = jacfwd(f)(x)
print(f"\nJacobian (forward mode):\n{J_fwd}")

print(f"\nMatch: {torch.allclose(J_rev, J_fwd)}")

print("\n" + "=" * 60)
print("5. HESSIAN")
print("=" * 60)

def g(x):
    return (x ** 3).sum() + (x[0] * x[1])

x = torch.tensor([1.0, 2.0, 3.0])
H = hessian(g)(x)
print(f"Hessian of sum(x^3) + x0*x1:")
print(f"{H}")
print(f"\nDiagonal (6*x): {torch.diag(H)}")

print("\n" + "=" * 60)
print("6. COMPOSING TRANSFORMS")
print("=" * 60)

# vmap + jacrev = batched Jacobian
batch_jacobian = vmap(jacrev(f))
x_batch = torch.randn(4, 3)
J_batch = batch_jacobian(x_batch)
print(f"Batched Jacobian: input {x_batch.shape} -> Jacobian {J_batch.shape}")

# vmap + grad = batched gradient
batch_grad = vmap(grad(lambda x: (x ** 2).sum()))
grads = batch_grad(x_batch)
print(f"Batched gradients: {grads.shape}")
print(f"Expected (2*x):\n{2 * x_batch}")
print(f"Got:\n{grads}")
print(f"Match: {torch.allclose(grads, 2 * x_batch)}")

print("\nDone!")
