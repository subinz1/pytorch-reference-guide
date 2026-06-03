"""
Functorch Transforms (torch.func): vmap, grad, jacrev, hessian
===============================================================

Demonstrates the core function transforms available in torch.func.
These allow you to write simple single-example code and transform it
into batched, differentiated, or higher-order derivative computations.
"""

import torch
from torch.func import vmap, grad, jacrev, jacfwd, hessian


def demo_vmap():
    """vmap: Vectorized map — automatic batching of functions."""
    print("=" * 60)
    print("VMAP: Vectorized Map")
    print("=" * 60)

    # Single-example function: compute the dot product of two vectors
    def dot_product(a, b):
        return torch.dot(a, b)

    # Without vmap: manual loop
    batch_a = torch.randn(5, 3)
    batch_b = torch.randn(5, 3)
    results_loop = torch.stack([
        dot_product(batch_a[i], batch_b[i]) for i in range(5)
    ])

    # With vmap: automatic batching over dimension 0
    results_vmap = vmap(dot_product)(batch_a, batch_b)

    print(f"  Loop result:  {results_loop}")
    print(f"  vmap result:  {results_vmap}")
    print(f"  Match: {torch.allclose(results_loop, results_vmap)}")

    # vmap with in_dims: control which dimension to vectorize over
    # Here, a is batched (dim 0) but b is the same for all samples
    shared_b = torch.randn(3)
    results = vmap(dot_product, in_dims=(0, None))(batch_a, shared_b)
    print(f"\n  vmap with shared vector: {results.shape}")

    # Nested vmap: vectorize over multiple dimensions
    matrix = torch.randn(4, 5)
    # Apply a per-element function across both dimensions
    abs_fn = lambda x: torch.abs(x)  # noqa: E731
    result = vmap(vmap(abs_fn))(matrix)
    print(f"  Nested vmap (element-wise abs): {result.shape}")


def demo_grad():
    """grad: Functional gradient computation."""
    print("\n" + "=" * 60)
    print("GRAD: Functional Gradients")
    print("=" * 60)

    # Simple scalar function
    def f(x):
        return torch.sin(x).sum()

    x = torch.tensor([0.0, 1.0, 2.0])
    gradient = grad(f)(x)
    expected = torch.cos(x)
    print(f"  f(x) = sin(x).sum()")
    print(f"  x =        {x.tolist()}")
    print(f"  grad f(x) = {gradient.tolist()}")
    print(f"  cos(x) =   {expected.tolist()}")
    print(f"  Match: {torch.allclose(gradient, expected)}")

    # grad of a function with multiple arguments
    def weighted_sum(x, w):
        return (x * w).sum()

    x = torch.tensor([1.0, 2.0, 3.0])
    w = torch.tensor([0.5, 0.3, 0.2])

    # argnums controls which argument to differentiate w.r.t.
    grad_x = grad(weighted_sum, argnums=0)(x, w)
    grad_w = grad(weighted_sum, argnums=1)(x, w)
    grad_both = grad(weighted_sum, argnums=(0, 1))(x, w)

    print(f"\n  f(x, w) = (x * w).sum()")
    print(f"  df/dx = {grad_x.tolist()} (should be w)")
    print(f"  df/dw = {grad_w.tolist()} (should be x)")
    print(f"  Both:   dx={grad_both[0].tolist()}, dw={grad_both[1].tolist()}")

    # Higher-order gradients via composition
    def g(x):
        return (x ** 3).sum()

    x = torch.tensor([1.0, 2.0])
    first = grad(g)(x)                    # 3x^2
    second = grad(lambda x: grad(g)(x).sum())(x)  # 6x  # noqa: E731
    print(f"\n  g(x) = x^3")
    print(f"  g'(x)  = {first.tolist()} (3x^2 = {(3 * x**2).tolist()})")
    print(f"  g''(x) = {second.tolist()} (6x = {(6 * x).tolist()})")


def demo_jacobian():
    """jacrev / jacfwd: Jacobian matrix computation."""
    print("\n" + "=" * 60)
    print("JACREV / JACFWD: Jacobian Computation")
    print("=" * 60)

    # Vector-valued function: R^3 -> R^2
    def f(x):
        return torch.stack([
            x[0] ** 2 + x[1] * x[2],   # f1 = x0^2 + x1*x2
            torch.sin(x[0]) + x[2] ** 2  # f2 = sin(x0) + x2^2
        ])

    x = torch.tensor([1.0, 2.0, 3.0])

    # The Jacobian J[i,j] = df_i/dx_j
    J_rev = jacrev(f)(x)
    J_fwd = jacfwd(f)(x)

    print(f"  f(x) = [x0^2 + x1*x2, sin(x0) + x2^2]")
    print(f"  x = {x.tolist()}")
    print(f"\n  Jacobian (reverse-mode):")
    print(f"    {J_rev}")
    print(f"\n  Jacobian (forward-mode):")
    print(f"    {J_fwd}")
    print(f"  Match: {torch.allclose(J_rev, J_fwd)}")

    # Analytical Jacobian for verification:
    # J = [[2*x0, x2, x1],
    #      [cos(x0), 0, 2*x2]]
    J_analytical = torch.tensor([
        [2 * x[0], x[2], x[1]],
        [torch.cos(x[0]), 0.0, 2 * x[2]],
    ])
    print(f"\n  Analytical Jacobian:")
    print(f"    {J_analytical}")
    print(f"  Match: {torch.allclose(J_rev, J_analytical)}")

    # Performance comparison: jacrev vs jacfwd
    # jacrev is O(output_dim), jacfwd is O(input_dim)
    print(f"\n  For f: R^3 -> R^2:")
    print(f"    jacrev: 2 backward passes (output_dim = 2)")
    print(f"    jacfwd: 3 forward passes (input_dim = 3)")
    print(f"    -> jacrev is more efficient here")


def demo_hessian():
    """hessian: Second-order derivative matrix."""
    print("\n" + "=" * 60)
    print("HESSIAN: Second-Order Derivatives")
    print("=" * 60)

    # Scalar function: R^3 -> R
    def f(x):
        return x[0] ** 2 * x[1] + x[1] ** 3 + x[2] ** 2

    x = torch.tensor([1.0, 2.0, 3.0])

    H = hessian(f)(x)
    print(f"  f(x) = x0^2 * x1 + x1^3 + x2^2")
    print(f"  x = {x.tolist()}")
    print(f"\n  Hessian H[i,j] = d^2f / dx_i dx_j:")
    print(f"    {H}")

    # Analytical Hessian:
    # df/dx0 = 2*x0*x1,  df/dx1 = x0^2 + 3*x1^2,  df/dx2 = 2*x2
    # H = [[2*x1, 2*x0, 0],
    #      [2*x0, 6*x1, 0],
    #      [0,    0,    2]]
    H_analytical = torch.tensor([
        [2 * x[1], 2 * x[0], 0.0],
        [2 * x[0], 6 * x[1], 0.0],
        [0.0, 0.0, 2.0],
    ])
    print(f"\n  Analytical Hessian:")
    print(f"    {H_analytical}")
    print(f"  Match: {torch.allclose(H, H_analytical)}")
    print(f"  Symmetric: {torch.allclose(H, H.T)}")


def demo_composition():
    """Composing transforms: vmap(jacrev(f)), batched Hessians, etc."""
    print("\n" + "=" * 60)
    print("COMPOSING TRANSFORMS")
    print("=" * 60)

    # Batched Jacobian: compute Jacobian for each sample in a batch
    def f(x):
        return torch.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])

    batch_x = torch.randn(8, 2)

    # vmap(jacrev(f)) = batched Jacobian
    batched_J = vmap(jacrev(f))(batch_x)
    print(f"  Batched Jacobian shape: {list(batched_J.shape)}")
    print(f"  (batch=8, output=3, input=2)")

    # Batched Hessian
    def g(x):
        return (x ** 3).sum()

    batch_x = torch.randn(4, 3)
    batched_H = vmap(hessian(g))(batch_x)
    print(f"\n  Batched Hessian shape: {list(batched_H.shape)}")
    print(f"  (batch=4, input=3, input=3)")

    # vmap(grad(f)): compute gradient for each element in a batch
    def loss_per_sample(x):
        return (x ** 2).sum()

    batch_x = torch.randn(16, 5)
    per_sample_grads = vmap(grad(loss_per_sample))(batch_x)
    expected = 2 * batch_x
    print(f"\n  Per-sample gradients shape: {list(per_sample_grads.shape)}")
    print(f"  Matches 2*x: {torch.allclose(per_sample_grads, expected)}")


if __name__ == "__main__":
    demo_vmap()
    demo_grad()
    demo_jacobian()
    demo_hessian()
    demo_composition()
    print("\n" + "=" * 60)
    print("All functorch transform demos completed successfully!")
    print("=" * 60)
