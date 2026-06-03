"""
Module 03: Higher-Order Gradients, Jacobians, and Hessians
===========================================================
Computing second derivatives, Jacobian matrices, Hessians, and
practical applications like gradient penalty and physics-informed NNs.

Run: python higher_order_gradients.py
"""

import torch
from torch.autograd.functional import jacobian, hessian, jvp, vjp

print("=" * 70)
print("PART 1: HIGHER-ORDER DERIVATIVES")
print("=" * 70)

# f(x) = x^4  →  f' = 4x^3  →  f'' = 12x^2  →  f''' = 24x  →  f'''' = 24
x = torch.tensor(2.0, requires_grad=True)
y = x ** 4

# First derivative: create_graph=True keeps the graph for further differentiation
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f(x) = x^4 at x = {x.item()}")
print(f"f'(x)   = 4x^3   = {dy_dx.item()}")

# Second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
print(f"f''(x)  = 12x^2  = {d2y_dx2.item()}")

# Third derivative
d3y_dx3 = torch.autograd.grad(d2y_dx2, x, create_graph=True)[0]
print(f"f'''(x) = 24x    = {d3y_dx3.item()}")

# Fourth derivative (no more create_graph needed since we stop here)
d4y_dx4 = torch.autograd.grad(d3y_dx3, x)[0]
print(f"f''''(x) = 24    = {d4y_dx4.item()}")

print(f"\nVerification:")
print(f"  4 * 2^3 = {4 * 2**3} ✓")
print(f"  12 * 2^2 = {12 * 2**2} ✓")
print(f"  24 * 2 = {24 * 2} ✓")
print(f"  24 = 24 ✓")


print("\n" + "=" * 70)
print("PART 2: MULTIVARIABLE SECOND DERIVATIVES")
print("=" * 70)

# f(x, y) = x^2*y + y^3*x
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

f = x ** 2 * y + y ** 3 * x
print(f"f(x, y) = x^2*y + y^3*x at ({x.item()}, {y.item()})")
print(f"f = {f.item()}")

# First partial derivatives
df_dx = torch.autograd.grad(f, x, create_graph=True)[0]
df_dy = torch.autograd.grad(f, y, create_graph=True)[0]
print(f"\ndf/dx = 2xy + y^3 = {df_dx.item()} (expected {2*1*2 + 8})")
print(f"df/dy = x^2 + 3y^2*x = {df_dy.item()} (expected {1 + 3*4*1})")

# Second partial derivatives
d2f_dxdx = torch.autograd.grad(df_dx, x, create_graph=True, retain_graph=True)[0]
d2f_dxdy = torch.autograd.grad(df_dx, y, retain_graph=True)[0]
d2f_dydx = torch.autograd.grad(df_dy, x, retain_graph=True)[0]
d2f_dydy = torch.autograd.grad(df_dy, y)[0]

print(f"\nHessian matrix:")
print(f"  d^2f/dx^2  = 2y     = {d2f_dxdx.item()} (expected {2*2})")
print(f"  d^2f/dxdy  = 2x+3y^2 = {d2f_dxdy.item()} (expected {2*1 + 3*4})")
print(f"  d^2f/dydx  = 2x+3y^2 = {d2f_dydx.item()} (expected {2*1 + 3*4})")
print(f"  d^2f/dy^2  = 6yx    = {d2f_dydy.item()} (expected {6*2*1})")
print(f"\nHessian is symmetric (mixed partials are equal): "
      f"{abs(d2f_dxdy.item() - d2f_dydx.item()) < 1e-6}")


print("\n" + "=" * 70)
print("PART 3: JACOBIAN COMPUTATION")
print("=" * 70)

def f_vec(x):
    """Vector function f: R^2 -> R^3."""
    return torch.stack([
        x[0] ** 2 + x[1],       # f1 = x0^2 + x1
        x[0] * x[1],             # f2 = x0 * x1
        torch.sin(x[0]) + x[1] ** 2  # f3 = sin(x0) + x1^2
    ])

x = torch.tensor([1.0, 2.0])
print(f"f: R^2 -> R^3")
print(f"x = {x.tolist()}")
print(f"f(x) = {f_vec(x).tolist()}")

J = jacobian(f_vec, x)
print(f"\nJacobian (3x2):\n{J}")
print(f"""
Expected Jacobian:
  [[2*x0,    1     ],     = [[2,    1   ],
   [x1,      x0    ],        [2,    1   ],
   [cos(x0), 2*x1  ]]        [{torch.cos(torch.tensor(1.0)).item():.4f}, 4   ]]
""")

# Verify one entry manually
x_v = torch.tensor([1.0, 2.0], requires_grad=True)
f_out = f_vec(x_v)
f_out[0].backward(retain_graph=True)
print(f"Verification: df1/dx = {x_v.grad.tolist()} (matches row 0 of Jacobian)")


print("\n" + "=" * 70)
print("PART 4: HESSIAN COMPUTATION")
print("=" * 70)

def f_scalar(x):
    """Scalar function f: R^3 -> R."""
    return x[0] ** 2 * x[1] + x[1] ** 3 + x[2] ** 2 * x[0]

x = torch.tensor([1.0, 2.0, 3.0])
print(f"f(x) = x0^2*x1 + x1^3 + x2^2*x0")
print(f"x = {x.tolist()}")
print(f"f(x) = {f_scalar(x).item()}")

H = hessian(f_scalar, x)
print(f"\nHessian (3x3):\n{H}")
print(f"""
Expected Hessian:
  d^2f/dx0^2 = 2*x1 = {2*2}
  d^2f/dx0dx1 = 2*x0 = {2*1}
  d^2f/dx0dx2 = 2*x2 = {2*3}
  d^2f/dx1^2 = 6*x1 = {6*2}
  d^2f/dx1dx2 = 0
  d^2f/dx2^2 = 2*x0 = {2*1}
""")

# Check positive definiteness (eigenvalues > 0?)
eigenvalues = torch.linalg.eigvalsh(H)
print(f"Hessian eigenvalues: {eigenvalues.tolist()}")
print(f"Positive definite (local minimum): {(eigenvalues > 0).all().item()}")


print("\n" + "=" * 70)
print("PART 5: JVP AND VJP")
print("=" * 70)

def f_multi(x):
    """f: R^3 -> R^2."""
    return torch.stack([
        x[0] * x[1] + x[2],
        x[0] ** 2 + x[1] * x[2]
    ])

x = torch.tensor([1.0, 2.0, 3.0])

# Full Jacobian (for reference)
J = jacobian(f_multi, x)
print(f"Full Jacobian:\n{J}\n")

# JVP: Jacobian-vector product (forward mode)
# Computes J @ v without materializing J
v = torch.tensor([1.0, 0.0, 0.0])
_, jvp_result = jvp(f_multi, (x,), (v,))
print(f"JVP with v=[1,0,0] (= first column of J): {jvp_result}")
print(f"Verify J @ v: {J @ v}")

# VJP: Vector-Jacobian product (reverse mode)
# Computes v^T @ J without materializing J
u = torch.tensor([1.0, 0.0])
_, vjp_fn = vjp(f_multi, x)
vjp_result = vjp_fn(u)
print(f"\nVJP with u=[1,0] (= first row of J): {vjp_result[0]}")
print(f"Verify u^T @ J: {u @ J}")

print(f"""
JVP computes one column of J per call (efficient for n >> m)
VJP computes one row of J per call (efficient for m >> n)
For neural networks (m=1 scalar loss, n=millions of params), VJP wins.
""")


print("=" * 70)
print("PART 6: GRADIENT PENALTY (WGAN-GP STYLE)")
print("=" * 70)

# In WGAN-GP, we penalize the gradient norm of the discriminator
# This requires computing gradients of gradients

# Simulate a simple "discriminator" function
def discriminator(x):
    return (x ** 2).sum()

# Compute gradient penalty
x = torch.randn(5, requires_grad=True)
output = discriminator(x)

# First: get gradients of output w.r.t. input
gradients = torch.autograd.grad(
    outputs=output,
    inputs=x,
    create_graph=True  # Need this to differentiate through the gradient!
)[0]

gradient_norm = gradients.norm(2)
gradient_penalty = (gradient_norm - 1) ** 2  # Penalize deviation from norm 1

print(f"Input x: {x.detach().round(decimals=3).tolist()}")
print(f"Discriminator output: {output.item():.4f}")
print(f"Gradient w.r.t. input: {gradients.detach().round(decimals=3).tolist()}")
print(f"Gradient norm: {gradient_norm.item():.4f}")
print(f"Gradient penalty: {gradient_penalty.item():.4f}")

# The penalty itself has gradients (thanks to create_graph=True)
gradient_penalty.backward()
print(f"Gradient of penalty w.r.t. x: {x.grad.round(decimals=4).tolist()}")
print("\nThis second-order gradient is what makes WGAN-GP training stable.")


print("\n" + "=" * 70)
print("PART 7: PHYSICS-INFORMED NEURAL NETWORK (PINN) EXAMPLE")
print("=" * 70)

# Solving a simple ODE: dy/dx = -2y, y(0) = 1
# Analytical solution: y = exp(-2x)

# Create a tiny neural network as the solution approximation
torch.manual_seed(42)
net = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

print("Training PINN to solve: dy/dx = -2y, y(0) = 1")
print(f"{'Epoch':>6} {'Physics Loss':>14} {'IC Loss':>10} {'Total':>10}")
print("-" * 45)

for epoch in range(2000):
    optimizer.zero_grad()

    # Collocation points (where we enforce the ODE)
    x_phys = torch.linspace(0, 2, 50).unsqueeze(1).requires_grad_(True)
    y_pred = net(x_phys)

    # Compute dy/dx using autograd (the key PINN ingredient!)
    dy_dx = torch.autograd.grad(
        outputs=y_pred,
        inputs=x_phys,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # Physics loss: dy/dx + 2y = 0
    physics_residual = dy_dx + 2 * y_pred
    physics_loss = (physics_residual ** 2).mean()

    # Initial condition: y(0) = 1
    x_ic = torch.tensor([[0.0]])
    y_ic = net(x_ic)
    ic_loss = (y_ic - 1.0) ** 2

    total_loss = physics_loss + 10 * ic_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(f"{epoch:6d} {physics_loss.item():14.6f} {ic_loss.item():10.6f} {total_loss.item():10.6f}")

# Evaluate
print("\nEvaluation:")
x_test = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
with torch.no_grad():
    y_test = net(x_test)

print(f"{'x':>6} {'PINN':>10} {'Exact':>10} {'Error':>10}")
print("-" * 40)
for i in range(len(x_test)):
    xi = x_test[i].item()
    exact = torch.exp(torch.tensor(-2.0 * xi)).item()
    pred = y_test[i].item()
    print(f"{xi:6.1f} {pred:10.4f} {exact:10.4f} {abs(pred - exact):10.4f}")


print("\n" + "=" * 70)
print("PART 8: COMPUTING THE FULL HESSIAN-VECTOR PRODUCT")
print("=" * 70)

def rosenbrock(xy):
    """The Rosenbrock function: a classic optimization test function.
    f(x, y) = (1-x)^2 + 100(y-x^2)^2
    Minimum at (1, 1) with f(1,1) = 0.
    """
    x, y = xy[0], xy[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

params = torch.tensor([0.5, 0.5], requires_grad=True)
print(f"Rosenbrock at {params.tolist()}: {rosenbrock(params).item():.4f}")

# Gradient
f = rosenbrock(params)
grad = torch.autograd.grad(f, params, create_graph=True)[0]
print(f"Gradient: {grad.tolist()}")

# Hessian-vector product without computing full Hessian
v = torch.tensor([1.0, 0.0])
hvp = torch.autograd.grad(grad, params, grad_outputs=v)[0]
print(f"Hessian-vector product (H @ [1,0]): {hvp.tolist()}")

# Full Hessian for comparison
H = hessian(rosenbrock, params)
print(f"\nFull Hessian:\n{H}")
print(f"H @ v = {(H @ v).tolist()} (matches HVP above)")


print("\n" + "=" * 70)
print("PART 9: AUTOGRAD HOOKS")
print("=" * 70)

# Tensor hooks
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
gradient_history = []

def save_gradient(grad):
    gradient_history.append(grad.clone())

handle = x.register_hook(save_gradient)

# Multiple backward passes
for scale in [1.0, 2.0, 3.0]:
    x_copy = x.detach().clone().requires_grad_(True)
    x_copy.register_hook(save_gradient)
    y = (x_copy * scale).sum()
    y.backward()

print("Gradient history (captured by hook):")
for i, g in enumerate(gradient_history):
    print(f"  Pass {i}: {g.tolist()}")

handle.remove()  # Clean up

# Gradient modification hook
print("\n--- Gradient clipping via hook ---")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def clip_gradient(grad):
    return torch.clamp(grad, -1.0, 1.0)

x.register_hook(clip_gradient)
y = (x ** 3).sum()  # Gradients would be [3, 12, 27]
y.backward()
print(f"x^3 gradient (unclipped): [3, 12, 27]")
print(f"x^3 gradient (clipped):   {x.grad.tolist()}")


print("\n" + "=" * 70)
print("PART 10: PRACTICAL — NEWTON'S METHOD WITH SECOND DERIVATIVES")
print("=" * 70)

# Newton's method: x_{n+1} = x_n - f'(x_n)/f''(x_n)
# Converges faster than gradient descent for smooth functions

def f(x):
    return (x - 3) ** 4 + (x - 3) ** 2

x = torch.tensor(0.0, requires_grad=True)
print("Minimizing f(x) = (x-3)^4 + (x-3)^2 using Newton's method")
print(f"{'Step':>4} {'x':>10} {'f(x)':>12} {'f_prime':>10} {'f_double':>10}")
print("-" * 50)

for step in range(15):
    loss = f(x)
    grad1 = torch.autograd.grad(loss, x, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1, x)[0]

    print(f"{step:4d} {x.item():10.6f} {loss.item():12.6f} {grad1.item():10.4f} {grad2.item():10.4f}")

    with torch.no_grad():
        if abs(grad2.item()) > 1e-8:
            x -= grad1.item() / grad2.item()  # Newton step
        else:
            break

    if abs(grad1.item()) < 1e-8:
        break

    x = x.detach().requires_grad_(True)

print(f"\nConverged to x = {x.item():.6f} (minimum near x = 3)")
print("Newton's method converges in far fewer steps than gradient descent!")

print("\n" + "=" * 70)
print("Higher-order gradients demonstration complete!")
print("=" * 70)
