"""
Module 01: Mathematical Foundations with PyTorch
================================================
This script demonstrates all the mathematical concepts essential for deep learning,
implemented using PyTorch. Every section is self-contained and prints its results.

Run: python math_with_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("PART 1: VECTOR OPERATIONS")
print("=" * 70)

# --- Dot Product ---
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
dot = torch.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"\nVectors: a = {a}, b = {b}")
print(f"Dot product (a·b): {dot.item()}")
print(f"Manual calculation: {(a * b).sum().item()}")

# --- Norms ---
v = torch.tensor([3.0, -4.0])
print(f"\nVector v = {v}")
print(f"L1 norm (sum of abs):    {torch.linalg.norm(v, ord=1).item()}")  # 7.0
print(f"L2 norm (Euclidean):     {torch.linalg.norm(v, ord=2).item()}")  # 5.0
print(f"L-inf norm (max abs):    {torch.linalg.norm(v, ord=float('inf')).item()}")  # 4.0

# --- Cosine Similarity ---
x = torch.tensor([1.0, 0.0, 0.0])
y = torch.tensor([0.0, 1.0, 0.0])
z = torch.tensor([1.0, 1.0, 0.0])

cos_sim = nn.CosineSimilarity(dim=0)
print(f"\nCosine similarity (perpendicular vectors): {cos_sim(x, y).item():.4f}")  # 0.0
print(f"Cosine similarity (45-degree vectors):     {cos_sim(x, z).item():.4f}")  # 0.7071
print(f"Cosine similarity (identical vectors):      {cos_sim(x, x).item():.4f}")  # 1.0

# --- Cross Product (3D only) ---
u = torch.tensor([1.0, 0.0, 0.0])
w = torch.tensor([0.0, 1.0, 0.0])
cross = torch.linalg.cross(u, w)
print(f"\nCross product of {u} x {w} = {cross}")  # [0, 0, 1]


print("\n" + "=" * 70)
print("PART 2: MATRIX OPERATIONS")
print("=" * 70)

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"\nMatrix A:\n{A}")
print(f"Matrix B:\n{B}")

# --- Matrix Multiplication ---
C = A @ B
print(f"\nA @ B (matrix multiply):\n{C}")

# --- Transpose ---
print(f"\nA transposed:\n{A.T}")
print(f"Verify: A[0,1] = {A[0, 1].item()}, A^T[1,0] = {A.T[1, 0].item()}")

# --- Determinant ---
det_A = torch.linalg.det(A)
print(f"\nDeterminant of A: {det_A.item()}")  # 1*4 - 2*3 = -2

# --- Inverse ---
A_inv = torch.linalg.inv(A)
print(f"\nInverse of A:\n{A_inv}")
print(f"A @ A^(-1) (should be identity):\n{(A @ A_inv).round()}")

# --- Trace ---
print(f"\nTrace of A (sum of diagonal): {torch.trace(A).item()}")  # 1 + 4 = 5

# --- Rank ---
rank_A = torch.linalg.matrix_rank(A)
print(f"Rank of A: {rank_A.item()}")  # 2 (full rank)

low_rank = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # Row 2 = 2 * Row 1
print(f"Rank of [[1,2],[2,4]]: {torch.linalg.matrix_rank(low_rank).item()}")  # 1


print("\n" + "=" * 70)
print("PART 3: EIGENDECOMPOSITION")
print("=" * 70)

S = torch.tensor([[4.0, 2.0], [2.0, 3.0]])  # Symmetric matrix
print(f"\nSymmetric matrix S:\n{S}")

eigenvalues, eigenvectors = torch.linalg.eigh(S)  # eigh for symmetric
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify: S @ v = lambda * v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    Sv = S @ v
    lam_v = lam * v
    print(f"\nEigenpair {i}: lambda={lam.item():.4f}")
    print(f"  S @ v     = {Sv}")
    print(f"  lambda * v = {lam_v}")
    print(f"  Match: {torch.allclose(Sv, lam_v, atol=1e-5)}")


print("\n" + "=" * 70)
print("PART 4: SINGULAR VALUE DECOMPOSITION (SVD)")
print("=" * 70)

M = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]], dtype=torch.float)
print(f"\nMatrix M (2x3):\n{M}")

U, S_vals, Vh = torch.linalg.svd(M, full_matrices=False)
print(f"\nU (left singular vectors), shape {U.shape}:\n{U}")
print(f"Singular values: {S_vals}")
print(f"Vh (right singular vectors), shape {Vh.shape}:\n{Vh}")

# Reconstruct: M = U @ diag(S) @ Vh
M_reconstructed = U @ torch.diag(S_vals) @ Vh
print(f"\nReconstructed M:\n{M_reconstructed.round()}")
print(f"Reconstruction matches: {torch.allclose(M, M_reconstructed, atol=1e-5)}")

# Low-rank approximation (keep only largest singular value)
k = 1
M_approx = U[:, :k] @ torch.diag(S_vals[:k]) @ Vh[:k, :]
print(f"\nRank-{k} approximation:\n{M_approx}")
error = torch.linalg.norm(M - M_approx)
print(f"Approximation error (Frobenius norm): {error.item():.4f}")


print("\n" + "=" * 70)
print("PART 5: QR DECOMPOSITION")
print("=" * 70)

Q_mat = torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]])
print(f"\nMatrix (3x2):\n{Q_mat}")

Q, R = torch.linalg.qr(Q_mat)
print(f"\nQ (orthogonal), shape {Q.shape}:\n{Q}")
print(f"R (upper triangular), shape {R.shape}:\n{R}")
print(f"Q^T @ Q (should be identity):\n{(Q.T @ Q).round()}")
print(f"Q @ R (should reconstruct original):\n{(Q @ R).round()}")


print("\n" + "=" * 70)
print("PART 6: torch.linalg FUNCTIONS")
print("=" * 70)

A_sys = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
b_sys = torch.tensor([9.0, 8.0])
print(f"\nSolving Ax = b:")
print(f"A = {A_sys}")
print(f"b = {b_sys}")

x_sol = torch.linalg.solve(A_sys, b_sys)
print(f"Solution x: {x_sol}")
print(f"Verify A @ x: {A_sys @ x_sol}")  # Should equal b

# Condition number
cond = torch.linalg.cond(A_sys)
print(f"\nCondition number of A: {cond.item():.4f}")
print("(Close to 1 = well-conditioned, large = ill-conditioned)")

# Cholesky decomposition (for positive definite matrices)
P = torch.tensor([[4.0, 2.0], [2.0, 3.0]])  # Positive definite
L = torch.linalg.cholesky(P)
print(f"\nCholesky decomposition of P:\n{P}")
print(f"L (lower triangular):\n{L}")
print(f"L @ L^T:\n{L @ L.T}")


print("\n" + "=" * 70)
print("PART 7: PROBABILITY DISTRIBUTIONS")
print("=" * 70)

from torch.distributions import Normal, Uniform, Bernoulli, Categorical, Beta

# Normal distribution
normal = Normal(loc=0.0, scale=1.0)
samples = normal.sample((10000,))
print(f"\nStandard Normal: mean={samples.mean():.4f}, std={samples.std():.4f}")
print(f"Log prob of x=0: {normal.log_prob(torch.tensor(0.0)).item():.4f}")
print(f"  (= log(1/sqrt(2*pi)) = {-0.5 * torch.log(torch.tensor(2 * 3.14159265)):.4f})")

# Uniform distribution
uniform = Uniform(low=0.0, high=1.0)
u_samples = uniform.sample((10000,))
print(f"\nUniform[0,1]: mean={u_samples.mean():.4f} (expect 0.5)")

# Bernoulli distribution
bernoulli = Bernoulli(probs=0.7)
b_samples = bernoulli.sample((10000,))
print(f"\nBernoulli(p=0.7): mean={b_samples.mean():.4f} (expect 0.7)")

# Categorical distribution
probs = torch.tensor([0.1, 0.3, 0.6])
cat = Categorical(probs=probs)
cat_samples = cat.sample((10000,))
for i in range(3):
    freq = (cat_samples == i).float().mean()
    print(f"Category {i}: freq={freq:.3f} (expect {probs[i]:.3f})")

# Reparameterization trick (used in VAEs)
mu = torch.tensor(5.0, requires_grad=True)
sigma = torch.tensor(2.0, requires_grad=True)
dist = Normal(mu, sigma)
z = dist.rsample()  # rsample allows gradient flow through sampling
loss = z ** 2
loss.backward()
print(f"\nReparameterization trick:")
print(f"  Sampled z: {z.item():.4f}")
print(f"  Gradient through mu: {mu.grad.item():.4f}")
print(f"  Gradient through sigma: {sigma.grad.item():.4f}")


print("\n" + "=" * 70)
print("PART 8: ENTROPY AND CROSS-ENTROPY")
print("=" * 70)

# Entropy: H(p) = -sum(p * log(p))
def entropy(p):
    return -(p * p.log()).sum()

uniform_p = torch.tensor([0.25, 0.25, 0.25, 0.25])
peaked_p = torch.tensor([0.9, 0.05, 0.025, 0.025])

print(f"\nEntropy of uniform distribution: {entropy(uniform_p):.4f}")
print(f"Entropy of peaked distribution: {entropy(peaked_p):.4f}")
print("(Uniform has maximum entropy = maximum uncertainty)")

# Cross-entropy: H(p, q) = -sum(p * log(q))
true_dist = torch.tensor([1.0, 0.0, 0.0])  # One-hot: class 0
good_pred = torch.tensor([0.9, 0.05, 0.05])
bad_pred = torch.tensor([0.1, 0.1, 0.8])

ce_good = -(true_dist * good_pred.log()).sum()
ce_bad = -(true_dist * bad_pred.log()).sum()
print(f"\nCross-entropy with good prediction: {ce_good:.4f}")
print(f"Cross-entropy with bad prediction:  {ce_bad:.4f}")
print("(Lower cross-entropy = better prediction)")

# Using PyTorch's built-in CrossEntropyLoss
logits = torch.tensor([[2.0, 0.5, 0.1]])  # Raw network outputs
target = torch.tensor([0])
ce_loss = nn.CrossEntropyLoss()(logits, target)
print(f"\nPyTorch CrossEntropyLoss: {ce_loss.item():.4f}")

# KL Divergence: D_KL(P || Q) = sum(P * log(P/Q))
p = torch.tensor([0.4, 0.3, 0.3])
q = torch.tensor([0.33, 0.33, 0.34])
kl = (p * (p / q).log()).sum()
print(f"\nKL divergence D_KL(P || Q): {kl.item():.6f}")
kl_reverse = (q * (q / p).log()).sum()
print(f"KL divergence D_KL(Q || P): {kl_reverse.item():.6f}")
print("(KL divergence is asymmetric!)")


print("\n" + "=" * 70)
print("PART 9: GRADIENT DESCENT FROM SCRATCH")
print("=" * 70)

print("\n--- Example 1: Minimize f(x) = (x - 3)^2 ---")
x = torch.tensor(0.0, requires_grad=True)
lr = 0.1
history = []

for step in range(30):
    loss = (x - 3) ** 2
    loss.backward()
    history.append((step, x.item(), loss.item()))
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

print(f"{'Step':>4} {'x':>10} {'Loss':>12}")
print("-" * 28)
for step, x_val, loss_val in history[::5]:
    print(f"{step:4d} {x_val:10.6f} {loss_val:12.8f}")
print(f"Final x = {x.item():.6f} (target: 3.0)")


print("\n--- Example 2: 2D Gradient Descent on f(x,y) = x^2 + 4*y^2 ---")
params = torch.tensor([5.0, 5.0], requires_grad=True)
lr = 0.1

print(f"{'Step':>4} {'x':>8} {'y':>8} {'Loss':>10}")
print("-" * 34)

for step in range(30):
    loss = params[0] ** 2 + 4 * params[1] ** 2
    loss.backward()
    if step % 5 == 0:
        print(f"{step:4d} {params[0].item():8.4f} {params[1].item():8.4f} {loss.item():10.4f}")
    with torch.no_grad():
        params -= lr * params.grad
    params.grad.zero_()

print(f"Final: ({params[0].item():.6f}, {params[1].item():.6f})")
print("(Target: (0, 0) — the minimum of x^2 + 4y^2)")


print("\n--- Example 3: Momentum vs Plain SGD ---")

def optimize(lr, momentum, steps=100):
    x = torch.tensor([5.0, 5.0], requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr, momentum=momentum)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = x[0] ** 2 + 10 * x[1] ** 2
        loss.backward()
        optimizer.step()
    return x.detach()

plain_result = optimize(lr=0.05, momentum=0.0, steps=50)
momentum_result = optimize(lr=0.05, momentum=0.9, steps=50)
print(f"\nPlain SGD after 50 steps: ({plain_result[0]:.6f}, {plain_result[1]:.6f})")
print(f"SGD+Momentum after 50 steps: ({momentum_result[0]:.6f}, {momentum_result[1]:.6f})")
print("(Momentum converges faster on elongated loss surfaces)")


print("\n--- Example 4: Adam Optimizer ---")
x = torch.tensor([5.0, 5.0], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=0.5)

for step in range(100):
    optimizer.zero_grad()
    loss = x[0] ** 2 + 10 * x[1] ** 2
    loss.backward()
    optimizer.step()

print(f"Adam after 100 steps: ({x[0].item():.6f}, {x[1].item():.6f})")


print("\n" + "=" * 70)
print("PART 10: SOLVING A LINEAR REGRESSION FROM SCRATCH")
print("=" * 70)

torch.manual_seed(42)

# Generate data: y = 3x + 7 + noise
X = torch.linspace(-5, 5, 100).unsqueeze(1)
y_true = 3 * X + 7 + torch.randn_like(X) * 0.5

# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.01

print(f"\nTrue parameters: w=3.0, b=7.0")
print(f"Initial parameters: w={w.item():.4f}, b={b.item():.4f}")

for epoch in range(200):
    y_pred = X * w + b
    loss = ((y_pred - y_true) ** 2).mean()  # MSE loss
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\nFinal: w={w.item():.4f} (true: 3.0), b={b.item():.4f} (true: 7.0)")

print("\n" + "=" * 70)
print("All mathematical foundations demonstrated successfully!")
print("=" * 70)
