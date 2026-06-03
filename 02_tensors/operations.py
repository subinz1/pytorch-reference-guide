"""
Module 02: Tensor Operations
=============================
Element-wise operations, reductions, and matrix operations with detailed output.

Run: python operations.py
"""

import torch

print("=" * 70)
print("PART 1: ELEMENT-WISE ARITHMETIC")
print("=" * 70)

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([10.0, 20.0, 30.0, 40.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"\na + b  = {a + b}")
print(f"a - b  = {a - b}")
print(f"a * b  = {a * b}")
print(f"a / b  = {a / b}")
print(f"a // b = {a // b}  (floor division)")
print(f"a % b  = {a % b}  (modulo)")
print(f"a ** 2 = {a ** 2}")

# Scalar-tensor operations
print(f"\na + 10 = {a + 10}")
print(f"a * 3  = {a * 3}")
print(f"2 ** a = {2 ** a}")


print("\n" + "=" * 70)
print("PART 2: MATHEMATICAL FUNCTIONS")
print("=" * 70)

x = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
print(f"x = {x}")

print(f"\ntorch.exp(x)    = {torch.exp(x)}")
print(f"torch.log(x+1)  = {torch.log(x + 1)}")
print(f"torch.log2(x+1) = {torch.log2(x + 1)}")
print(f"torch.log10(x+1)= {torch.log10(x + 1)}")
print(f"torch.sqrt(x)   = {torch.sqrt(x)}")
print(f"torch.square(x) = {torch.square(x)}")

# Trigonometric
angles = torch.tensor([0.0, torch.pi / 6, torch.pi / 4, torch.pi / 3, torch.pi / 2])
print(f"\nangles (radians) = {angles}")
print(f"sin = {torch.sin(angles).round(decimals=4)}")
print(f"cos = {torch.cos(angles).round(decimals=4)}")
print(f"tan = {torch.tan(angles[:4]).round(decimals=4)}")

# Activation functions
x = torch.linspace(-3, 3, 7)
print(f"\nx = {x}")
print(f"sigmoid(x) = {torch.sigmoid(x).round(decimals=3)}")
print(f"tanh(x)    = {torch.tanh(x).round(decimals=3)}")
print(f"relu(x)    = {torch.relu(x)}")
print(f"softplus(x)= {torch.nn.functional.softplus(x).round(decimals=3)}")


print("\n" + "=" * 70)
print("PART 3: COMPARISON AND LOGICAL OPERATIONS")
print("=" * 70)

a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([5, 4, 3, 2, 1])

print(f"a = {a}")
print(f"b = {b}")
print(f"\na > b:  {a > b}")
print(f"a >= b: {a >= b}")
print(f"a < b:  {a < b}")
print(f"a == b: {a == b}")
print(f"a != b: {a != b}")

# torch.where: conditional selection
result = torch.where(a > b, a, b)
print(f"\ntorch.where(a > b, a, b): {result}  (element-wise max)")

# Logical operations
x = torch.tensor([True, True, False, False])
y = torch.tensor([True, False, True, False])
print(f"\nx = {x}")
print(f"y = {y}")
print(f"x & y (AND): {x & y}")
print(f"x | y (OR):  {x | y}")
print(f"~x (NOT):    {~x}")
print(f"x ^ y (XOR): {x ^ y}")


print("\n" + "=" * 70)
print("PART 4: CLAMPING AND ROUNDING")
print("=" * 70)

x = torch.tensor([-3.7, -1.2, 0.0, 1.5, 4.9])
print(f"x = {x}")

print(f"\nclamp(min=0):        {torch.clamp(x, min=0)}")
print(f"clamp(max=2):        {torch.clamp(x, max=2)}")
print(f"clamp(min=-1, max=3): {torch.clamp(x, min=-1, max=3)}")
print(f"abs(x):              {torch.abs(x)}")
print(f"sign(x):             {torch.sign(x)}")
print(f"floor(x):            {torch.floor(x)}")
print(f"ceil(x):             {torch.ceil(x)}")
print(f"round(x):            {torch.round(x)}")
print(f"trunc(x):            {torch.trunc(x)}  (round toward zero)")
print(f"frac(x):             {torch.frac(x)}  (fractional part)")


print("\n" + "=" * 70)
print("PART 5: REDUCTION OPERATIONS")
print("=" * 70)

x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
print(f"x =\n{x}")
print(f"Shape: {x.shape}")

print(f"\n--- Global reductions (no dim) ---")
print(f"sum:  {x.sum().item()}")
print(f"mean: {x.mean().item()}")
print(f"max:  {x.max().item()}")
print(f"min:  {x.min().item()}")
print(f"prod: {x.prod().item()}")
print(f"std:  {x.std().item():.4f}")
print(f"var:  {x.var().item():.4f}")

print(f"\n--- Reduce along dim=0 (collapse rows → one row per column) ---")
print(f"sum(dim=0):  {x.sum(dim=0)}")
print(f"mean(dim=0): {x.mean(dim=0)}")
print(f"max(dim=0):  values={x.max(dim=0).values}, indices={x.max(dim=0).indices}")

print(f"\n--- Reduce along dim=1 (collapse columns → one value per row) ---")
print(f"sum(dim=1):  {x.sum(dim=1)}")
print(f"mean(dim=1): {x.mean(dim=1)}")
print(f"max(dim=1):  values={x.max(dim=1).values}, indices={x.max(dim=1).indices}")

print(f"\n--- keepdim=True preserves the reduced dimension ---")
print(f"sum(dim=1):             shape={x.sum(dim=1).shape}    {x.sum(dim=1)}")
print(f"sum(dim=1, keepdim):    shape={x.sum(dim=1, keepdim=True).shape} {x.sum(dim=1, keepdim=True)}")

# argmax/argmin
print(f"\n--- argmax and argmin ---")
print(f"argmax (global):  {x.argmax().item()} (flattened index)")
print(f"argmax (dim=0): {x.argmax(dim=0)}")
print(f"argmax (dim=1): {x.argmax(dim=1)}")


print("\n" + "=" * 70)
print("PART 6: NORM OPERATIONS")
print("=" * 70)

x = torch.tensor([[3.0, -4.0],
                   [5.0, 12.0]])
print(f"x =\n{x}")

print(f"\nFrobenius norm (default): {torch.linalg.norm(x).item():.4f}")
print(f"L1 norm (row): {torch.linalg.norm(x, ord=1, dim=1)}")
print(f"L2 norm (row): {torch.linalg.norm(x, ord=2, dim=1)}")
print(f"L-inf norm (row): {torch.linalg.norm(x, ord=float('inf'), dim=1)}")


print("\n" + "=" * 70)
print("PART 7: MATRIX OPERATIONS")
print("=" * 70)

A = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                   [7.0, 8.0]])
print(f"A =\n{A}")
print(f"B =\n{B}")

print(f"\nA @ B (matmul) =\n{A @ B}")
print(f"torch.mm(A, B) =\n{torch.mm(A, B)}")
print(f"torch.matmul(A, B) =\n{torch.matmul(A, B)}")

# Vector-matrix
v = torch.tensor([1.0, 2.0])
print(f"\nv = {v}")
print(f"v @ A = {v @ A}  (1x2 @ 2x2 → 1x2)")
print(f"A @ v = {A @ v}  (2x2 @ 2x1 → 2x1)")

# Batch matrix multiply
batch_A = torch.randn(5, 3, 4)
batch_B = torch.randn(5, 4, 2)
batch_C = torch.bmm(batch_A, batch_B)
print(f"\nBatch matmul: {batch_A.shape} @ {batch_B.shape} → {batch_C.shape}")


print("\n" + "=" * 70)
print("PART 8: EINSUM EXAMPLES")
print("=" * 70)

A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
v = torch.tensor([1., 2.])
w = torch.tensor([3., 4.])

print(f"A =\n{A}")
print(f"B =\n{B}")
print(f"v = {v}, w = {w}")

print(f"\nDot product (i,i->): {torch.einsum('i,i->', v, w).item()}")
print(f"Outer product (i,j->ij):\n{torch.einsum('i,j->ij', v, w)}")
print(f"Matrix multiply (ik,kj->ij):\n{torch.einsum('ik,kj->ij', A, B)}")
print(f"Trace (ii->): {torch.einsum('ii->', A).item()}")
print(f"Matrix-vector (ij,j->i): {torch.einsum('ij,j->i', A, v)}")
print(f"Element-wise product sum (ij,ij->): {torch.einsum('ij,ij->', A, B).item()}")

# Batch operations
batch = torch.randn(3, 2, 2)
print(f"\nBatch trace (bii->b): {torch.einsum('bii->b', batch)}")
print(f"Batch diagonal (bii->bi):\n{torch.einsum('bii->bi', batch)}")


print("\n" + "=" * 70)
print("PART 9: SORTING AND TOP-K")
print("=" * 70)

x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
print(f"x = {x}")

vals, idxs = torch.sort(x)
print(f"\nSorted (ascending):  values={vals}, indices={idxs}")
vals, idxs = torch.sort(x, descending=True)
print(f"Sorted (descending): values={vals}, indices={idxs}")

top_vals, top_idxs = torch.topk(x, k=3)
print(f"\nTop 3: values={top_vals}, indices={top_idxs}")

bot_vals, bot_idxs = torch.topk(x, k=3, largest=False)
print(f"Bottom 3: values={bot_vals}, indices={bot_idxs}")

# unique
x_dup = torch.tensor([1, 3, 2, 1, 3, 3, 2])
unique, counts = torch.unique(x_dup, return_counts=True)
print(f"\nUnique values: {unique}, counts: {counts}")

# cumulative sum
x = torch.tensor([1, 2, 3, 4, 5])
print(f"\ncumsum: {torch.cumsum(x, dim=0)}")
print(f"cumprod: {torch.cumprod(x, dim=0)}")

print("\n" + "=" * 70)
print("Tensor operations demonstration complete!")
print("=" * 70)
