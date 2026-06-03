"""
Tensor Manipulation — Reshape, Index, Broadcast, Views
=======================================================
Covers: reshaping, slicing, advanced indexing, broadcasting, views vs copies.
"""

import torch

print("=" * 60)
print("1. RESHAPING")
print("=" * 60)

x = torch.arange(12).float()
print(f"Original: {x}, shape: {x.shape}")

# view — must be contiguous
y = x.view(3, 4)
print(f"view(3,4):\n{y}")

# reshape — works even if not contiguous
y = x.reshape(4, 3)
print(f"reshape(4,3):\n{y}")

# flatten
y = torch.randn(2, 3, 4)
print(f"\nflatten(): {y.flatten().shape}")
print(f"flatten(1): {y.flatten(1).shape}")

print("\n" + "=" * 60)
print("2. TRANSPOSE & PERMUTE")
print("=" * 60)

x = torch.randn(2, 3, 4)
print(f"Original shape: {x.shape}")

# Transpose two dims
y = x.transpose(0, 2)
print(f"transpose(0,2): {y.shape}")

# Arbitrary permutation
y = x.permute(2, 0, 1)
print(f"permute(2,0,1): {y.shape}")

# Matrix transpose (last 2 dims)
m = torch.randn(3, 4)
print(f"\nmT of (3,4): {m.mT.shape}")

print("\n" + "=" * 60)
print("3. SQUEEZE & UNSQUEEZE")
print("=" * 60)

x = torch.randn(3, 4)
print(f"Original: {x.shape}")

y = x.unsqueeze(0)
print(f"unsqueeze(0): {y.shape}")

y = x.unsqueeze(-1)
print(f"unsqueeze(-1): {y.shape}")

z = torch.randn(1, 3, 1, 4)
print(f"\nBefore squeeze: {z.shape}")
print(f"After squeeze:  {z.squeeze().shape}")
print(f"squeeze(0):     {z.squeeze(0).shape}")

print("\n" + "=" * 60)
print("4. CONCATENATION & STACKING")
print("=" * 60)

a = torch.randn(2, 3)
b = torch.randn(2, 3)

cat0 = torch.cat([a, b], dim=0)
cat1 = torch.cat([a, b], dim=1)
stack = torch.stack([a, b], dim=0)

print(f"a: {a.shape}, b: {b.shape}")
print(f"cat(dim=0): {cat0.shape}")
print(f"cat(dim=1): {cat1.shape}")
print(f"stack(dim=0): {stack.shape}")

print("\n" + "=" * 60)
print("5. SPLIT & CHUNK")
print("=" * 60)

x = torch.arange(12).reshape(4, 3)
print(f"x:\n{x}")

chunks = x.chunk(2, dim=0)
print(f"\nchunk(2, dim=0): {[c.shape for c in chunks]}")

parts = x.split(1, dim=0)
print(f"split(1, dim=0): {[p.shape for p in parts]}")

print("\n" + "=" * 60)
print("6. INDEXING")
print("=" * 60)

x = torch.arange(20).reshape(4, 5).float()
print(f"x:\n{x}")

print(f"\nRow 0:      {x[0]}")
print(f"Col 0:      {x[:, 0]}")
print(f"Slice [1:3, 2:]: \n{x[1:3, 2:]}")

# Boolean indexing (returns copy)
mask = x > 10
print(f"\nx > 10: {x[mask]}")

# Fancy indexing (returns copy)
idx = torch.tensor([0, 3])
print(f"Rows [0,3]:\n{x[idx]}")

print("\n" + "=" * 60)
print("7. BROADCASTING")
print("=" * 60)

a = torch.randn(3, 4)
b = torch.randn(4)       # Broadcasts to (3, 4)
c = a + b
print(f"(3,4) + (4,) = {c.shape}")

a = torch.randn(3, 1, 4)
b = torch.randn(1, 5, 4)
c = a + b
print(f"(3,1,4) + (1,5,4) = {c.shape}")

print("\n" + "=" * 60)
print("8. VIEWS vs COPIES")
print("=" * 60)

x = torch.randn(3, 4)

# View — shares memory
y = x.view(4, 3)
y[0, 0] = 999.0
print(f"After modifying view, original x[0,0] = {x[0, 0]}")

# Clone — independent copy
z = x.clone()
z[0, 0] = -1.0
print(f"After modifying clone, original x[0,0] = {x[0, 0]} (unchanged)")

# Contiguity
t = torch.randn(3, 4)
t_transposed = t.transpose(0, 1)
print(f"\nTransposed is contiguous: {t_transposed.is_contiguous()}")
t_contig = t_transposed.contiguous()
print(f"After .contiguous():      {t_contig.is_contiguous()}")

print("\n" + "=" * 60)
print("9. EINSUM")
print("=" * 60)

# Matrix multiply
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ij,jk->ik', A, B)
print(f"einsum matmul: {A.shape} x {B.shape} = {C.shape}")

# Batch matmul
A = torch.randn(8, 3, 4)
B = torch.randn(8, 4, 5)
C = torch.einsum('bij,bjk->bik', A, B)
print(f"einsum batch matmul: {C.shape}")

# Trace
M = torch.randn(4, 4)
trace = torch.einsum('ii->', M)
print(f"einsum trace: {trace:.4f} vs torch.trace: {torch.trace(M):.4f}")

# Outer product
u = torch.randn(3)
v = torch.randn(4)
outer = torch.einsum('i,j->ij', u, v)
print(f"einsum outer product: {outer.shape}")

print("\nDone!")
