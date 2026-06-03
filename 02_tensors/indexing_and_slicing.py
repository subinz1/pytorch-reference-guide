"""
Module 02: Indexing and Slicing
================================
Every form of tensor indexing, from basic to advanced.

Run: python indexing_and_slicing.py
"""

import torch

print("=" * 70)
print("PART 1: BASIC INDEXING")
print("=" * 70)

x = torch.arange(20).reshape(4, 5)
print(f"x =\n{x}\n")

print(f"x[0]      = {x[0]}           (first row)")
print(f"x[-1]     = {x[-1]}      (last row)")
print(f"x[2]      = {x[2]}      (third row)")
print(f"x[1, 3]   = {x[1, 3].item()}                  (row 1, col 3)")
print(f"x[-1, -1] = {x[-1, -1].item()}                 (bottom-right corner)")

# Basic indexing returns views (shared memory)
row = x[0]
row[0] = 999
print(f"\nAfter modifying x[0][0] via a view:")
print(f"x[0] = {x[0]}  (modified through the view)")
x[0, 0] = 0  # Restore


print("\n" + "=" * 70)
print("PART 2: SLICING")
print("=" * 70)

x = torch.arange(20).reshape(4, 5)
print(f"x =\n{x}\n")

print(f"x[1:3]     = (rows 1 and 2)\n{x[1:3]}\n")
print(f"x[:2]      = (first 2 rows)\n{x[:2]}\n")
print(f"x[2:]      = (from row 2 onward)\n{x[2:]}\n")
print(f"x[:, 1:4]  = (columns 1, 2, 3)\n{x[:, 1:4]}\n")
print(f"x[1:3, 2:5]= (submatrix)\n{x[1:3, 2:5]}\n")

# Step slicing
print(f"x[::2]     = (every other row)\n{x[::2]}\n")
print(f"x[:, ::2]  = (every other column)\n{x[:, ::2]}\n")
print(f"x[::2, ::2]= (checkerboard)\n{x[::2, ::2]}\n")

# Negative step (reverse)
print(f"x.flip(0)  = (reversed rows)\n{x.flip(0)}\n")
print(f"x.flip(1)  = (reversed columns)\n{x.flip(1)}\n")


print("=" * 70)
print("PART 3: BOOLEAN (MASK) INDEXING")
print("=" * 70)

torch.manual_seed(42)
x = torch.randn(3, 4).round(decimals=2)
print(f"x =\n{x}\n")

# Create boolean mask
mask = x > 0
print(f"Mask (x > 0):\n{mask}\n")

# Select elements where mask is True
positives = x[mask]
print(f"Positive elements: {positives}")
print(f"Number of positive elements: {positives.numel()}")

# Conditional assignment
x_relu = x.clone()
x_relu[x_relu < 0] = 0
print(f"\nAfter zeroing negatives (manual ReLU):\n{x_relu}")

# Multiple conditions
x_abs = x.clone()
big_mask = (x_abs > -0.5) & (x_abs < 0.5)
print(f"\nElements between -0.5 and 0.5: {x[big_mask]}")

# torch.where for conditional selection
replaced = torch.where(x > 0, x, torch.zeros_like(x))
print(f"\ntorch.where(x > 0, x, 0):\n{replaced}")


print("\n" + "=" * 70)
print("PART 4: FANCY (ADVANCED) INDEXING")
print("=" * 70)

x = torch.arange(20).reshape(4, 5)
print(f"x =\n{x}\n")

# Index with a list of indices
rows = torch.tensor([0, 2, 3])
print(f"x[rows] = x[[0, 2, 3]] = (select rows)\n{x[rows]}\n")

cols = torch.tensor([1, 3, 4])
print(f"x[:, cols] = x[:, [1,3,4]] = (select columns)\n{x[:, cols]}\n")

# Paired indexing: select specific (row, col) pairs
r = torch.tensor([0, 1, 2, 3])
c = torch.tensor([4, 3, 2, 1])
print(f"Pairs: rows={r.tolist()}, cols={c.tolist()}")
print(f"x[r, c] = {x[r, c]}  (diagonal from top-right)")

# Index with 2D indices
idx = torch.tensor([[0, 1], [2, 3]])
print(f"\n2D index tensor:\n{idx}")
print(f"x[idx] = (index rows using 2D index, result is 2x2x5)\n{x[idx]}")


print("\n" + "=" * 70)
print("PART 5: GATHER AND SCATTER")
print("=" * 70)

# gather: pick elements from src along a dimension using index
src = torch.tensor([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]])
print(f"Source:\n{src}\n")

# For each row, pick elements at specific column indices
index = torch.tensor([[0, 2],
                       [1, 0],
                       [2, 1]])
gathered = torch.gather(src, dim=1, index=index)
print(f"Gather index (dim=1):\n{index}")
print(f"Gathered result:\n{gathered}")
print("Explanation: row 0 picks cols [0,2]=[10,30], row 1 picks cols [1,0]=[50,40], etc.")

# scatter_: the inverse operation
print(f"\n--- Scatter ---")
dst = torch.zeros(3, 3, dtype=torch.long)
values = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]])
dst.scatter_(dim=1, index=index, src=values)
print(f"Scatter values:\n{values}")
print(f"Into indices:\n{index}")
print(f"Result:\n{dst}")


print("\n" + "=" * 70)
print("PART 6: INDEX_SELECT AND MASKED_SELECT")
print("=" * 70)

x = torch.arange(20).reshape(4, 5)
print(f"x =\n{x}\n")

# index_select: select along a dimension
selected_rows = torch.index_select(x, dim=0, index=torch.tensor([0, 3]))
print(f"index_select(dim=0, [0,3]):\n{selected_rows}\n")

selected_cols = torch.index_select(x, dim=1, index=torch.tensor([1, 4]))
print(f"index_select(dim=1, [1,4]):\n{selected_cols}\n")

# masked_select: select elements where mask is True (returns 1D)
mask = x > 12
selected = torch.masked_select(x, mask)
print(f"Mask (x > 12):\n{mask}")
print(f"masked_select result: {selected}")


print("\n" + "=" * 70)
print("PART 7: ASSIGNMENT WITH INDEXING")
print("=" * 70)

x = torch.zeros(4, 5, dtype=torch.long)
print(f"Starting with zeros:\n{x}\n")

# Assign a single element
x[0, 0] = 1
print(f"After x[0,0] = 1:\n{x}\n")

# Assign a row
x[1] = torch.arange(5)
print(f"After x[1] = [0,1,2,3,4]:\n{x}\n")

# Assign with boolean mask
x[x == 0] = -1
print(f"After x[x==0] = -1:\n{x}\n")

# Assign with fancy indexing
x[torch.tensor([2, 3]), torch.tensor([2, 3])] = 99
print(f"After x[[2,3], [2,3]] = 99:\n{x}\n")

# Assign a column
x[:, -1] = 42
print(f"After x[:, -1] = 42:\n{x}")


print("\n" + "=" * 70)
print("PART 8: ADVANCED — NONZERO AND MASKING PATTERNS")
print("=" * 70)

x = torch.tensor([[0, 1, 0],
                   [2, 0, 3],
                   [0, 4, 0]])
print(f"x =\n{x}\n")

# Find indices of non-zero elements
nz = torch.nonzero(x)
print(f"Nonzero indices:\n{nz}")
print(f"Values at those indices: {x[nz[:, 0], nz[:, 1]]}")

# Create a mask from indices
mask = torch.zeros(3, 3, dtype=torch.bool)
mask[nz[:, 0], nz[:, 1]] = True
print(f"\nReconstructed mask:\n{mask}")

# Triangular masks (useful for causal attention)
print(f"\nLower triangular mask:\n{torch.tril(torch.ones(4, 4))}")
print(f"\nUpper triangular mask:\n{torch.triu(torch.ones(4, 4))}")
print(f"\nStrict lower (k=-1):\n{torch.tril(torch.ones(4, 4), diagonal=-1)}")

# Diagonal extraction and creation
diag = torch.diag(torch.tensor([1, 2, 3, 4]))
print(f"\nDiagonal matrix from [1,2,3,4]:\n{diag}")
print(f"Extract diagonal: {torch.diag(diag)}")

print("\n" + "=" * 70)
print("Indexing and slicing demonstration complete!")
print("=" * 70)
