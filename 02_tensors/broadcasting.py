"""
Module 02: Broadcasting
========================
Step-by-step exploration of PyTorch broadcasting rules with visual explanations.

Run: python broadcasting.py
"""

import torch

print("=" * 70)
print("BROADCASTING RULES")
print("=" * 70)
print("""
Broadcasting lets you operate on tensors with different shapes.
PyTorch aligns shapes from the RIGHT and applies these rules:

  Rule 1: If dimensions differ in count, prepend 1s to the smaller shape.
  Rule 2: For each dimension, sizes must be equal OR one must be 1.
  Rule 3: Size-1 dimensions are "stretched" to match the other.

If Rule 2 fails for any dimension, broadcasting is impossible.
""")


print("=" * 70)
print("EXAMPLE 1: Scalar + Tensor")
print("=" * 70)

a = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])  # Shape: (2, 3)
b = 10.0  # Scalar, shape: ()

result = a + b
print(f"a (2, 3):\n{a}")
print(f"b (scalar): {b}")
print(f"\na + b (2, 3):\n{result}")
print("""
Step-by-step:
  a shape: (2, 3)
  b shape: ()      → pad with 1s → (1, 1)
  Compare:  2 vs 1 → stretch to 2
            3 vs 1 → stretch to 3
  Result shape: (2, 3)
""")


print("=" * 70)
print("EXAMPLE 2: Vector + Matrix (row broadcasting)")
print("=" * 70)

A = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])  # Shape: (3, 3)
row = torch.tensor([10.0, 20.0, 30.0])  # Shape: (3,)

result = A + row
print(f"A (3, 3):\n{A}")
print(f"row (3,): {row}")
print(f"\nA + row (3, 3):\n{result}")
print("""
Step-by-step:
  A shape:   (3, 3)
  row shape: (3,)   → pad with 1s → (1, 3)
  Compare:    3 vs 1 → stretch row to 3 rows
              3 vs 3 → match!
  Result shape: (3, 3)
  The row vector is added to EVERY row of A.
""")


print("=" * 70)
print("EXAMPLE 3: Column vector + Row vector = Matrix")
print("=" * 70)

col = torch.tensor([[1.0],
                     [2.0],
                     [3.0]])  # Shape: (3, 1)
row = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # Shape: (1, 4)

result = col + row
print(f"col (3, 1):\n{col}")
print(f"row (1, 4): {row}")
print(f"\ncol + row (3, 4):\n{result}")
print("""
Step-by-step:
  col shape: (3, 1)
  row shape: (1, 4)
  Compare:    3 vs 1 → stretch row to 3 rows
              1 vs 4 → stretch col to 4 columns
  Result shape: (3, 4)
  This creates an "outer sum" — every combination of col + row values.
""")


print("=" * 70)
print("EXAMPLE 4: Feature normalization (common pattern)")
print("=" * 70)

torch.manual_seed(42)
data = torch.randn(4, 3)  # 4 samples, 3 features
print(f"Data (4 samples, 3 features):\n{data.round(decimals=3)}\n")

# Normalize each feature (column) to zero mean, unit variance
mean = data.mean(dim=0)              # Shape: (3,) — one mean per feature
std = data.std(dim=0)                # Shape: (3,) — one std per feature
normalized = (data - mean) / std     # Broadcasting: (4, 3) - (3,) → (4, 3)

print(f"Feature means: {mean.round(decimals=3)}")
print(f"Feature stds:  {std.round(decimals=3)}")
print(f"\nNormalized:\n{normalized.round(decimals=3)}")
print(f"Normalized means: {normalized.mean(dim=0).round(decimals=3)} (should be ~0)")
print(f"Normalized stds:  {normalized.std(dim=0).round(decimals=3)} (should be ~1)")


print("\n" + "=" * 70)
print("EXAMPLE 5: keepdim=True for safe broadcasting")
print("=" * 70)

x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])  # Shape: (2, 3)
print(f"x (2, 3):\n{x}\n")

# Without keepdim: shape (2,) — ambiguous for broadcasting
row_sum = x.sum(dim=1)
print(f"sum(dim=1): shape={row_sum.shape}, values={row_sum}")
print("  Cannot safely broadcast (2,) against (2, 3) — which dim to match?\n")

# With keepdim: shape (2, 1) — unambiguous
row_sum_kd = x.sum(dim=1, keepdim=True)
print(f"sum(dim=1, keepdim=True): shape={row_sum_kd.shape}")
print(f"  values:\n{row_sum_kd}")

# Now we can safely divide
proportions = x / row_sum_kd
print(f"\nRow proportions (x / row_sum):\n{proportions.round(decimals=4)}")
print(f"Row sums of proportions: {proportions.sum(dim=1)}")  # Should be 1.0


print("\n" + "=" * 70)
print("EXAMPLE 6: Softmax by hand using broadcasting")
print("=" * 70)

logits = torch.tensor([[2.0, 1.0, 0.1],
                        [1.0, 3.0, 0.5]])  # Shape: (2, 3)
print(f"Logits:\n{logits}\n")

# For numerical stability, subtract the max per row
max_vals = logits.max(dim=1, keepdim=True).values  # Shape: (2, 1)
shifted = logits - max_vals                         # Broadcasting: (2, 3) - (2, 1)
exp_vals = torch.exp(shifted)                       # Shape: (2, 3)
sum_exp = exp_vals.sum(dim=1, keepdim=True)          # Shape: (2, 1)
softmax = exp_vals / sum_exp                         # Broadcasting: (2, 3) / (2, 1)

print(f"Max per row: {max_vals.squeeze()}")
print(f"Softmax:\n{softmax.round(decimals=4)}")
print(f"Row sums: {softmax.sum(dim=1)} (should be 1.0)")
print(f"\nVerify with torch.softmax:\n{torch.softmax(logits, dim=1).round(decimals=4)}")


print("\n" + "=" * 70)
print("EXAMPLE 7: Pairwise distance using broadcasting")
print("=" * 70)

# Compute all pairwise L2 distances between two sets of points
points_a = torch.tensor([[0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])  # 3 points in 2D

points_b = torch.tensor([[1.0, 1.0],
                          [2.0, 2.0]])  # 2 points in 2D

# Reshape for broadcasting:
# a: (3, 1, 2) — each of 3 points broadcast against all of b's points
# b: (1, 2, 2) — each of 2 points broadcast against all of a's points
diff = points_a.unsqueeze(1) - points_b.unsqueeze(0)  # Shape: (3, 2, 2)
distances = torch.linalg.norm(diff, dim=2)  # Shape: (3, 2)

print(f"Points A (3 points):\n{points_a}")
print(f"Points B (2 points):\n{points_b}")
print(f"\nPairwise distances (3x2):\n{distances.round(decimals=4)}")
print("""
Shape analysis:
  points_a.unsqueeze(1): (3, 1, 2) — "for each of my 3 points..."
  points_b.unsqueeze(0): (1, 2, 2) — "...compare against each of 2 points"
  diff:                  (3, 2, 2) — all 3*2=6 difference vectors
  norm(dim=2):           (3, 2)    — L2 distance for each pair
""")


print("=" * 70)
print("EXAMPLE 8: Broadcasting failures")
print("=" * 70)

print("\nThese shapes are INCOMPATIBLE for broadcasting:")
incompatible_pairs = [
    ((3, 4), (5,)),
    ((2, 3), (3, 2)),
    ((2, 1, 3), (2, 4, 1)),
]

for shape_a, shape_b in incompatible_pairs:
    a = torch.randn(*shape_a)
    b = torch.randn(*shape_b)
    try:
        _ = a + b
        print(f"  {shape_a} + {shape_b} → SUCCEEDED (unexpected)")
    except RuntimeError as e:
        print(f"  {shape_a} + {shape_b} → FAILED: {str(e)[:60]}...")

print("\nThese shapes ARE compatible:")
compatible_pairs = [
    ((3, 4), (4,)),       # (3,4) + (1,4) → (3,4)
    ((3, 1), (1, 4)),     # → (3, 4)
    ((5, 3, 1), (1, 1, 4)),  # → (5, 3, 4)
    ((1,), (5, 3)),       # (1,1) + (5,3) → (5, 3)
]

for shape_a, shape_b in compatible_pairs:
    a = torch.randn(*shape_a)
    b = torch.randn(*shape_b)
    c = a + b
    print(f"  {shape_a} + {shape_b} → {tuple(c.shape)}")


print("\n" + "=" * 70)
print("EXAMPLE 9: Common broadcasting mistake and fix")
print("=" * 70)

# Task: subtract row means from each row
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

# WRONG: row_means has shape (2,), which broadcasts against columns, not rows
row_means_bad = x.mean(dim=1)  # Shape: (2,)
print(f"x shape: {x.shape}")
print(f"row_means (no keepdim) shape: {row_means_bad.shape}")
print("  This would subtract along the WRONG dimension!")

# RIGHT: keepdim=True gives shape (2, 1), which broadcasts correctly
row_means_good = x.mean(dim=1, keepdim=True)  # Shape: (2, 1)
result = x - row_means_good
print(f"\nrow_means (keepdim=True) shape: {row_means_good.shape}")
print(f"x - row_means:\n{result}")
print(f"Verify row means are 0: {result.mean(dim=1)}")

print("\n" + "=" * 70)
print("Broadcasting demonstration complete!")
print("=" * 70)
