"""
Module 02: Views, Strides, and Memory Layout
=============================================
Deep dive into how PyTorch stores tensors in memory and how views work.

Run: python views_strides_memory.py
"""

import torch

print("=" * 70)
print("PART 1: MEMORY LAYOUT AND STRIDES")
print("=" * 70)

x = torch.arange(12).reshape(3, 4)
print(f"x =\n{x}\n")

print(f"Shape:    {x.shape}")
print(f"Strides:  {x.stride()}")
print(f"  stride[0]={x.stride()[0]} means: skip 4 elements to move down one row")
print(f"  stride[1]={x.stride()[1]} means: skip 1 element to move right one column")

# Physical memory is a flat 1D array
print(f"\nPhysical memory (flat): {x.storage().tolist()}")
print(f"Storage size: {x.storage().size()} elements")
print(f"Storage offset: {x.storage_offset()}")

# How element access works: x[i, j] is at storage[offset + i*stride[0] + j*stride[1]]
print("\nManual element access using strides:")
for i in range(3):
    for j in range(4):
        flat_idx = x.storage_offset() + i * x.stride()[0] + j * x.stride()[1]
        print(f"  x[{i},{j}] = storage[{flat_idx}] = {x.storage()[flat_idx]}", end="")
        assert x[i, j].item() == x.storage()[flat_idx]
    print()


print("\n" + "=" * 70)
print("PART 2: VIEWS — OPERATIONS THAT SHARE MEMORY")
print("=" * 70)

x = torch.arange(12).reshape(3, 4)
print(f"Original x:\n{x}")
print(f"  data_ptr: {x.data_ptr()}\n")

# View: reshape without copying
v = x.view(4, 3)
print(f"x.view(4, 3):\n{v}")
print(f"  data_ptr: {v.data_ptr()} (same!)")
print(f"  shares memory: {x.data_ptr() == v.data_ptr()}")

# Prove shared memory
v[0, 0] = 999
print(f"\nAfter v[0,0] = 999:")
print(f"  v[0,0] = {v[0, 0].item()}")
print(f"  x[0,0] = {x[0, 0].item()} (also changed!)")
v[0, 0] = 0  # Restore

print("\n--- Operations that return views ---")
operations = [
    ("x.view(4, 3)", lambda: x.view(4, 3)),
    ("x.reshape(4, 3)", lambda: x.reshape(4, 3)),
    ("x.T", lambda: x.T),
    ("x.transpose(0, 1)", lambda: x.transpose(0, 1)),
    ("x[0]", lambda: x[0]),
    ("x[:2]", lambda: x[:2]),
    ("x.unsqueeze(0)", lambda: x.unsqueeze(0)),
    ("x.expand(2, 3, 4)", lambda: x.unsqueeze(0).expand(2, 3, 4)),
    ("x.narrow(0, 0, 2)", lambda: x.narrow(0, 0, 2)),
    ("x.flatten()", lambda: x.flatten()),
]

for name, op in operations:
    result = op()
    is_view = result.data_ptr() == x.data_ptr() or result.storage().data_ptr() == x.storage().data_ptr()
    print(f"  {name:<25} shape={str(result.shape):<15} view={is_view}")


print("\n" + "=" * 70)
print("PART 3: COPIES — OPERATIONS THAT ALLOCATE NEW MEMORY")
print("=" * 70)

x = torch.arange(12).reshape(3, 4)
print(f"Original data_ptr: {x.data_ptr()}\n")

copy_operations = [
    ("x.clone()", lambda: x.clone()),
    ("x[x > 5]", lambda: x[x > 5]),
    ("x[[0, 2]]", lambda: x[[0, 2]]),
    ("x.repeat(2, 1)", lambda: x.repeat(2, 1)),
    ("x + 0", lambda: x + 0),
    ("x.contiguous()", lambda: x.contiguous()),
]

for name, op in copy_operations:
    result = op()
    same_storage = result.storage().data_ptr() == x.storage().data_ptr()
    print(f"  {name:<25} same_storage={same_storage} (copy={not same_storage})")

# Special case: contiguous() on an already-contiguous tensor returns the same tensor
print(f"\n  x.is_contiguous() = {x.is_contiguous()}")
c = x.contiguous()
print(f"  x.contiguous() returns same object: {c.data_ptr() == x.data_ptr()}")

xt = x.T
print(f"  x.T.is_contiguous() = {xt.is_contiguous()}")
xtc = xt.contiguous()
print(f"  x.T.contiguous() returns same object: {xtc.data_ptr() == xt.data_ptr()}")
print(f"  (It copies because the transpose is not contiguous)")


print("\n" + "=" * 70)
print("PART 4: TRANSPOSE AND STRIDES")
print("=" * 70)

x = torch.arange(6).reshape(2, 3)
print(f"x =\n{x}")
print(f"  strides: {x.stride()}, contiguous: {x.is_contiguous()}")
print(f"  memory:  {x.storage().tolist()}")

xt = x.T
print(f"\nx.T =\n{xt}")
print(f"  strides: {xt.stride()}, contiguous: {xt.is_contiguous()}")
print(f"  memory:  {xt.storage().tolist()} (same as x!)")

print("""
The data in memory is: [0, 1, 2, 3, 4, 5]

x reads it as 2×3 with strides (3, 1):
  x[0,0]=0  x[0,1]=1  x[0,2]=2
  x[1,0]=3  x[1,1]=4  x[1,2]=5

x.T reads SAME data as 3×2 with strides (1, 3):
  xt[0,0]=0  xt[0,1]=3
  xt[1,0]=1  xt[1,1]=4
  xt[2,0]=2  xt[2,1]=5

No data was copied — only the interpretation changed!
""")


print("=" * 70)
print("PART 5: CONTIGUITY")
print("=" * 70)

x = torch.arange(12).reshape(3, 4)
print(f"x (contiguous):")
print(f"  shape={x.shape}, strides={x.stride()}, is_contiguous={x.is_contiguous()}")

xt = x.T
print(f"\nx.T (NOT contiguous):")
print(f"  shape={xt.shape}, strides={xt.stride()}, is_contiguous={xt.is_contiguous()}")

# view() requires contiguous memory
try:
    xt.view(-1)
except RuntimeError as e:
    print(f"\nx.T.view(-1) fails: {e}")
    print("Fix: use .contiguous().view() or .reshape()")

xt_c = xt.contiguous()
print(f"\nAfter .contiguous():")
print(f"  strides: {xt_c.stride()}, is_contiguous: {xt_c.is_contiguous()}")
print(f"  Now view works: {xt_c.view(-1)}")

# reshape handles this automatically
xt_r = xt.reshape(-1)
print(f"\n.reshape(-1) works directly: {xt_r}")


print("\n" + "=" * 70)
print("PART 6: SLICING AND STRIDES")
print("=" * 70)

x = torch.arange(20).reshape(4, 5)
print(f"x =\n{x}\n")

# Normal slice
s = x[1:3]
print(f"x[1:3] (rows 1 and 2):")
print(f"  {s}")
print(f"  strides={s.stride()}, offset={s.storage_offset()}")
print(f"  contiguous={s.is_contiguous()}")

# Step slice
s2 = x[::2]
print(f"\nx[::2] (every other row):")
print(f"  {s2}")
print(f"  strides={s2.stride()}, offset={s2.storage_offset()}")
print(f"  Note stride[0]={s2.stride()[0]} (double the original)")

# Column slice
s3 = x[:, ::2]
print(f"\nx[:, ::2] (every other column):")
print(f"  {s3}")
print(f"  strides={s3.stride()}")
print(f"  Note stride[1]={s3.stride()[1]} (double the original)")

# Double step
s4 = x[::2, ::2]
print(f"\nx[::2, ::2] (checkerboard):")
print(f"  {s4}")
print(f"  strides={s4.stride()} (both doubled)")


print("\n" + "=" * 70)
print("PART 7: EXPAND vs REPEAT")
print("=" * 70)

x = torch.tensor([[1], [2], [3]])  # Shape: (3, 1)
print(f"x (3, 1):\n{x}\n")

# expand: zero-copy, uses stride trick (stride=0 for expanded dims)
x_expanded = x.expand(3, 4)
print(f"x.expand(3, 4):\n{x_expanded}")
print(f"  strides: {x_expanded.stride()}")
print(f"  stride[1]=0 means the column dimension uses NO memory!")
print(f"  same storage: {x_expanded.storage().data_ptr() == x.storage().data_ptr()}")
print(f"  storage size: {x_expanded.storage().size()} (still just 3 elements!)")

# repeat: copies data
x_repeated = x.repeat(1, 4)
print(f"\nx.repeat(1, 4):\n{x_repeated}")
print(f"  strides: {x_repeated.stride()}")
print(f"  same storage: {x_repeated.storage().data_ptr() == x.storage().data_ptr()}")
print(f"  storage size: {x_repeated.storage().size()} (12 elements — data copied)")

# Modifying expanded tensor affects all "copies"
expanded = x.expand(3, 4).clone()  # Clone to make it safe
print(f"\nExpanded (cloned for safety):\n{expanded}")


print("\n" + "=" * 70)
print("PART 8: STORAGE SHARING VISUALIZATION")
print("=" * 70)

base = torch.arange(24).reshape(4, 6)
print(f"Base tensor (4x6):\n{base}\n")

views = {
    "base[1:3, 2:5]": base[1:3, 2:5],
    "base.T": base.T,
    "base.flatten()": base.flatten(),
    "base.unsqueeze(0)": base.unsqueeze(0),
}

for name, view in views.items():
    print(f"{name}:")
    print(f"  shape={view.shape}, strides={view.stride()}")
    print(f"  offset={view.storage_offset()}, contiguous={view.is_contiguous()}")
    print(f"  shares storage: {view.storage().data_ptr() == base.storage().data_ptr()}")
    print()


print("=" * 70)
print("PART 9: PRACTICAL PATTERNS — as_strided")
print("=" * 70)

# as_strided is the low-level primitive behind views
x = torch.arange(10, dtype=torch.float)
print(f"x = {x}\n")

# Create a sliding window view (like unfold)
window_size = 3
windows = x.unfold(0, window_size, 1)
print(f"Sliding windows (size={window_size}, step=1):\n{windows}")
print(f"  shape: {windows.shape}")
print(f"  strides: {windows.stride()}")
print(f"  This is a VIEW — no data copied!")

# This is equivalent to:
windows2 = torch.as_strided(x, size=(8, 3), stride=(1, 1))
print(f"\nSame via as_strided:\n{windows2}")
print(f"  Equal: {torch.equal(windows, windows2)}")


print("\n" + "=" * 70)
print("PART 10: MEMORY DEBUGGING TIPS")
print("=" * 70)

x = torch.randn(1000, 1000)
print(f"Tensor: shape={x.shape}")
print(f"  Memory: {x.nelement() * x.element_size() / 1e6:.1f} MB")
print(f"  data_ptr: {x.data_ptr()}")

# Check if two tensors share memory
y = x.view(500, 2000)
z = x.clone()

print(f"\ny = x.view(500, 2000)")
print(f"  Shares memory with x: {x.storage().data_ptr() == y.storage().data_ptr()}")
print(f"  Modifying y modifies x: True (same storage)")

print(f"\nz = x.clone()")
print(f"  Shares memory with x: {x.storage().data_ptr() == z.storage().data_ptr()}")
print(f"  Independent copy: True")

# Total memory used
print(f"\nMemory accounting:")
print(f"  x uses: {x.nelement() * x.element_size() / 1e6:.1f} MB")
print(f"  y uses: 0 MB (view of x)")
print(f"  z uses: {z.nelement() * z.element_size() / 1e6:.1f} MB (clone)")
print(f"  Total:  {(x.nelement() + z.nelement()) * x.element_size() / 1e6:.1f} MB")

print("\n" + "=" * 70)
print("Views, strides, and memory demonstration complete!")
print("=" * 70)
