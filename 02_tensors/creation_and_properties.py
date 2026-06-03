"""
Module 02: Tensor Creation and Properties
==========================================
Demonstrates every way to create tensors and inspect their properties.

Run: python creation_and_properties.py
"""

import torch
import numpy as np

print("=" * 70)
print("PART 1: CREATING TENSORS FROM PYTHON DATA")
print("=" * 70)

# From scalars, lists, and nested lists
scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3, 4, 5])
matrix = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                           [[5, 6], [7, 8]]])

print(f"Scalar:  value={scalar.item()}, shape={scalar.shape}, ndim={scalar.ndim}")
print(f"Vector:  {vector}, shape={vector.shape}, ndim={vector.ndim}")
print(f"Matrix:\n{matrix}")
print(f"  shape={matrix.shape}, ndim={matrix.ndim}")
print(f"3D tensor:\n{tensor_3d}")
print(f"  shape={tensor_3d.shape}, ndim={tensor_3d.ndim}")

# Explicit dtype
t_float = torch.tensor([1, 2, 3], dtype=torch.float32)
t_double = torch.tensor([1, 2, 3], dtype=torch.float64)
t_int = torch.tensor([1.7, 2.3, 3.9], dtype=torch.int32)
print(f"\nfloat32: {t_float} (dtype={t_float.dtype})")
print(f"float64: {t_double} (dtype={t_double.dtype})")
print(f"int32 (truncates!): {t_int} (dtype={t_int.dtype})")


print("\n" + "=" * 70)
print("PART 2: CONSTANT-FILL TENSORS")
print("=" * 70)

zeros = torch.zeros(2, 3)
ones = torch.ones(3, 4)
full = torch.full((2, 5), fill_value=3.14)
eye = torch.eye(4)

print(f"zeros(2, 3):\n{zeros}")
print(f"\nones(3, 4):\n{ones}")
print(f"\nfull((2, 5), 3.14):\n{full}")
print(f"\neye(4):\n{eye}")


print("\n" + "=" * 70)
print("PART 3: RANDOM TENSORS")
print("=" * 70)

torch.manual_seed(42)

rand_uniform = torch.rand(2, 3)         # Uniform [0, 1)
rand_normal = torch.randn(2, 3)         # Normal (mean=0, std=1)
rand_int = torch.randint(0, 10, (2, 3)) # Integers in [0, 10)
rand_perm = torch.randperm(8)            # Random permutation of 0..7

print(f"rand (uniform [0,1)):\n{rand_uniform}")
print(f"\nrandn (normal):\n{rand_normal}")
print(f"\nrandint (0 to 9):\n{rand_int}")
print(f"\nrandperm(8): {rand_perm}")

# Reproducibility
torch.manual_seed(42)
a = torch.randn(3)
torch.manual_seed(42)
b = torch.randn(3)
print(f"\nSame seed → same values: {a} == {b} → {torch.equal(a, b)}")


print("\n" + "=" * 70)
print("PART 4: SEQUENCES")
print("=" * 70)

arange = torch.arange(0, 10)
arange_step = torch.arange(0, 1, 0.2)
linspace = torch.linspace(0, 1, steps=5)
logspace = torch.logspace(0, 3, steps=4)

print(f"arange(0, 10):     {arange}")
print(f"arange(0, 1, 0.2): {arange_step}")
print(f"linspace(0, 1, 5): {linspace}")
print(f"logspace(0, 3, 4): {logspace}  (= 10^0, 10^1, 10^2, 10^3)")


print("\n" + "=" * 70)
print("PART 5: 'LIKE' CONSTRUCTORS")
print("=" * 70)

template = torch.randn(2, 3, dtype=torch.float32)
print(f"Template: shape={template.shape}, dtype={template.dtype}")

zl = torch.zeros_like(template)
ol = torch.ones_like(template)
rl = torch.randn_like(template)
fl = torch.full_like(template, 99.0)

print(f"zeros_like:\n{zl}")
print(f"ones_like:\n{ol}")
print(f"randn_like:\n{rl}")
print(f"full_like (99.0):\n{fl}")


print("\n" + "=" * 70)
print("PART 6: FROM NUMPY")
print("=" * 70)

np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t_from_np = torch.from_numpy(np_arr)
print(f"NumPy array:\n{np_arr}")
print(f"Torch tensor:\n{t_from_np}")
print(f"Dtype: {t_from_np.dtype} (inherited from NumPy's float64)")

# Shared memory demonstration
np_arr[0, 0] = 999
print(f"\nAfter modifying NumPy array[0,0] = 999:")
print(f"Torch tensor[0,0] = {t_from_np[0, 0].item()} (shared memory!)")


print("\n" + "=" * 70)
print("PART 7: TENSOR PROPERTIES")
print("=" * 70)

x = torch.randn(3, 4, 5)
print(f"Tensor shape: {x.shape}")
print(f"Tensor size(): {x.size()}")
print(f"Dimensions (ndim): {x.ndim}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")
print(f"Total elements (numel): {x.numel()}")
print(f"Bytes per element: {x.element_size()}")
print(f"Total bytes: {x.nelement() * x.element_size()}")
print(f"Strides: {x.stride()}")
print(f"Is contiguous: {x.is_contiguous()}")
print(f"Requires grad: {x.requires_grad}")
print(f"Layout: {x.layout}")

# Size of individual dimension
print(f"\nDimension sizes:")
for i in range(x.ndim):
    print(f"  dim {i}: {x.size(i)}")


print("\n" + "=" * 70)
print("PART 8: DATA TYPE SIZES AND RANGES")
print("=" * 70)

dtypes = [
    (torch.float16, "float16"),
    (torch.bfloat16, "bfloat16"),
    (torch.float32, "float32"),
    (torch.float64, "float64"),
    (torch.int8, "int8"),
    (torch.int16, "int16"),
    (torch.int32, "int32"),
    (torch.int64, "int64"),
    (torch.bool, "bool"),
]

print(f"{'Type':<12} {'Bytes':>5} {'Sample'}")
print("-" * 40)
for dtype, name in dtypes:
    t = torch.tensor([1], dtype=dtype)
    print(f"{name:<12} {t.element_size():>5}   {t}")

# Special: complex types
c = torch.tensor([1 + 2j], dtype=torch.complex64)
print(f"\ncomplex64: {c}, element_size={c.element_size()} bytes")


print("\n" + "=" * 70)
print("PART 9: TYPE CASTING")
print("=" * 70)

x = torch.tensor([1, 2, 3])
print(f"Original: {x} (dtype={x.dtype})")
print(f".float():  {x.float()} (dtype={x.float().dtype})")
print(f".double(): {x.double()} (dtype={x.double().dtype})")
print(f".half():   {x.half()} (dtype={x.half().dtype})")
print(f".bool():   {x.bool()} (dtype={x.bool().dtype})")
print(f".to(torch.float32): {x.to(torch.float32)}")

# Casting with loss of precision
f = torch.tensor([1.7, 2.3, 3.9])
print(f"\nfloat {f} → int: {f.int()} (truncates toward zero!)")
print(f"float {f} → long: {f.long()}")


print("\n" + "=" * 70)
print("PART 10: MEMORY LAYOUT")
print("=" * 70)

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(f"Shape: {x.shape}")
print(f"Strides: {x.stride()}")
print(f"Data pointer: {x.data_ptr()}")
print(f"Is contiguous: {x.is_contiguous()}")
print(f"\nUnderlying storage (flat): {x.storage().tolist()}")
print(f"Storage size: {x.storage().size()} elements")
print(f"Storage offset: {x.storage_offset()}")

# A view into the same storage
y = x[1:]  # Skip first row
print(f"\ny = x[1:], shape={y.shape}")
print(f"y storage offset: {y.storage_offset()}")
print(f"Same storage? {x.storage().data_ptr() == y.storage().data_ptr()}")

print("\n" + "=" * 70)
print("Tensor creation and properties demonstration complete!")
print("=" * 70)
