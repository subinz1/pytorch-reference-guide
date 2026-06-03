"""
Tensor Basics — Creation, Properties, and Operations
=====================================================
Covers: tensor creation, dtypes, device management, basic math, and properties.
"""

import torch

print("=" * 60)
print("1. TENSOR CREATION")
print("=" * 60)

# From Python data
x = torch.tensor([1, 2, 3])
print(f"From list: {x}")

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"2D tensor:\n{x}")

# Common creation functions
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
randn = torch.randn(3, 4)      # Normal(0, 1)
rand = torch.rand(3, 4)        # Uniform(0, 1)
empty = torch.empty(3, 4)      # Uninitialized
full = torch.full((3, 4), fill_value=3.14)
eye = torch.eye(4)             # Identity matrix

print(f"\nzeros(3,4) shape: {zeros.shape}")
print(f"eye(4):\n{eye}")

# Ranges
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, steps=5)
logspace = torch.logspace(0, 3, steps=4)

print(f"\narange(0,10,2): {arange}")
print(f"linspace(0,1,5): {linspace}")
print(f"logspace(0,3,4): {logspace}")

# Like variants (same shape/device/dtype)
x = torch.randn(3, 4)
y = torch.zeros_like(x)
z = torch.ones_like(x)
print(f"\nzeros_like shape: {y.shape}, dtype: {y.dtype}")

print("\n" + "=" * 60)
print("2. TENSOR PROPERTIES")
print("=" * 60)

x = torch.randn(3, 4, 5)
print(f"Shape:       {x.shape}")
print(f"Dimensions:  {x.ndim}")
print(f"Dtype:       {x.dtype}")
print(f"Device:      {x.device}")
print(f"Numel:       {x.numel()}")
print(f"Strides:     {x.stride()}")
print(f"Contiguous:  {x.is_contiguous()}")

print("\n" + "=" * 60)
print("3. DTYPE CASTING")
print("=" * 60)

x = torch.randn(3)
print(f"Default:   {x.dtype}")
print(f"float16:   {x.half().dtype}")
print(f"bfloat16:  {x.bfloat16().dtype}")
print(f"float64:   {x.double().dtype}")
print(f"int32:     {x.int().dtype}")
print(f"bool:      {(x > 0).dtype}")

print("\n" + "=" * 60)
print("4. DEVICE MANAGEMENT")
print("=" * 60)

x = torch.randn(3)
print(f"CPU tensor device: {x.device}")

if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(f"GPU tensor device: {x_gpu.device}")
    x_back = x_gpu.cpu()
    print(f"Back to CPU: {x_back.device}")
else:
    print("CUDA not available, skipping GPU examples")

print("\n" + "=" * 60)
print("5. NUMPY INTEROP")
print("=" * 60)

import numpy as np

np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)
print(f"From numpy: {t}")

t_np = t.numpy()
print(f"To numpy:   {t_np}")

# They share memory!
np_arr[0] = 999
print(f"After modifying numpy: tensor = {t}")

print("\n" + "=" * 60)
print("6. BASIC MATH OPERATIONS")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b     = {a + b}")
print(f"a * b     = {a * b}")
print(f"a / b     = {a / b}")
print(f"a ** 2    = {a ** 2}")
print(f"sqrt(a)   = {torch.sqrt(a)}")
print(f"exp(a)    = {torch.exp(a)}")
print(f"dot(a,b)  = {torch.dot(a, b)}")

print("\n" + "=" * 60)
print("7. REDUCTIONS")
print("=" * 60)

x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(f"x:\n{x}")
print(f"sum():     {x.sum()}")
print(f"sum(dim=0): {x.sum(dim=0)}")
print(f"sum(dim=1): {x.sum(dim=1)}")
print(f"mean():    {x.mean()}")
print(f"max():     {x.max()}")
print(f"argmax():  {x.argmax()}")
print(f"max(dim=1): values={x.max(dim=1).values}, indices={x.max(dim=1).indices}")

print("\nDone!")
