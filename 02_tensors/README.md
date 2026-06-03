# Module 02: Tensors — The Complete Guide

## Table of Contents
1. [What is a Tensor?](#what-is-a-tensor)
2. [Tensor Creation](#tensor-creation)
3. [Data Types](#data-types)
4. [Device Management](#device-management)
5. [Tensor Properties](#tensor-properties)
6. [Element-wise Operations](#element-wise-operations)
7. [Reduction Operations](#reduction-operations)
8. [Matrix Operations](#matrix-operations)
9. [Tensor Manipulation](#tensor-manipulation)
10. [Indexing Deep Dive](#indexing-deep-dive)
11. [Broadcasting](#broadcasting)
12. [Views vs Copies](#views-vs-copies)
13. [In-place Operations](#in-place-operations)
14. [Strides Explained](#strides-explained)
15. [NumPy Interop](#numpy-interop)

---

## What is a Tensor?

A tensor is a multi-dimensional array — the fundamental data structure of PyTorch.
If you know NumPy, a tensor is like `np.ndarray` but with two superpowers:
GPU acceleration and automatic differentiation.

The word "tensor" comes from mathematics, where it refers to objects that obey
certain transformation rules. In deep learning, we use the word more loosely to
mean "an n-dimensional array of numbers."

**Dimensionality taxonomy:**

| Dimensions | Math Name | PyTorch Shape | Example |
|-----------|-----------|---------------|---------|
| 0 | Scalar | `torch.Size([])` | A single loss value: 0.543 |
| 1 | Vector | `torch.Size([n])` | A word embedding: [0.2, -0.1, ...] |
| 2 | Matrix | `torch.Size([m, n])` | A linear layer's weights |
| 3 | 3-tensor | `torch.Size([a, b, c])` | A batch of sequences (batch, seq, embed) |
| 4 | 4-tensor | `torch.Size([a, b, c, d])` | A batch of images (batch, channels, H, W) |
| N | N-tensor | `torch.Size([...])` | Anything higher-dimensional |

```python
import torch

scalar = torch.tensor(3.14)                  # 0-D
vector = torch.tensor([1, 2, 3])             # 1-D
matrix = torch.tensor([[1, 2], [3, 4]])      # 2-D
cube = torch.randn(2, 3, 4)                  # 3-D

print(scalar.ndim, vector.ndim, matrix.ndim, cube.ndim)
# 0, 1, 2, 3
```

**Why "tensor" instead of "array"?** The name emphasizes that these objects carry
metadata (dtype, device, gradient tracking) and participate in PyTorch's autograd
system. A NumPy array can't run on a GPU or compute gradients.

---

## Tensor Creation

PyTorch offers many ways to create tensors. Each serves a specific purpose:

### From Python data

```python
# From a list
t = torch.tensor([1, 2, 3])          # int64 by default
t = torch.tensor([1.0, 2.0, 3.0])    # float32 by default

# From nested lists (matrix)
t = torch.tensor([[1, 2], [3, 4]])

# With explicit dtype
t = torch.tensor([1, 2, 3], dtype=torch.float32)
```

### Constant-fill tensors

```python
# Zeros and ones
torch.zeros(3, 4)         # 3x4 matrix of zeros
torch.ones(2, 3, 5)       # 2x3x5 tensor of ones
torch.full((2, 3), 7.0)   # 2x3 matrix filled with 7.0

# Identity matrix
torch.eye(4)               # 4x4 identity matrix
```

**Why these exist**: `torch.zeros` is used to initialize bias terms, accumulation
buffers, and masks. `torch.ones` creates multiplicative identities. `torch.eye`
creates identity matrices for residual connections and initialization schemes.

### Random tensors

```python
# Uniform [0, 1)
torch.rand(3, 4)

# Standard normal (mean=0, std=1)
torch.randn(3, 4)

# Random integers
torch.randint(low=0, high=10, size=(3, 4))

# Random permutation
torch.randperm(10)  # A random shuffling of [0, 1, ..., 9]
```

**Why random tensors matter**: Weight initialization is crucial for training.
`torch.randn` is the basis of Gaussian initialization. `torch.randperm` is used
for shuffling dataset indices. Setting `torch.manual_seed(42)` makes random
operations reproducible.

### Sequences

```python
# Integer sequence [0, 1, 2, ..., 9]
torch.arange(10)
torch.arange(2, 10)       # [2, 3, ..., 9]
torch.arange(0, 1, 0.1)   # [0.0, 0.1, ..., 0.9]

# Evenly spaced (specify count, not step)
torch.linspace(0, 1, steps=5)    # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.logspace(0, 2, steps=3)    # [10^0, 10^1, 10^2] = [1, 10, 100]
```

**Why these exist**: `torch.arange` creates position indices. `torch.linspace` is
used for evaluating functions over a range (plotting, interpolation). `torch.logspace`
creates logarithmic learning rate schedules.

### Uninitialized tensors

```python
# Uninitialized — contains whatever was in memory!
torch.empty(3, 4)
```

**Warning**: `torch.empty` does NOT fill with zeros. It allocates memory and returns
whatever garbage was there. Use it only when you'll immediately overwrite all values,
as it avoids the cost of zero-filling. This matters in performance-critical code.

### Like-functions (matching shape/dtype/device of an existing tensor)

```python
x = torch.randn(3, 4, dtype=torch.float32)
torch.zeros_like(x)      # Same shape, dtype, device — filled with zeros
torch.ones_like(x)       # Same but with ones
torch.randn_like(x)      # Same but with random normal values
torch.empty_like(x)      # Same but uninitialized
torch.full_like(x, 5.0)  # Same but filled with 5.0
```

**Why these exist**: When writing generic code (custom layers, loss functions),
you often need to create a tensor with the same properties as an input. These
functions handle dtype and device automatically, avoiding common bugs.

---

## Data Types

Every tensor has a dtype (data type). Choosing the right one affects memory,
speed, and numerical precision:

| dtype | Bits | Range/Precision | Use Case |
|-------|------|-----------------|----------|
| `torch.float32` (default) | 32 | ~7 decimal digits | Standard training |
| `torch.float64` | 64 | ~15 decimal digits | Numerical verification, scientific computing |
| `torch.float16` | 16 | ~3 decimal digits | Mixed-precision training (with loss scaling) |
| `torch.bfloat16` | 16 | ~3 decimal digits, wider range | LLM training (Transformer-preferred) |
| `torch.int8` | 8 | [-128, 127] | Quantized inference |
| `torch.int16` | 16 | [-32768, 32767] | Rarely used |
| `torch.int32` | 32 | [-2^31, 2^31-1] | Indices, counts |
| `torch.int64` (default for ints) | 64 | [-2^63, 2^63-1] | Default integer type |
| `torch.bool` | 8 | True/False | Masks, conditions |
| `torch.complex64` | 64 | Two float32 | Signal processing, FFT |
| `torch.complex128` | 128 | Two float64 | High-precision complex math |
| `torch.float8_e4m3fn` | 8 | ~2 digits, narrow | Transformer engine inference |
| `torch.float8_e5m2` | 8 | ~1 digit, wider range | Transformer engine training |

### float32 vs float16 vs bfloat16

This choice is one of the most important practical decisions in training:

- **float32**: Full precision. Use for the optimizer state and any numerically
  sensitive operations (loss computation, normalization).

- **float16**: Half precision. 2x memory savings, faster on GPUs with tensor cores.
  BUT has a limited range (max ~65504), which means gradients can overflow.
  Requires loss scaling for stable training.

- **bfloat16**: Same memory as float16 but with the same exponent range as float32.
  This means it almost never overflows. Preferred for LLM training because you
  get memory savings without the numerical headaches of float16.

```python
x = torch.randn(1000, 1000)
print(f"float32: {x.element_size()} bytes per element, {x.nelement() * x.element_size() / 1e6:.1f} MB")

x16 = x.to(torch.float16)
print(f"float16: {x16.element_size()} bytes per element, {x16.nelement() * x16.element_size() / 1e6:.1f} MB")

xbf = x.to(torch.bfloat16)
print(f"bfloat16: {xbf.element_size()} bytes per element")
```

### Casting

```python
x = torch.tensor([1, 2, 3])          # int64
x_float = x.float()                   # → float32
x_half = x.half()                     # → float16
x_double = x.double()                 # → float64
x_int = x_float.int()                 # → int32
x_long = x_float.long()               # → int64
x_bool = x.bool()                     # → bool (0=False, nonzero=True)
x_cast = x.to(torch.float32)          # General casting
```

---

## Device Management

Tensors live on a specific device. Operations between tensors require them to be
on the same device — this is a common source of runtime errors.

```python
# CPU (default)
x_cpu = torch.randn(3, 4)
print(x_cpu.device)  # cpu

# Check for GPU availability
if torch.cuda.is_available():
    x_gpu = torch.randn(3, 4, device='cuda')
    x_gpu = x_cpu.to('cuda')      # Move to GPU
    x_gpu = x_cpu.cuda()          # Shorthand
    x_back = x_gpu.cpu()          # Move back to CPU

# Apple Silicon
if torch.backends.mps.is_available():
    x_mps = x_cpu.to('mps')

# Context manager for default device
with torch.device('cpu'):
    x = torch.randn(3, 4)  # Created on CPU
```

**Common error**: `RuntimeError: Expected all tensors to be on the same device`.
This happens when you mix CPU and GPU tensors in an operation. Always check
`.device` when debugging.

---

## Tensor Properties

Every tensor carries metadata that you can inspect:

```python
x = torch.randn(2, 3, 4, dtype=torch.float32)

print(f"Shape: {x.shape}")              # torch.Size([2, 3, 4])
print(f"Size (same): {x.size()}")       # torch.Size([2, 3, 4])
print(f"Dimensions: {x.ndim}")          # 3
print(f"Data type: {x.dtype}")          # torch.float32
print(f"Device: {x.device}")            # cpu
print(f"Total elements: {x.numel()}")   # 24 (= 2*3*4)
print(f"Element size: {x.element_size()} bytes")  # 4 (float32)
print(f"Total memory: {x.nelement() * x.element_size()} bytes")  # 96
print(f"Strides: {x.stride()}")         # (12, 4, 1)
print(f"Is contiguous: {x.is_contiguous()}")  # True
print(f"Requires grad: {x.requires_grad}")    # False
```

**Understanding shape vs size**: They're identical. `x.shape` is a property,
`x.size()` is a method. Most people use `.shape` (following NumPy convention).
You can index into shape: `x.shape[0]` gives the first dimension.

---

## Element-wise Operations

Element-wise operations apply a function independently to each element. The
shapes of input tensors must be compatible (same shape or broadcastable).

### Arithmetic

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a + b        # tensor([5., 7., 9.])     — or torch.add(a, b)
a - b        # tensor([-3., -3., -3.])   — or torch.sub(a, b)
a * b        # tensor([4., 10., 18.])    — or torch.mul(a, b)
a / b        # tensor([0.25, 0.4, 0.5])  — or torch.div(a, b)
a ** 2       # tensor([1., 4., 9.])      — or torch.pow(a, 2)
a // b       # Floor division
a % b        # Modulo
```

### Mathematical functions

```python
x = torch.tensor([0.0, 1.0, 2.0])

torch.exp(x)      # e^x:  [1.0, 2.718, 7.389]
torch.log(x + 1)  # ln(x+1) — add 1 to avoid log(0)
torch.sqrt(x)     # [0.0, 1.0, 1.414]
torch.abs(x - 1)  # |x-1|: [1.0, 0.0, 1.0]
torch.sin(x)      # Sine
torch.cos(x)      # Cosine
torch.tanh(x)     # Hyperbolic tangent (activation function)
torch.sigmoid(x)  # 1 / (1 + e^(-x)) (activation function)
```

### Clamping

```python
x = torch.tensor([-3.0, -1.0, 0.5, 2.0, 5.0])
torch.clamp(x, min=0.0)             # ReLU! [0, 0, 0.5, 2, 5]
torch.clamp(x, min=-1.0, max=1.0)   # Clip to [-1, 1]
```

**Why clamping matters**: `torch.clamp(x, min=0)` is literally the ReLU activation
function. Gradient clipping uses `torch.clamp` on gradient norms. Value clipping
prevents numerical instability.

---

## Reduction Operations

Reductions collapse one or more dimensions by aggregating values. Understanding
the `dim` parameter is critical.

### The `dim` parameter

When you specify `dim=k`, that dimension is "collapsed" (removed from the output):

```python
x = torch.tensor([[1., 2., 3.],
                   [4., 5., 6.]])  # Shape: (2, 3)

x.sum()            # 21.0 — all elements (scalar output)
x.sum(dim=0)       # [5., 7., 9.] — collapse rows → shape (3,)
x.sum(dim=1)       # [6., 15.] — collapse columns → shape (2,)
x.sum(dim=1, keepdim=True)  # [[6.], [15.]] — shape (2, 1) — keeps the dim
```

**Mental model for `dim`**: Think "I'm reducing ALONG this axis." `dim=0` means
"go down the rows" (collapse them). `dim=1` means "go across the columns."
`keepdim=True` keeps the reduced dimension as size 1, which is essential for
broadcasting the result back.

### Common reductions

```python
x = torch.tensor([[1., 2., 3.],
                   [4., 5., 6.]])

x.mean(dim=1)           # [2., 5.] — row means
x.std(dim=1)            # Standard deviation per row
x.var(dim=1)            # Variance per row
x.max(dim=1)            # Returns (values, indices) — both the max and where
x.min(dim=0)            # Returns (values, indices) along dim 0
x.argmax(dim=1)         # Index of max in each row: [2, 2]
x.argmin(dim=0)         # Index of min in each column: [0, 0, 0]
x.prod(dim=1)           # Product: [6., 120.]
torch.norm(x, dim=1)    # L2 norm per row
```

### The max/min return type

`max` and `min` return a named tuple with `.values` and `.indices`:

```python
vals, idxs = x.max(dim=1)
print(f"Max values: {vals}")    # [3., 6.]
print(f"Max indices: {idxs}")   # [2, 2]
```

---

## Matrix Operations

### Matrix multiplication varieties

```python
# 2D @ 2D: standard matrix multiply
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                # Shape: (3, 5)
C = torch.mm(A, B)       # Equivalent, only for 2D
C = torch.matmul(A, B)   # Most general

# Batched matrix multiply: each batch is independent
A = torch.randn(10, 3, 4)   # 10 matrices of shape 3x4
B = torch.randn(10, 4, 5)   # 10 matrices of shape 4x5
C = A @ B                    # Shape: (10, 3, 5) — 10 independent matmuls
C = torch.bmm(A, B)          # Equivalent, only for 3D

# Vector-matrix products
v = torch.randn(4)
M = torch.randn(4, 5)
result = v @ M               # Shape: (5,) — vector × matrix
result = M.T @ v              # Shape: (5,) — equivalent
```

### Einstein summation (einsum)

`einsum` is the Swiss army knife of tensor operations. It uses subscript notation
to specify arbitrary contractions:

```python
# Matrix multiply: "ik,kj->ij" means sum over k
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ik,kj->ij', A, B)  # Same as A @ B

# Batch matrix multiply
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
C = torch.einsum('bik,bkj->bij', A, B)  # Same as torch.bmm

# Dot product
a = torch.randn(5)
b = torch.randn(5)
d = torch.einsum('i,i->', a, b)  # Same as torch.dot

# Outer product
outer = torch.einsum('i,j->ij', a, b)  # Shape: (5, 5)

# Trace
M = torch.randn(4, 4)
tr = torch.einsum('ii->', M)  # Same as torch.trace(M)

# Batch diagonal
B_diag = torch.einsum('bii->bi', torch.randn(3, 4, 4))  # Diagonal of each batch
```

**Why einsum?** It makes complex tensor operations readable. Instead of chains of
`transpose`, `reshape`, and `matmul`, one einsum string describes the operation
declaratively. Attention mechanisms are often clearest in einsum notation.

---

## Tensor Manipulation

### Reshaping

```python
x = torch.arange(12)  # [0, 1, 2, ..., 11]

# view: returns a view (shares memory) — requires contiguous input
x.view(3, 4)       # Shape: (3, 4)
x.view(2, 2, 3)    # Shape: (2, 2, 3)
x.view(-1, 4)      # -1 is inferred: (3, 4)
x.view(-1)         # Flatten: (12,)

# reshape: like view but works on non-contiguous tensors (may copy)
x.reshape(3, 4)
x.reshape(-1, 6)   # (2, 6)

# flatten: collapse dimensions
y = torch.randn(2, 3, 4)
y.flatten()              # Shape: (24,) — all dims
y.flatten(1)             # Shape: (2, 12) — flatten from dim 1 onward
y.flatten(start_dim=1, end_dim=2)  # Shape: (2, 12)
```

**view vs reshape**: `view` always returns a view (no memory copy), but requires
the tensor to be contiguous in memory. `reshape` returns a view if possible, but
will copy data if necessary. **Rule of thumb**: use `reshape` unless you specifically
need to guarantee no copy (then use `view` and handle the contiguity yourself).

### Transposing and permuting

```python
x = torch.randn(2, 3, 4)

# transpose: swap exactly two dimensions
x.transpose(0, 1)   # Shape: (3, 2, 4)
x.transpose(1, 2)   # Shape: (2, 4, 3)

# For 2D matrices, .T is shorthand
m = torch.randn(3, 4)
m.T                  # Shape: (4, 3)

# permute: reorder ALL dimensions at once
x.permute(2, 0, 1)  # Shape: (4, 2, 3) — moved dim 2 to front
```

**Common use case**: Images come as (batch, H, W, C) from some libraries but
PyTorch expects (batch, C, H, W). Fix with: `img.permute(0, 3, 1, 2)`.

### Squeezing and unsqueezing

```python
x = torch.randn(1, 3, 1, 4)

x.squeeze()       # Remove ALL size-1 dims → shape (3, 4)
x.squeeze(0)      # Remove dim 0 if size 1 → shape (3, 1, 4)
x.squeeze(2)      # Remove dim 2 if size 1 → shape (1, 3, 4)

y = torch.randn(3, 4)
y.unsqueeze(0)    # Add dim at position 0 → shape (1, 3, 4)
y.unsqueeze(1)    # Add dim at position 1 → shape (3, 1, 4)
y.unsqueeze(-1)   # Add dim at end → shape (3, 4, 1)
```

**Why these matter**: Many PyTorch operations expect specific numbers of dimensions.
A single image (3, H, W) needs `unsqueeze(0)` to become a batch of 1 (1, 3, H, W)
for a model. After processing, `squeeze(0)` removes the batch dimension.

### Concatenation and stacking

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# cat: join along EXISTING dimension
torch.cat([a, b], dim=0)   # Shape: (4, 3) — stack vertically
torch.cat([a, b], dim=1)   # Shape: (2, 6) — join horizontally

# stack: join along NEW dimension
torch.stack([a, b], dim=0)  # Shape: (2, 2, 3) — new dim 0 indexes a vs b
torch.stack([a, b], dim=1)  # Shape: (2, 2, 3) — new dim inserted at 1
```

**Key difference**: `cat` glues tensors along an existing dimension (dimensions
must match elsewhere). `stack` creates a new dimension and places tensors along
it (all tensors must have exactly the same shape).

### Splitting and chunking

```python
x = torch.arange(12).reshape(4, 3)

# split: split into pieces of given size
pieces = torch.split(x, 2, dim=0)  # Two pieces of size 2 each
# pieces[0] shape: (2, 3), pieces[1] shape: (2, 3)

# chunk: split into N roughly equal pieces
chunks = torch.chunk(x, 3, dim=0)  # Three chunks (sizes 2, 1, 1)
```

### Expanding and repeating

```python
x = torch.tensor([[1], [2], [3]])  # Shape: (3, 1)

# expand: view-based (no memory copy), only expands size-1 dims
x.expand(3, 4)    # Shape: (3, 4) — [1,1,1,1], [2,2,2,2], [3,3,3,3]
x.expand(-1, 4)   # -1 means "keep this dim's size"

# repeat: actually copies data
x.repeat(1, 4)    # Shape: (3, 4) — same result but data is copied
x.repeat(2, 3)    # Shape: (6, 3) — repeat 2x along dim 0, 3x along dim 1
```

**expand vs repeat**: `expand` is free (just changes strides, no memory allocation).
`repeat` copies data. Always prefer `expand` when possible. But be careful:
modifying an expanded tensor affects all "copies" since they share memory.

---

## Indexing Deep Dive

### Basic indexing (returns views)

```python
x = torch.arange(20).reshape(4, 5)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

x[0]        # Row 0: [0, 1, 2, 3, 4]
x[0, 2]     # Element at row 0, col 2: 2
x[-1]       # Last row: [15, 16, 17, 18, 19]
x[-1, -1]   # Last element: 19
```

### Slicing (returns views)

```python
x[1:3]        # Rows 1 and 2
x[:, 2:4]     # All rows, columns 2 and 3
x[::2]        # Every other row (stride 2)
x[:, ::-1]    # Reverse columns (or use torch.flip)
x[1:3, 2:5]   # Submatrix: rows 1-2, cols 2-4
```

### Boolean (mask) indexing (returns copies)

```python
x = torch.randn(4, 4)
mask = x > 0
positives = x[mask]       # 1-D tensor of all positive values
x[mask] = 0               # Set all positive values to 0
x[x < -1] = -1            # Clamp from below

# Common pattern: conditional replacement
scores = torch.randn(5)
scores[scores < 0] = 0    # ReLU by hand
```

**Why boolean indexing returns copies**: The selected elements aren't contiguous
in memory, so they can't form a view. Any modification to the result won't
affect the original tensor (but assigning back with `x[mask] = val` does work).

### Fancy (advanced) indexing

```python
x = torch.arange(20).reshape(4, 5)
rows = torch.tensor([0, 2, 3])
cols = torch.tensor([1, 3, 4])

x[rows]              # Select rows 0, 2, 3 → shape (3, 5)
x[rows, cols]         # Select (0,1), (2,3), (3,4) → [1, 13, 19]
x[:, [0, 2, 4]]      # Select columns 0, 2, 4 → shape (4, 3)
```

### gather and scatter

`gather` collects values from a tensor according to an index tensor.
Think of it as "for each position in the index, fetch the value at that index
from the source."

```python
# gather: collect elements
src = torch.tensor([[1, 2, 3],
                     [4, 5, 6]])
index = torch.tensor([[0, 2],
                       [1, 0]])
result = torch.gather(src, dim=1, index=index)
# result[0,0] = src[0, index[0,0]] = src[0, 0] = 1
# result[0,1] = src[0, index[0,1]] = src[0, 2] = 3
# result = tensor([[1, 3], [5, 4]])

# scatter: the inverse of gather — place values at index positions
dst = torch.zeros(2, 3, dtype=torch.long)
dst.scatter_(1, index, src=torch.tensor([[10, 20], [30, 40]]))
```

**Where gather is used**: Gathering token embeddings by index, selecting log
probabilities at target positions (for computing cross-entropy manually),
implementing top-k selection.

### index_select and masked_select

```python
x = torch.randn(4, 5)
torch.index_select(x, dim=0, index=torch.tensor([0, 3]))  # Select rows 0 and 3
torch.index_select(x, dim=1, index=torch.tensor([1, 4]))  # Select cols 1 and 4

mask = x > 0
selected = torch.masked_select(x, mask)  # 1-D tensor of all True-mask elements
```

---

## Broadcasting

Broadcasting lets you operate on tensors with different shapes without explicit
copying. PyTorch follows NumPy's broadcasting rules.

### The Rules (applied right-to-left)

1. If tensors have different numbers of dimensions, prepend 1s to the smaller
   tensor's shape until they match.
2. For each dimension, sizes must either be equal or one of them must be 1.
3. A dimension of size 1 is "stretched" to match the other.

```python
# Example: (3, 4) + (4,) → (3, 4) + (1, 4) → (3, 4)
A = torch.randn(3, 4)
b = torch.randn(4)        # Automatically becomes (1, 4), then broadcast to (3, 4)
C = A + b                  # Works! Each row of A gets b added

# Example: (3, 1) + (1, 4) → (3, 4)
col = torch.tensor([[1.], [2.], [3.]])   # Shape: (3, 1)
row = torch.tensor([[10., 20., 30., 40.]])  # Shape: (1, 4)
result = col + row  # Shape: (3, 4) — addition table!
# [[11, 21, 31, 41],
#  [12, 22, 32, 42],
#  [13, 23, 33, 43]]
```

### Common broadcasting patterns

```python
# Subtract the mean from each row (feature normalization)
x = torch.randn(100, 10)       # 100 samples, 10 features
mean = x.mean(dim=0)            # Shape: (10,)
x_centered = x - mean           # Broadcasting: (100, 10) - (10,) → (100, 10)

# Subtract mean from each column
col_mean = x.mean(dim=1, keepdim=True)  # Shape: (100, 1)
x_col_centered = x - col_mean            # (100, 10) - (100, 1) → (100, 10)
```

**Why keepdim=True matters**: Without it, `x.mean(dim=1)` has shape `(100,)`.
You can't subtract shape `(100,)` from shape `(100, 10)` — it's ambiguous.
With `keepdim=True`, shape is `(100, 1)`, which broadcasts unambiguously.

### Broadcasting failure

```python
# This FAILS:
a = torch.randn(3, 4)
b = torch.randn(5)
# a + b → Error! 4 ≠ 5 and neither is 1
```

---

## Views vs Copies

Understanding when PyTorch shares memory vs copies it is crucial for both
performance and correctness.

### Operations that return views (share memory)

```python
x = torch.arange(12).reshape(3, 4)

# These all share memory with x:
y = x.view(4, 3)         # Reshape (requires contiguous)
y = x.reshape(4, 3)      # Reshape (may return view)
y = x.T                   # Transpose
y = x.transpose(0, 1)     # Transpose
y = x[0]                  # Basic indexing
y = x[:2]                 # Slicing
y = x.unsqueeze(0)        # Add dimension
y = x.squeeze()           # Remove size-1 dims
y = x.expand(2, 3, 4)     # Expand size-1 dims
y = x.narrow(0, 0, 2)     # Narrowing

# Modifying y also modifies x!
y = x.view(4, 3)
y[0, 0] = 999
print(x[0, 0])  # 999 — shared memory!
```

### Operations that return copies

```python
x = torch.arange(12).reshape(3, 4)

# These create new tensors (independent memory):
y = x.clone()             # Explicit copy
y = x.contiguous()        # Copy if not already contiguous
y = x[x > 5]              # Boolean indexing
y = x[[0, 2]]             # Fancy indexing
y = x.repeat(2, 1)        # Repeat (not expand)
```

### How to check

```python
x = torch.arange(6).reshape(2, 3)
y = x.view(3, 2)
z = x.clone()

print(x.data_ptr() == y.data_ptr())  # True — same memory
print(x.data_ptr() == z.data_ptr())  # False — different memory
print(x.storage().data_ptr() == y.storage().data_ptr())  # True
```

---

## In-place Operations

In-place operations modify a tensor's data directly without allocating new memory.
They are indicated by a trailing underscore:

```python
x = torch.tensor([1.0, 2.0, 3.0])

x.add_(1)          # x is now [2, 3, 4] — no new tensor created
x.mul_(2)          # x is now [4, 6, 8]
x.zero_()          # x is now [0, 0, 0]
x.fill_(5)         # x is now [5, 5, 5]
x.clamp_(min=0)    # ReLU in-place
x.uniform_()       # Fill with uniform random numbers
x.normal_()        # Fill with normal random numbers
```

### Autograd implications

In-place operations can break autograd because they modify data that the
computation graph may need for backward:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
y.add_(1)     # This MAY cause an error during backward!
# PyTorch tracks in-place modifications via a version counter.
# If backward() needs y's original value, this will fail.
```

**Rule of thumb**: Avoid in-place operations on tensors that require gradients,
unless you're explicitly operating on gradient buffers (like zeroing them) or
on detached tensors. The one exception is `param.data` manipulation, which
bypasses autograd entirely.

---

## Strides Explained

Strides are the mechanism that makes views, transposes, and slices work without
copying data. A stride tells PyTorch how many elements to skip in physical
memory to advance one position along each dimension.

### How strides work

```python
x = torch.arange(12).reshape(3, 4)
# Memory layout: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Shape: (3, 4), Strides: (4, 1)
#
# To go from x[0,0] to x[1,0]: skip 4 elements (stride of dim 0)
# To go from x[0,0] to x[0,1]: skip 1 element (stride of dim 1)

print(f"Strides: {x.stride()}")  # (4, 1)
```

### Transpose changes strides, not data

```python
x = torch.arange(6).reshape(2, 3)
print(f"x strides: {x.stride()}")     # (3, 1) — row-major

xt = x.T
print(f"x.T strides: {xt.stride()}")  # (1, 3) — column-major
print(f"x.T is contiguous: {xt.is_contiguous()}")  # False!

# The data in memory hasn't changed! Only the interpretation (strides) changed.
# x:   [0, 1, 2, 3, 4, 5] read as 2x3 with strides (3, 1)
# x.T: [0, 1, 2, 3, 4, 5] read as 3x2 with strides (1, 3)
```

**Why contiguity matters**: Some operations (like `view`) require contiguous
memory. If a tensor isn't contiguous, call `.contiguous()` first (which
copies data to a contiguous layout) or use `.reshape()` which handles
this automatically.

### Strides enable zero-copy slicing

```python
x = torch.arange(20).reshape(4, 5)
y = x[::2, ::2]  # Every other row, every other column
print(f"y strides: {y.stride()}")  # (10, 2)
# Stride of 10 means skip 10 elements (2 rows of 5) to get next row
# Stride of 2 means skip 2 elements to get next column
# No data was copied!
```

---

## NumPy Interop

PyTorch and NumPy can share memory, enabling seamless interoperation.

### Conversion

```python
import numpy as np

# NumPy → PyTorch (shares memory by default!)
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)
np_array[0] = 999
print(tensor[0])  # 999.0 — shared memory!

# PyTorch → NumPy (shares memory by default!)
tensor = torch.tensor([1.0, 2.0, 3.0])
np_array = tensor.numpy()
tensor[0] = 999
print(np_array[0])  # 999.0 — shared memory!
```

### The shared memory warning

**This is the single most common source of subtle bugs** when mixing PyTorch and
NumPy. If you modify one, the other changes too. To get an independent copy:

```python
# Safe conversion (independent copy)
np_array = tensor.clone().numpy()           # PyTorch → NumPy (safe)
tensor = torch.from_numpy(np_array.copy())  # NumPy → PyTorch (safe)
```

### GPU tensors and NumPy

NumPy only works on CPU. You must move GPU tensors to CPU first:

```python
# If x is on GPU:
# x.numpy()          # ERROR! Can't convert CUDA tensor to numpy
# x.cpu().numpy()    # OK — move to CPU first
# x.detach().cpu().numpy()  # OK — also detaches from autograd
```

### dtype compatibility

```python
# NumPy float64 → PyTorch float64 (NOT the default float32!)
np_f64 = np.array([1.0, 2.0])  # numpy default is float64
t = torch.from_numpy(np_f64)
print(t.dtype)  # torch.float64

# If you want float32, cast explicitly
t32 = torch.from_numpy(np_f64).float()
```

---

## What's Next?

With tensors mastered, Module 03 covers autograd — PyTorch's automatic
differentiation engine that makes all of deep learning possible. You'll learn how
PyTorch tracks operations on tensors and computes gradients automatically.

Run the example files in this directory to practice:
- `creation_and_properties.py` — tensor creation and inspection
- `operations.py` — element-wise and reduction operations
- `indexing_and_slicing.py` — all forms of indexing
- `broadcasting.py` — broadcasting rules with examples
- `views_strides_memory.py` — views, strides, and memory layout
