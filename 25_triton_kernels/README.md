<div align="center">

[← Previous Module](../24_masked_tensor/) | [🏠 Home](../README.md) | Next Module →

</div>

---

> **Module 25** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 07 — Training Pipelines](../07_training/), [Module 08 — torch.compile](../08_torch_compile/)
> **Time to complete:** ~3 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — Triton programming model, kernels, PyTorch integration, autotuning |
| [`triton_basics.py`](triton_basics.py) | Vector add, fused add+ReLU, fused softmax kernels with benchmarks |
| [`triton_with_pytorch.py`](triton_with_pytorch.py) | torch.library registration, autograd, torch.compile, autotuning |

---

# Custom Triton Kernels — GPU Programming in Python

## Table of Contents

1. [What is Triton?](#1-what-is-triton)
2. [Why Custom Triton Kernels?](#2-why-custom-triton-kernels)
3. [Triton Programming Model](#3-triton-programming-model)
4. [Hello World: Vector Addition](#4-hello-world-vector-addition)
5. [Fused Add + ReLU](#5-fused-add--relu)
6. [Fused Softmax](#6-fused-softmax)
7. [Matrix Multiplication](#7-matrix-multiplication)
8. [Integrating Triton Kernels with PyTorch](#8-integrating-triton-kernels-with-pytorch)
9. [Autotuning](#9-autotuning)
10. [Grid Functions](#10-grid-functions)
11. [Common Patterns](#11-common-patterns)
12. [Triton vs CUDA](#12-triton-vs-cuda)
13. [How TorchInductor Uses Triton](#13-how-torchinductor-uses-triton)
14. [Upstream Updates (June 2026)](#14-upstream-updates-june-2026)
15. [Summary & Next Steps](#15-summary--next-steps)

---

## 1. What is Triton?

**Triton** is OpenAI's open-source language and compiler for writing GPU kernels in Python. It sits between the ease of PyTorch and the raw power of CUDA C++:

```
Ease of use:   PyTorch  >  Triton  >  CUDA C++
Performance:   CUDA C++ ≈  Triton  >  PyTorch (eager)
```

Key facts about Triton:

- **Python syntax** — you write GPU kernels that look like Python (with NumPy-like operations), but they compile to PTX/SASS and run directly on NVIDIA GPUs.
- **Near-peak performance** — Triton's compiler handles tiling, shared memory, register allocation, and memory coalescing automatically. Well-written Triton kernels achieve 80-95% of hand-tuned CUDA performance.
- **PyTorch uses Triton internally** — TorchInductor (the `torch.compile` backend) generates Triton code for fused operations. When you `torch.compile` a model, the generated kernels are Triton.
- **Custom kernel integration** — you can write your own Triton kernels and register them as PyTorch custom ops, complete with autograd support, shape inference, and `torch.compile` compatibility.

### Installation

Triton ships with PyTorch on Linux (CUDA builds). You can also install it standalone:

```bash
pip install triton
```

> **Note:** Triton requires an NVIDIA GPU (Compute Capability 7.0+, i.e., Volta or later). All examples in this module detect GPU availability and provide explanations when running on CPU.

---

## 2. Why Custom Triton Kernels?

### The Memory Bandwidth Problem

Modern GPUs have enormous compute throughput (e.g., A100: 312 TFLOPS for FP16) but relatively limited memory bandwidth (e.g., A100: 2 TB/s). For many operations, the bottleneck is not compute — it is moving data between GPU global memory and the compute units.

Consider a simple `y = relu(x + bias)`:

```
Eager PyTorch (2 separate kernels):
  Kernel 1: Read x, Read bias → Compute add → Write temp       (2 reads + 1 write)
  Kernel 2: Read temp         → Compute relu → Write y          (1 read + 1 write)
  Total memory traffic: 3 reads + 2 writes = 5 memory ops
```

```
Fused Triton kernel (1 kernel):
  Kernel: Read x, Read bias → Compute add+relu → Write y        (2 reads + 1 write)
  Total memory traffic: 2 reads + 1 write = 3 memory ops
```

The fused version does 40% less memory traffic. For larger fusion chains (common in Transformers), the savings are even greater.

### Use Cases

| Use Case | Example |
|----------|---------|
| **Fuse operations** | Combine elementwise ops, reductions, and activations into one kernel |
| **Custom ops** | Implement operations that don't exist in PyTorch (novel attention variants, custom normalizations) |
| **Eliminate overhead** | Remove Python/dispatch overhead by running everything in a single GPU launch |
| **Prototyping** | Iterate on GPU kernel ideas 10x faster than CUDA C++ |
| **Match Inductor** | Write kernels that equal or beat what `torch.compile` generates |

---

## 3. Triton Programming Model

### Block-Based Execution

Triton programs run as a **grid of blocks** (called "programs"). Each block processes a chunk of data independently and in parallel:

```
Data:    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Block 0: [0, 1, 2, 3]          pid=0, processes indices 0-3
Block 1: [4, 5, 6, 7]          pid=1, processes indices 4-7
Block 2: [8, 9, 10, 11]        pid=2, processes indices 8-11
Block 3: [12, 13, 14, 15]      pid=3, processes indices 12-15
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| `@triton.jit` | Decorator that compiles a Python function into a GPU kernel |
| `tl.program_id(axis)` | Returns the index of the current block along the given axis |
| `BLOCK_SIZE` | Number of elements each block processes (a `tl.constexpr`) |
| `tl.arange(0, N)` | Creates a range `[0, 1, ..., N-1]` within a block (like `torch.arange`) |
| `tl.load(ptr, mask)` | Loads data from GPU memory. The mask handles out-of-bounds indices |
| `tl.store(ptr, val, mask)` | Stores data to GPU memory with an optional mask |
| Grid | Total number of blocks to launch — `grid = (num_blocks,)` |

### The `tl.constexpr` Annotation

Parameters marked as `tl.constexpr` are compile-time constants. Triton compiles a separate kernel for each unique value. This allows the compiler to make aggressive optimizations (unrolling, constant folding):

```python
@triton.jit
def my_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE is known at compile time
    # Triton can fully unroll loops over BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
```

### Memory Model

Unlike CUDA, Triton **automatically manages shared memory (SRAM)**. When you `tl.load` data, the compiler decides whether to stage it through shared memory for reuse. You focus on the algorithm; the compiler handles the memory hierarchy.

---

## 4. Hello World: Vector Addition

The simplest Triton kernel — adding two vectors element by element:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,          # Pointer to first input vector
    y_ptr,          # Pointer to second input vector
    out_ptr,        # Pointer to output vector
    n,              # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Elements per block (compile-time)
):
    # Step 1: Which block am I?
    pid = tl.program_id(0)

    # Step 2: Compute which indices this block handles
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Step 3: Mask to avoid out-of-bounds access
    mask = offsets < n

    # Step 4: Load inputs (masked — out-of-bounds loads return 0)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Step 5: Compute
    result = x + y

    # Step 6: Store output (masked — out-of-bounds stores are skipped)
    tl.store(out_ptr + offsets, result, mask=mask)
```

### Line-by-Line Explanation

1. **`pid = tl.program_id(0)`** — Gets the block index along axis 0. For a 1D grid with 4 blocks, this returns 0, 1, 2, or 3.

2. **`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`** — Computes the global indices this block processes. Block 0 gets `[0, 1, ..., BS-1]`, block 1 gets `[BS, BS+1, ..., 2*BS-1]`, etc.

3. **`mask = offsets < n`** — Creates a boolean mask. The last block may extend past the array — the mask prevents reading/writing garbage.

4. **`tl.load(x_ptr + offsets, mask=mask)`** — Loads elements from GPU memory. Pointer arithmetic in Triton is element-wise (like C). The mask ensures out-of-bounds addresses are not accessed.

5. **`result = x + y`** — Standard addition. This happens in registers — no memory traffic.

6. **`tl.store(out_ptr + offsets, result, mask=mask)`** — Writes results back to GPU memory.

### Launching the Kernel

```python
import torch

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # Ceiling division
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output

# Usage
x = torch.randn(100_000, device='cuda')
y = torch.randn(100_000, device='cuda')
z = triton_add(x, y)
assert torch.allclose(z, x + y)
```

The `kernel[grid](args...)` syntax launches the kernel over the grid. `triton.cdiv(n, BLOCK_SIZE)` computes `ceil(n / BLOCK_SIZE)` — the number of blocks needed.

---

## 5. Fused Add + ReLU

Fusion is Triton's killer feature. Instead of two kernel launches, we do everything in one pass:

```python
@triton.jit
def fused_add_relu_kernel(
    x_ptr, y_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Fused: add + relu in one pass (no intermediate memory write)
    result = tl.maximum(x + y, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)
```

### Why This is Faster

```
PyTorch eager:
  temp = x + y          # Kernel 1: read x,y → write temp (allocate temp tensor!)
  out  = relu(temp)     # Kernel 2: read temp → write out
  # 2 kernel launches, 1 temporary allocation, 5 memory transactions

Triton fused:
  out = max(x + y, 0)   # 1 kernel: read x,y → write out
  # 1 kernel launch, 0 temporary allocations, 3 memory transactions
```

For a 10M element tensor in FP32, the temporary tensor alone is 40 MB of wasted memory bandwidth. On an A100 (2 TB/s bandwidth), that's ~20 microseconds of pure overhead eliminated.

---

## 6. Fused Softmax

A real-world kernel: computing softmax over rows of a matrix in a single pass.

### The Algorithm

For each row `x`:
1. `max_val = max(x)` — for numerical stability
2. `x = x - max_val` — shift
3. `x = exp(x)` — exponentiate
4. `sum_val = sum(x)` — normalize
5. `out = x / sum_val`

In eager PyTorch, this involves multiple intermediate tensors. In Triton, we do it all in registers:

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one row
    row_idx = tl.program_id(0)

    # Pointers to the start of this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the entire row into SRAM
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))

    # Compute softmax in registers
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    # Write back
    out_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(out_start_ptr + col_offsets, softmax_out, mask=mask)
```

### Key Details

- **One block per row** — the grid size equals the number of rows.
- **`other=float('-inf')`** — masked-out positions get negative infinity, so `exp(-inf) = 0` and they don't contribute to the sum.
- **Everything in registers/SRAM** — the row is loaded once, all computation happens locally, and the result is written once. Eager PyTorch would create temporaries for each step.

### Launching

```python
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        x, output,
        n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

> **Limitation:** This simple kernel requires `BLOCK_SIZE >= n_cols` (the whole row must fit in one block). For very wide rows, you'd need a two-pass approach. In practice, Triton's maximum block size (up to 64K elements depending on dtype) handles most use cases.

---

## 7. Matrix Multiplication

Matrix multiplication demonstrates Triton's tiling model. We compute `C = A @ B` where `A` is `(M, K)` and `B` is `(K, N)`:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: each block computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator (initialized to zero)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load tiles of A and B
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Accumulate: BLOCK_M x BLOCK_K @ BLOCK_K x BLOCK_N
        acc += tl.dot(a, b)

    # Store the output tile
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
```

### Tiling Explained

```
Matrix C (M x N):
┌─────────────────────────┐
│ Block(0,0) │ Block(0,1) │  ← Each block computes a BLOCK_M x BLOCK_N tile
│            │            │
├────────────┼────────────┤
│ Block(1,0) │ Block(1,1) │
│            │            │
└─────────────────────────┘

For each tile of C, we iterate over K in chunks of BLOCK_K:
  acc += A_tile @ B_tile   (repeated K/BLOCK_K times)
```

- **2D grid** — `grid = (M // BLOCK_M, N // BLOCK_N)`. Each block is identified by `(pid_m, pid_n)`.
- **`tl.dot`** — hardware-accelerated matrix multiply on Tensor Cores (FP16/BF16/TF32).
- **Accumulator in FP32** — even if inputs are FP16, we accumulate in FP32 for precision.
- **Shared memory is implicit** — Triton automatically stages loaded tiles through SRAM. You never manually manage `__shared__` memory.

---

## 8. Integrating Triton Kernels with PyTorch

Raw Triton kernels are useful, but to work with PyTorch's autograd, `torch.compile`, and other features, you need to register them as custom ops.

### Step 1: Define the Kernel

```python
@triton.jit
def _fused_gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    out = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, out, mask=mask)
```

### Step 2: Register as a Custom Op

```python
@torch.library.custom_op("mylib::fused_gelu", mutates_args=())
def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    _fused_gelu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output
```

### Step 3: Add Meta (Fake Tensor) Implementation

For `torch.compile` to trace through your op, it needs to know the output shape without running the kernel:

```python
@fused_gelu.register_fake
def fused_gelu_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### Step 4: Add Autograd Support

```python
def fused_gelu_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)

def fused_gelu_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    # GELU derivative (could also be a Triton kernel)
    grad_input = grad_output * (
        0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
        + 0.5 * x * (1.0 - torch.tanh(0.7978845608 * (x + 0.044715 * x**3))**2)
        * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x**2)
    )
    return grad_input

fused_gelu.register_autograd(fused_gelu_backward, setup_context=fused_gelu_setup_context)
```

### Step 5: Use with torch.compile

```python
@torch.compile
def model_forward(x):
    return torch.ops.mylib.fused_gelu(x)  # Seamlessly compiled

x = torch.randn(1024, requires_grad=True, device='cuda')
y = model_forward(x)
y.sum().backward()  # Autograd works!
```

The full pipeline: Triton kernel → `custom_op` → `register_fake` → `register_autograd` → works with `torch.compile`.

---

## 9. Autotuning

Different GPUs and problem sizes perform best with different block sizes. Triton provides built-in autotuning:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n'],  # Re-tune when 'n' changes
)
@triton.jit
def add_kernel_autotuned(
    x_ptr, y_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

### How It Works

1. **First call** — Triton benchmarks all configs and picks the fastest one for the given `n`.
2. **Subsequent calls** with the same `n` — uses the cached best config.
3. **Different `n`** — re-benchmarks (since different sizes may have different optimal configs).

### Matmul Autotuning

For more complex kernels, you can tune multiple parameters simultaneously:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_autotuned(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

The `num_warps` parameter controls how many CUDA warps (groups of 32 threads) execute each block. More warps can hide memory latency but increase register pressure.

---

## 10. Grid Functions

The grid tells Triton how many blocks to launch. For simple kernels, you compute it directly:

```python
grid = (triton.cdiv(n, BLOCK_SIZE),)
kernel[grid](...)
```

With autotuning, `BLOCK_SIZE` is chosen at runtime, so you need a **lambda grid**:

```python
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
kernel[grid](x_ptr, y_ptr, out_ptr, n)
```

The `meta` dict contains all `tl.constexpr` parameters. The lambda is called after autotuning selects a config.

### 2D Grids

For matmul-style kernels with two-dimensional tiling:

```python
grid = lambda meta: (
    triton.cdiv(M, meta['BLOCK_M']),
    triton.cdiv(N, meta['BLOCK_N']),
)
```

### Grid Considerations

| Factor | Guidance |
|--------|----------|
| Too few blocks | GPU SMs sit idle. Aim for at least `num_SMs * 4` blocks |
| Too many blocks | Minor overhead from scheduling. Generally harmless |
| Block size too small | Instruction overhead dominates. Use 256+ for elementwise |
| Block size too large | Register spill, reduced occupancy |

---

## 11. Common Patterns

### Reduction (Sum)

```python
@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single block reduction (for n <= BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr, total)
```

For large arrays, you need a two-pass approach: each block reduces a chunk, then a second kernel reduces the partial sums.

### Elementwise with Multiple Inputs

```python
@triton.jit
def fused_bias_dropout_relu(
    x_ptr, bias_ptr, out_ptr, n,
    p_drop,  # dropout probability
    seed,    # random seed
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % x.shape[0], mask=mask)

    # Fused: bias + dropout + relu
    x = x + bias
    random = tl.rand(seed, offsets)
    x = tl.where(random > p_drop, x / (1 - p_drop), 0.0)
    x = tl.maximum(x, 0.0)

    tl.store(out_ptr + offsets, x, mask=mask)
```

### Online Softmax (Numerically Stable, Two-Pass in Registers)

The fused softmax kernel in Section 6 uses the standard approach. An **online softmax** computes max and sum in a single pass using the log-sum-exp trick, which is more register-efficient for very long rows.

### Tiled Operations

For 2D operations (convolutions, attention), use 2D indexing:

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
# Use offs_m[:, None] and offs_n[None, :] for 2D indexing
```

---

## 12. Triton vs CUDA

| Aspect | Triton | CUDA C++ |
|--------|--------|----------|
| **Language** | Python | C++ |
| **Iteration speed** | Fast (Python workflow, auto-compile) | Slow (compile, link, debug cycle) |
| **Shared memory** | Automatic (compiler-managed) | Manual (`__shared__`, bank conflict avoidance) |
| **Thread-level control** | Block-level only | Full warp/thread control |
| **Performance** | ~80-95% of hand-tuned CUDA | 100% (by definition) |
| **Warp primitives** | Limited (`tl.atomic_*`, basic reductions) | Full (`__shfl_*`, warp vote, cooperative groups) |
| **Tensor Cores** | Via `tl.dot` (automatic) | Via `wmma` or `mma.sync` (manual) |
| **Portability** | NVIDIA GPUs (AMD ROCm support WIP) | NVIDIA GPUs |
| **Debugging** | Print + assert | CUDA-GDB, Nsight Compute |

### When Triton is Sufficient

- Elementwise operations (any complexity)
- Reductions (sum, max, mean, argmax)
- Matrix multiply (including with epilogues like bias + activation)
- Softmax, layer norm, RMS norm
- Attention variants (forward pass)
- Most operations you'd find in a Transformer

### When You Need CUDA

- Warp-level primitives (warp shuffle, ballot)
- Complex thread synchronization patterns
- Custom memory access patterns (e.g., cross-warp communication)
- Extreme tuning for specific GPU architectures
- Operations on non-NVIDIA hardware (outside ROCm support)

---

## 13. How TorchInductor Uses Triton

When you call `torch.compile(model)`, TorchInductor:

1. **Traces** the model with Dynamo to get an FX graph
2. **Lowers** operations to a scheduling IR
3. **Fuses** compatible operations into groups
4. **Generates Triton kernels** for each fused group
5. **Compiles** the Triton kernels to PTX
6. **Caches** the compiled kernels for reuse

### Viewing Generated Triton Code

```python
import torch

# Set environment variable BEFORE running
# TORCH_LOGS="output_code" python my_script.py

# Or programmatically:
import torch._logging
torch._logging.set_logs(output_code=True)

@torch.compile
def f(x, y):
    return torch.relu(x + y)

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
f(x, y)  # Check logs for generated Triton code
```

The generated code looks like:

```python
# (Simplified example of Inductor-generated Triton)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.maximum(tmp2, 0)
    tl.store(out_ptr0 + x0, tmp3, xmask)
```

Notice it automatically fused `add + relu` — the same optimization we wrote by hand in Section 5!

### Your Kernels + Inductor

When you register a Triton kernel as a `custom_op` with a `register_fake` implementation, Inductor can:

- **Schedule** your kernel alongside its generated kernels
- **Fuse** operations before/after your kernel (if applicable)
- **Apply** autotuning and caching

If you don't register as a custom op, your kernel appears as a graph break to Dynamo.

---

## 14. Upstream Updates (June 2026)

Recent PyTorch commits relevant to this module's topics (June 16-17, 2026):

| PR | Area | Summary |
|----|------|---------|
| [#187402](https://github.com/pytorch/pytorch/pull/187402) | Optimizers | **Muon optimizer: `spectral_unclamped` scaling** — new scaling strategy for the Muon optimizer that avoids clamping spectral norms, improving convergence for certain architectures |
| [#186300](https://github.com/pytorch/pytorch/pull/186300) | Distributed | **c10d abort hooks and pre/post collective hooks** — new extensibility points for distributed collectives: register callbacks before and after collectives, and abort hooks for cleanup |
| [#187387](https://github.com/pytorch/pytorch/pull/187387) | Distributed | **Public `torch.distributed.set_timeout`** — exposes a public API for setting distributed operation timeouts, replacing internal-only mechanisms |
| [#183838](https://github.com/pytorch/pytorch/pull/183838) | Inductor | **Unbacked FlexAttention predicates** — Inductor now supports FlexAttention score_mod/mask_mod with unbacked SymInt predicates, enabling more dynamic attention patterns |
| [#187406](https://github.com/pytorch/pytorch/pull/187406) | Testing | **torchfuzz ~190 ops coverage expansion** — the torchfuzz fuzzing framework now covers approximately 190 PyTorch operators, up from the initial set |
| [#186976](https://github.com/pytorch/pytorch/pull/186976) | Dynamo | **`object()` support** — Dynamo can now trace through code that creates and compares `object()` sentinels, eliminating a common source of graph breaks |

These updates reflect the ongoing evolution of PyTorch's compilation stack, distributed infrastructure, and testing tooling — all areas that interact with custom Triton kernel development.

---

## 15. Summary & Next Steps

### What We Learned

| Topic | Key Takeaway |
|-------|-------------|
| **Triton** | Write GPU kernels in Python with near-CUDA performance |
| **Programming model** | Grid of blocks, `program_id`, `BLOCK_SIZE`, load/store with masks |
| **Fusion** | Combine operations to eliminate memory bandwidth waste |
| **Softmax** | Practical kernel: load row, compute in registers, write once |
| **Matmul** | Tiled approach with `tl.dot` for Tensor Core utilization |
| **PyTorch integration** | `custom_op` → `register_fake` → `register_autograd` pipeline |
| **Autotuning** | `@triton.autotune` automatically finds the best config |
| **TorchInductor** | Generates Triton code from `torch.compile` — your kernels can interact with it |

### When to Write Custom Triton Kernels

1. **torch.compile already fuses your ops** — check first! `TORCH_LOGS="output_code"` shows what Inductor generates. Often it's already optimal.
2. **Custom logic** — novel attention, custom normalization, or domain-specific ops that PyTorch doesn't support natively.
3. **Squeeze the last 10%** — when profiling shows a specific kernel is the bottleneck and you can beat Inductor's generated code.

### Further Resources

- [Triton Documentation](https://triton-lang.org/main/index.html) — official tutorials and API reference
- [Triton GitHub](https://github.com/triton-lang/triton) — source code and examples
- [PyTorch Custom Operators](https://pytorch.org/docs/stable/library.html) — `torch.library` API reference
- [TorchInductor Deep Dive](../08_torch_compile/) — how `torch.compile` works under the hood
- [Training Pipelines](../07_training/) — where custom kernels fit in the training loop

---

<div align="center">

[← Previous Module](../24_masked_tensor/) | [🏠 Home](../README.md) | Next Module →

**Notebook**: [`25_triton_kernels.ipynb`](../notebooks/25_triton_kernels.ipynb)

</div>
