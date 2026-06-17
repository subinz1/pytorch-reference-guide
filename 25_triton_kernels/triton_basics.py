"""
Custom Triton Kernels — Basics
==============================
Module 25: Vector addition, fused add+ReLU, fused softmax kernels
with line-by-line explanations and benchmarks against PyTorch eager.

Requirements: pip install torch triton (NVIDIA GPU required for kernel execution)
"""

import time
import torch

# ---------------------------------------------------------------------------
# Check Triton availability
# ---------------------------------------------------------------------------
HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

HAS_CUDA = torch.cuda.is_available()

print("=" * 70)
print("Module 25: Custom Triton Kernels — Basics")
print("=" * 70)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {HAS_CUDA}")
print(f"Triton available: {HAS_TRITON}")
if HAS_CUDA:
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
print("=" * 70)


# ===================================================================
# SECTION 1: Triton Programming Model (Conceptual)
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 1: Triton Programming Model")
print("=" * 70)

print("""
Triton Kernel Execution Model
------------------------------

A Triton kernel runs as a GRID of BLOCKS (also called "programs").
Each block processes a chunk of data independently and in parallel.

Example: Processing 16 elements with BLOCK_SIZE=4

  Data:    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  Block 0: [0, 1, 2, 3]          pid=0
  Block 1: [4, 5, 6, 7]          pid=1
  Block 2: [8, 9, 10, 11]        pid=2
  Block 3: [12, 13, 14, 15]      pid=3

Key primitives:
  @triton.jit          — Compile a Python function into a GPU kernel
  tl.program_id(axis)  — Which block am I? (0-indexed)
  tl.arange(0, N)      — Range [0..N-1] within a block
  tl.load(ptr, mask)   — Load from GPU memory (mask for bounds)
  tl.store(ptr, v, m)  — Store to GPU memory (mask for bounds)

Grid launch syntax:
  kernel[grid](arg1, arg2, ..., BLOCK_SIZE=1024)
""")

# ===================================================================
# SECTION 2: Vector Addition Kernel
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 2: Vector Addition — Hello World Triton Kernel")
print("=" * 70)

if HAS_TRITON and HAS_CUDA:

    # ---------------------------------------------------------------
    # Kernel: element-wise addition of two vectors
    #
    # Each block (identified by program_id) computes a contiguous
    # chunk of BLOCK_SIZE elements from the output vector.
    # ---------------------------------------------------------------
    @triton.jit
    def add_kernel(
        x_ptr,      # pointer to first input vector in GPU memory
        y_ptr,      # pointer to second input vector in GPU memory
        out_ptr,    # pointer to output vector in GPU memory
        n,          # total number of elements
        BLOCK_SIZE: tl.constexpr,  # elements per block (compile-time constant)
    ):
        # Which block is this? For a 1D grid, axis=0.
        pid = tl.program_id(0)

        # Compute the range of global indices this block handles.
        # Block 0 → [0, 1, ..., BS-1], Block 1 → [BS, BS+1, ..., 2*BS-1], ...
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Boolean mask: True for valid indices, False for out-of-bounds.
        # The last block may extend past the array boundary.
        mask = offsets < n

        # Load from global memory. Masked-out lanes get 0 by default.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        # Compute in registers (no memory traffic for this step).
        result = x + y

        # Write result back to global memory (masked stores skip OOB).
        tl.store(out_ptr + offsets, result, mask=mask)

    def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Launch the Triton vector addition kernel."""
        assert x.is_cuda and y.is_cuda and x.shape == y.shape
        output = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        # Grid: number of blocks = ceil(n / BLOCK_SIZE)
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output

    # Test correctness
    N = 100_000
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    z_triton = triton_add(x, y)
    z_pytorch = x + y
    assert torch.allclose(z_triton, z_pytorch, atol=1e-6), "Mismatch!"
    print(f"[Vector Add] N={N:,}: Triton matches PyTorch ✓")

    # Benchmark
    def benchmark_fn(fn, *args, warmup=25, rep=100):
        """Simple GPU benchmark: returns median time in ms."""
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        times = []
        for _ in range(rep):
            start = time.perf_counter()
            fn(*args)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        return times[len(times) // 2]

    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    print(f"\n{'Size':>12s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}")
    print("-" * 55)
    for sz in sizes:
        x = torch.randn(sz, device="cuda")
        y = torch.randn(sz, device="cuda")
        t_pt = benchmark_fn(torch.add, x, y)
        t_tr = benchmark_fn(triton_add, x, y)
        speedup = t_pt / t_tr if t_tr > 0 else float("inf")
        print(f"{sz:>12,d} | {t_pt:>12.4f} | {t_tr:>12.4f} | {speedup:>7.2f}x")
    print()

else:
    print("""
[CPU-only mode] Vector Addition kernel structure:

  @triton.jit
  def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)                              # block index
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # global indices
      mask = offsets < n                                    # bounds check
      x = tl.load(x_ptr + offsets, mask=mask)              # load x
      y = tl.load(y_ptr + offsets, mask=mask)              # load y
      tl.store(out_ptr + offsets, x + y, mask=mask)        # store x+y

  Grid launch: add_kernel[(ceil(n/BS),)](x, y, out, n, BLOCK_SIZE=1024)

  Each block of 1024 threads processes 1024 elements. For n=100,000
  that's ceil(100000/1024) = 98 blocks running in parallel on the GPU.
""")


# ===================================================================
# SECTION 3: Fused Add + ReLU Kernel
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 3: Fused Add + ReLU — Why Fusion Matters")
print("=" * 70)

print("""
Memory bandwidth is the bottleneck for elementwise ops.

Unfused (PyTorch eager):
  temp = x + y       # Kernel 1: read x,y → write temp  (3 mem ops)
  out  = relu(temp)  # Kernel 2: read temp → write out   (2 mem ops)
  Total: 5 memory ops + 1 temp allocation

Fused (Triton):
  out = max(x + y, 0) # 1 kernel: read x,y → write out  (3 mem ops)
  Total: 3 memory ops + 0 temp allocations

Savings: 40% less memory traffic, no temporary tensor.
""")

if HAS_TRITON and HAS_CUDA:

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

        # Fused operation: add then ReLU, no intermediate memory write
        result = tl.maximum(x + y, 0.0)

        tl.store(out_ptr + offsets, result, mask=mask)

    def triton_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda and y.is_cuda
        output = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        fused_add_relu_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output

    def pytorch_add_relu(x, y):
        return torch.relu(x + y)

    # Correctness
    N = 100_000
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    assert torch.allclose(triton_add_relu(x, y), pytorch_add_relu(x, y), atol=1e-6)
    print(f"[Fused Add+ReLU] N={N:,}: Triton matches PyTorch ✓")

    # Benchmark
    print(f"\n{'Size':>12s} | {'Eager (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}")
    print("-" * 55)
    for sz in sizes:
        x = torch.randn(sz, device="cuda")
        y = torch.randn(sz, device="cuda")
        t_pt = benchmark_fn(pytorch_add_relu, x, y)
        t_tr = benchmark_fn(triton_add_relu, x, y)
        speedup = t_pt / t_tr if t_tr > 0 else float("inf")
        print(f"{sz:>12,d} | {t_pt:>12.4f} | {t_tr:>12.4f} | {speedup:>7.2f}x")
    print()

else:
    print("""
[CPU-only mode] Fused Add+ReLU kernel:

  @triton.jit
  def fused_add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n
      x = tl.load(x_ptr + offsets, mask=mask)
      y = tl.load(y_ptr + offsets, mask=mask)
      result = tl.maximum(x + y, 0.0)          # <-- fused: add + relu
      tl.store(out_ptr + offsets, result, mask=mask)

  The key insight: x+y is computed in registers and immediately
  passed to max(., 0). No intermediate tensor is ever written to
  global memory. This saves both memory and bandwidth.
""")


# ===================================================================
# SECTION 4: Fused Softmax Kernel
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 4: Fused Softmax — A Practical Kernel")
print("=" * 70)

print("""
Softmax algorithm for each row x:
  1. max_val = max(x)           — numerical stability
  2. x = x - max_val            — shift
  3. x = exp(x)                 — exponentiate
  4. sum_val = sum(x)           — normalization constant
  5. out = x / sum_val          — normalize

In eager PyTorch, each step creates a temporary tensor.
In Triton, we load the row once, compute everything in
registers/SRAM, and write the result once.
""")

if HAS_TRITON and HAS_CUDA:

    @triton.jit
    def softmax_kernel(
        input_ptr, output_ptr,
        n_cols,
        input_row_stride, output_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each block processes one row of the matrix
        row_idx = tl.program_id(0)

        # Pointers to this row's data
        row_start = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load entire row into SRAM (masked positions get -inf so exp(-inf)=0)
        row = tl.load(row_start + col_offsets, mask=mask, other=float("-inf"))

        # Step 1-2: subtract max for numerical stability
        row_max = tl.max(row, axis=0)
        row = row - row_max

        # Step 3: exponentiate
        numerator = tl.exp(row)

        # Step 4: sum
        denominator = tl.sum(numerator, axis=0)

        # Step 5: normalize
        softmax_out = numerator / denominator

        # Write result
        out_start = output_ptr + row_idx * output_row_stride
        tl.store(out_start + col_offsets, softmax_out, mask=mask)

    def triton_softmax(x: torch.Tensor) -> torch.Tensor:
        """Row-wise softmax using Triton."""
        n_rows, n_cols = x.shape
        # BLOCK_SIZE must be >= n_cols (whole row in one block)
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

    # Correctness
    M, K = 1024, 256
    x = torch.randn(M, K, device="cuda")
    out_triton = triton_softmax(x)
    out_pytorch = torch.softmax(x, dim=-1)
    assert torch.allclose(out_triton, out_pytorch, atol=1e-5), "Mismatch!"
    print(f"[Fused Softmax] ({M}x{K}): Triton matches PyTorch ✓")

    # Benchmark for different matrix shapes
    shapes = [(256, 128), (1024, 256), (4096, 512), (8192, 1024)]
    print(f"\n{'Shape':>14s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}")
    print("-" * 58)
    for m, k in shapes:
        x = torch.randn(m, k, device="cuda")
        t_pt = benchmark_fn(lambda t: torch.softmax(t, dim=-1), x)
        t_tr = benchmark_fn(triton_softmax, x)
        speedup = t_pt / t_tr if t_tr > 0 else float("inf")
        print(f"{str((m,k)):>14s} | {t_pt:>12.4f} | {t_tr:>12.4f} | {speedup:>7.2f}x")

    # Show that rows sum to 1
    print(f"\nRow sums (should be ~1.0): {out_triton.sum(dim=-1)[:5].tolist()}")
    print()

else:
    print("""
[CPU-only mode] Fused Softmax kernel:

  @triton.jit
  def softmax_kernel(input_ptr, output_ptr, n_cols,
                     input_row_stride, output_row_stride,
                     BLOCK_SIZE: tl.constexpr):
      row_idx = tl.program_id(0)        # one block per row
      row_start = input_ptr + row_idx * input_row_stride
      col_offsets = tl.arange(0, BLOCK_SIZE)
      mask = col_offsets < n_cols

      row = tl.load(row_start + col_offsets, mask=mask, other=float('-inf'))
      row_max = tl.max(row, axis=0)
      numerator = tl.exp(row - row_max)
      denominator = tl.sum(numerator, axis=0)
      softmax_out = numerator / denominator

      out_start = output_ptr + row_idx * output_row_stride
      tl.store(out_start + col_offsets, softmax_out, mask=mask)

  Grid: (n_rows,) — one block per row.
  BLOCK_SIZE = next_power_of_2(n_cols) so the entire row fits in one block.

  The row is loaded once, all 5 softmax steps happen in registers,
  and the normalized result is written once. Eager PyTorch would
  allocate 3-4 temporary tensors for the intermediate results.
""")


# ===================================================================
# SECTION 5: Grid Launch Pattern
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 5: Grid Launch Patterns")
print("=" * 70)

print("""
The grid tells Triton how many blocks to launch.

1D Grid (elementwise):
  grid = (triton.cdiv(n, BLOCK_SIZE),)
  # Example: n=10000, BS=1024 → grid=(10,) → 10 blocks

2D Grid (matmul, attention):
  grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
  # Each block computes a BLOCK_M x BLOCK_N tile of the output

Lambda Grid (with autotuning):
  # BLOCK_SIZE is chosen at runtime by the autotuner
  grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
  kernel[grid](x_ptr, y_ptr, out_ptr, n)
  # The lambda is called after autotuning selects a config

Grid sizing guidelines:
  - Too few blocks  → GPU SMs sit idle
  - Too many blocks → minor scheduling overhead (usually fine)
  - Aim for >= 4 * num_SMs blocks for good occupancy
""")

if HAS_TRITON and HAS_CUDA:
    n = 100_000
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    print(f"Example: n={n:,}, BLOCK_SIZE={BLOCK_SIZE}")
    print(f"  Grid = ({num_blocks},) → {num_blocks} blocks launched")
    print(f"  GPU SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"  Blocks per SM: ~{num_blocks / torch.cuda.get_device_properties(0).multi_processor_count:.1f}")
    print()


# ===================================================================
# SECTION 6: Summary
# ===================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
What we covered:
  1. Triton programming model: grid of blocks, program_id, load/store
  2. Vector addition kernel with line-by-line explanation
  3. Fused add+ReLU showing the bandwidth savings of fusion
  4. Fused softmax — a real-world kernel doing 5 ops in one pass
  5. Grid launch patterns for 1D, 2D, and autotuned kernels

Key takeaways:
  - Triton kernels are Python functions decorated with @triton.jit
  - Each block processes BLOCK_SIZE elements identified by program_id
  - Masks handle boundary conditions (last block may be partial)
  - Fusion eliminates temporary tensors and reduces memory traffic
  - Triton handles shared memory, register allocation automatically

Next: triton_with_pytorch.py — integrate kernels with torch.library,
      add autograd, use with torch.compile, and autotune.
""")
