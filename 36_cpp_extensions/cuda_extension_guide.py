"""
Module 36: CUDA Extension Guide — Writing GPU Kernels for PyTorch
=================================================================

This script shows CUDA kernel patterns as annotated source code strings,
explains GPU programming concepts, and compares CUDA C++ with Triton.
Everything runs on CPU (source code is printed, not compiled).

Run: python cuda_extension_guide.py
"""

import torch


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# Section 1: CUDA Kernel Basics
# ============================================================================
section("1. CUDA Kernel Anatomy")

ELEMENTWISE_KERNEL = r"""
// elementwise_ops.cu — Fused Add + ReLU CUDA kernel
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// __global__: marks this as a GPU kernel (called from CPU, runs on GPU)
// Each thread processes one element
template <typename scalar_t>
__global__ void fused_add_relu_kernel(
    const scalar_t* __restrict__ a,      // Input tensor A (read-only)
    const scalar_t* __restrict__ b,      // Input tensor B (read-only)
    scalar_t* __restrict__ output,       // Output tensor  (write-only)
    int64_t size                         // Number of elements
) {
    // Thread's global index:
    //   blockIdx.x  = which block this thread is in
    //   blockDim.x  = threads per block (e.g., 256)
    //   threadIdx.x = thread's position within its block
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: some threads in the last block may be past the array end
    if (idx < size) {
        scalar_t val = a[idx] + b[idx];
        output[idx] = val > 0 ? val : 0;  // ReLU
    }
}

// Host function — called from C++, launches the kernel on the GPU
torch::Tensor fused_add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be CUDA");
    TORCH_CHECK(b.device().is_cuda(), "b must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Shape mismatch");

    auto output = torch::empty_like(a);
    const int64_t size = a.numel();

    const int threads = 256;                          // Threads per block
    const int blocks = (size + threads - 1) / threads;  // Ceil division

    // Dispatch by dtype — generates code for float and double
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "fused_add_relu", [&] {
        fused_add_relu_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });

    return output;
}
"""

print(ELEMENTWISE_KERNEL)
print("Line-by-line breakdown:")
print("  __global__           → Kernel function, runs on GPU")
print("  __restrict__         → Pointers don't alias (compiler optimization)")
print("  blockIdx.x           → Block index in the grid")
print("  blockDim.x           → Number of threads per block")
print("  threadIdx.x          → Thread index within its block")
print("  <<<blocks, threads>>>→ Launch config: grid size, block size")
print("  AT_DISPATCH_*        → Generate code for each dtype")


# ============================================================================
# Section 2: Grid/Block Sizing
# ============================================================================
section("2. Grid and Block Sizing")

print("""
CUDA organizes threads in a hierarchy:

  Grid (all blocks)
  ├── Block 0: [Thread 0, Thread 1, ..., Thread 255]
  ├── Block 1: [Thread 256, Thread 257, ..., Thread 511]
  ├── Block 2: [Thread 512, Thread 513, ..., Thread 767]
  └── ...

Key constraints:
  - Max threads per block: 1024
  - Common choice: 256 threads/block (good occupancy)
  - Threads within a block can share memory and synchronize
  - Threads across blocks cannot synchronize (except via kernel boundary)

Warp = 32 threads that execute in lockstep on the GPU
  - Always choose block sizes that are multiples of 32
  - Common choices: 128, 256, 512
""")

# Demonstrate grid/block calculation
for num_elements in [1000, 10000, 1_000_000, 100_000_000]:
    threads = 256
    blocks = (num_elements + threads - 1) // threads
    total_threads = blocks * threads
    wasted = total_threads - num_elements
    print(f"  Elements: {num_elements:>12,}  →  Blocks: {blocks:>8,}  "
          f"×  {threads} threads  =  {total_threads:>12,} total  "
          f"({wasted:,} idle)")


# ============================================================================
# Section 3: Reduction Kernel Pattern
# ============================================================================
section("3. CUDA Kernel Pattern: Reduction (Sum)")

REDUCTION_KERNEL = r"""
// reduction.cu — Parallel sum reduction using shared memory
template <typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size
) {
    // Shared memory — fast on-chip storage shared within a block
    extern __shared__ char shared_mem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element into shared memory
    sdata[tid] = (gid < size) ? input[gid] : 0;
    __syncthreads();  // Wait for all threads to finish loading

    // Tree reduction within the block
    //   Step 1: stride=128 → threads 0-127 add elements 128-255
    //   Step 2: stride=64  → threads 0-63  add elements 64-127
    //   ...
    //   Step 8: stride=1   → thread 0 adds element 1
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the block's partial sum
    if (tid == 0) {
        atomicAdd(&output[0], sdata[0]);
    }
}

// Launch: one block per chunk of 256 elements
// sum_reduction_kernel<<<blocks, 256, 256 * sizeof(float)>>>(...)
//                                      ^^^ shared memory size
"""

print(REDUCTION_KERNEL)
print("Key concepts:")
print("  __shared__     → On-chip memory shared within a block (~48KB)")
print("  __syncthreads()→ Barrier: all threads in block must reach this point")
print("  atomicAdd()    → Thread-safe addition to global memory")
print("  Tree reduction → O(log n) parallel reduction within a block")


# ============================================================================
# Section 4: Tiled Matrix Multiplication Pattern
# ============================================================================
section("4. CUDA Kernel Pattern: Tiled Matrix Multiplication")

TILED_MATMUL = r"""
// tiled_matmul.cu — Matrix multiply using shared memory tiling
// Each block computes a TILE_SIZE x TILE_SIZE output tile
#define TILE_SIZE 16

template <typename scalar_t>
__global__ void tiled_matmul_kernel(
    const scalar_t* __restrict__ A,  // [M, K]
    const scalar_t* __restrict__ B,  // [K, N]
    scalar_t* __restrict__ C,        // [M, N]
    int M, int K, int N
) {
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    scalar_t sum = 0;

    // Loop over tiles along K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaboratively load tile from A and B into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Launch with 2D grid
// dim3 threads(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 threads
// dim3 blocks((N+15)/16, (M+15)/16);  // Enough blocks to cover output
// tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, K, N);
"""

print(TILED_MATMUL)
print("Why tiling?")
print("  - Global memory access: ~400 cycles latency")
print("  - Shared memory access: ~5 cycles latency")
print("  - Each element of A and B is loaded once per tile, not once per output")
print("  - Reduces global memory reads by factor of TILE_SIZE")


# ============================================================================
# Section 5: Memory Coalescing
# ============================================================================
section("5. Memory Coalescing")

print("""
Coalesced access: adjacent threads access adjacent memory addresses.
The GPU batches these into fewer, wider memory transactions.

Good (coalesced):
  Thread 0 reads element 0    ┐
  Thread 1 reads element 1    ├─ One 128-byte transaction
  Thread 2 reads element 2    │
  ...                         │
  Thread 31 reads element 31  ┘

Bad (strided):
  Thread 0 reads element 0      → 1 transaction
  Thread 1 reads element 100    → 1 transaction
  Thread 2 reads element 200    → 1 transaction
  ...                              32 separate transactions!
""")

COALESCING_EXAMPLE = r"""
// Good: coalesced access (row-major, threads iterate along columns)
// output[row][col] = input[row][col] * 2
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;  // Adjacent threads → adjacent cols
output[row * width + col] = input[row * width + col] * 2;

// Bad: strided access (threads iterate along rows)
// Each thread processes a different row → non-contiguous memory
int idx = blockIdx.x * blockDim.x + threadIdx.x;
for (int j = 0; j < width; j++) {
    output[idx * width + j] = input[idx * width + j] * 2;  // Sequential in 1 thread
}
"""

print(COALESCING_EXAMPLE)

# Demonstrate with PyTorch memory layout
t = torch.randn(4, 8)
print("PyTorch tensor memory layout (row-major / C-contiguous):")
print(f"  Shape:   {t.shape}")
print(f"  Strides: {t.stride()} (row stride={t.stride(0)}, col stride={t.stride(1)})")
print(f"  Contiguous: {t.is_contiguous()}")
print(f"  Adjacent elements in memory are adjacent columns (stride-1 dim)")


# ============================================================================
# Section 6: CUDA Extension setup.py
# ============================================================================
section("6. CUDA Extension setup.py")

CUDA_SETUP_PY = """
# setup.py for a CUDA extension with multiple kernels
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_cuda_ops",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="my_cuda_ops._C",
            sources=[
                "csrc/bindings.cpp",         # pybind11 module definition
                "csrc/elementwise.cu",       # Elementwise kernels
                "csrc/reduction.cu",         # Reduction kernels
                "csrc/matmul.cu",            # Matmul kernels
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_90,code=sm_90",  # H100
                    "--threads=4",  # Parallel compilation
                ],
            },
            include_dirs=["csrc/include"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
"""

print(CUDA_SETUP_PY)

print("Architecture flags explained:")
print("  compute_80/sm_80 → NVIDIA A100 (Ampere)")
print("  compute_90/sm_90 → NVIDIA H100 (Hopper)")
print("  --use_fast_math  → Allow approximations (faster, less precise)")
print("  --threads=4      → Compile 4 files in parallel")


# ============================================================================
# Section 7: Fused Dropout + Scale Kernel
# ============================================================================
section("7. Complete Example: Fused Dropout + Scale CUDA Kernel")

FUSED_DROPOUT = r"""
// fused_dropout_scale.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>   // For random number generation on GPU

template <typename scalar_t>
__global__ void fused_dropout_scale_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    uint8_t* __restrict__ mask,         // Store mask for backward pass
    int64_t size,
    float dropout_prob,
    float scale,                        // 1.0 / (1.0 - dropout_prob)
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Initialize per-thread random state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Generate uniform random number in [0, 1)
    float rand = curand_uniform(&state);

    // Apply dropout: zero out with probability dropout_prob, scale rest
    if (rand < dropout_prob) {
        output[idx] = 0;
        mask[idx] = 0;
    } else {
        output[idx] = input[idx] * static_cast<scalar_t>(scale);
        mask[idx] = 1;
    }
}

std::tuple<torch::Tensor, torch::Tensor> fused_dropout_scale_cuda(
    torch::Tensor input,
    double dropout_prob,
    bool training
) {
    if (!training || dropout_prob == 0.0) {
        return {input, torch::ones_like(input, torch::kUInt8)};
    }

    auto output = torch::empty_like(input);
    auto mask = torch::empty(input.sizes(), input.options().dtype(torch::kUInt8));
    float scale = 1.0f / (1.0f - static_cast<float>(dropout_prob));

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    // Use a random seed (in practice, use PyTorch's generator)
    unsigned long long seed = std::chrono::system_clock::now()
        .time_since_epoch().count();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "fused_dropout_scale", [&] {
            fused_dropout_scale_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                mask.data_ptr<uint8_t>(),
                input.numel(),
                static_cast<float>(dropout_prob),
                scale,
                seed
            );
        }
    );

    return {output, mask};
}
"""

print(FUSED_DROPOUT)
print("Why fuse dropout + scale?")
print("  - Standard: read input → write dropout mask → read mask → write scaled output")
print("  - Fused: read input → generate random → write scaled output + mask")
print("  - Eliminates one read + one write pass over the entire tensor")
print("  - For a 1B parameter model, that's ~8GB of saved memory bandwidth per layer")


# ============================================================================
# Section 8: When to Use CUDA C++ vs Triton
# ============================================================================
section("8. CUDA C++ vs Triton — Decision Guide")

print("""
┌────────────────────────┬───────────────────────┬─────────────────────────┐
│ Criteria               │ CUDA C++              │ Triton                  │
├────────────────────────┼───────────────────────┼─────────────────────────┤
│ Language               │ C++/CUDA              │ Python                  │
│ Learning curve         │ Steep                 │ Moderate                │
│ Compilation            │ nvcc (manual setup)   │ JIT (automatic)         │
│ Shared memory          │ Manual management     │ Automatic tiling        │
│ Memory coalescing      │ Manual                │ Automatic               │
│ Warp-level ops         │ Full access           │ Limited                 │
│ Autotuning             │ Manual grid search    │ @triton.autotune        │
│ torch.compile          │ Needs registration    │ Native support          │
│ Debugging              │ cuda-gdb, NSight      │ Python debugger         │
│ Tensor cores           │ Manual (WMMA/MMA)     │ Automatic (tl.dot)      │
│ Cross-platform         │ NVIDIA only           │ Multi-backend (WIP)     │
│ Max performance        │ Slightly higher       │ ~90-95% of handwritten  │
│ Development speed      │ Slow                  │ Fast                    │
└────────────────────────┴───────────────────────┴─────────────────────────┘

Choose CUDA C++ when you need:
  ✓ Warp-level primitives (__shfl_sync, cooperative groups)
  ✓ Complex shared memory patterns (double buffering, bank conflict avoidance)
  ✓ Integration with existing CUDA libraries (cuBLAS, cuDNN, CUTLASS)
  ✓ Hardware-specific intrinsics (TMA, async copies on Hopper)

Choose Triton when:
  ✓ Writing elementwise, reduction, or attention-like kernels
  ✓ You want autotuning without manual grid search
  ✓ Development speed matters more than last-5% performance
  ✓ You need torch.compile compatibility out of the box
""")


# ============================================================================
# Section 9: The triton_op Alternative
# ============================================================================
section("9. The triton_op Alternative")

TRITON_OP_EXAMPLE = """
import torch
import triton
import triton.language as tl

# Step 1: Write the Triton kernel (pure Python!)
@triton.jit
def add_relu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)

# Step 2: Register as a custom op via torch.library
@torch.library.custom_op("myops::add_relu", mutates_args=())
def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_relu_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

# Step 3: Register fake tensor implementation (for torch.compile)
@add_relu.register_fake
def _(x, y):
    return torch.empty_like(x)

# Step 4: Use it — works with autograd and torch.compile!
x = torch.randn(1000, device="cuda")
y = torch.randn(1000, device="cuda")
result = torch.ops.myops.add_relu(x, y)

# Works with torch.compile
@torch.compile
def f(x, y):
    return torch.ops.myops.add_relu(x, y)
"""

print(TRITON_OP_EXAMPLE)
print("Advantages of registering Triton kernels via torch.library:")
print("  - Full dispatcher integration (autograd, torch.compile, FSDP)")
print("  - torch.compile can fuse surrounding ops with your kernel")
print("  - Proper fake tensor support for tracing")
print("  - Can be called via torch.ops.myops.add_relu()")


# ============================================================================
# Section 10: Performance Considerations
# ============================================================================
section("10. CUDA Performance Considerations")

print("""
1. OCCUPANCY
   - More active warps per SM → better latency hiding
   - Limited by: registers/thread, shared memory/block, threads/block
   - Use nvcc --ptxas-options=-v to see resource usage
   - Target: >50% occupancy (but not always the bottleneck)

2. MEMORY BANDWIDTH
   - Most kernels are memory-bound, not compute-bound
   - Global memory bandwidth: ~2 TB/s (H100), ~2 TB/s (A100)
   - Shared memory bandwidth: ~19 TB/s (H100)
   - Key metric: achieved bandwidth / peak bandwidth

3. KERNEL LAUNCH OVERHEAD
   - Each kernel launch costs ~5-10 μs
   - For small tensors, launch overhead dominates
   - Solution: fuse operations into fewer kernels
   - CUDA Graphs can amortize launch overhead (Module 21)

4. MEMORY COALESCING
   - Adjacent threads should access adjacent addresses
   - Stride-1 access → coalesced → one transaction
   - Strided access → multiple transactions → bandwidth wasted

5. BANK CONFLICTS (shared memory)
   - Shared memory has 32 banks
   - Threads in a warp accessing same bank → serialization
   - Solution: pad arrays, permute access patterns

6. BRANCH DIVERGENCE
   - Threads in a warp execute in lockstep
   - If threads take different branches → both paths execute
   - Minimize data-dependent branching within a warp
""")

# Show the performance difference between fused and unfused ops (Python demo)
print("Demonstrating fusion benefit (Python/CPU):")
x = torch.randn(10_000_000)
y = torch.randn(10_000_000)

import time

# Unfused: 3 separate operations
start = time.perf_counter()
for _ in range(10):
    temp = x + y
    temp2 = temp * 2.0
    result = torch.relu(temp2)
unfused_time = (time.perf_counter() - start) / 10

# Fused equivalent (PyTorch may optimize internally)
start = time.perf_counter()
for _ in range(10):
    result = torch.relu((x + y) * 2.0)
fused_time = (time.perf_counter() - start) / 10

print(f"  Unfused (3 ops): {unfused_time*1000:.2f} ms")
print(f"  Expression:      {fused_time*1000:.2f} ms")
print(f"  (A true CUDA fused kernel would be even faster — one memory pass)")


# ============================================================================
# Section 11: Complete Extension Structure
# ============================================================================
section("11. Complete CUDA Extension File Structure")

print("""
my_cuda_ops/
├── setup.py                        # CUDAExtension + BuildExtension
├── my_cuda_ops/
│   ├── __init__.py                 # Python API
│   ├── csrc/
│   │   ├── bindings.cpp            # PYBIND11_MODULE — all function registrations
│   │   ├── elementwise.cu          # Elementwise kernels (add_relu, scale, etc.)
│   │   ├── reduction.cu            # Reduction kernels (sum, max, mean)
│   │   ├── attention.cu            # Fused attention kernel
│   │   └── include/
│   │       ├── common.h            # Shared macros, error checking
│   │       └── launch_config.h     # Grid/block size helpers
│   └── functional.py               # Python wrappers with nice API
├── tests/
│   ├── test_elementwise.py         # Compare with torch.relu(a + b)
│   ├── test_reduction.py           # Compare with torch.sum()
│   └── test_gradients.py           # torch.autograd.gradcheck
└── benchmarks/
    └── bench.py                    # torch.utils.benchmark comparison
""")

BINDINGS_CPP = r"""
// csrc/bindings.cpp — Central binding file
#include <torch/extension.h>

// Forward declarations from .cu files
torch::Tensor fused_add_relu_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor parallel_sum_cuda(torch::Tensor input);
std::tuple<torch::Tensor, torch::Tensor> fused_dropout_cuda(
    torch::Tensor input, double p, bool training);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_relu", &fused_add_relu_cuda, "Fused add + ReLU (CUDA)");
    m.def("parallel_sum", &parallel_sum_cuda, "Parallel reduction sum (CUDA)");
    m.def("fused_dropout", &fused_dropout_cuda, "Fused dropout + scale (CUDA)");
}
"""
print("=== csrc/bindings.cpp ===")
print(BINDINGS_CPP)


# ============================================================================
# Section 12: Summary
# ============================================================================
section("12. Summary")

print("""
CUDA extension development workflow:

  1. Prototype in Python/PyTorch → get correctness right
  2. Profile to find the bottleneck → is it worth a custom kernel?
  3. Consider Triton first → simpler, autotuning, torch.compile-ready
  4. If you need CUDA C++:
     a. Write kernel in .cu file
     b. Write host function that validates inputs and launches kernel
     c. Write bindings in .cpp file
     d. Build with load() (dev) or setup.py (distribution)
  5. Verify: compare outputs with PyTorch reference
  6. Profile: nvprof/NSight to measure achieved bandwidth

Key files in this module:
  - README.md                → Full theory and API reference
  - cpp_extension_basics.py  → C++ extension infrastructure (this file's sibling)
  - cuda_extension_guide.py  → CUDA kernel patterns (this file)

See also:
  - Module 25: Custom Triton Kernels (often simpler for GPU ops)
  - Module 35: The Dispatcher (how custom ops integrate with PyTorch)
  - Module 21: CUDA Graphs (amortizing kernel launch overhead)
""")
