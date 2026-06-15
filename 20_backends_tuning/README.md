<div align="center">

[← Previous Module](../19_torch_function_dispatch/) | [🏠 Home](../README.md) | [Next Module →](../21_cuda_graphs/)

</div>

---

# Module 20: torch.backends — Performance Tuning

> **Prerequisites**: [Module 07 (Training Pipelines)](../07_training/)
> **Time**: ~2 hours
> **Level**: Intermediate → Advanced

---

## Overview

PyTorch's `torch.backends` module is a **configuration layer** that controls hardware-specific optimizations. These settings determine which algorithms run under the hood for convolutions, matrix multiplications, attention, and parallelism—often yielding **2–10x speedups** with a single line of code.

Most tutorials never mention these knobs. This module changes that.

## Files in This Module

| File | Description |
|------|-------------|
| `backends_tuning.py` | Runnable script demonstrating all backend settings |

---

## 1. What Are torch.backends?

`torch.backends` exposes runtime configuration for the hardware libraries PyTorch uses:

```
torch.backends
├── cudnn          # NVIDIA cuDNN (convolutions, RNNs)
├── cuda           # NVIDIA CUDA (matmul, SDPA)
├── mkldnn         # Intel oneDNN (CPU conv, linear)
├── mkl            # Intel MKL (BLAS/LAPACK)
├── openmp         # OpenMP (CPU threading)
├── opt_einsum     # Optimized einsum contraction
└── mps            # Apple Metal (M1/M2/M3)
```

Each backend exposes **flags** you can toggle at runtime. No recompilation needed.

**Key principle**: backends control the *how*, not the *what*. The mathematical operation stays the same; the algorithm, precision, or parallelism strategy changes.

---

## 2. torch.backends.cudnn

cuDNN is NVIDIA's deep neural network library. It provides optimized implementations for convolutions, pooling, normalization, and RNNs.

### 2.1 cudnn.enabled

```python
torch.backends.cudnn.enabled  # default: True
```

When `True`, PyTorch uses cuDNN for supported operations. Disabling it falls back to slower native implementations. You almost never want to disable this.

### 2.2 cudnn.benchmark

```python
torch.backends.cudnn.benchmark = True  # default: False
```

**What it does**: Before the first convolution at each input size, cuDNN runs multiple algorithm variants and selects the fastest. Results are cached for the session.

**When it helps**:
- Fixed input sizes (standard training with constant batch size and image dimensions)
- Repeated convolutions with the same shapes
- Training CNNs (ResNet, EfficientNet, etc.)

**When it hurts**:
- Variable input sizes (NLP with different sequence lengths, object detection with varying image sizes)
- Short-lived scripts (benchmarking overhead > savings)
- First iteration is slower (paying the auto-tuning cost)

```python
# Typical training setup for CNNs with fixed input
torch.backends.cudnn.benchmark = True

# Disable for variable-size inputs
torch.backends.cudnn.benchmark = False
```

### 2.3 cudnn.deterministic

```python
torch.backends.cudnn.deterministic = True  # default: False
```

Forces cuDNN to use deterministic algorithms. Non-deterministic algorithms are often faster because they can exploit parallelism without worrying about reduction order.

**Tradeoffs**:
| | Deterministic=False | Deterministic=True |
|---|---|---|
| Speed | Faster | Slower (sometimes 2–3x) |
| Reproducibility | Run-to-run variance | Bit-exact results |
| Use case | Normal training | Debugging, CI, research requiring exact reproduction |

For full determinism, also call `torch.use_deterministic_algorithms(True)`.

### 2.4 cudnn.allow_tf32

```python
torch.backends.cudnn.allow_tf32 = True  # default: True (PyTorch 2.x)
```

Controls whether cuDNN can use TF32 precision for convolutions on Ampere+ GPUs. See Section 9 for details on TF32.

---

## 3. torch.backends.cuda

Configuration for CUDA operations beyond cuDNN (matrix multiplications, attention).

### 3.1 matmul.allow_tf32

```python
torch.backends.cuda.matmul.allow_tf32 = True  # default: True (PyTorch 2.x)
```

Allows cuBLAS to use TF32 for float32 matrix multiplications. On Ampere (A100) and later, this can **double matmul throughput** with minimal precision loss.

### 3.2 matmul.allow_fp16_reduced_precision_reduction

```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
```

Controls whether fp16 GEMMs can use reduced precision for internal accumulation. Faster but potentially less accurate for very large matrices.

### 3.3 matmul.allow_bf16_reduced_precision_reduction

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

Same as above but for bfloat16 operations. Relevant for training on Ampere+ GPUs where bf16 is preferred over fp16.

### 3.4 Flash SDP (Scaled Dot-Product Attention)

```python
# Check if Flash Attention is enabled
torch.backends.cuda.flash_sdp_enabled()  # True/False

# Enable/disable Flash Attention
torch.backends.cuda.enable_flash_sdp(True)

# Other SDPA backends
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

Flash Attention is the fastest SDPA implementation for most cases. You might disable it for debugging or to force a specific backend.

### 3.5 preferred_blas_library

```python
torch.backends.cuda.preferred_blas_library()           # get current
torch.backends.cuda.preferred_blas_library("cublas")   # set preference
# Options: "cublas", "cublaslt", "hipblaslt" (AMD)
```

Select which BLAS library handles matrix multiplications. cuBLASLt supports more fused operations and epilogues.

---

## 4. torch.backends.mkldnn

Intel oneDNN (formerly MKL-DNN) provides optimized CPU implementations for convolutions, linear layers, and batch normalization.

```python
torch.backends.mkldnn.enabled  # default: True on x86

# Check if available
torch.backends.mkldnn.is_available()
```

When enabled, PyTorch automatically routes supported operations through oneDNN for faster CPU execution. This is especially impactful for inference on Intel CPUs.

---

## 5. torch.backends.mkl

Intel Math Kernel Library (MKL) provides optimized BLAS/LAPACK routines.

```python
# Enable verbose mode to see which MKL routines are called
torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON)
# VERBOSE_OFF, VERBOSE_ON
```

**Use case**: Profiling CPU performance to confirm MKL is being used for linear algebra operations.

---

## 6. torch.backends.openmp

Controls OpenMP threading for CPU parallelism.

```python
import torch

# Get/set number of threads
torch.get_num_threads()        # current thread count
torch.set_num_threads(4)       # set to 4 threads

# Also controlled via environment variable (before import):
# OMP_NUM_THREADS=4
# MKL_NUM_THREADS=4
```

**Guidelines**:
- For training: set to number of physical cores (not hyperthreads)
- For inference with batching: reduce threads to allow concurrent requests
- For DataLoader workers: reduce to avoid oversubscription

```python
import os
os.environ["OMP_NUM_THREADS"] = "4"  # must be set BEFORE importing torch
```

---

## 7. torch.backends.opt_einsum

Optimized path planning for `torch.einsum` operations.

```python
torch.backends.opt_einsum.enabled       # default: True if opt_einsum installed
torch.backends.opt_einsum.strategy      # default: "auto"
# Strategies: "auto", "greedy", "optimal", "branch-all", "branch-2", "dp"
```

**What it does**: For complex einsum expressions with 3+ tensors, finding the optimal contraction order is NP-hard. `opt_einsum` uses heuristics to find near-optimal orderings that can be **orders of magnitude faster**.

```python
# Example: without optimization, this could be O(N^5)
# With optimal contraction order, it's O(N^3)
result = torch.einsum("ij,jk,kl->il", A, B, C)
```

**Strategies**:
| Strategy | Speed | Quality | Use case |
|----------|-------|---------|----------|
| `"greedy"` | Fast | Good | Default for most cases |
| `"optimal"` | Slow | Best | Small expressions (<10 indices) |
| `"dp"` | Medium | Good | Balanced for larger expressions |
| `"auto"` | Adaptive | Best tradeoff | Recommended |

---

## 8. torch.backends.mps

Apple Metal Performance Shaders backend for M1/M2/M3/M4 chips.

```python
torch.backends.mps.is_available()  # True on Apple Silicon with macOS 12.3+
torch.backends.mps.is_built()      # True if PyTorch was compiled with MPS
```

MPS provides GPU acceleration on Apple hardware. While not as fast as CUDA for large models, it enables GPU training on Apple laptops and desktops.

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = torch.randn(1000, 1000, device=device)
```

---

## 9. TF32 Precision

### What is TF32?

TF32 (TensorFloat-32) is a **19-bit floating point format** introduced with NVIDIA Ampere (A100):

```
Format comparison:
FP32:  1 sign + 8 exponent + 23 mantissa = 32 bits
TF32:  1 sign + 8 exponent + 10 mantissa = 19 bits
FP16:  1 sign + 5 exponent + 10 mantissa = 16 bits
BF16:  1 sign + 8 exponent +  7 mantissa = 16 bits
```

TF32 has the **range of FP32** (8-bit exponent) with the **precision of FP16** (10-bit mantissa). It's used internally by tensor cores—inputs are read as FP32, rounded to TF32 for computation, and results are accumulated in FP32.

### How TF32 Affects Operations

| Operation | Setting | Speedup (A100) | Precision Loss |
|-----------|---------|---------------|----------------|
| Conv2d | `cudnn.allow_tf32=True` | ~2x | ~0.1% relative |
| matmul | `cuda.matmul.allow_tf32=True` | ~2–3x | ~0.1% relative |
| Linear | Via matmul setting | ~2–3x | ~0.1% relative |

### Enable/Disable TF32

```python
# Enable TF32 everywhere (recommended for training)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Disable TF32 for full FP32 precision (validation, debugging)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
```

### When to Disable TF32

- Numerical validation (comparing against reference implementations)
- Scientific computing requiring full FP32 precision
- Debugging convergence issues
- Unit tests checking exact numerical equality

---

## 10. torch.set_float32_matmul_precision()

A high-level API that controls matmul precision globally:

```python
torch.set_float32_matmul_precision("highest")  # No TF32, full FP32
torch.set_float32_matmul_precision("high")     # TF32 on Ampere+
torch.set_float32_matmul_precision("medium")   # TF32 + reduced precision reductions
```

| Level | Matmul Precision | Speed | Use Case |
|-------|-----------------|-------|----------|
| `"highest"` | Full FP32 | Baseline | Debugging, validation |
| `"high"` | TF32 on Ampere+ | ~2–3x faster | Standard training |
| `"medium"` | TF32 + BF16 reductions | Fastest | Large-batch training |

```python
# Recommended: set once at the top of your training script
torch.set_float32_matmul_precision("high")
```

This is what `torch.compile` hints at when it logs: *"TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled."*

---

## 11. torch.backends.flags() Context Manager

Temporarily override backend settings within a scope:

```python
with torch.backends.flags(
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    cudnn_enabled=True,
    allow_tf32=True
):
    # These settings active only inside this block
    output = model(input)

# Original settings restored here
```

**Use cases**:
- Temporarily disabling TF32 for a validation step
- Enabling benchmark mode for a specific layer
- Writing tests that need specific backend states

---

## 12. Performance Checklist

### Training Settings (Maximum Speed)

```python
# Set at top of training script
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

### Inference Settings (Maximum Throughput)

```python
# Set before inference
torch.backends.cudnn.benchmark = True       # if input sizes are fixed
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.set_num_threads(num_physical_cores)
```

### Debugging/Reproducibility Settings

```python
# Set for exact reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
```

### Quick Reference Table

| Setting | Training | Inference | Debug |
|---------|:--------:|:---------:|:-----:|
| `cudnn.benchmark` | ✅ (fixed sizes) | ✅ (fixed sizes) | ❌ |
| `cudnn.deterministic` | ❌ | ❌ | ✅ |
| `cudnn.allow_tf32` | ✅ | ✅ | ❌ |
| `cuda.matmul.allow_tf32` | ✅ | ✅ | ❌ |
| `float32_matmul_precision` | "high" | "high" | "highest" |
| `cudnn.enabled` | ✅ | ✅ | ✅ |
| `use_deterministic_algorithms` | ❌ | ❌ | ✅ |

---

## 13. Upstream Updates (June 10–11, 2026)

Recent PyTorch commits that affect backend behavior and performance:

### cuDNN SDPA d=256 Support (#185553)

cuDNN's SDPA backend now supports head dimension d=256, enabling Flash Attention-like performance for larger attention heads without falling back to the math kernel. This benefits models using large head dimensions (e.g., certain MoE architectures).

### ARM scatter/gather Optimization (#156161)

Scatter and gather operations now use optimized ARM NEON intrinsics on aarch64, improving CPU performance on ARM servers (AWS Graviton, Apple M-series) by 2–4x for these ops.

### Dynamo Polyfills for itertools (#186240)

`torch.compile` now handles `itertools.chain`, `itertools.islice`, and other itertools functions as graph-safe polyfills, reducing graph breaks in models that use standard library iteration patterns.

### DTensor Single-Dim Strategies Migration (#186667)

Internal migration of DTensor sharding strategies to a single-dimension representation, improving compilation speed and reducing memory for distributed models using DeviceMesh.

### AOTI torch.cond/while_loop Support (#184736)

AOTInductor (the ahead-of-time compiler) now supports `torch.cond` and `torch.while_loop` control flow operations, enabling export of models with conditional branches and loops.

---

## Summary

```
torch.backends — what to remember:

1. cudnn.benchmark = True     → auto-tune conv algorithms (fixed sizes)
2. allow_tf32 = True          → 2–3x matmul/conv speedup on Ampere+
3. set_float32_matmul_precision("high") → same as TF32, cleaner API
4. set_num_threads(N)         → match physical cores for CPU
5. opt_einsum.strategy        → optimize multi-tensor contractions
6. deterministic = True       → reproducibility at the cost of speed
7. backends.flags()           → scope backend settings temporarily
```

---

## Further Reading

- [PyTorch backends documentation](https://pytorch.org/docs/stable/backends.html)
- [NVIDIA TF32 documentation](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)
- [cuDNN developer guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [Intel oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html)

---

<div align="center">

[← Previous Module](../19_torch_function_dispatch/) | [🏠 Home](../README.md) | [Next Module →](../21_cuda_graphs/)

**Notebook**: [`20_backends_tuning.ipynb`](../notebooks/20_backends_tuning.ipynb)

</div>
