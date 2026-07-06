# Module 36: Custom C++ Extensions

<div align="center">

[← Previous Module (The Dispatcher)](../35_dispatcher/) | [🏠 Home](../README.md) | [Next Module (torch.export Deep Dive) →](../37_export_deep_dive/)

**Notebook**: [`36_cpp_extensions.ipynb`](../notebooks/36_cpp_extensions.ipynb)

</div>

---

> **Prerequisites**: [Module 04 — Neural Networks](../04_neural_networks/), [Module 35 — The Dispatcher](../35_dispatcher/)
> **Time**: ~3 hours
> **Files**: `cpp_extension_basics.py`, `cuda_extension_guide.py`

---

## Table of Contents

1. [Why C++ Extensions?](#1-why-c-extensions)
2. [Two Ways to Build](#2-two-ways-to-build)
3. [JIT Compilation with load()](#3-jit-compilation-with-load)
4. [Writing a C++ Extension (CPU)](#4-writing-a-c-extension-cpu)
5. [Accessing Tensor Data in C++](#5-accessing-tensor-data-in-c)
6. [Writing a CUDA Extension](#6-writing-a-cuda-extension)
7. [CppExtension vs CUDAExtension in setup.py](#7-cppextension-vs-cudaextension-in-setuppy)
8. [Integrating with Autograd](#8-integrating-with-autograd)
9. [Error Handling](#9-error-handling)
10. [Performance Tips](#10-performance-tips)
11. [Packaging and Distribution](#11-packaging-and-distribution)
12. [Modern Alternative: triton_op](#12-modern-alternative-triton_op)
13. [Upstream Updates (July 1-3, 2026)](#13-upstream-updates-july-1-3-2026)

---

## 1. Why C++ Extensions?

Python is wonderful for prototyping, but sometimes it's not fast enough:

**When Python is too slow:**
- Custom operators with tight inner loops that Python's overhead kills
- Operations that need to iterate over individual tensor elements
- Complex control flow that can't be expressed as PyTorch ops

**When you need CUDA kernels:**
- Novel GPU algorithms not covered by existing PyTorch ops
- Fused operations that eliminate memory round-trips
- Hardware-specific optimizations (shared memory, warp-level primitives)

**When wrapping existing libraries:**
- Integrating C/C++ numerical libraries (BLAS variants, custom solvers)
- Using vendor-specific GPU libraries alongside PyTorch
- Porting existing research code to the PyTorch ecosystem

PyTorch makes this easy with `torch.utils.cpp_extension`, which handles:
- Compiler invocation (gcc/g++, nvcc for CUDA)
- Include path management (Python, PyTorch, pybind11 headers)
- ABI compatibility across PyTorch versions
- Caching compiled shared objects

The extension mechanism builds on **pybind11** for Python-C++ bindings and integrates with PyTorch's tensor library, autograd engine, and dispatcher.

---

## 2. Two Ways to Build

### Option A: JIT Compilation with `load()`

```python
from torch.utils.cpp_extension import load

module = load(
    name="my_extension",
    sources=["my_extension.cpp"],
)
```

| Pros | Cons |
|------|------|
| No setup.py needed | Compiles on first import (slow) |
| Great for development/iteration | Must have compiler on target machine |
| Automatic caching | Can't pip install |
| Verbose mode for debugging | Not suitable for distribution |

### Option B: Setuptools with `setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="my_extension",
    ext_modules=[CppExtension("my_extension", ["my_extension.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
```

| Pros | Cons |
|------|------|
| Standard Python packaging | Requires setup.py boilerplate |
| `pip install .` works | Must rebuild after changes |
| Can build wheels for distribution | ABI compatibility concerns |
| Integrates with conda/pip ecosystem | More complex build configuration |

**Rule of thumb**: Use `load()` during development, switch to `setup.py` for distribution.

---

## 3. JIT Compilation with `load()`

The `load()` function is the fastest way to get C++ code running from Python:

```python
from torch.utils.cpp_extension import load

module = load(
    name="my_ext",           # Name of the compiled module
    sources=["my_ext.cpp"],  # Source files
    verbose=True,            # Print compilation commands
)
```

### How it works

1. **First call**: Compiles sources → shared library (`.so` on Linux, `.pyd` on Windows)
2. **Subsequent calls**: Loads cached `.so` from `~/.cache/torch_extensions/`
3. **Recompiles** only when source files change (timestamp-based)

### Common parameters

```python
module = load(
    name="fused_ops",
    sources=["fused_ops.cpp", "fused_ops_kernel.cu"],
    extra_include_paths=["/path/to/headers"],
    extra_cflags=["-O3", "-march=native"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=["-L/path/to/lib", "-lmylib"],
    verbose=True,
    with_cuda=True,  # Auto-detected from .cu files
)
```

### load_inline() for quick experiments

For small extensions, you can skip writing files entirely:

```python
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
"""

module = load_inline(
    name="inline_ext",
    cpp_sources=cpp_source,
    functions=["my_add"],
    verbose=True,
)
```

### Cache management

```python
import torch.utils.cpp_extension as ext

# Default cache location
print(ext._get_build_directory("my_ext", verbose=False))

# Clear cache to force recompilation
import shutil
shutil.rmtree(ext._get_build_directory("my_ext", verbose=False))
```

---

## 4. Writing a C++ Extension (CPU)

### Anatomy of an extension file

```cpp
// my_add.cpp
#include <torch/extension.h>

// The operation itself — uses PyTorch's C++ tensor API
torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Size mismatch");
    return a + b;
}

// Python bindings via pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_add, "Element-wise addition");
}
```

### Key components explained

**`#include <torch/extension.h>`** — The single header that includes:
- `<torch/torch.h>` — Full PyTorch C++ API (tensors, autograd, nn)
- `<pybind11/pybind11.h>` — Python binding utilities
- Macro definitions for `TORCH_EXTENSION_NAME`, `TORCH_CHECK`, etc.

**`torch::Tensor`** — C++ equivalent of Python's `torch.Tensor`. Same underlying storage, same operations:

```cpp
torch::Tensor result = torch::zeros({3, 4});
result = result + 1;                        // Broadcasting works
auto sliced = result.index({Slice(), 0});   // Indexing
```

**`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`** — Creates the Python module. `TORCH_EXTENSION_NAME` is automatically set to the `name` argument passed to `load()` or `CppExtension()`.

**`m.def("name", &function, "docstring")`** — Registers a C++ function as a Python callable. Pybind11 automatically converts between Python and C++ types (tensors, scalars, strings, lists, etc.).

### A more realistic example: fused linear + ReLU

```cpp
#include <torch/extension.h>

torch::Tensor fused_linear_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");

    auto output = torch::mm(input, weight.t());
    if (bias.defined()) {
        output = output + bias.unsqueeze(0);
    }
    return torch::relu(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_relu", &fused_linear_relu,
          "Fused Linear + ReLU (CPU)");
}
```

From Python:

```python
from torch.utils.cpp_extension import load

ext = load(name="fused_ops", sources=["fused_linear_relu.cpp"])

# Use it like any Python function
output = ext.fused_linear_relu(x, weight, bias)
```

---

## 5. Accessing Tensor Data in C++

### Raw pointer access with `data_ptr<T>()`

The fastest but most dangerous — no bounds checking:

```cpp
float* data = tensor.data_ptr<float>();
for (int i = 0; i < tensor.numel(); i++) {
    data[i] *= 2.0f;
}
```

**Critical**: Always ensure contiguity first:

```cpp
auto t = tensor.contiguous();  // Copy if not contiguous
float* data = t.data_ptr<float>();
```

### Safe strided access with `accessor<T, N>()`

Handles non-contiguous tensors correctly:

```cpp
// For CPU tensors — includes bounds checking in debug mode
auto accessor = tensor.accessor<float, 2>();  // 2D float tensor

for (int i = 0; i < accessor.size(0); i++) {
    for (int j = 0; j < accessor.size(1); j++) {
        accessor[i][j] += 1.0f;
    }
}
```

For CUDA tensors, use `packed_accessor32<T, N>()` or `packed_accessor64<T, N>()`:

```cpp
// In CUDA kernel — uses 32-bit indexing (faster, limits to ~2B elements)
auto acc = tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
```

### Dtype dispatch with `AT_DISPATCH_FLOATING_TYPES`

Your C++ code needs to handle multiple dtypes. The `AT_DISPATCH_*` macros generate code for each supported type:

```cpp
torch::Tensor scale_tensor(torch::Tensor input, double factor) {
    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scale_tensor", [&] {
        // 'scalar_t' is the C++ type (float, double)
        auto inp_a = input.accessor<scalar_t, 1>();
        auto out_a = output.accessor<scalar_t, 1>();

        for (int64_t i = 0; i < input.size(0); i++) {
            out_a[i] = inp_a[i] * static_cast<scalar_t>(factor);
        }
    });

    return output;
}
```

Available dispatch macros:

| Macro | Types |
|-------|-------|
| `AT_DISPATCH_FLOATING_TYPES` | float, double |
| `AT_DISPATCH_FLOATING_TYPES_AND_HALF` | float, double, Half |
| `AT_DISPATCH_ALL_TYPES` | all integer + float + double |
| `AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, ...)` | all + specified extras |
| `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES` | float, double, complex |

### Shape and stride access

```cpp
auto sizes = tensor.sizes();      // IntArrayRef — shape
auto strides = tensor.strides();  // IntArrayRef — strides
int64_t dim = tensor.dim();       // Number of dimensions
int64_t n = tensor.numel();       // Total elements
bool contig = tensor.is_contiguous();
auto device = tensor.device();    // Device (cpu, cuda:0, ...)
auto dtype = tensor.scalar_type();
```

---

## 6. Writing a CUDA Extension

CUDA extensions have two parts:
1. **`.cu` file** — CUDA kernels (compiled by nvcc)
2. **`.cpp` file** — Python bindings and dispatch logic (compiled by gcc)

### Example: fused add + ReLU CUDA kernel

**fused_add_relu_kernel.cu:**

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel — runs on GPU, one thread per element
template <typename scalar_t>
__global__ void fused_add_relu_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    int64_t size
) {
    // Global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = a[idx] + b[idx];
        output[idx] = val > 0 ? val : 0;  // ReLU
    }
}

// Host function — called from C++, launches kernel
torch::Tensor fused_add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Size mismatch");

    auto output = torch::empty_like(a);
    const int64_t size = a.numel();

    // Grid/block configuration
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

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
```

**fused_add_relu.cpp:**

```cpp
#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor fused_add_relu_cuda(torch::Tensor a, torch::Tensor b);

// Dispatch based on device
torch::Tensor fused_add_relu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on same device");

    if (a.is_cuda()) {
        return fused_add_relu_cuda(a, b);
    }
    // CPU fallback
    return torch::relu(a + b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_relu", &fused_add_relu, "Fused add + ReLU");
}
```

### Key CUDA concepts

**`__global__`** — Marks a function as a CUDA kernel (called from host, runs on device).

**`<<<blocks, threads>>>`** — Kernel launch configuration:
- `blocks` = number of thread blocks in the grid
- `threads` = number of threads per block (max 1024)
- Total threads = blocks × threads

**Thread indexing:**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 1D
int row = blockIdx.y * blockDim.y + threadIdx.y;   // 2D
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

**Grid sizing** — ensure enough threads to cover all elements:
```cpp
const int threads = 256;  // Common choice (multiple of warp size 32)
const int blocks = (num_elements + threads - 1) / threads;
```

**`__restrict__`** — Tells the compiler pointers don't alias (enables optimizations).

### Building the CUDA extension

```python
# JIT
module = load(
    name="fused_ops",
    sources=["fused_add_relu.cpp", "fused_add_relu_kernel.cu"],
    verbose=True,
)

# Or in setup.py
from torch.utils.cpp_extension import CUDAExtension
ext_modules = [CUDAExtension("fused_ops", [
    "fused_add_relu.cpp",
    "fused_add_relu_kernel.cu",
])]
```

---

## 7. CppExtension vs CUDAExtension in setup.py

### CPU-only extension

```python
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="my_cpu_ext",
    ext_modules=[
        CppExtension(
            name="my_cpu_ext",
            sources=["my_ext.cpp"],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### CUDA extension

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_ext",
            sources=[
                "my_ext.cpp",            # Host code (gcc)
                "my_ext_kernel.cu",      # Device code (nvcc)
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### Key differences

| Feature | `CppExtension` | `CUDAExtension` |
|---------|----------------|-----------------|
| Compiler | gcc/g++ only | gcc + nvcc |
| Source files | `.cpp`, `.c` | `.cpp`, `.cu` |
| CUDA headers | Not included | Auto-included |
| GPU support | No | Yes |
| Requires CUDA toolkit | No | Yes |

### BuildExtension

`BuildExtension` is a custom setuptools command class that:
- Detects the C++ compiler and its capabilities
- Manages mixed compilation (C++ and CUDA)
- Handles ABI compatibility flags
- Passes the correct PyTorch include paths

Install and test:

```bash
pip install .
python -c "import my_ext; print(my_ext.my_function(torch.ones(3)))"
```

---

## 8. Integrating with Autograd

To support `backward()`, you need to write a custom autograd function in C++.

### C++ autograd function

```cpp
#include <torch/extension.h>

using namespace torch::autograd;

class FusedLinearReLUFunction : public Function<FusedLinearReLUFunction> {
public:
    static torch::Tensor forward(
        AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias
    ) {
        auto output = torch::mm(input, weight.t()) + bias;
        auto relu_output = torch::relu(output);

        // Save tensors needed for backward
        ctx->save_for_backward({input, weight, relu_output});

        return relu_output;
    }

    static tensor_list backward(
        AutogradContext* ctx,
        tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto relu_output = saved[2];

        auto grad_output = grad_outputs[0];

        // ReLU backward: zero gradient where output was zero
        auto grad_relu = grad_output * (relu_output > 0).to(grad_output.dtype());

        // Linear backward
        auto grad_input = torch::mm(grad_relu, weight);
        auto grad_weight = torch::mm(grad_relu.t(), input);
        auto grad_bias = grad_relu.sum(0);

        return {grad_input, grad_weight, grad_bias};
    }
};

torch::Tensor fused_linear_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    return FusedLinearReLUFunction::apply(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_relu", &fused_linear_relu);
}
```

### Using the custom op with autograd

```python
import torch
from torch.utils.cpp_extension import load

ext = load(name="fused_ops", sources=["fused_linear_relu.cpp"])

x = torch.randn(4, 8, requires_grad=True)
w = torch.randn(16, 8, requires_grad=True)
b = torch.randn(16, requires_grad=True)

out = ext.fused_linear_relu(x, w, b)
loss = out.sum()
loss.backward()  # Calls C++ backward()

print(x.grad.shape)  # torch.Size([4, 8])
```

### Registering with the dispatcher (modern approach)

For better integration with `torch.compile` and other transforms, register ops via `torch.library`:

```python
import torch

torch.library.define("myops::fused_linear_relu", "(Tensor x, Tensor w, Tensor b) -> Tensor")

@torch.library.impl("myops::fused_linear_relu", "cpu")
def fused_linear_relu_cpu(x, w, b):
    return torch.relu(x @ w.t() + b)

@torch.library.register_fake("myops::fused_linear_relu")
def fused_linear_relu_fake(x, w, b):
    return x.new_empty(x.shape[0], w.shape[0])
```

---

## 9. Error Handling

### TORCH_CHECK — user-facing errors

Use for input validation and expected error conditions:

```cpp
TORCH_CHECK(input.dim() == 2,
    "Expected 2D input, got ", input.dim(), "D");

TORCH_CHECK(input.device().is_cuda(),
    "Input must be a CUDA tensor, got ", input.device());

TORCH_CHECK(input.scalar_type() == torch::kFloat32,
    "Expected float32, got ", input.scalar_type());

TORCH_CHECK(weight.size(1) == input.size(1),
    "Weight columns (", weight.size(1),
    ") must match input columns (", input.size(1), ")");
```

`TORCH_CHECK` throws a `c10::Error` which becomes a Python `RuntimeError`.

### TORCH_INTERNAL_ASSERT — developer-facing errors

Use for invariants that should never be violated (bugs in your code):

```cpp
TORCH_INTERNAL_ASSERT(output.numel() == input.numel(),
    "Output size mismatch — this is a bug");
```

In release builds, `TORCH_INTERNAL_ASSERT` can be compiled away for performance.

### CUDA error checking

```cpp
#define CUDA_CHECK(call)                                   \
    do {                                                   \
        cudaError_t err = call;                            \
        TORCH_CHECK(err == cudaSuccess,                    \
            "CUDA error: ", cudaGetErrorString(err));      \
    } while (0)

// Usage
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
```

### Best practices

- Use `TORCH_CHECK` for all user-facing validation
- Include the actual values in error messages (not just "size mismatch")
- Check device, dtype, shape, and contiguity at the entry point
- Use `TORCH_INTERNAL_ASSERT` sparingly for true invariants

---

## 10. Performance Tips

### 1. Use `AT_DISPATCH_*` macros for dtype support

Don't write separate functions per dtype — the dispatch macros generate templatized code:

```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "my_kernel", [&] {
        my_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }
);
```

### 2. Avoid unnecessary copies

```cpp
// Bad — copies data
auto input_contig = input.contiguous();  // Might copy

// Better — check first
if (!input.is_contiguous()) {
    input = input.contiguous();
}

// Best for read-only access — use accessor (handles strides)
auto acc = input.accessor<float, 2>();
```

### 3. Use `torch::NoGradGuard` for inference-only code

```cpp
torch::Tensor my_inference_op(torch::Tensor input) {
    torch::NoGradGuard no_grad;  // Disables autograd tracking
    return torch::relu(input);
}
```

### 4. Ensure contiguity before raw pointer access

```cpp
auto t = tensor.contiguous();  // MUST do this first
float* ptr = t.data_ptr<float>();  // Now safe
```

### 5. OpenMP for CPU parallelism

```cpp
#include <omp.h>

AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "parallel_op", [&] {
    auto data = input.data_ptr<scalar_t>();
    #pragma omp parallel for
    for (int64_t i = 0; i < input.numel(); i++) {
        data[i] = std::exp(data[i]);
    }
});
```

Compile with OpenMP:
```python
load(name="ext", sources=["ext.cpp"], extra_cflags=["-fopenmp"],
     extra_ldflags=["-lgomp"])
```

### 6. Minimize kernel launches (CUDA)

Fuse operations into a single kernel instead of launching multiple kernels:

```cuda
// Bad: 3 kernel launches
auto temp = a + b;
auto temp2 = temp * c;
auto output = torch::relu(temp2);

// Good: 1 kernel launch (fused)
fused_add_mul_relu_kernel<<<blocks, threads>>>(a, b, c, output, size);
```

### 7. Memory coalescing (CUDA)

Adjacent threads should access adjacent memory locations:

```cuda
// Good: coalesced (threads access consecutive elements)
output[idx] = input[idx] * 2;

// Bad: strided access (threads access non-consecutive elements)
output[idx] = input[idx * stride] * 2;
```

---

## 11. Packaging and Distribution

### Building wheels

```bash
# Build a wheel
python setup.py bdist_wheel

# Install the wheel
pip install dist/my_ext-0.1-cp310-cp310-linux_x86_64.whl
```

### Conda packages

```yaml
# meta.yaml
package:
  name: my-pytorch-ext
  version: "0.1.0"

requirements:
  build:
    - python
    - setuptools
    - pytorch
  run:
    - python
    - pytorch
```

### ABI compatibility

PyTorch extensions must match the ABI of the PyTorch installation:

```python
import torch
print(torch._C._GLIBCXX_USE_CXX11_ABI)  # 0 or 1
```

`BuildExtension` handles this automatically, but pre-built wheels must match:
- **CXX11 ABI** (most pip installs): `_GLIBCXX_USE_CXX11_ABI=1`
- **Pre-CXX11 ABI** (some conda installs): `_GLIBCXX_USE_CXX11_ABI=0`

### Version compatibility

```python
setup(
    name="my_ext",
    install_requires=[
        "torch>=2.0",
    ],
    # ...
)
```

For CUDA extensions, also check CUDA version compatibility:

```python
import torch
print(torch.version.cuda)  # e.g., "12.4"
```

### Directory structure for a distributable extension

```
my_ext/
├── setup.py
├── my_ext/
│   ├── __init__.py
│   ├── _C.cpp          # C++ bindings
│   ├── _C_cuda.cu      # CUDA kernels (optional)
│   └── ops.py          # Python wrappers
├── tests/
│   └── test_ops.py
└── README.md
```

The `__init__.py` loads the compiled module:

```python
from torch.utils.cpp_extension import load
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_C = load(
    name="my_ext_C",
    sources=[os.path.join(_dir, "_C.cpp")],
)

def my_op(x, y):
    return _C.my_op(x, y)
```

---

## 12. Modern Alternative: triton_op

For many GPU kernels, **Triton** (see [Module 25](../25_triton_kernels/)) is easier than writing CUDA C++:

| Aspect | CUDA C++ Extension | Triton Kernel |
|--------|-------------------|---------------|
| Language | C++/CUDA | Python |
| Compilation | nvcc (complex setup) | JIT (automatic) |
| Autotuning | Manual | Built-in `@triton.autotune` |
| torch.compile | Needs dispatcher registration | Native support |
| Debugging | gdb, cuda-gdb | Python debugger |
| Portability | NVIDIA only | Multi-backend (experimental) |

### Registering Triton kernels as native ops

`torch._native.triton` provides `triton_op` for registering Triton kernels so they integrate with the dispatcher, autograd, and `torch.compile`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)

@torch.library.custom_op("myops::add_relu", mutates_args=())
def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_relu_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

@add_relu.register_fake
def _(x, y):
    return torch.empty_like(x)
```

### When to choose which

**Use CUDA C++ when:**
- You need shared memory, warp-level primitives, or hardware intrinsics
- The algorithm requires complex thread synchronization
- You're wrapping an existing CUDA library
- Maximum performance is critical and you need full hardware control

**Use Triton when:**
- Writing elementwise, reduction, or matmul-like kernels
- You want autotuning without manual grid search
- Rapid iteration matters more than squeezing the last 5% of performance
- You need `torch.compile` integration

---

## 13. Upstream Updates (July 1-3, 2026)

Recent PyTorch changes relevant to C++ extensions and custom ops:

### OpaqueBase → CustomClassBase rename (#188455)

The `OpaqueBase` class used for registering custom C++ classes has been renamed to `CustomClassBase` for clarity. If you have code using `torch::OpaqueBase`, update to `torch::CustomClassBase`:

```cpp
// Before
class MyClass : public torch::OpaqueBase { ... };

// After
class MyClass : public torch::CustomClassBase { ... };
```

### register_opaque_type → register_custom_class (#188456)

The function for registering custom C++ types with the dispatcher has been renamed:

```cpp
// Before
torch::register_opaque_type<MyClass>("MyClass");

// After
torch::register_custom_class<MyClass>("MyClass");
```

### triton_op schema support (#188722)

Enhanced schema validation for `triton_op` registrations. Triton ops now support richer type annotations in their schemas, improving integration with `torch.compile` and the dispatcher.

### Native instrumentation module

New `torch/_native/instrumentation.py` provides a standardized instrumentation API for native ops, enabling profiling and tracing of custom kernels registered through `torch.library`.

### Dynamo hasattr unification (#187226)

`torch.compile` (Dynamo) now handles `hasattr` checks uniformly across Python objects, improving compatibility when custom C++ extensions use `hasattr`-based feature detection in Python wrappers.

### Inductor graph partition naming (#188700)

Improved naming for graph partitions in TorchInductor, making it easier to identify which partition corresponds to which custom op when debugging compiled code that includes C++ extensions.

---

## Key Takeaways

1. **`torch.utils.cpp_extension`** provides two build paths: `load()` for development, `setup.py` for distribution
2. **`torch/extension.h`** is the single include for all PyTorch C++ functionality
3. **`AT_DISPATCH_*` macros** handle dtype-generic code without manual template instantiation
4. **CUDA extensions** split into `.cpp` (host) and `.cu` (device) files
5. **Autograd integration** works through `torch::autograd::Function` in C++ or `torch.library` in Python
6. **`TORCH_CHECK`** provides clean error messages that surface as Python exceptions
7. **Always ensure contiguity** before calling `data_ptr<T>()`
8. **Triton is often simpler** than CUDA C++ for standard GPU patterns — consider it first
9. **ABI compatibility** matters for distribution — use `BuildExtension` to handle it

Understanding the dispatcher ([Module 35](../35_dispatcher/)) is essential — C++ extensions ultimately register kernels in the same dispatch system that powers all of PyTorch.

---

### Further Resources

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html) — official tutorial
- [pybind11 Documentation](https://pybind11.readthedocs.io/) — Python-C++ binding details
- [Module 25 — Triton Kernels](../25_triton_kernels/) — the modern GPU alternative
- [Module 35 — The Dispatcher](../35_dispatcher/) — how ops integrate with PyTorch internals

---

<div align="center">

[← Previous Module (The Dispatcher)](../35_dispatcher/) | [🏠 Home](../README.md) | [Next Module (torch.export Deep Dive) →](../37_export_deep_dive/)

**Notebook**: [`36_cpp_extensions.ipynb`](../notebooks/36_cpp_extensions.ipynb)

</div>
