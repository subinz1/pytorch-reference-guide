"""
Module 36: C++ Extension Basics — Building Custom PyTorch Ops in C++
====================================================================

This script demonstrates PyTorch's C++ extension infrastructure.
Since we can't compile C++ in all environments, we:
  1. Print the C++ source code that would be written
  2. Show the Python-side API calls
  3. Implement equivalent pure-Python ops for correctness comparison
  4. Walk through the compilation process step by step

Run: python cpp_extension_basics.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# Section 1: Check if cpp_extension is available
# ============================================================================
section("1. C++ Extension Infrastructure")

from torch.utils.cpp_extension import (  # noqa: E402
    CppExtension,
    CUDAExtension,
    BuildExtension,
)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:    {torch.version.cuda}")

try:
    from torch.utils.cpp_extension import _get_build_directory
    build_dir = _get_build_directory("test_ext", verbose=False)
    print(f"Build cache dir: {build_dir}")
except Exception:
    print("Build cache dir: (not available in this environment)")

# Check for compiler
import shutil
compilers = ["g++", "gcc", "c++", "clang++"]
found = {c: shutil.which(c) for c in compilers if shutil.which(c)}
print(f"Available C++ compilers: {found if found else 'None found'}")

print(f"\nCppExtension class:  {CppExtension.__module__}.{CppExtension.__name__}")
print(f"CUDAExtension class: {CUDAExtension.__module__}.{CUDAExtension.__name__}")
print(f"BuildExtension class: {BuildExtension.__module__}.{BuildExtension.__name__}")


# ============================================================================
# Section 2: Anatomy of a C++ Extension — Source Code
# ============================================================================
section("2. Anatomy of a C++ Extension (Source Code)")

MY_ADD_CPP = r"""
// my_add.cpp — Minimal C++ extension
#include <torch/extension.h>

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(),
        "Size mismatch: ", a.sizes(), " vs ", b.sizes());
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_add, "Element-wise addition");
}
"""

print("=== my_add.cpp ===")
print(MY_ADD_CPP)

print("Key components:")
print("  #include <torch/extension.h>    — Single header: PyTorch C++ API + pybind11")
print("  torch::Tensor                   — C++ tensor type (same as Python torch.Tensor)")
print("  TORCH_CHECK(cond, msg)          — Throws RuntimeError on failure")
print("  PYBIND11_MODULE(NAME, m)        — Creates Python module bindings")
print("  TORCH_EXTENSION_NAME            — Auto-set to name passed to load()/CppExtension()")


# ============================================================================
# Section 3: JIT Compilation with load()
# ============================================================================
section("3. JIT Compilation with load()")

LOAD_EXAMPLE = """
from torch.utils.cpp_extension import load

# Basic usage — compiles on first call, caches the .so
module = load(
    name="my_add",              # Module name
    sources=["my_add.cpp"],     # Source files
    verbose=True,               # Print compile commands
)

result = module.my_add(torch.ones(3), torch.ones(3))
# tensor([2., 2., 2.])
"""
print("=== Python-side load() call ===")
print(LOAD_EXAMPLE)

LOAD_ADVANCED = """
# Advanced load() with extra flags
module = load(
    name="fast_ops",
    sources=["ops.cpp", "kernels.cu"],
    extra_include_paths=["/usr/local/include/mylib"],
    extra_cflags=["-O3", "-march=native", "-fopenmp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=["-lgomp"],
    verbose=True,
    with_cuda=True,
)
"""
print("=== Advanced load() with optimization flags ===")
print(LOAD_ADVANCED)

LOAD_INLINE_EXAMPLE = """
# load_inline() — no separate files needed
from torch.utils.cpp_extension import load_inline

cpp_source = '''
torch::Tensor double_tensor(torch::Tensor x) {
    return x * 2;
}
'''

module = load_inline(
    name="inline_ext",
    cpp_sources=cpp_source,
    functions=["double_tensor"],
    verbose=True,
)
"""
print("=== load_inline() for quick experiments ===")
print(LOAD_INLINE_EXAMPLE)


# ============================================================================
# Section 4: Compilation Process Step by Step
# ============================================================================
section("4. Compilation Process (Step by Step)")

print("""
When you call load(name="my_ext", sources=["my_ext.cpp"]):

Step 1: Check cache
  └─ Look in ~/.cache/torch_extensions/py{VERSION}/my_ext/
  └─ If .so exists and sources haven't changed → load and return

Step 2: Determine compiler and flags
  └─ C++ compiler: g++ (or from CXX env var)
  └─ CUDA compiler: nvcc (if .cu files present)
  └─ Flags: -O3, -shared, -fPIC, -std=c++17
  └─ Include paths: Python.h, torch headers, pybind11 headers

Step 3: Compile each source file → object files
  └─ g++ -c my_ext.cpp -o my_ext.o [flags] [includes]
  └─ nvcc -c kernel.cu -o kernel.o [cuda_flags] (if CUDA)

Step 4: Link object files → shared library
  └─ g++ -shared my_ext.o kernel.o -o my_ext.so [ldflags]

Step 5: Load the shared library
  └─ importlib.import_module("my_ext")
  └─ Return the module object to Python
""")

print("PyTorch include paths that get added automatically:")
from torch.utils.cpp_extension import include_paths
for p in include_paths():
    print(f"  {p}")


# ============================================================================
# Section 5: setup.py for Distribution
# ============================================================================
section("5. setup.py for Distribution")

SETUP_CPU = """
# setup.py — CPU-only extension
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="my_cpu_ext",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="my_cpu_ext",
            sources=["my_ext.cpp"],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
print("=== setup.py (CPU-only) ===")
print(SETUP_CPU)

SETUP_CUDA = """
# setup.py — CUDA extension
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_cuda_ext",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_ext",
            sources=[
                "csrc/ops.cpp",           # Host code (gcc/g++)
                "csrc/kernels.cu",        # Device code (nvcc)
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math",
                         "-gencode=arch=compute_80,code=sm_80"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
print("=== setup.py (CUDA) ===")
print(SETUP_CUDA)

print("Install commands:")
print("  pip install .              # Install from source")
print("  pip install -e .           # Editable install (for development)")
print("  python setup.py bdist_wheel  # Build distributable wheel")


# ============================================================================
# Section 6: Fused Linear+ReLU — Complete C++ Source
# ============================================================================
section("6. Complete Example: Fused Linear+ReLU C++ Extension")

FUSED_LINEAR_RELU_CPP = r"""
// fused_linear_relu.cpp — Fused Linear + ReLU with autograd support
#include <torch/extension.h>

using namespace torch::autograd;

class FusedLinearReLUFunction : public Function<FusedLinearReLUFunction> {
public:
    static torch::Tensor forward(
        AutogradContext* ctx,
        torch::Tensor input,    // [batch, in_features]
        torch::Tensor weight,   // [out_features, in_features]
        torch::Tensor bias      // [out_features]
    ) {
        TORCH_CHECK(input.dim() == 2, "Input must be 2D");
        TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
        TORCH_CHECK(input.size(1) == weight.size(1),
            "Input features (", input.size(1),
            ") must match weight columns (", weight.size(1), ")");

        // Fused: output = relu(input @ weight.T + bias)
        auto linear_out = torch::mm(input, weight.t());
        if (bias.defined()) {
            linear_out = linear_out + bias.unsqueeze(0);
        }
        auto output = torch::relu(linear_out);

        ctx->save_for_backward({input, weight, output});
        return output;
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

        // ReLU backward: mask out where relu_output == 0
        auto grad_relu = grad_output * (relu_output > 0).to(grad_output.dtype());

        // Linear backward
        auto grad_input = torch::mm(grad_relu, weight);
        auto grad_weight = torch::mm(grad_relu.t(), input);
        auto grad_bias = grad_relu.sum(0);

        return {grad_input, grad_weight, grad_bias};
    }
};

torch::Tensor fused_linear_relu(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias
) {
    return FusedLinearReLUFunction::apply(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_relu", &fused_linear_relu,
          "Fused Linear + ReLU with autograd");
}
"""

print("=== fused_linear_relu.cpp ===")
print(FUSED_LINEAR_RELU_CPP)


# ============================================================================
# Section 7: Pure-Python Equivalent for Correctness Comparison
# ============================================================================
section("7. Pure-Python Equivalent (Correctness Reference)")


def fused_linear_relu_python(input: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor) -> torch.Tensor:
    """Pure Python equivalent of the C++ fused linear+relu."""
    return F.relu(input @ weight.t() + bias)


class FusedLinearReLU(torch.autograd.Function):
    """Python autograd Function equivalent of the C++ version."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        output = F.relu(input @ weight.t() + bias)
        ctx.save_for_backward(input, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, relu_output = ctx.saved_tensors
        grad_relu = grad_output * (relu_output > 0).float()
        grad_input = grad_relu @ weight
        grad_weight = grad_relu.t() @ input
        grad_bias = grad_relu.sum(0)
        return grad_input, grad_weight, grad_bias


# Test the Python version
batch, in_f, out_f = 4, 8, 16
x = torch.randn(batch, in_f, requires_grad=True)
w = torch.randn(out_f, in_f, requires_grad=True)
b = torch.randn(out_f, requires_grad=True)

# Forward pass
out_simple = fused_linear_relu_python(x, w, b)
out_autograd = FusedLinearReLU.apply(x, w, b)

print(f"Input shape:  {x.shape}")
print(f"Weight shape: {w.shape}")
print(f"Bias shape:   {b.shape}")
print(f"Output shape: {out_simple.shape}")
print(f"Outputs match: {torch.allclose(out_simple, out_autograd)}")

# Backward pass
loss = out_autograd.sum()
loss.backward()
print(f"x.grad shape: {x.grad.shape}")
print(f"w.grad shape: {w.grad.shape}")
print(f"b.grad shape: {b.grad.shape}")
print(f"Gradients computed successfully: True")


# ============================================================================
# Section 8: Tensor Data Access Patterns (C++ code reference)
# ============================================================================
section("8. Accessing Tensor Data in C++ (Patterns)")

DATA_ACCESS_CPP = r"""
// Pattern 1: Raw pointer (fastest, unsafe)
auto t = tensor.contiguous();   // MUST ensure contiguity first
float* data = t.data_ptr<float>();
for (int i = 0; i < t.numel(); i++) {
    data[i] *= 2.0f;
}

// Pattern 2: Accessor (safe, handles strides)
auto acc = tensor.accessor<float, 2>();  // 2D float tensor
for (int i = 0; i < acc.size(0); i++) {
    for (int j = 0; j < acc.size(1); j++) {
        acc[i][j] += 1.0f;
    }
}

// Pattern 3: packed_accessor for CUDA kernels
auto pacc = tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
// Pass pacc to kernel — uses 32-bit indexing

// Pattern 4: AT_DISPATCH for dtype-generic code
AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "my_op", [&] {
    // scalar_t is float or double
    auto data = tensor.data_ptr<scalar_t>();
    for (int64_t i = 0; i < tensor.numel(); i++) {
        data[i] = std::sqrt(data[i]);
    }
});

// Shape and metadata queries
auto sizes = tensor.sizes();          // Shape (IntArrayRef)
auto strides = tensor.strides();      // Strides
int64_t dim = tensor.dim();           // Number of dimensions
int64_t n = tensor.numel();           // Total elements
bool contig = tensor.is_contiguous(); // Contiguity check
auto device = tensor.device();        // cpu, cuda:0, ...
auto dtype = tensor.scalar_type();    // Float, Double, Half, ...
"""

print(DATA_ACCESS_CPP)

# Python equivalents
t = torch.randn(3, 4)
print("Python equivalents of C++ tensor metadata:")
print(f"  tensor.shape:          {t.shape}")
print(f"  tensor.stride():       {t.stride()}")
print(f"  tensor.dim():          {t.dim()}")
print(f"  tensor.numel():        {t.numel()}")
print(f"  tensor.is_contiguous():{t.is_contiguous()}")
print(f"  tensor.device:         {t.device}")
print(f"  tensor.dtype:          {t.dtype}")
print(f"  tensor.data_ptr():     {t.data_ptr()} (memory address)")


# ============================================================================
# Section 9: C++ Autograd Function Template
# ============================================================================
section("9. C++ Autograd Function Template")

AUTOGRAD_TEMPLATE = r"""
#include <torch/extension.h>

using namespace torch::autograd;

class MyCustomFunction : public Function<MyCustomFunction> {
public:
    static torch::Tensor forward(
        AutogradContext* ctx,
        torch::Tensor input,
        double scale_factor
    ) {
        // Save non-tensor data
        ctx->saved_data["scale"] = scale_factor;
        // Save tensors
        ctx->save_for_backward({input});

        return input * scale_factor;
    }

    static tensor_list backward(
        AutogradContext* ctx,
        tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        double scale = ctx->saved_data["scale"].toDouble();
        auto grad_output = grad_outputs[0];

        // grad_input = grad_output * d(input * scale)/d(input) = grad_output * scale
        auto grad_input = grad_output * scale;

        // Return one gradient per forward() input
        // scale_factor is not a tensor, return empty
        return {grad_input, torch::Tensor()};
    }
};
"""

print(AUTOGRAD_TEMPLATE)

print("Key points about C++ autograd functions:")
print("  - Inherit from torch::autograd::Function<YourClass>")
print("  - forward() receives AutogradContext* for saving state")
print("  - backward() returns one gradient per forward() input")
print("  - ctx->save_for_backward() for tensors")
print("  - ctx->saved_data[] for non-tensor values (scalars, strings)")
print("  - Return torch::Tensor() (undefined) for non-differentiable inputs")


# ============================================================================
# Section 10: Error Handling Patterns
# ============================================================================
section("10. Error Handling in C++ Extensions")

ERROR_HANDLING_CPP = r"""
// TORCH_CHECK — user-facing errors (becomes Python RuntimeError)
TORCH_CHECK(input.dim() == 2,
    "Expected 2D input, got ", input.dim(), "D");

TORCH_CHECK(input.device().is_cuda(),
    "Input must be on CUDA, got ", input.device());

TORCH_CHECK(input.scalar_type() == torch::kFloat32,
    "Expected float32, got ", input.scalar_type());

TORCH_CHECK(weight.size(1) == input.size(1),
    "Weight cols (", weight.size(1),
    ") must match input cols (", input.size(1), ")");

// TORCH_INTERNAL_ASSERT — invariants (bugs in your code)
TORCH_INTERNAL_ASSERT(output.numel() == input.numel(),
    "Output/input size mismatch — this is a bug");

// CUDA error checking
cudaError_t err = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice);
TORCH_CHECK(err == cudaSuccess,
    "CUDA error: ", cudaGetErrorString(err));
"""

print(ERROR_HANDLING_CPP)

# Demonstrate the Python-side behavior
print("Python-side: TORCH_CHECK becomes RuntimeError")
try:
    t = torch.randn(3)
    assert t.dim() == 2, f"Expected 2D input, got {t.dim()}D"
except AssertionError as e:
    print(f"  Caught: {e}")


# ============================================================================
# Section 11: Directory Structure for Distributable Extension
# ============================================================================
section("11. Distributable Extension Structure")

print("""
my_pytorch_ext/
├── setup.py                  # Build configuration
├── README.md
├── my_ext/
│   ├── __init__.py           # Loads compiled module, exposes Python API
│   ├── csrc/
│   │   ├── ops.cpp           # C++ bindings and dispatch
│   │   ├── cpu_kernels.cpp   # CPU implementations
│   │   └── cuda_kernels.cu   # CUDA implementations (optional)
│   └── functional.py         # Python wrappers for the C++ ops
├── tests/
│   ├── test_correctness.py   # Compare with PyTorch reference
│   └── test_gradients.py     # Gradient checking
└── benchmarks/
    └── bench_ops.py           # Performance comparison
""")

INIT_PY = """
# my_ext/__init__.py
import torch

# Option 1: Pre-built (installed via pip install)
try:
    from my_ext._C import my_op, my_other_op
except ImportError:
    # Option 2: JIT fallback (for development)
    import os
    from torch.utils.cpp_extension import load
    _dir = os.path.dirname(os.path.abspath(__file__))
    _C = load(
        name="my_ext_C",
        sources=[
            os.path.join(_dir, "csrc", "ops.cpp"),
            os.path.join(_dir, "csrc", "cpu_kernels.cpp"),
        ],
    )
    my_op = _C.my_op
    my_other_op = _C.my_other_op
"""
print("=== my_ext/__init__.py ===")
print(INIT_PY)


# ============================================================================
# Section 12: Verifying Correctness
# ============================================================================
section("12. Verifying Correctness (Testing Strategy)")

print("When you have a working C++ extension, verify like this:\n")

VERIFY_CODE = """
import torch
from torch.utils.cpp_extension import load

# Load the C++ extension
ext = load(name="my_ops", sources=["my_ops.cpp"])

# Reference implementation in pure PyTorch
def reference_fused_add_relu(a, b):
    return torch.relu(a + b)

# Test with random inputs
for trial in range(100):
    a = torch.randn(64, 128)
    b = torch.randn(64, 128)

    result_cpp = ext.fused_add_relu(a, b)
    result_ref = reference_fused_add_relu(a, b)

    assert torch.allclose(result_cpp, result_ref, atol=1e-6), (
        f"Mismatch on trial {trial}"
    )

# Gradient check
from torch.autograd import gradcheck

a = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
b = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
assert gradcheck(ext.fused_add_relu, (a, b), eps=1e-6, atol=1e-4)

print("All correctness checks passed!")
"""
print(VERIFY_CODE)

# Demonstrate with our Python version
print("Running correctness check with Python equivalent...")
for trial in range(100):
    a = torch.randn(64, 128)
    b = torch.randn(64, 128)
    result = F.relu(a + b)
    reference = torch.relu(a + b)
    assert torch.allclose(result, reference)

print("  100 random trials passed!")

# Gradient check with our Python autograd function
a = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
w = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
b = torch.randn(16, dtype=torch.float64, requires_grad=True)
passed = torch.autograd.gradcheck(FusedLinearReLU.apply, (a, w, b), eps=1e-6, atol=1e-4)
print(f"  Gradient check passed: {passed}")


# ============================================================================
# Section 13: Dispatcher Registration (Modern Approach)
# ============================================================================
section("13. Dispatcher Registration (torch.library)")

print("Modern approach: register ops with the dispatcher for torch.compile support\n")

DISPATCHER_EXAMPLE = """
import torch

# Step 1: Define the op schema
torch.library.define(
    "myops::fused_linear_relu",
    "(Tensor input, Tensor weight, Tensor bias) -> Tensor"
)

# Step 2: Register CPU implementation
@torch.library.impl("myops::fused_linear_relu", "cpu")
def fused_linear_relu_cpu(input, weight, bias):
    return torch.relu(input @ weight.t() + bias)

# Step 3: Register fake (meta) implementation for torch.compile
@torch.library.register_fake("myops::fused_linear_relu")
def fused_linear_relu_fake(input, weight, bias):
    return input.new_empty(input.shape[0], weight.shape[0])

# Step 4: (Optional) Register autograd formula
def fused_linear_relu_backward(ctx, grad):
    input, weight, output = ctx.saved_tensors
    grad_relu = grad * (output > 0).float()
    return grad_relu @ weight, grad_relu.t() @ input, grad_relu.sum(0)

torch.library.impl_autograd(
    "myops::fused_linear_relu", fused_linear_relu_cpu,
    setup_context=lambda ctx, inputs, output: ctx.save_for_backward(
        inputs[0], inputs[1], output
    ),
    backward=fused_linear_relu_backward,
)

# Usage — works with torch.compile!
result = torch.ops.myops.fused_linear_relu(x, w, b)
"""
print(DISPATCHER_EXAMPLE)


# ============================================================================
# Section 14: Summary
# ============================================================================
section("14. Summary — When to Use What")

print("""
┌─────────────────────────┬──────────────────────────────────────────────┐
│ Approach                │ When to Use                                  │
├─────────────────────────┼──────────────────────────────────────────────┤
│ Pure Python + PyTorch   │ Default choice. PyTorch ops are already fast │
│ torch.compile           │ Fuse Python-level ops automatically          │
│ Triton kernel           │ Custom GPU kernels in Python                  │
│ C++ Extension (CPU)     │ Custom CPU ops, wrapping C libraries         │
│ CUDA Extension          │ Custom GPU ops needing hardware intrinsics   │
│ torch.library + custom  │ Full dispatcher integration (compile-ready)  │
└─────────────────────────┴──────────────────────────────────────────────┘

Decision flow:
  1. Can you do it with existing PyTorch ops?  → Yes → stop
  2. Can torch.compile fuse it for you?        → Yes → stop
  3. Is it a GPU kernel?
     → Yes → Try Triton first (simpler), CUDA C++ if you need low-level control
  4. Is it a CPU-intensive op?
     → Yes → C++ extension with OpenMP
  5. Do you need torch.compile compatibility?
     → Yes → Register via torch.library
""")

print("Files demonstrated:")
print("  - my_add.cpp                    (minimal C++ extension)")
print("  - fused_linear_relu.cpp         (C++ autograd function)")
print("  - setup.py                      (CPU and CUDA variants)")
print("  - Pure-Python equivalents       (for correctness comparison)")
print("  - Dispatcher registration       (torch.library approach)")

print("\nDone! See cuda_extension_guide.py for CUDA-specific patterns.")
