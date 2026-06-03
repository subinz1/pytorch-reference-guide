# Module 13: Advanced PyTorch Features

This module covers advanced PyTorch capabilities that go beyond standard model
building and training. These are the tools that separate "I can train a model"
from "I can build production-quality ML systems."

---

## Functorch (torch.func): Functional Transformations

`torch.func` (formerly the standalone `functorch` library) provides composable
function transforms inspired by JAX. The core idea: transform a plain Python
function into a new function with different behavior.

### vmap — Vectorized Map

`vmap` automatically vectorizes a function over a batch dimension. Instead of
writing explicit batch loops or reshaping tensors, you write the function for
a single example and `vmap` handles batching:

```python
import torch
from torch.func import vmap

def compute_norm(x):
    """Compute L2 norm of a single vector."""
    return torch.sqrt(torch.sum(x ** 2))

# Without vmap: need to handle batch dimension explicitly
batch = torch.randn(32, 10)
norms_loop = torch.stack([compute_norm(batch[i]) for i in range(32)])

# With vmap: automatic batching
norms_vmap = vmap(compute_norm)(batch)
```

Why use `vmap` instead of just writing batched code? Three reasons:
1. **Clarity**: Write single-example logic, get batched execution
2. **Correctness**: No batch dimension bugs
3. **Composition**: Combine with `grad`, `jacrev`, etc.

### grad — Functional Gradient

`torch.func.grad` computes gradients functionally, without modifying tensors
in-place or using `.backward()`:

```python
from torch.func import grad

def f(x):
    return torch.sin(x).sum()

# grad returns a function that computes the gradient
grad_f = grad(f)
x = torch.tensor([1.0, 2.0, 3.0])
print(grad_f(x))  # cos(x) = [0.5403, -0.4161, -0.9900]
```

`grad` is particularly useful when composed with other transforms.

### jacrev / jacfwd — Jacobian Computation

The Jacobian matrix contains all partial derivatives of a vector-valued function:

```python
from torch.func import jacrev, jacfwd

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])

x = torch.tensor([1.0, 2.0])
J_rev = jacrev(f)(x)   # Reverse-mode: efficient when output dim < input dim
J_fwd = jacfwd(f)(x)   # Forward-mode: efficient when input dim < output dim
```

Rule of thumb:
- Use `jacrev` when output dimension is smaller than input dimension
- Use `jacfwd` when input dimension is smaller than output dimension

### hessian — Second Derivatives

The Hessian matrix is the Jacobian of the gradient. `torch.func.hessian` is
syntactic sugar for `jacrev(jacrev(f))` or `jacfwd(jacrev(f))`:

```python
from torch.func import hessian

def f(x):
    return (x ** 3).sum()

H = hessian(f)(torch.tensor([1.0, 2.0, 3.0]))
# H[i,j] = d^2f / dx_i dx_j
```

### Composing Transforms

The real power is in composition:

```python
from torch.func import vmap, grad, jacrev

# Per-sample gradients: grad of loss for each sample in a batch
def loss_fn(params, x, y):
    pred = model_fn(params, x)
    return ((pred - y) ** 2).sum()

# Batched Jacobian: Jacobian for each sample in a batch
batched_jacobian = vmap(jacrev(model_fn), in_dims=(None, 0))
```

See `functorch_transforms.py` for complete examples.

---

## Per-Sample Gradients

The classic `vmap + grad` use case. Standard training computes the *average*
gradient across a batch. But sometimes you need the gradient for *each sample
individually* — for example, in differential privacy (DP-SGD), where you need
to clip per-sample gradients before averaging.

Without `vmap`, you'd need to loop over samples or use inefficient tricks.
With `vmap`:

```python
from torch.func import vmap, grad
from torch import nn

model = nn.Linear(10, 1)
params = dict(model.named_parameters())

def compute_loss(params, x, y):
    # Stateless function: takes params explicitly
    pred = torch.func.functional_call(model, params, (x,))
    return ((pred - y) ** 2).squeeze()

# grad w.r.t. params for a single sample
grad_fn = grad(compute_loss)

# vmap over the batch dimension of x and y
per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0))(params, X_batch, Y_batch)
```

See `per_sample_gradients.py` for a complete walkthrough.

---

## Sparse Tensors

Sparse tensors store only nonzero elements, saving memory and computation for
data that is mostly zeros (e.g., adjacency matrices, bag-of-words features).

### COO (Coordinate) Format

Stores row and column indices alongside values. Good for construction and
conversion, less efficient for arithmetic:

```python
indices = torch.tensor([[0, 1, 2], [1, 0, 2]])  # (2, nnz)
values = torch.tensor([3.0, 4.0, 5.0])
sparse_coo = torch.sparse_coo_tensor(indices, values, size=(3, 3))
```

### CSR (Compressed Sparse Row) Format

Stores row pointers, column indices, and values. Efficient for row-slicing
and matrix-vector products:

```python
crow_indices = torch.tensor([0, 1, 2, 3])  # row pointers
col_indices = torch.tensor([1, 0, 2])       # column indices
values = torch.tensor([3.0, 4.0, 5.0])
sparse_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))
```

### BSR (Block Sparse Row) Format

Like CSR but stores dense blocks instead of individual elements. Useful when
sparsity has block structure (common in structured pruning):

```python
crow_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([0, 1])
values = torch.randn(2, 2, 2)  # two 2x2 blocks
sparse_bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(4, 4))
```

When to use each:
- **COO**: Building sparse tensors, format conversion, unstructured updates
- **CSR**: Sparse matrix-vector products, row-based access patterns
- **BSR**: Block-structured sparsity, GPU-friendly operations

### Sparse Operations

```python
# Matrix multiply (sparse @ dense)
result = torch.sparse.mm(sparse_csr, dense_matrix)

# Element-wise operations
sparse_sum = sparse_coo + sparse_coo
sparse_scaled = sparse_coo * 2.0

# Convert between formats
dense = sparse_coo.to_dense()
csr = sparse_coo.to_sparse_csr()
```

---

## Complex Numbers

PyTorch natively supports complex tensors, essential for signal processing,
quantum computing simulations, and Fourier analysis.

```python
# Creating complex tensors
z = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
z = torch.tensor([1+3j, 2+4j])  # Python complex literals

# Operations
z.real      # real part
z.imag      # imaginary part
z.abs()     # magnitude
z.angle()   # phase angle
z.conj()    # complex conjugate

# FFT
signal = torch.randn(1000)
spectrum = torch.fft.fft(signal)
freqs = torch.fft.fftfreq(1000)

# Inverse FFT
reconstructed = torch.fft.ifft(spectrum).real
```

See `sparse_and_complex.py` for complete examples.

---

## Quantization Overview

Quantization reduces model size and increases inference speed by using lower
precision (e.g., INT8 instead of FP32).

### Why Quantize?

- **Memory**: INT8 uses 4x less memory than FP32
- **Speed**: Integer arithmetic is faster, especially on mobile/edge devices
- **Accuracy**: Modern quantization preserves most of the model's accuracy

### Quantization Approaches

**Post-Training Quantization (PTQ)**: Quantize a pre-trained model without
retraining. Fast but may lose more accuracy.

**Quantization-Aware Training (QAT)**: Simulate quantization during training
so the model learns to be robust to quantization noise. Better accuracy but
requires retraining.

**PT2E Quantization Flow**: The modern PyTorch 2 Export quantization approach.
Uses `torch.export` to capture the model as a graph, applies quantization
annotations, and lowers to optimized backends:

```python
# Conceptual PT2E flow (simplified)
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

model = MyModel()
exported = torch.export.export(model, example_inputs)
prepared = prepare_pt2e(exported, quantizer)
# Calibrate with representative data
for batch in calibration_data:
    prepared(batch)
quantized = convert_pt2e(prepared)
```

**torchao**: A library for architecture optimization, including quantization,
sparsity, and low-precision training. Offers simple APIs:

```python
# Conceptual torchao usage
import torchao
torchao.quantize_(model, torchao.quantization.int8_weight_only())
```

---

## Custom Operators (torch.library)

When PyTorch's built-in ops don't cover your needs, you can define custom
operators with proper integration into autograd, `torch.compile`, and
`torch.export`.

```python
import torch
from torch.library import Library, impl

# Create a library namespace for your custom ops
my_lib = Library("myops", "DEF")

# Define the op signature
my_lib.define("my_relu(Tensor x) -> Tensor")

# Register a CPU implementation
@impl(my_lib, "my_relu", "CPU")
def my_relu_cpu(x):
    return x.clamp(min=0)

# Register a Meta (shape-only) implementation for torch.compile
@impl(my_lib, "my_relu", "Meta")
def my_relu_meta(x):
    return torch.empty_like(x)

# Use the custom op
x = torch.randn(5)
result = torch.ops.myops.my_relu(x)
```

### Custom Autograd for Custom Ops

```python
# Register autograd formula
def my_relu_backward(ctx, grad_output):
    x, = ctx.saved_tensors
    return grad_output * (x > 0).float()

torch.library.impl_abstract("myops::my_relu", my_relu_meta)
```

See `custom_operators.py` for a complete example.

---

## C++ Extensions

For performance-critical code, you can write custom C++ (and CUDA) extensions:

```python
from torch.utils.cpp_extension import load

# JIT compilation: compiles C++ code on first use
my_extension = load(
    name="my_extension",
    sources=["my_extension.cpp"],
    verbose=True,
)
```

The C++ side uses the PyTorch C++ API (LibTorch):

```cpp
#include <torch/extension.h>

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_add, "Custom add");
}
```

---

## Profiling Deep Dive

PyTorch's profiler helps identify performance bottlenecks.

### Basic Profiling

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("model_inference"):
        output = model(input_tensor)

# Print a summary table sorted by CPU time
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### Chrome Trace

Export a trace viewable in Chrome's `chrome://tracing` or Perfetto UI:

```python
prof.export_chrome_trace("trace.json")
```

### record_function

Annotate specific code regions for fine-grained profiling:

```python
with record_function("my_attention"):
    attn_output = attention(q, k, v)

with record_function("my_ffn"):
    ffn_output = feed_forward(attn_output)
```

### TensorBoard Integration

```python
with profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dir"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()
```

See `profiling.py` for runnable examples.

---

## Memory Profiling

### Basic Memory Tracking

```python
# Note: these require CUDA, shown for reference
torch.cuda.memory_allocated()      # current memory used by tensors
torch.cuda.max_memory_allocated()  # peak memory since last reset
torch.cuda.memory_reserved()       # total memory reserved by allocator
torch.cuda.reset_peak_memory_stats()
```

### Finding Memory Leaks

Common causes:
1. **Storing tensors in a list that grows**: solution is to `.detach()` or
   store `.item()` for scalar values
2. **Not clearing gradients**: call `optimizer.zero_grad()` each step
3. **Keeping computation graph alive**: use `.detach()` or `with torch.no_grad()`

```python
# BAD: keeps entire computation graph in memory
losses = []
for batch in dataloader:
    loss = model(batch).sum()
    losses.append(loss)  # holds onto graph!

# GOOD: detach the scalar value
losses = []
for batch in dataloader:
    loss = model(batch).sum()
    losses.append(loss.item())  # just a float, no graph
```

---

## Debugging Techniques

### Anomaly Detection

Detects the operation that produced a NaN or Inf gradient:

```python
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # will print a traceback if NaN/Inf is detected
```

Warning: this is SLOW. Only use for debugging.

### Gradient Checking

Verify your autograd implementation by comparing with finite differences:

```python
from torch.autograd import gradcheck

func = MyCustomFunction.apply
input = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
assert gradcheck(func, input, eps=1e-6, atol=1e-4)
```

### Common Error Messages and Fixes

**"one of the variables needed for gradient computation has been modified
by an inplace operation"**
- Cause: in-place operation (like `x += 1`) on a tensor that requires grad
- Fix: use `x = x + 1` (out-of-place) instead

**"Trying to backward through the graph a second time"**
- Cause: calling `.backward()` twice without `retain_graph=True`
- Fix: either use `retain_graph=True` or restructure to avoid double backward

**"Expected all tensors to be on the same device"**
- Cause: mixing CPU and GPU tensors in one operation
- Fix: ensure all tensors are on the same device with `.to(device)`

**"RuntimeError: mat1 and mat2 shapes cannot be multiplied"**
- Cause: shape mismatch in linear layers
- Fix: print shapes before the operation to identify the mismatch

See `debugging_tips.py` for practical debugging examples.

---

## torch.fx: Symbolic Tracing and Graph Transformation

`torch.fx` symbolically traces a model to produce a graph IR (intermediate
representation) that you can analyze and transform.

### Basic Tracing

```python
import torch.fx

class MyModel(nn.Module):
    def forward(self, x):
        x = torch.relu(x)
        x = x + 1
        return x

model = MyModel()
traced = torch.fx.symbolic_trace(model)
print(traced.graph)  # shows the operations as a graph
```

### Writing Custom Passes

```python
def replace_relu_with_gelu(module):
    """Replace all ReLU calls with GELU."""
    traced = torch.fx.symbolic_trace(module)
    for node in traced.graph.nodes:
        if node.op == "call_function" and node.target == torch.relu:
            node.target = torch.nn.functional.gelu
    traced.graph.lint()  # validate the graph
    traced.recompile()
    return traced
```

### Use Cases

- **Quantization**: Analyze the graph to determine where to insert quant/dequant nodes
- **Fusion**: Merge compatible operations (e.g., Conv + BN)
- **Shape inference**: Propagate shapes through the graph without running data
- **Visualization**: Understand model structure programmatically

---

## Meta Device: Shape Inference Without Memory

The `meta` device lets you analyze models without allocating real memory.
Tensors on the `meta` device have shapes and dtypes but no data:

```python
# Create a model on the meta device (no memory allocated)
with torch.device("meta"):
    model = nn.Linear(1000, 1000)
    # model.weight.shape == (1000, 1000) but uses 0 bytes

# Analyze input/output shapes
x = torch.empty(32, 1000, device="meta")
out = model(x)
print(out.shape)  # torch.Size([32, 1000])
```

Use cases:
- **Model analysis**: Count parameters and compute shapes for huge models
  that don't fit in memory
- **Architecture prototyping**: Verify shapes without waiting for memory
  allocation
- **Deferred initialization**: Create model structure on meta, then
  materialize weights on the target device

```python
# Count parameters of a huge model without any memory
with torch.device("meta"):
    huge_model = nn.Sequential(
        nn.Linear(10000, 10000),
        nn.ReLU(),
        nn.Linear(10000, 10000),
    )

total_params = sum(p.numel() for p in huge_model.parameters())
memory_gb = total_params * 4 / 1e9  # FP32 = 4 bytes
print(f"Parameters: {total_params:,}, Memory: {memory_gb:.2f} GB")
```

---

## Summary

| Feature | Use Case | Key API |
|---------|----------|---------|
| vmap | Batch any function | `torch.func.vmap` |
| grad | Functional gradients | `torch.func.grad` |
| jacrev/jacfwd | Jacobian matrices | `torch.func.jacrev` |
| Per-sample grads | DP-SGD, influence functions | `vmap(grad(...))` |
| Sparse tensors | Graphs, sparse data | `torch.sparse_coo_tensor` |
| Complex numbers | FFT, signal processing | `torch.complex`, `torch.fft` |
| Custom ops | Extending PyTorch | `torch.library` |
| Profiling | Performance optimization | `torch.profiler` |
| Anomaly detection | Debugging NaN/Inf | `torch.autograd.detect_anomaly` |
| torch.fx | Graph transforms | `torch.fx.symbolic_trace` |
| Meta device | Shape analysis | `torch.device("meta")` |

## Files in This Module

- `functorch_transforms.py` — vmap, grad, jacrev, hessian demonstrations
- `per_sample_gradients.py` — Per-sample gradient computation with vmap+grad
- `custom_operators.py` — Defining custom ops with torch.library
- `profiling.py` — Profiler usage, timing, and analysis
- `sparse_and_complex.py` — Sparse tensors, complex numbers, and FFT
- `debugging_tips.py` — Anomaly detection, gradient flow checking, and common fixes
