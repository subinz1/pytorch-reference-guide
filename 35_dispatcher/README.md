# Module 35: PyTorch Internals — The Dispatcher

<div align="center">

[← Previous Module (LLM Fine-Tuning)](../34_llm_finetuning/) | [🏠 Home](../README.md) | Next Module → (none)

**Deep Dive**: How every PyTorch operation gets routed to the right kernel

</div>

---

> **Prerequisites**: [Module 02 (Tensors)](../02_tensors/), [Module 04 (Neural Networks)](../04_neural_networks/), [Module 08 (torch.compile)](../08_torch_compile/), [Module 19 (Tensor Dispatch)](../19_torch_function_dispatch/)
>
> **Time**: ~3 hours
>
> **Files**: `dispatch_keys.py`, `custom_dispatch.py`

---

## Table of Contents

1. [What is the Dispatcher?](#1-what-is-the-dispatcher)
2. [The Journey of `torch.add(x, y)`](#2-the-journey-of-torchaddx-y)
3. [Dispatch Keys](#3-dispatch-keys)
4. [How Dispatch Keys Are Determined](#4-how-dispatch-keys-are-determined)
5. [The Priority Chain](#5-the-priority-chain)
6. [Fallthrough Keys](#6-fallthrough-keys)
7. [CompositeImplicitAutograd](#7-compositeimplicitautograd)
8. [CompositeExplicitAutograd](#8-compositeexplicitautograd)
9. [torch.library — Registering Custom Ops](#9-torchlibrary--registering-custom-ops)
10. [@custom_op — The Modern API](#10-custom_op--the-modern-api)
11. [Viewing Dispatch Tables](#11-viewing-dispatch-tables)
12. [Structured Kernels](#12-structured-kernels)
13. [How torch.compile Interacts](#13-how-torchcompile-interacts)
14. [How Autograd Uses Dispatch](#14-how-autograd-uses-dispatch)
15. [Upstream Updates (June 30 - July 1, 2026)](#15-upstream-updates-june-30---july-1-2026)

---

## 1. What is the Dispatcher?

The **dispatcher** is the central routing mechanism of PyTorch. Every single operator call — `torch.add`, `torch.mm`, `tensor.relu()` — flows through it. The dispatcher examines the input tensors, determines which "features" are active (autograd? autocast? vmap?), and routes to the correct kernel implementation.

This is what makes PyTorch extensible. New backends (XPU, MPS, custom hardware), autograd, torch.compile, vmap, autocast — all work by registering **dispatch keys** in the dispatcher. No single subsystem needs to know about the others; they all plug into the same routing table.

```
┌──────────────────────────────────────────────────────┐
│                    User Code                          │
│              z = torch.add(x, y)                     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                   Dispatcher                         │
│                                                      │
│  1. Collect dispatch keys from inputs                │
│  2. Walk priority chain (high → low)                 │
│  3. Find first key with registered kernel            │
│  4. Execute kernel (may redispatch to lower keys)    │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │Autograd │ │Autocast │ │   CPU   │
     │ kernel  │ │ kernel  │ │ kernel  │
     └─────────┘ └─────────┘ └─────────┘
```

### Why Does This Matter?

Understanding the dispatcher helps you:
- **Debug** why an operation behaves differently than expected
- **Write custom ops** that integrate cleanly with autograd, compile, etc.
- **Understand performance** — dispatch overhead, kernel selection
- **Extend PyTorch** with new backends or functional transforms

---

## 2. The Journey of `torch.add(x, y)`

Let's trace exactly what happens when you call `torch.add(x, y)` where `x` is a CUDA tensor with `requires_grad=True`:

```
Python: torch.add(x, y)
  │
  ├─ 1. Python binding → C++ at::add(x, y)
  │
  ├─ 2. Dispatcher examines tensors:
  │     x.dispatch_keyset() = {CUDA, AutogradCUDA}
  │     y.dispatch_keyset() = {CUDA, AutogradCUDA}
  │     combined = x.keys | y.keys = {CUDA, AutogradCUDA}
  │
  ├─ 3. Walk priority chain (highest first):
  │     AutogradCUDA → has kernel? YES → execute
  │
  ├─ 4. AutogradCUDA kernel:
  │     - Save x, y for backward
  │     - Create AddBackward0 node
  │     - Redispatch to CUDA key (exclude AutogradCUDA)
  │
  ├─ 5. CUDA kernel:
  │     - Launch element-wise add kernel on GPU
  │     - Return result tensor
  │
  └─ 6. Result propagates back:
       - Attach grad_fn to output
       - Return to Python
```

The key insight: **the Autograd kernel doesn't compute the addition itself**. It records the operation for backward, then *redispatches* to the actual compute backend. This separation of concerns is what makes the system composable.

---

## 3. Dispatch Keys

Each tensor carries a **dispatch key set** — a bitset where each bit represents a "feature" or "backend" that should handle operations on that tensor.

### The Full Priority Table

| Priority | Key | Purpose |
|----------|-----|---------|
| Highest | PythonTLSSnapshot | Thread-local state snapshot |
| High | PythonDispatcher | Python-level dispatch (torch.compile) |
| | FuncTorchDynamicLayerFront | Front guard for functorch |
| | Functionalize | Convert mutations to functional ops |
| | Autocast | Mixed precision dtype casting |
| | AutogradCPU | Record op for backward (CPU tensors) |
| | AutogradCUDA | Record op for backward (CUDA tensors) |
| | AutogradMPS | Record op for backward (MPS tensors) |
| | AutogradXPU | Record op for backward (XPU tensors) |
| | ADInplaceOrView | Track in-place ops and views for autograd |
| | FuncTorchBatched | vmap batching rules |
| | FuncTorchVmapMode | vmap mode (outer) |
| | BackendSelect | Route to correct backend for factory ops |
| Low | CPU | Actual computation on CPU |
| Low | CUDA | Actual computation on CUDA |
| Low | MPS | Actual computation on MPS |
| Low | XPU | Actual computation on XPU |
| Low | Meta | Shape/dtype computation (no data) |
| Lowest | CompositeImplicitAutograd | Default decompositions (autograd-aware) |
| Lowest | CompositeExplicitAutograd | Decompositions with explicit autograd |

### Viewing Keys on a Tensor

```python
import torch

x = torch.randn(3, 3)
print(torch._C._dispatch_keys(x))
# DispatchKeySet(CPU, AutogradCPU)

y = torch.randn(3, 3, device='cuda', requires_grad=True)
print(torch._C._dispatch_keys(y))
# DispatchKeySet(CUDA, AutogradCUDA)

m = torch.randn(3, 3, device='meta')
print(torch._C._dispatch_keys(m))
# DispatchKeySet(Meta, AutogradMeta)
```

---

## 4. How Dispatch Keys Are Determined

Dispatch keys come from multiple sources:

### From Tensor Properties

| Property | Key Added |
|----------|-----------|
| `device='cpu'` | CPU |
| `device='cuda'` | CUDA |
| `device='mps'` | MPS |
| `device='meta'` | Meta |
| `requires_grad=True` | AutogradCPU/CUDA/... (matches device) |
| Is a view or in-place result | ADInplaceOrView |

### From Thread-Local State

| Context | Key Added |
|---------|-----------|
| Inside `torch.autocast(...)` | Autocast |
| Inside `torch.vmap(...)` | FuncTorchBatched |
| Inside `torch._dynamo` | PythonDispatcher |
| Custom `TorchDispatchMode` active | Python |

### Key Set Computation

When an op is called with multiple tensor arguments, the dispatcher computes the **union** of all input key sets:

```python
# x has keys {CPU, AutogradCPU}
# y has keys {CPU, AutogradCPU}
# Combined: {CPU, AutogradCPU}

z = torch.add(x, y)  # Dispatcher uses combined key set
```

For factory functions (no tensor inputs), `BackendSelect` routes based on the `device` argument.

---

## 5. The Priority Chain

The dispatcher walks keys from **highest priority to lowest**. The first key with a registered kernel for that op wins.

```
┌────────────────────────────────────────┐
│ Key Set: {AutogradCUDA, Autocast, CUDA}│
└────────────────────┬───────────────────┘
                     │
     Priority walk:  │
                     ▼
         Autocast ──── has kernel? ─── YES ──→ Execute
              │                                    │
              │ (if no)                            │ redispatch
              ▼                                    ▼
         AutogradCUDA ─ has kernel? ─── YES ──→ Execute
              │                                    │
              │ (if no)                            │ redispatch
              ▼                                    ▼
            CUDA ────── has kernel? ─── YES ──→ Execute (final)
```

### Redispatch

After a higher-priority kernel does its work, it **redispatches** to the remaining keys by excluding itself:

```cpp
// Inside the Autocast kernel for add:
at::AutoDispatchBelowAutocast guard;  // Excludes Autocast from key set
return at::add(self, other);          // Redispatches with remaining keys
```

This is how features compose — Autocast casts dtypes, then Autograd records the op, then the backend computes it.

---

## 6. Fallthrough Keys

Not every dispatch key has a kernel registered for every op. When a key has no kernel, the dispatcher **falls through** to the next key in the priority chain.

```python
# BackendSelect only has kernels for factory ops (torch.randn, torch.empty, etc.)
# For torch.add, BackendSelect falls through to the backend key (CPU/CUDA)
```

### Types of Fallthrough

1. **No registration** — key is skipped entirely
2. **Explicit fallthrough** — kernel registered that simply redispatches
3. **Default/catch-all** — CompositeImplicitAutograd provides fallback decompositions

### Example: `torch.randn`

```
torch.randn(3, 3, device='cuda')
  → BackendSelect kernel routes to CUDA
  → CUDA kernel allocates memory + fills with random values
```

`BackendSelect` is needed here because `torch.randn` has no input tensors — the dispatcher can't infer the backend from inputs.

---

## 7. CompositeImplicitAutograd

Ops registered at `CompositeImplicitAutograd` are **decomposed into other ops**. The autograd graph is built from the decomposed primitives — no custom backward formula needed.

```python
# torch.addmm decomposes into mm + add:
def addmm(input, mat1, mat2, beta=1, alpha=1):
    return beta * input + alpha * (mat1 @ mat2)
```

Since `mm` and `add` each have their own autograd formulas, the chain rule composes them automatically.

### When to Use CompositeImplicit

- The decomposition is numerically stable
- Performance of the decomposed version is acceptable
- You don't need a specialized backward pass

### Implications

- Works on ALL backends without backend-specific code
- Autograd "just works" through the decomposition
- torch.compile can see through the decomposition and fuse

---

## 8. CompositeExplicitAutograd

Ops at `CompositeExplicitAutograd` have a **custom backward formula** registered separately. Used when:

- The naive decomposition is numerically unstable (e.g., `log_softmax`)
- A custom backward is more memory-efficient (recompute vs store)
- The mathematical gradient simplifies significantly

```python
# log_softmax: naive decomposition has numerical issues
# Custom backward avoids computing exp() twice and is more stable

# Naive (unstable):
def log_softmax_naive(x):
    return torch.log(torch.softmax(x, dim=-1))

# Actual implementation uses log-sum-exp trick for stability
```

### Registration Pattern

```python
# Forward registered at CompositeExplicitAutograd (works on all backends)
# Backward registered at AutogradCPU, AutogradCUDA, etc.
# This lets the forward decompose freely while backward is specialized
```

---

## 9. torch.library — Registering Custom Ops

The `torch.library` module provides the Python API to register ops in the dispatcher.

### Step 1: Define the Op Schema

```python
from torch.library import Library, impl

lib = Library("mylib", "DEF")
lib.define("my_op(Tensor x, float scale) -> Tensor")
```

The schema uses PyTorch's operator schema language — it specifies argument types, return types, and optional mutability annotations.

### Step 2: Register Backend Implementations

```python
@impl(lib, "my_op", "CPU")
def my_op_cpu(x, scale):
    return x * scale + x.sin()

@impl(lib, "my_op", "CUDA")
def my_op_cuda(x, scale):
    # Could call a custom CUDA kernel here
    return x * scale + x.sin()
```

### Step 3: Register Meta Implementation

Meta kernels compute output shape/dtype without actual data — required for `torch.compile` and `torch.export`:

```python
@impl(lib, "my_op", "Meta")
def my_op_meta(x, scale):
    return torch.empty_like(x)
```

### Step 4: Register Autograd

```python
class MyOpAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return torch.ops.mylib.my_op(x, scale)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output * (ctx.scale + x.cos())
        return grad_x, None

def my_op_autograd(x, scale):
    return MyOpAutograd.apply(x, scale)

lib_autograd = Library("mylib", "IMPL")
lib_autograd.impl("my_op", my_op_autograd, "AutogradCPU")
```

### Step 5: Use the Op

```python
x = torch.randn(4, 4, requires_grad=True)
y = torch.ops.mylib.my_op(x, scale=2.0)
y.sum().backward()  # Autograd works!
```

---

## 10. @custom_op — The Modern API

PyTorch 2.4+ provides a simpler decorator-based API for custom ops:

```python
@torch.library.custom_op("mylib::fast_gelu", mutates_args=())
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)
```

### Register Fake (Meta) Implementation

```python
@fast_gelu.register_fake
def fast_gelu_fake(x):
    return torch.empty_like(x)
```

### Register Autograd

```python
def fast_gelu_setup_context(ctx, inputs, output):
    x, = inputs
    ctx.save_for_backward(x)

def fast_gelu_backward(ctx, grad_output):
    x, = ctx.saved_tensors
    sigmoid_val = torch.sigmoid(1.702 * x)
    grad = sigmoid_val + 1.702 * x * sigmoid_val * (1 - sigmoid_val)
    return (grad_output * grad,)

fast_gelu.register_autograd(fast_gelu_backward, setup_context=fast_gelu_setup_context)
```

### Advantages of @custom_op

| Feature | Library API | @custom_op |
|---------|-------------|------------|
| Boilerplate | High | Low |
| Schema inference | Manual | Automatic from type hints |
| torch.compile | Manual Meta reg | `register_fake` |
| Autograd | Manual Function class | `register_autograd` |
| Composability | Manual | Built-in |

---

## 11. Viewing Dispatch Tables

PyTorch exposes tools to inspect what kernels are registered for any operator:

### Dump Full Table

```python
print(torch._C._dispatch_dump("aten::add.Tensor"))
```

Output shows every dispatch key and its registered kernel:

```
Registered Kernels:
  CompositeImplicitAutograd[alias]: ...
  CPU[kernel]: at::native::add(...)
  CUDA[kernel]: at::native::add_cuda(...)
  Meta[kernel]: at::native::add_meta(...)
  AutogradCPU[autograd]: ...
  AutogradCUDA[autograd]: ...
  ...
```

### Check Specific Key

```python
# Check if a key has a kernel for an op
print(torch._C._dispatch_has_kernel_for_dispatch_key(
    "aten::add.Tensor", "CPU"
))  # True
```

### List All Registered Ops

```python
# All ops in a namespace
ops = [op for op in dir(torch.ops.aten) if not op.startswith('_')]
print(f"aten namespace has {len(ops)} ops")
```

---

## 12. Structured Kernels

Structured kernels are the modern pattern for implementing ops in C++. They split the kernel into two parts:

1. **Meta function** — computes output shape/dtype, allocates output tensor
2. **Impl function** — fills the output tensor with computed values

```cpp
// Meta function (shared across all backends)
TORCH_META_FUNC(add)(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    // Compute output shape via broadcasting
    auto output_shape = infer_size(self.sizes(), other.sizes());
    set_output_raw_strided(0, output_shape, {}, self.options());
}

// CPU implementation
TORCH_IMPL_FUNC(add_out_cpu)(const Tensor& self, const Tensor& other,
                              const Scalar& alpha, const Tensor& result) {
    // Fill result with self + alpha * other
    add_kernel(kCPU, *this);  // Dispatches to vectorized CPU code
}

// CUDA implementation
TORCH_IMPL_FUNC(add_out_cuda)(const Tensor& self, const Tensor& other,
                               const Scalar& alpha, const Tensor& result) {
    add_kernel(kCUDA, *this);  // Dispatches to CUDA kernel
}
```

### Benefits

- Shape/dtype logic written once
- Output allocation handled uniformly
- Each backend only writes the compute
- Meta backend gets the meta function for free

---

## 13. How torch.compile Interacts

torch.compile operates **above** the dispatcher in most cases:

```
┌─────────────────────────────────────────────────────┐
│ TorchDynamo (Python bytecode analysis)              │
│   Intercepts Python code BEFORE it hits dispatcher  │
│   Captures a graph of operations                    │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ AOTAutograd                                         │
│   Traces through Autograd dispatch keys             │
│   Produces forward + backward graphs               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Inductor                                            │
│   Generates fused kernels                           │
│   Bypasses dispatcher entirely at runtime           │
└─────────────────────────────────────────────────────┘
```

### Key Points

1. **Dynamo** intercepts at the Python level — it sees `torch.add` calls before they reach C++
2. **AOTAutograd** traces through the autograd dispatch keys to produce explicit forward/backward graphs
3. **Inductor** generates code that calls compute kernels directly — no dispatch overhead at runtime
4. Custom ops with `register_fake` work seamlessly — Dynamo uses the fake implementation for tracing

### Dispatch Keys Relevant to Compile

- `PythonDispatcher` — active when Dynamo is tracing
- `FakeTensor` — uses Meta kernels to track shapes during tracing
- `ProxyTorchDispatchMode` — captures ops as a graph during AOTAutograd

---

## 14. How Autograd Uses Dispatch

The Autograd dispatch keys (AutogradCPU, AutogradCUDA, etc.) implement automatic differentiation as a **dispatch layer**:

### Forward Pass

```
AutogradCUDA kernel for add(x, y):
  1. x and y have requires_grad=True
  2. Save inputs needed for backward (none for add)
  3. Create AddBackward0 node
  4. Exclude AutogradCUDA from key set
  5. Redispatch → CUDA kernel computes result
  6. Attach grad_fn to result tensor
  7. Return result
```

### Backward Pass

When `.backward()` is called, the engine walks the autograd graph and calls each node's backward function. This does NOT go through the dispatcher again for the top-level backward — but the individual gradient computations (e.g., `grad * weight.T` inside a linear backward) do go through the dispatcher.

### ADInplaceOrView

This key handles a subtle case: in-place operations and views.

```python
x = torch.randn(3, 3, requires_grad=True)
y = x.view(9)       # ADInplaceOrView tracks this
y.add_(1.0)         # In-place on a view — must update version counter
x.backward(...)     # Still works because of proper tracking
```

The `ADInplaceOrView` key ensures that:
- View operations record the view relationship
- In-place ops increment version counters
- Autograd can detect illegal in-place modifications

---

## 15. Upstream Updates (June 30 - July 1, 2026)

Recent PyTorch changes relevant to the dispatcher:

### CPU Flash SDPA Non-Contiguous Fix (#187506)

Fixed a bug where the CPU implementation of Flash Scaled Dot Product Attention produced incorrect results for non-contiguous input tensors. The dispatcher correctly routed to the CPU SDPA kernel, but the kernel itself assumed contiguous memory layout. Now properly handles strided inputs.

### DTensor linspace (#187933)

Added dispatcher registration for `linspace` in the DTensor subsystem. DTensor implements its own dispatch key to intercept operations and distribute them across a device mesh. This PR ensures `torch.linspace` works correctly in distributed tensor contexts.

### c10d setSequenceNumberForGroup Deprecation (#188611)

Deprecated `setSequenceNumberForGroup` in favor of a new sequence tracking mechanism. This affects the distributed dispatch keys (`c10d`) that handle collective operations. The dispatcher routes collective ops (all_reduce, broadcast, etc.) through dedicated dispatch keys.

### MPS F.linear Bias Fix (#188619)

Fixed incorrect results from `F.linear` on MPS backend when bias is provided. The MPS dispatch key routes to Apple Metal kernels — this fix corrects the bias addition in the MPS-specific linear kernel.

### Control Collectives Removal (#188617)

Removed the experimental control collectives dispatch mechanism. This simplifies the dispatch key space by removing keys that were used for prototype distributed control flow. Demonstrates that dispatch keys can be added and removed as the system evolves.

---

## Files in This Module

| File | Description | Lines |
|------|-------------|-------|
| `README.md` | This guide — dispatcher internals explained | 400+ |
| `dispatch_keys.py` | Explore dispatch keys, priority chains, tables | 250+ |
| `custom_dispatch.py` | Register custom ops, autograd, compile integration | 250+ |

---

## Key Takeaways

1. **Every op goes through the dispatcher** — it's the central nervous system of PyTorch
2. **Dispatch keys are a bitset** on each tensor, representing active features
3. **Priority chain** determines which kernel runs — highest priority with a registered kernel wins
4. **Redispatch** is how features compose — Autocast → Autograd → Backend
5. **CompositeImplicit** ops decompose into primitives — autograd comes free
6. **torch.library** and `@custom_op` let you register ops that work with all PyTorch features
7. **torch.compile bypasses most dispatch** — it captures a graph, then generates code that calls kernels directly
8. **The dispatcher is extensible** — new backends/features just register new keys

Understanding the dispatcher transforms PyTorch from a "magic box" into a transparent, debuggable system.

---

### Further Resources

- [PyTorch Dispatcher Deep Dive (Edward Yang)](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — the definitive blog post
- [torch.library documentation](https://pytorch.org/docs/stable/library.html) — official custom ops guide
- [Module 19 — Tensor Dispatch](../19_torch_function_dispatch/) — `__torch_function__` and `__torch_dispatch__`
- [Module 08 — torch.compile](../08_torch_compile/) — how the compiler interacts with dispatch

---

<div align="center">

[← Previous Module (LLM Fine-Tuning)](../34_llm_finetuning/) | [🏠 Home](../README.md) | Next Module → (none)

**Notebook**: [`35_dispatcher.ipynb`](../notebooks/35_dispatcher.ipynb)

</div>
