# PyTorch: The Definitive Master Reference (2026 Edition)
## From Mathematical Foundations to Production Deployment

**Version:** 2.13+ (Main Branch, June 2026)
**Last Updated:** June 2026

---

> **About This Document**
>
> This is a single, comprehensive, updated PyTorch reference covering the entire framework from first principles to production deployment. It consolidates and updates all previous reference documents with the latest APIs, new features, and best practices from the PyTorch main branch (v2.13+).
>
> **What's New in This Edition:**
> - FlexAttention API (`torch.nn.attention.flex_attention`)
> - Flash Attention 3 & 4 backends
> - FSDP2 (`fully_shard`) — the composable FSDP API
> - DTensor moved to `torch.distributed.tensor` (stable)
> - Pipeline Parallelism schedules (1F1B, GPipe, ZeroBubble, DualPipeV)
> - NativeRT — C++ inference engine for exported models
> - `torch.export` maturity improvements
> - Compiled Autograd and Compiled Optimizers
> - DeviceMesh as the standard for multi-device coordination
> - New pipeline schedules: `ScheduleZBVZeroBubble`, `ScheduleDualPipeV`
> - SymmetricMemory for efficient intra-node communication
> - Updated `torch.compile` with improved dynamic shapes and caching
>
> **Converting to PDF:**
> ```bash
> pandoc PYTORCH_MASTER_REFERENCE_2026.md -o pytorch_reference_2026.pdf \
>   --pdf-engine=xelatex \
>   --toc --toc-depth=3 \
>   -V geometry:margin=1in \
>   -V fontsize=10pt \
>   -V mainfont="DejaVu Sans" \
>   -V monofont="DejaVu Sans Mono" \
>   --highlight-style=tango
> ```

---

# MASTER TABLE OF CONTENTS

## Part 0: Mathematical Foundations
- 0.1 Linear Algebra (Vectors, Matrices, Norms, Decompositions)
- 0.2 Calculus for Deep Learning (Chain Rule, Jacobians, Hessians)
- 0.3 Probability & Statistics (Distributions, Expectation, Variance)
- 0.4 Optimization Theory (Gradient Descent, Momentum, Adam)
- 0.5 Information Theory (Entropy, Cross-Entropy, KL Divergence)

## Part I: Introduction & Setup
- 1.1 What is PyTorch?
- 1.2 Installation & Environment
- 1.3 Core Concepts & Design Philosophy
- 1.4 Tensor Fundamentals
- 1.5 Dynamic Computation Graphs

## Part II: PyTorch Architecture & Core Internals
- 2.1 Layer Architecture Overview
- 2.2 C10 Core Library (Device, ScalarType, Storage, TensorImpl)
- 2.3 ATen Tensor Library (Operators, Code Generation)
- 2.4 Dispatcher System (Dispatch Keys, Priority Chain, Structured Kernels)
- 2.5 Memory Management (CPU Allocator, CUDA Caching Allocator)
- 2.6 Custom Operators (torch.library, TORCH_LIBRARY)
- 2.7 Meta Device & Shape Inference

## Part III: Tensor Operations
- 3.1 Tensor Creation & Properties
- 3.2 Mathematical Operations (Element-wise, Reductions, Linear Algebra)
- 3.3 Tensor Manipulation (Reshape, Concat, Split)
- 3.4 Broadcasting Rules
- 3.5 Indexing & Advanced Indexing
- 3.6 Views vs Copies, Strides, Contiguity
- 3.7 In-place Operations
- 3.8 Einsum

## Part IV: Automatic Differentiation
- 4.1 Autograd Architecture (Engine, Graph, Nodes)
- 4.2 Computation Graph (grad_fn, Leaf Tensors)
- 4.3 Backward Pass Implementation (C++ Engine)
- 4.4 Gradient Mathematics (Chain Rule, Backpropagation Derivation)
- 4.5 Custom Autograd Functions
- 4.6 Higher-Order Gradients (Jacobian, Hessian, JVP, VJP)
- 4.7 Compiled Autograd

## Part V: Neural Networks (torch.nn)
- 5.1 nn.Module Base Class (Lifecycle, Parameters, Buffers, Hooks)
- 5.2 Container Modules (Sequential, ModuleList, ModuleDict)
- 5.3 Linear Layers
- 5.4 Convolutional Layers (Conv1d/2d/3d, ConvTranspose, Depthwise)
- 5.5 Pooling Layers
- 5.6 Normalization Layers (BatchNorm, LayerNorm, GroupNorm, RMSNorm)
- 5.7 Activation Functions (ReLU, GELU, SiLU, Mish, Softmax)
- 5.8 Dropout & Regularization
- 5.9 Recurrent Layers (RNN, LSTM, GRU)
- 5.10 Transformer Layers (MultiheadAttention, Encoder, Decoder)
- 5.11 Embedding Layers
- 5.12 Loss Functions
- 5.13 Attention Mechanisms & FlexAttention
- 5.14 Functional API (torch.nn.functional)
- 5.15 Weight Initialization

## Part VI: Optimization
- 6.1 Optimizer Base Class & Parameter Groups
- 6.2 SGD (Momentum, Nesterov)
- 6.3 Adam Family (Adam, AdamW, LAMB)
- 6.4 Learning Rate Schedulers
- 6.5 Compiled Optimizers

## Part VII: Data Loading & Processing
- 7.1 Dataset & DataLoader Architecture
- 7.2 Custom Datasets & Collate Functions
- 7.3 Data Augmentation (Transforms, MixUp, CutMix)
- 7.4 Distributed Sampling

## Part VIII: Training Pipelines
- 8.1 Complete Training Loop Template
- 8.2 Mixed Precision Training (AMP, GradScaler, BFloat16)
- 8.3 Gradient Accumulation
- 8.4 Gradient Checkpointing (Activation Checkpointing)
- 8.5 Transfer Learning & Fine-Tuning
- 8.6 Knowledge Distillation
- 8.7 EMA, SWA, Label Smoothing

## Part IX: Compilation & Performance (torch.compile)
- 9.1 torch.compile Overview
- 9.2 TorchDynamo (Graph Capture, Guards, Graph Breaks)
- 9.3 AOTAutograd & FX Graph IR
- 9.4 TorchInductor (Triton Codegen, C++ Backend, Fusion)
- 9.5 Dynamic Shapes & Compilation Cache
- 9.6 torch.compile Config & Debugging
- 9.7 Writing Custom Triton Kernels
- 9.8 Compiled Autograd
- 9.9 Custom Backends

## Part X: Distributed Training
- 10.1 Distributed Overview (DDP vs FSDP1 vs FSDP2 vs Tensor Parallel)
- 10.2 DeviceMesh — The Foundation
- 10.3 DistributedDataParallel (DDP)
- 10.4 FSDP1 (Legacy)
- 10.5 FSDP2 — fully_shard (The New Standard)
- 10.6 DTensor (torch.distributed.tensor)
- 10.7 Tensor Parallelism
- 10.8 Pipeline Parallelism (1F1B, GPipe, ZeroBubble, DualPipeV)
- 10.9 Collective Communications
- 10.10 torchrun & Elastic Training
- 10.11 SymmetricMemory

## Part XI: torch.export & Deployment
- 11.1 torch.export API
- 11.2 Export Constraints & Dynamic Shapes
- 11.3 AOTInductor (C++ Deployment)
- 11.4 NativeRT — C++ Inference Engine
- 11.5 ONNX Export
- 11.6 TorchServe
- 11.7 Mobile & Edge Deployment

## Part XII: Hardware Acceleration
- 12.1 CUDA Programming (Streams, Events, Graphs, Memory)
- 12.2 cuDNN Integration
- 12.3 Metal Performance Shaders (MPS)
- 12.4 Intel GPU (XPU)
- 12.5 ROCm Support

## Part XIII: Advanced Features
- 13.1 torch.fx — Graph Transformation
- 13.2 Functorch (vmap, grad, jacrev, hessian)
- 13.3 Sparse Tensors (COO, CSR, BSR)
- 13.4 Complex Numbers
- 13.5 Quantization (Post-Training, QAT, PT2E)
- 13.6 torch.ao — Architecture Optimization

## Part XIV: Model Architectures
- 14.1 ResNet (Complete Implementation)
- 14.2 Transformer (Encoder-Decoder with Flash Attention)
- 14.3 GPT (Decoder-Only with FlexAttention)
- 14.4 Vision Transformer (ViT)

## Part XV: Testing & Debugging
- 15.1 Testing Framework (TestCase, run_tests)
- 15.2 OpInfo Framework
- 15.3 Profiling (PyTorch Profiler, torch.profiler)
- 15.4 Common Errors & Solutions
- 15.5 Reproducibility Guide

## Part XVI: Build System & Contributing
- 16.1 CMake Build System
- 16.2 Code Generation (torchgen)
- 16.3 Adding New Operators
- 16.4 C++ Extensions

## Appendices
- A: Complete Mathematical Derivations
- B: Complete torch.nn Module Reference
- C: torch.linalg Reference
- D: torch.distributions Reference
- E: torch.fft / torch.special Reference
- F: Serialization (torch.save/load, SafeTensors)
- G: Performance Optimization Checklist
- H: Debugging Guide
- I: Quick Reference Tables

---

# Part 0: Mathematical Foundations

## 0.1 Linear Algebra

### Vectors and Vector Spaces

A vector space $V$ over field $\mathbb{R}$ is a set with addition and scalar multiplication satisfying 8 axioms (associativity, commutativity, identity, inverse for addition; compatibility, identity, distributivity for scalar multiplication).

**Inner Product**: $\langle u, v \rangle = u^T v = \sum_i u_i v_i$

**Norms**:
- L2 (Euclidean): $\|v\|_2 = \sqrt{\sum_i v_i^2}$
- L1 (Manhattan): $\|v\|_1 = \sum_i |v_i|$
- L∞ (Max): $\|v\|_\infty = \max_i |v_i|$

```python
import torch

u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, 5.0, 6.0])

dot_product = torch.dot(u, v)      # 32.0
l2_norm = torch.linalg.norm(u)     # 3.7417
l1_norm = torch.linalg.norm(u, 1)  # 6.0
linf = torch.linalg.norm(u, float('inf'))  # 3.0

print(f"Dot product: {dot_product}")
print(f"L2 norm: {l2_norm:.4f}")
print(f"Cosine similarity: {torch.nn.functional.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0))}")
```

### Matrices and Decompositions

**Matrix Multiplication**: For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$:
$C = AB$ where $C_{ij} = \sum_k A_{ik} B_{kj}$

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # or torch.matmul(A, B) → shape (3, 5)

# Eigendecomposition
M = torch.randn(3, 3)
M = M @ M.T  # Make symmetric
eigenvalues, eigenvectors = torch.linalg.eigh(M)

# SVD
U, S, Vh = torch.linalg.svd(A)
print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")

# QR decomposition
Q, R = torch.linalg.qr(A)

# Cholesky decomposition (for positive definite matrices)
L = torch.linalg.cholesky(M)

# Solve linear system Ax = b
A_sq = torch.randn(3, 3)
b = torch.randn(3, 1)
x = torch.linalg.solve(A_sq, b)
```

## 0.2 Calculus for Deep Learning

### Chain Rule (Backpropagation Foundation)

For $f(g(x))$: $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$

For neural networks with layers $y = f_n(f_{n-1}(\ldots f_1(x)\ldots))$:
$$\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_{i+1}}{\partial f_i} \cdot \frac{\partial f_i}{\partial \theta_i}$$

### Jacobian Matrix

For $f: \mathbb{R}^n \to \mathbb{R}^m$:
$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

```python
x = torch.randn(3, requires_grad=True)

def f(x):
    return torch.stack([x[0]**2 + x[1], x[1]*x[2], x[0] + x[2]**3])

J = torch.autograd.functional.jacobian(f, x)
print(f"Jacobian shape: {J.shape}")  # (3, 3)
```

### Hessian Matrix

For scalar-valued $f: \mathbb{R}^n \to \mathbb{R}$:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

```python
x = torch.randn(3, requires_grad=True)
def g(x):
    return (x ** 3).sum()

H = torch.autograd.functional.hessian(g, x)
print(f"Hessian shape: {H.shape}")  # (3, 3)
```

## 0.3 Probability & Statistics

### Key Distributions in Deep Learning

| Distribution | PDF/PMF | PyTorch |
|---|---|---|
| Normal | $\frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | `torch.distributions.Normal(mu, sigma)` |
| Bernoulli | $p^k(1-p)^{1-k}$ | `torch.distributions.Bernoulli(probs)` |
| Categorical | $\prod_i p_i^{[y=i]}$ | `torch.distributions.Categorical(probs)` |
| Uniform | $\frac{1}{b-a}$ | `torch.distributions.Uniform(a, b)` |

```python
from torch.distributions import Normal, Categorical

# Normal distribution
dist = Normal(loc=0.0, scale=1.0)
samples = dist.sample((1000,))
log_probs = dist.log_prob(samples)
print(f"Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")

# Reparameterization trick (used in VAEs)
mu = torch.zeros(10, requires_grad=True)
sigma = torch.ones(10, requires_grad=True)
dist = Normal(mu, sigma)
z = dist.rsample()  # Differentiable sampling
loss = z.sum()
loss.backward()
print(f"Gradient flows through rsample: {mu.grad is not None}")  # True
```

## 0.4 Optimization Theory

### Gradient Descent

$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$

### Momentum

$v_{t+1} = \beta v_t + \nabla L(\theta_t)$
$\theta_{t+1} = \theta_t - \eta v_{t+1}$

### Adam

$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
$\hat{m}_t = m_t / (1 - \beta_1^t)$
$\hat{v}_t = v_t / (1 - \beta_2^t)$
$\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

### AdamW (Decoupled Weight Decay)

Same as Adam but weight decay is applied directly to parameters, not through the gradient:
$\theta_{t+1} = (1 - \lambda\eta)\theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

## 0.5 Information Theory

**Entropy**: $H(p) = -\sum_i p_i \log p_i$

**Cross-Entropy**: $H(p, q) = -\sum_i p_i \log q_i$

**KL Divergence**: $D_{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} = H(p, q) - H(p)$

```python
import torch.nn.functional as F

# Cross-entropy loss (combines log_softmax + nll_loss)
logits = torch.randn(4, 10)  # 4 samples, 10 classes
targets = torch.randint(0, 10, (4,))
loss = F.cross_entropy(logits, targets)
print(f"Cross-entropy loss: {loss:.4f}")

# KL divergence
p = F.softmax(torch.randn(100), dim=0)
q = F.softmax(torch.randn(100), dim=0)
kl = F.kl_div(q.log(), p, reduction='sum')
print(f"KL divergence: {kl:.4f}")
```

---

# Part I: Introduction & Setup

## 1.1 What is PyTorch?

PyTorch is an open-source machine learning framework originally developed by Meta AI. Key characteristics:

- **Eager execution by default**: Operations execute immediately, making debugging intuitive
- **Dynamic computation graphs**: Graph is built on-the-fly during forward pass
- **Python-first**: Deep Python integration with C++/CUDA backend for performance
- **torch.compile**: Optional graph compilation for 2x+ speedup without changing model code
- **Production-ready**: Export, compile, and deploy models via torch.export, AOTInductor, and NativeRT

### PyTorch Architecture Stack

```
┌─────────────────────────────────────────────────┐
│               Python Frontend                     │
│    torch.nn  │  torch.optim  │  torch.utils      │
├─────────────────────────────────────────────────┤
│              torch.compile                        │
│    Dynamo  │  AOTAutograd  │  Inductor            │
├─────────────────────────────────────────────────┤
│           Autograd Engine (C++)                   │
├─────────────────────────────────────────────────┤
│              ATen (Tensor Library)                 │
│    Operators  │  Dispatch  │  Code Generation     │
├─────────────────────────────────────────────────┤
│              C10 (Core Library)                    │
│    Device  │  ScalarType  │  Storage  │ Allocator │
├─────────────────────────────────────────────────┤
│            Hardware Backends                       │
│    CPU  │  CUDA  │  MPS  │  XPU  │  ROCm         │
└─────────────────────────────────────────────────┘
```

## 1.2 Installation & Environment

```bash
# Stable release (pip)
pip install torch torchvision torchaudio

# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Verify
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 1.3 Core Concepts & Design Philosophy

**Eager Mode**: Default execution — operations run immediately:
```python
x = torch.randn(3, 4)
y = x + 1       # Executes immediately
z = y.relu()    # Executes immediately
print(z)        # Result is available now
```

**Compiled Mode**: `torch.compile` captures operations into a graph for optimization:
```python
@torch.compile
def fast_fn(x):
    y = x + 1
    return y.relu()

z = fast_fn(torch.randn(3, 4))  # First call: trace + compile. Subsequent: fast execution
```

**Key Principle**: Write idiomatic Python. Use `torch.compile` to make it fast. No code changes needed.

## 1.4 Tensor Fundamentals

```python
import torch

# Creation
x = torch.tensor([1, 2, 3])               # From list
x = torch.zeros(3, 4)                      # Zeros
x = torch.ones(3, 4)                       # Ones
x = torch.randn(3, 4)                      # Normal distribution
x = torch.empty(3, 4)                      # Uninitialized
x = torch.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, steps=5)          # [0.0, 0.25, 0.5, 0.75, 1.0]
x = torch.eye(3)                           # 3×3 identity

# Properties
print(f"Shape: {x.shape}")
print(f"Dtype: {x.dtype}")         # torch.float32
print(f"Device: {x.device}")       # cpu
print(f"Strides: {x.stride()}")
print(f"Contiguous: {x.is_contiguous()}")

# Dtype casting
x = x.to(torch.float16)
x = x.float()      # → float32
x = x.half()       # → float16
x = x.bfloat16()   # → bfloat16

# Device transfer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# Gradient tracking
x = torch.randn(3, 4, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(f"Gradient: {x.grad}")  # dy/dx = 2x
```

## 1.5 Dynamic Computation Graphs

Unlike static graph frameworks, PyTorch builds the computation graph dynamically during the forward pass:

```python
def dynamic_forward(x, use_relu=True):
    y = x @ torch.randn(x.shape[1], 10)
    if use_relu:        # Standard Python control flow
        y = y.relu()
    else:
        y = y.sigmoid()
    return y

# The graph is different each call
x = torch.randn(5, 20, requires_grad=True)
out1 = dynamic_forward(x, use_relu=True)   # Graph has relu
out2 = dynamic_forward(x, use_relu=False)  # Graph has sigmoid
```

---

# Part II: PyTorch Architecture & Core Internals

## 2.1 Layer Architecture Overview

PyTorch's architecture is layered:

1. **C10** (caffe2/c10): Core library — `Device`, `ScalarType`, `Storage`, `TensorImpl`, memory allocators
2. **ATen** (aten/): Tensor library — all operators (add, matmul, conv, etc.), code generation
3. **Autograd**: Automatic differentiation engine (C++ core with Python bindings)
4. **Dispatcher**: Central routing system — maps operator calls to the correct kernel by dispatch key
5. **Python Frontend**: `torch`, `torch.nn`, `torch.optim` — the user-facing API
6. **torch.compile**: JIT compilation stack — Dynamo (capture) → AOTAutograd (differentiation) → Inductor (codegen)

## 2.2 C10 Core Library

C10 provides the fundamental types used across all of PyTorch:

### Device
```python
cpu_device = torch.device('cpu')
cuda_device = torch.device('cuda', 0)
mps_device = torch.device('mps')
xpu_device = torch.device('xpu', 0)
meta_device = torch.device('meta')  # Shape-only, no data
```

### ScalarType (dtype)
```python
# Standard types
torch.float16    # Half precision
torch.bfloat16   # Brain floating point
torch.float32    # Single precision (default)
torch.float64    # Double precision
torch.int8
torch.int16
torch.int32
torch.int64
torch.bool
torch.complex64
torch.complex128

# Float8 types (for quantized training/inference)
torch.float8_e4m3fn
torch.float8_e5m2
torch.float8_e4m3fnuz
torch.float8_e5m2fnuz

# Unsigned integer types
torch.uint8
torch.uint16
torch.uint32
torch.uint64
```

### Storage and TensorImpl

Every tensor is backed by a `Storage` (contiguous block of memory) and described by a `TensorImpl` (shape, strides, dtype, device).

```python
x = torch.randn(3, 4)
print(f"Storage size: {x.storage().size()}")  # 12 elements
print(f"Storage offset: {x.storage_offset()}")
print(f"Strides: {x.stride()}")  # (4, 1) — row-major

# View shares storage
y = x.view(4, 3)
y[0, 0] = 999
print(x[0, 0])  # 999 — same storage
```

## 2.3 ATen Tensor Library

ATen provides the implementation of ~2000+ tensor operations. Operators are defined in YAML files and code-generated:

- `aten/src/ATen/native/native_functions.yaml` — operator signatures
- `torchgen/` — code generation tool
- `aten/src/ATen/native/` — C++ kernel implementations

### Operator Categories

| Category | Examples |
|---|---|
| Pointwise | `add`, `mul`, `relu`, `sigmoid`, `sin`, `cos` |
| Reduction | `sum`, `mean`, `max`, `argmax`, `norm` |
| Linear Algebra | `mm`, `bmm`, `matmul`, `linalg.solve` |
| Comparison | `eq`, `gt`, `lt`, `where` |
| Shape | `view`, `reshape`, `permute`, `cat`, `split` |
| Indexing | `index`, `index_put`, `gather`, `scatter` |
| Convolution | `conv1d`, `conv2d`, `conv3d` |
| Normalization | `batch_norm`, `layer_norm`, `group_norm` |

## 2.4 Dispatcher System

The dispatcher is PyTorch's central routing mechanism. When you call `torch.add(x, y)`, the dispatcher selects the right kernel based on **dispatch keys**:

```
Priority order (highest to lowest):
  Autocast          → Casts dtypes for mixed precision
  AutogradCPU/CUDA  → Records operation for backward pass
  ADInplaceOrView   → Tracks aliasing for autograd
  BackendSelect     → Routes to correct backend
  CPU/CUDA/MPS/XPU  → Actual computation kernel
  CompositeImplicit  → Default decomposition
```

```python
# Dispatch key priority for a CUDA tensor with requires_grad=True:
# 1. AutogradCUDA → records op in autograd graph
# 2. CUDA         → runs the actual CUDA kernel

# With torch.compile:
# Dynamo intercepts BEFORE dispatch, captures the graph,
# and the compiled kernel may bypass most dispatch keys
```

### torch.library — Custom Operators

```python
import torch
from torch.library import Library, impl

lib = Library("myops", "DEF")

# Define operator schema
lib.define("custom_relu(Tensor x) -> Tensor")

# Register CPU implementation
@impl(lib, "custom_relu", "CPU")
def custom_relu_cpu(x):
    return x.clamp(min=0)

# Register meta implementation (for shape inference)
@impl(lib, "custom_relu", "Meta")
def custom_relu_meta(x):
    return torch.empty_like(x)

# Now use it
x = torch.randn(3, 4)
y = torch.ops.myops.custom_relu(x)
```

## 2.5 Memory Management

### CPU Allocator
- Uses standard `malloc`/`free` with optional jemalloc/mimalloc
- Aligned allocations for SIMD operations

### CUDA Caching Allocator
- Maintains a cache of previously allocated CUDA memory blocks
- Avoids expensive `cudaMalloc`/`cudaFree` calls
- Memory is returned to the cache, not the OS

```python
# Memory monitoring
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Force cache cleanup
torch.cuda.empty_cache()

# Memory snapshot for debugging
torch.cuda.memory._record_memory_history()
# ... run your code ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

## 2.7 Meta Device & Shape Inference

The `meta` device creates tensors with shape/dtype but no actual data. Useful for model analysis without GPU memory:

```python
# Analyze model memory without allocating anything
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
).to('meta')

total_params = sum(p.numel() for p in model.parameters())
param_memory_mb = total_params * 4 / 1e6  # float32
print(f"Parameters: {total_params:,}, Memory: {param_memory_mb:.2f} MB")

# Trace activation shapes
x = torch.randn(32, 784, device='meta')
for layer in model:
    x = layer(x)
    print(f"{layer.__class__.__name__}: output={list(x.shape)}, "
          f"memory={x.numel() * 4 / 1024:.1f} KB")
```

---

# Part III: Tensor Operations

## 3.1 Tensor Creation & Properties

```python
import torch

# Creation functions
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x = torch.zeros(3, 4, dtype=torch.float32, device='cpu')
x = torch.ones_like(x)
x = torch.randn(3, 4)          # Normal(0, 1)
x = torch.rand(3, 4)           # Uniform(0, 1)
x = torch.randint(0, 10, (3, 4))
x = torch.full((3, 4), fill_value=3.14)
x = torch.arange(0, 10, step=0.5)
x = torch.linspace(0, 1, steps=100)
x = torch.logspace(0, 3, steps=4)  # [1, 10, 100, 1000]
x = torch.eye(4)

# From numpy
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
x = torch.from_numpy(np_arr)   # Shares memory!
np_back = x.numpy()            # Shares memory (CPU only)
```

## 3.2 Mathematical Operations

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Element-wise
c = a + b                    # Addition
c = a * b                    # Multiplication
c = a / b                    # Division
c = a ** 2                   # Power
c = torch.sqrt(a.abs())      # Square root
c = torch.exp(a)             # Exponential
c = torch.log(a.abs())       # Logarithm
c = torch.sin(a)             # Trigonometric
c = torch.clamp(a, -1, 1)    # Clamp

# Reductions
s = a.sum()                  # Scalar sum
s = a.sum(dim=1)             # Sum along dim 1
s = a.mean(dim=0)            # Mean along dim 0
s = a.max(dim=1)             # Max + argmax along dim 1
s = a.argmax(dim=-1)         # Argmax

# Linear algebra
x = torch.randn(3, 4)
y = torch.randn(4, 5)
z = x @ y                   # Matrix multiply → (3, 5)
z = torch.matmul(x, y)      # Same
z = torch.bmm(x.unsqueeze(0), y.unsqueeze(0))  # Batch matmul

# Batch matrix multiply
batch = torch.randn(8, 3, 4)
batch2 = torch.randn(8, 4, 5)
result = torch.bmm(batch, batch2)  # (8, 3, 5)
```

## 3.3 Tensor Manipulation

```python
x = torch.randn(2, 3, 4)

# Reshape
y = x.view(6, 4)          # Must be contiguous
y = x.reshape(6, 4)       # Works even if not contiguous
y = x.flatten()            # → (24,)
y = x.flatten(1)           # → (2, 12)

# Transpose / Permute
y = x.transpose(0, 1)     # Swap dims 0 and 1 → (3, 2, 4)
y = x.permute(2, 0, 1)    # Arbitrary permutation → (4, 2, 3)
y = x.T                   # Transpose (2D only shorthand)
y = x.mT                  # Matrix transpose (last two dims)

# Squeeze / Unsqueeze
y = x.unsqueeze(0)         # → (1, 2, 3, 4)
y = x.unsqueeze(-1)        # → (2, 3, 4, 1)
z = y.squeeze()            # Remove all dims of size 1

# Concatenation / Stacking
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.cat([a, b], dim=0)    # → (4, 3)
c = torch.cat([a, b], dim=1)    # → (2, 6)
c = torch.stack([a, b], dim=0)  # → (2, 2, 3) — new dim

# Split / Chunk
chunks = x.chunk(3, dim=1)     # Split into 3 along dim 1
parts = x.split(2, dim=2)      # Split into sizes of 2 along dim 2
```

## 3.4 Broadcasting Rules

PyTorch follows NumPy broadcasting semantics:
1. Align shapes from the right
2. Dimensions are compatible if equal or one of them is 1
3. Missing dimensions are treated as 1

```python
a = torch.randn(3, 4)
b = torch.randn(4)       # Broadcasts to (3, 4)
c = a + b                # Works: (3, 4) + (4,) → (3, 4)

a = torch.randn(3, 1, 4)
b = torch.randn(1, 5, 4)
c = a + b                # → (3, 5, 4)
```

## 3.5 Indexing & Advanced Indexing

```python
x = torch.randn(5, 4)

# Basic indexing (returns views)
row = x[0]          # First row
col = x[:, 0]       # First column
sub = x[1:3, 2:]    # Slice

# Boolean indexing (returns copy)
mask = x > 0
pos = x[mask]        # All positive elements

# Fancy indexing
idx = torch.tensor([0, 2, 4])
selected = x[idx]    # Rows 0, 2, 4

# Gather / Scatter
src = torch.randn(3, 4)
index = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 0, 0]])
gathered = torch.gather(src, dim=1, index=index)
```

## 3.6 Views vs Copies, Strides, Contiguity

```python
x = torch.randn(3, 4)

# Views share memory
y = x.view(4, 3)
y[0, 0] = 999
assert x[0, 0] == 999  # Same storage

# Operations that return views
y = x[0]            # Slice → view
y = x.transpose(0, 1)  # Transpose → view
y = x.unsqueeze(0)  # Unsqueeze → view
y = x.expand(2, 3, 4)  # Expand → view

# Operations that return copies
y = x[torch.tensor([0, 1])]  # Fancy indexing → copy
y = x[x > 0]       # Boolean indexing → copy
y = x.clone()       # Explicit copy

# Contiguity
x = torch.randn(3, 4)
y = x.transpose(0, 1)        # Not contiguous
print(f"Contiguous: {y.is_contiguous()}")  # False
z = y.contiguous()            # Makes a contiguous copy
```

## 3.8 Einsum

```python
# Einstein summation notation — expressive and efficient
a = torch.randn(3, 4)
b = torch.randn(4, 5)

# Matrix multiply
c = torch.einsum('ij,jk->ik', a, b)

# Batch matrix multiply
A = torch.randn(8, 3, 4)
B = torch.randn(8, 4, 5)
C = torch.einsum('bij,bjk->bik', A, B)

# Trace
M = torch.randn(4, 4)
trace = torch.einsum('ii->', M)

# Outer product
u = torch.randn(3)
v = torch.randn(4)
outer = torch.einsum('i,j->ij', u, v)

# Attention scores
Q = torch.randn(2, 8, 32, 64)  # (B, H, N, D)
K = torch.randn(2, 8, 32, 64)
scores = torch.einsum('bhnd,bhmd->bhnm', Q, K)  # (B, H, N, M)
```

---

# Part IV: Automatic Differentiation

## 4.1 Autograd Architecture

PyTorch's autograd engine builds a **directed acyclic graph (DAG)** during the forward pass, then traverses it in reverse during backward:

```
Forward:  x → [mul] → [add] → [relu] → loss
Backward: x.grad ← [MulBackward] ← [AddBackward] ← [ReluBackward] ← 1.0
```

Each tensor tracks its `grad_fn` — the function that created it:

```python
x = torch.randn(3, requires_grad=True)  # Leaf tensor
y = x * 2
z = y + 3
loss = z.sum()

print(f"y.grad_fn: {y.grad_fn}")       # MulBackward0
print(f"z.grad_fn: {z.grad_fn}")       # AddBackward0
print(f"loss.grad_fn: {loss.grad_fn}") # SumBackward0

loss.backward()
print(f"x.grad: {x.grad}")  # tensor([2., 2., 2.])
```

## 4.2 Gradient Control

```python
# Disable gradient tracking
with torch.no_grad():
    y = model(x)  # No graph built, faster

# Inference mode (even faster — disallows grad-requiring ops)
with torch.inference_mode():
    y = model(x)

# Detach from graph
y = x.detach()  # New tensor, no grad_fn

# Selective gradient computation
for param in model.parameters():
    param.requires_grad = False  # Freeze

# Gradient accumulation
optimizer.zero_grad()  # or optimizer.zero_grad(set_to_none=True) — faster
for i, (data, target) in enumerate(dataloader):
    loss = criterion(model(data), target) / accumulation_steps
    loss.backward()  # Gradients accumulate
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

## 4.5 Custom Autograd Functions

```python
class CustomSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid_x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_x = ctx.saved_tensors
        grad = sigmoid_x * (1 + x * (1 - sigmoid_x))
        return grad_output * grad

# Usage
x = torch.randn(5, requires_grad=True)
y = CustomSiLU.apply(x)
y.sum().backward()

# Verify with gradcheck
from torch.autograd import gradcheck
x = torch.randn(5, dtype=torch.float64, requires_grad=True)
assert gradcheck(CustomSiLU.apply, (x,), eps=1e-6)
```

## 4.6 Higher-Order Gradients

```python
import torch
from torch.func import jacrev, jacfwd, vmap, hessian

# Jacobian via functorch
def f(x):
    return torch.stack([x[0]**2 + x[1], x[1]*x[2]])

x = torch.randn(3)
J = jacrev(f)(x)      # Reverse-mode Jacobian
J_fwd = jacfwd(f)(x)  # Forward-mode Jacobian

# Hessian
def g(x):
    return (x ** 3).sum()

H = hessian(g)(x)

# Per-sample gradients (functorch vmap)
def compute_loss(params, x, y):
    pred = torch.func.functional_call(model, params, x)
    return torch.nn.functional.cross_entropy(pred, y)

params = dict(model.named_parameters())
ft_compute_grad = torch.func.grad(compute_loss)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
per_sample_grads = ft_compute_sample_grad(params, batch_x, batch_y)
```

## 4.7 Compiled Autograd

Compiled Autograd captures the backward pass graph for compilation, enabling further optimizations:

```python
import torch._dynamo.compiled_autograd

# Enable compiled autograd
with torch._dynamo.compiled_autograd.enable(torch.compile(backend="inductor")):
    model = torch.compile(model)
    loss = model(x).sum()
    loss.backward()  # Backward is also compiled!
```

---

# Part V: Neural Networks (torch.nn)

## 5.1 nn.Module Base Class

Every neural network in PyTorch is a subclass of `nn.Module`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Non-trainable state
        self.register_buffer('step_count', torch.tensor(0))

    def forward(self, x):
        x = self.act(self.bn(self.fc1(x)))
        self.step_count += 1
        return self.fc2(x)

model = MyModel(784, 256, 10)

# Key properties
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Named params: {[n for n, _ in model.named_parameters()]}")
print(f"Buffers: {[n for n, _ in model.named_buffers()]}")
print(f"Modules: {[n for n, _ in model.named_modules()]}")
```

### Module Lifecycle & Hooks

```python
# Forward hooks
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__}: input={input[0].shape}, output={output.shape}")

handle = model.fc1.register_forward_hook(hook_fn)
model(torch.randn(4, 784))
handle.remove()

# Save / Load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', weights_only=True))

# Device / Dtype management
model = model.to('cuda')
model = model.to(torch.bfloat16)
model = model.half()    # → float16
model = model.float()   # → float32
```

## 5.2 Container Modules

```python
# Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# ModuleList — for dynamic indexing
class MultiHead(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_heads)])

    def forward(self, x):
        return [head(x) for head in self.heads]

# ModuleDict — for named access
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleDict({
            'vision': nn.Linear(512, 256),
            'text': nn.Linear(768, 256),
        })

    def forward(self, x, modality):
        return self.experts[modality](x)
```

## 5.3 Linear Layers

```python
# y = xW^T + b
linear = nn.Linear(in_features=256, out_features=128, bias=True)
# Parameters: weight (128, 256), bias (128,)
# Input: (*, 256) → Output: (*, 128)

# Bilinear: y = x1^T A x2 + b
bilinear = nn.Bilinear(in1_features=20, in2_features=30, out_features=40)

# LazyLinear — infers in_features from first input
lazy = nn.LazyLinear(out_features=128)
```

## 5.4 Convolutional Layers

```python
# Conv2d: standard convolution
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,      # 'same' padding for stride=1
    bias=False
)
# Input: (N, 3, H, W) → Output: (N, 64, H, W)

# Depthwise convolution
depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)

# ConvTranspose2d (upsampling)
deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# Input: (N, 64, H, W) → Output: (N, 32, 2H, 2W)
```

Output size formula: $H_{out} = \lfloor \frac{H_{in} + 2P - D(K-1) - 1}{S} + 1 \rfloor$

## 5.5 Pooling Layers

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)        # Halves spatial dims
pool = nn.AvgPool2d(kernel_size=2, stride=2)
pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))      # Global average pooling
pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))
```

## 5.6 Normalization Layers

```python
# BatchNorm — normalizes over batch dimension
bn = nn.BatchNorm2d(64)          # For (N, 64, H, W)
# y = γ * (x - μ_batch) / √(σ²_batch + ε) + β

# LayerNorm — normalizes over last dims
ln = nn.LayerNorm(512)           # For (..., 512)
ln = nn.LayerNorm([64, 32, 32]) # For (N, 64, 32, 32)

# GroupNorm — normalizes over groups of channels
gn = nn.GroupNorm(num_groups=8, num_channels=64)

# RMSNorm — root mean square normalization (used in LLMs)
rms = nn.RMSNorm(512)           # For (..., 512)
# y = x / RMS(x) * γ  where RMS(x) = √(mean(x²) + ε)
```

## 5.7 Activation Functions

```python
# Common activations
relu = nn.ReLU()                 # max(0, x)
gelu = nn.GELU()                 # x * Φ(x) — used in Transformers
silu = nn.SiLU()                 # x * σ(x) — aka Swish, used in LLMs
mish = nn.Mish()                 # x * tanh(softplus(x))
leaky = nn.LeakyReLU(0.01)       # max(0.01x, x)

# Derivatives (for reference)
# ReLU: 0 if x < 0, 1 if x > 0
# GELU: Φ(x) + x·φ(x) where Φ is CDF, φ is PDF of N(0,1)
# SiLU: σ(x) + x·σ(x)·(1-σ(x)) = σ(x)·(1 + x·(1-σ(x)))

# Output activations
sigmoid = nn.Sigmoid()           # σ(x) = 1/(1+e^(-x)) — binary classification
softmax = nn.Softmax(dim=-1)     # e^xi / Σ e^xj — multi-class
log_softmax = nn.LogSoftmax(dim=-1)
```

## 5.8 Dropout & Regularization

```python
dropout = nn.Dropout(p=0.1)         # Zeroes elements with probability p
dropout2d = nn.Dropout2d(p=0.1)     # Zeroes entire channels
alpha_drop = nn.AlphaDropout(p=0.1) # For SELU networks

# Dropout is only active during training
model.train()   # dropout active
model.eval()    # dropout disabled
```

## 5.9 Recurrent Layers

```python
# LSTM
lstm = nn.LSTM(
    input_size=256,
    hidden_size=512,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.1
)
# Input: (N, L, 256) → Output: (N, L, 1024), (h_n, c_n)

# GRU
gru = nn.GRU(input_size=256, hidden_size=512, num_layers=2, batch_first=True)
```

## 5.10 Transformer Layers

```python
# Built-in Transformer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-norm (recommended)
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# MultiheadAttention (uses SDPA internally — Flash Attention when available)
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)
# attn_output, attn_weights = mha(query, key, value)
```

## 5.11 Embedding Layers

```python
# Standard embedding
emb = nn.Embedding(num_embeddings=50000, embedding_dim=512, padding_idx=0)
# Input: (N, L) of indices → Output: (N, L, 512)

# Embedding bag (efficient for bag-of-words)
emb_bag = nn.EmbeddingBag(num_embeddings=50000, embedding_dim=512, mode='mean')
```

## 5.12 Loss Functions

```python
# Classification
ce_loss = nn.CrossEntropyLoss()             # Combines LogSoftmax + NLLLoss
bce_loss = nn.BCEWithLogitsLoss()           # Binary cross-entropy
focal_loss = None                            # Not built-in, see below

# Regression
mse_loss = nn.MSELoss()                     # Mean squared error
l1_loss = nn.L1Loss()                       # Mean absolute error
huber_loss = nn.HuberLoss(delta=1.0)        # Smooth L1

# Distribution
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Ranking / Metric Learning
triplet_loss = nn.TripletMarginLoss(margin=1.0)
cosine_loss = nn.CosineEmbeddingLoss()

# Focal loss implementation (for class imbalance)
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()
```

## 5.13 Attention Mechanisms & FlexAttention

### Scaled Dot-Product Attention (SDPA)

```python
import torch.nn.functional as F

Q = torch.randn(2, 8, 1024, 64)  # (B, H, N, D)
K = torch.randn(2, 8, 1024, 64)
V = torch.randn(2, 8, 1024, 64)

# Automatically selects best backend: Flash, Memory-Efficient, cuDNN, or Math
output = F.scaled_dot_product_attention(
    Q, K, V,
    is_causal=True,
    dropout_p=0.0
)
```

### SDPA Backend Control

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force a specific backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Available backends:
# SDPBackend.FLASH_ATTENTION      — O(N) memory, fastest for long sequences
# SDPBackend.EFFICIENT_ATTENTION  — Memory-efficient attention
# SDPBackend.CUDNN_ATTENTION      — cuDNN backend
# SDPBackend.MATH                 — Standard implementation (fallback)
# SDPBackend.OVERRIDEABLE         — For extensions
```

### FlexAttention — Programmable Attention

FlexAttention allows arbitrary attention patterns through `score_mod` and `mask_mod` functions:

```python
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
    or_masks,
    noop_mask,
)

# Basic usage — causal attention
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=2, H=8, Q_LEN=1024, KV_LEN=1024)
output = flex_attention(Q, K, V, block_mask=block_mask)

# Score modification — relative position bias
def alibi_score_mod(score, b, h, q_idx, kv_idx):
    bias = -torch.abs(q_idx - kv_idx).float()
    slope = 1.0 / (2 ** ((h + 1) * 8.0 / n_heads))
    return score + bias * slope

output = flex_attention(Q, K, V, score_mod=alibi_score_mod, block_mask=block_mask)

# Sliding window attention
def sliding_window_mask(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= window_size

# Combine masks
combined = and_masks(causal_mask, sliding_window_mask)
block_mask = create_block_mask(combined, B=2, H=8, Q_LEN=1024, KV_LEN=1024)
```

FlexAttention compiles the score/mask functions into fused Triton kernels for maximum performance, enabling custom attention patterns without manual CUDA code.

## 5.14 Functional API

```python
import torch.nn.functional as F

# Activations
y = F.relu(x)
y = F.gelu(x)
y = F.silu(x)
y = F.softmax(x, dim=-1)

# Losses
loss = F.cross_entropy(logits, targets)
loss = F.mse_loss(pred, target)
loss = F.binary_cross_entropy_with_logits(logits, targets)

# Layers
y = F.linear(x, weight, bias)
y = F.conv2d(x, weight, bias, stride=1, padding=1)
y = F.batch_norm(x, running_mean, running_var, weight, bias, training=True)
y = F.layer_norm(x, normalized_shape=[512], weight=w, bias=b)
y = F.dropout(x, p=0.1, training=True)

# Attention
y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

## 5.15 Weight Initialization

```python
import torch.nn.init as init

# Xavier/Glorot (good for tanh, sigmoid)
init.xavier_uniform_(linear.weight)
init.xavier_normal_(linear.weight)

# Kaiming/He (good for ReLU family)
init.kaiming_uniform_(conv.weight, mode='fan_out', nonlinearity='relu')
init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

# Custom initialization for a model
def init_weights(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        init.ones_(module.weight)
        init.zeros_(module.bias)

model.apply(init_weights)
```

---

# Part VI: Optimization

## 6.1 Optimizer Base Class & Parameter Groups

```python
# Basic usage
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Parameter groups — different LR for different parts
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3},
], weight_decay=0.01)

# Optimizer loop
for data, target in dataloader:
    optimizer.zero_grad(set_to_none=True)   # More efficient
    loss = criterion(model(data), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

## 6.2 SGD

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,          # Momentum
    weight_decay=1e-4,     # L2 regularization
    nesterov=True          # Nesterov momentum
)
```

Algorithm:
$v_{t+1} = \mu v_t + g_t + \lambda\theta_t$ (with weight decay)
$\theta_{t+1} = \theta_t - \eta v_{t+1}$ (Nesterov: uses lookahead gradient)

## 6.3 Adam Family

```python
# Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW (decoupled weight decay — recommended)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Key difference:
# Adam: adds weight_decay * param to gradient (L2 regularization)
# AdamW: subtracts weight_decay * lr * param directly (true weight decay)
```

## 6.4 Learning Rate Schedulers

```python
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts,
    LinearLR, SequentialLR
)

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# OneCycle (recommended for training from scratch)
scheduler = OneCycleLR(
    optimizer, max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

# Warmup + Cosine decay
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# Reduce on plateau (for fine-tuning)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# scheduler.step(val_loss)  — called with metric value
```

## 6.5 Additional Optimizers

```python
# Adafactor — memory-efficient optimizer (no per-parameter momentum storage)
from torch.optim import Adafactor
optimizer = Adafactor(model.parameters(), lr=1e-3)

# Muon — momentum-based optimizer with unit normalization
from torch.optim._muon import Muon
optimizer = Muon(model.parameters(), lr=0.02)
```

## 6.6 Compiled Optimizers

For use with `torch.compile`, optimizers can be compiled for better fusion with the backward pass:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# The fused/foreach variants are used automatically when possible
# For explicit compiled optimizer usage with torch.compile:
model = torch.compile(model)

# The optimizer step is automatically fused with backward when using
# compiled autograd
```

---

# Part VII: Data Loading & Processing

## 7.1 Dataset & DataLoader Architecture

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# DataLoader with best practices
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,                # Shuffle for training
    num_workers=4,               # Parallel data loading
    pin_memory=True,             # Faster CPU→GPU transfer
    persistent_workers=True,     # Keep workers alive between epochs
    prefetch_factor=2,           # Prefetch 2 batches per worker
    drop_last=True               # Drop incomplete last batch
)
```

## 7.2 Custom Collate Functions

```python
def variable_length_collate(batch):
    """Custom collate for variable-length sequences."""
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

loader = DataLoader(dataset, collate_fn=variable_length_collate)
```

## 7.3 Data Augmentation

```python
import torchvision.transforms.v2 as T

transform_train = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MixUp and CutMix
def mixup(data, targets, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    indices = torch.randperm(data.size(0))
    mixed_data = lam * data + (1 - lam) * data[indices]
    return mixed_data, targets, targets[indices], lam
```

---

# Part VIII: Training Pipelines

## 8.1 Complete Training Loop Template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def train(config, model, train_loader, val_loader):
    device = config.device
    model = model.to(device)

    # Optional: compile model
    if config.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs
    )
    scaler = GradScaler('cuda') if config.use_amp and device.type == 'cuda' else None

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast('cuda', dtype=torch.float16):
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device, scaler is not None)

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with autocast('cuda', enabled=use_amp):
            output = model(data)
            loss = F.cross_entropy(output, target)
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, 100.0 * correct / total
```

## 8.2 Mixed Precision Training

```python
from torch.amp import autocast, GradScaler

# Modern API (device-agnostic)
scaler = GradScaler('cuda')

for data, target in loader:
    optimizer.zero_grad(set_to_none=True)

    with autocast('cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# BFloat16 — no scaler needed (no inf/nan issues)
with autocast('cuda', dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
```

## 8.4 Gradient Checkpointing (Activation Checkpointing)

Trades compute for memory — recomputes activations during backward instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=1024, nhead=16, batch_first=True)
            for _ in range(24)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

## 8.5 Transfer Learning & Fine-Tuning

```python
# Load pretrained and freeze backbone
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace and train only the head
model.fc = nn.Linear(2048, num_classes)

# Unfreeze later for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Use different learning rates
optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

## 8.7 EMA (Exponential Moving Average)

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])
```

---

# Part IX: Compilation & Performance (torch.compile)

## 9.1 torch.compile Overview

`torch.compile` is PyTorch's JIT compiler that optimizes models without code changes:

```python
model = MyModel()
compiled_model = torch.compile(model)

# Or as decorator
@torch.compile
def fast_fn(x):
    return x.sin() + x.cos()

# Compilation modes
torch.compile(model, mode="default")                    # Balanced
torch.compile(model, mode="reduce-overhead")             # CUDA graphs for minimal overhead
torch.compile(model, mode="max-autotune")                # Max optimization (slower compile)
torch.compile(model, mode="max-autotune-no-cudagraphs")  # Autotune without CUDA graphs
torch.compile(model, mode="lite")                        # Selective decomposition / regional compile

# Compiler stances — control recompilation behavior
torch.compiler.set_stance("default")              # Normal behavior
torch.compiler.set_stance("eager_on_recompile")   # Fall back to eager on recompile
torch.compiler.set_stance("fail_on_recompile")    # Error if recompilation needed
torch.compiler.set_stance("force_eager")           # Disable compilation entirely
```

### How torch.compile Works

```
Python Code
    │
    ▼
┌─────────────────────┐
│  TorchDynamo        │  → Captures Python bytecode into FX graph
│  (Graph Capture)    │  → Generates "guards" for when to recompile
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  AOTAutograd         │  → Traces forward AND backward
│  (Differentiation)  │  → Produces separate forward/backward graphs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  TorchInductor       │  → Generates optimized Triton/C++ code
│  (Code Generation)  │  → Operator fusion, memory planning
└─────────────────────┘
    │
    ▼
Optimized Kernel (runs on GPU/CPU)
```

## 9.2 TorchDynamo — Graph Capture

Dynamo intercepts Python bytecode execution and converts it to an FX graph:

```python
import torch._dynamo as dynamo

# What causes graph breaks (falling back to eager)
def graph_break_example(x):
    y = x + 1
    print(y)          # Graph break! (side effect)
    z = y * 2
    return z

# Avoid graph breaks
def no_break(x):
    y = x + 1
    z = y * 2
    return z

# Force fullgraph (error on graph breaks)
@torch.compile(fullgraph=True)
def strict_fn(x):
    return x.sin() + x.cos()

# Inspect what Dynamo captures
def explain_fn(x):
    return x + 1

explanation = torch._dynamo.explain(explain_fn)(torch.randn(10))
print(explanation)
```

### Guards and Recompilation

```python
@torch.compile
def fn(x):
    return x + 1

# Call 1: shape (3,4) → compile (cache miss)
fn(torch.randn(3, 4))

# Call 2: same shape → cache hit (fast!)
fn(torch.randn(3, 4))

# Call 3: different shape → may recompile
fn(torch.randn(5, 6))

# Dynamic shapes to avoid recompilation
@torch.compile(dynamic=True)
def fn_dynamic(x):
    return x + 1

# One compilation works for ANY shape
for size in [16, 32, 64, 128]:
    fn_dynamic(torch.randn(size, 256))
```

## 9.3 AOTAutograd & FX Graph IR

AOTAutograd traces both forward and backward at compile time:

```python
# FX Graph IR — the intermediate representation
import torch.fx

def example(x, y):
    z = x + y
    return z.relu()

# Trace to FX graph
graph = torch.fx.symbolic_trace(example)
print(graph.graph)
# graph():
#   %x : [#users=1] = placeholder[target=x]
#   %y : [#users=1] = placeholder[target=y]
#   %add : [#users=1] = call_function[target=operator.add](args=(%x, %y))
#   %relu : [#users=1] = call_method[target=relu](args=(%add,))
#   return relu
```

## 9.4 TorchInductor — Code Generation

Inductor generates optimized Triton kernels for GPU and C++/OpenMP for CPU:

```python
# See generated code
import torch._inductor.config as inductor_config
inductor_config.debug = True  # Prints generated Triton/C++ code

@torch.compile
def fused_fn(x, y):
    return (x + y).relu().mul(2)

# Inductor fuses this into a SINGLE Triton kernel:
# @triton.jit
# def fused_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel):
#     xoffset = tl.program_id(0) * XBLOCK
#     xindex = xoffset + tl.arange(0, XBLOCK)
#     x0 = tl.load(in_ptr0 + xindex)
#     x1 = tl.load(in_ptr1 + xindex)
#     tmp = x0 + x1
#     tmp = tl.where(tmp > 0, tmp, 0)  # relu
#     tmp = tmp * 2
#     tl.store(out_ptr0 + xindex, tmp)
```

### Inductor Backends

| Backend | Target | Description |
|---|---|---|
| Triton | NVIDIA GPU | Default GPU backend, generates Triton kernels |
| C++ | CPU | Generates OpenMP-parallelized C++ code |
| Halide | CPU/GPU | Experimental Halide-based backend |
| CUDA | NVIDIA GPU | Direct CUDA kernel generation |

## 9.5 Dynamic Shapes & Compilation Cache

```python
# Automatic dynamic shapes
@torch.compile
def fn(x):
    return x.sum(dim=-1)

# After seeing different shapes, Dynamo marks dimensions as dynamic
fn(torch.randn(3, 4))   # Compile
fn(torch.randn(3, 5))   # Recompile, mark dim 1 as dynamic
fn(torch.randn(3, 100)) # Cache hit! (dim 1 is dynamic)

# Explicit dynamic shapes
from torch._dynamo import mark_dynamic
x = torch.randn(32, 256)
mark_dynamic(x, 0)  # Batch dim is dynamic
fn(x)

# Persistent compilation cache
torch._inductor.config.fx_graph_cache = True        # In-memory cache
torch._inductor.config.autotune_in_subproc = True   # Parallel autotuning
```

## 9.6 torch.compile Config & Debugging

```python
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config

# Dynamo config
dynamo_config.cache_size_limit = 8        # Max cached compilations per site
dynamo_config.suppress_errors = False      # Set True in production

# Inductor config
inductor_config.debug = False              # Print generated code
inductor_config.max_autotune = False       # Try all implementations

# Debugging
torch._dynamo.config.verbose = True
torch._logging.set_logs(dynamo=logging.DEBUG)

# Reset compiled state
torch._dynamo.reset()

# Explain compilation
explanation = torch._dynamo.explain(model)(sample_input)
```

## 9.7 Writing Custom Triton Kernels

```python
import triton
import triton.language as tl

@triton.jit
def fused_add_relu_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    result = tl.where(x + y > 0, x + y, 0.0)
    tl.store(out_ptr + offset, result, mask=mask)

def fused_add_relu(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_add_relu_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out

# Register as custom op for use with torch.compile
from torch.library import custom_op

@custom_op("mylib::fused_add_relu", mutates_args=())
def fused_add_relu_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return fused_add_relu(x, y)
```

---

# Part X: Distributed Training

## 10.1 Distributed Overview

| Strategy | When to Use | Memory | Communication |
|---|---|---|---|
| DDP | Data fits on one GPU | Full model per GPU | Gradient all-reduce |
| FSDP2 | Model too large for one GPU | Sharded parameters | All-gather/reduce-scatter |
| Tensor Parallel | Single layer too large | Split within layers | All-reduce per layer |
| Pipeline Parallel | Very deep models | Split by layers | Point-to-point |

## 10.2 DeviceMesh — The Foundation

DeviceMesh is the standard way to describe device topologies in PyTorch distributed:

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh (simple data parallelism)
mesh = init_device_mesh("cuda", (world_size,))

# 2D mesh (data parallel + tensor parallel)
mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# Access sub-meshes
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]

# 3D mesh (DP + TP + PP)
mesh_3d = init_device_mesh(
    "cuda", (dp_size, pp_size, tp_size),
    mesh_dim_names=("dp", "pp", "tp")
)
```

## 10.3 DistributedDataParallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training is the same as single-GPU
for data, target in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(data.cuda()), target.cuda())
    loss.backward()
    optimizer.step()

dist.destroy_process_group()
```

Launch:
```bash
torchrun --nproc_per_node=4 train.py
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=host0 --master_port=29500 train.py
```

## 10.5 FSDP2 — fully_shard (The New Standard)

FSDP2 is the recommended API for sharded training, replacing FSDP1:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

# Mixed precision policy
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
)

# Apply FSDP2 composably
model = MyLargeModel()

# Shard individual submodules first
for layer in model.transformer.layers:
    fully_shard(layer, mp_policy=mp_policy)

# Then shard the root
fully_shard(model, mp_policy=mp_policy)

# Training is the same as single-GPU!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for data, target in train_loader:
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP2 Key Features

- **Composable**: Works with tensor parallel, pipeline parallel, compiled autograd
- **Per-parameter sharding**: Shards each parameter independently (no FlatParameter)
- **DTensor integration**: Uses DTensor for parameter representation
- **CPU Offload**: `CPUOffloadPolicy` to offload parameters to CPU

## 10.6 DTensor (torch.distributed.tensor)

DTensor is the distributed tensor abstraction used by FSDP2 and tensor parallelism:

```python
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (4,))

# Distribute a tensor
tensor = torch.randn(16, 32)
dtensor = DTensor.from_local(tensor, mesh, placements=[Shard(0)])

# Placement types:
# Shard(dim)   — tensor is sharded along given dimension
# Replicate()  — tensor is replicated on all devices
# Partial()    — tensor has partial results (needs reduction)

# distribute_tensor for automatic distribution
from torch.distributed.tensor import distribute_tensor
dtensor = distribute_tensor(tensor, mesh, placements=[Shard(0)])
```

## 10.7 Tensor Parallelism

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

# Parallelize a Transformer layer
tp_mesh = mesh_2d["tp"]

parallelize_module(
    model.transformer_layer,
    tp_mesh,
    {
        "attention.q_proj": ColwiseParallel(),
        "attention.k_proj": ColwiseParallel(),
        "attention.v_proj": ColwiseParallel(),
        "attention.out_proj": RowwiseParallel(),
        "ffn.fc1": ColwiseParallel(),
        "ffn.fc2": RowwiseParallel(),
    }
)
```

## 10.8 Pipeline Parallelism

```python
from torch.distributed.pipelining import (
    pipeline, SplitPoint, build_stage, PipelineStage,
    Schedule1F1B, ScheduleGPipe, ScheduleInterleavedZeroBubble,
    ScheduleZBVZeroBubble, ScheduleDualPipeV,
)

# Split model into stages
pipe = pipeline(
    model,
    num_chunks=4,   # Microbatches
    example_args=(torch.randn(16, 512),),
    split_spec={
        "layer_4": SplitPoint.END,    # Split after layer_4
        "layer_8": SplitPoint.END,    # Split after layer_8
    }
)

# Build stages for each rank
stage = build_stage(pipe, rank, device)

# Choose a schedule
schedule = Schedule1F1B(stage, n_microbatches=4)
# or ScheduleGPipe, ScheduleInterleavedZeroBubble, etc.

# Execute
if rank == 0:
    schedule.step(input_batch)
elif rank == last_rank:
    output = schedule.step()
else:
    schedule.step()
```

### Pipeline Schedules

| Schedule | Bubble Ratio | Memory | Description |
|---|---|---|---|
| GPipe | High | High (all micro) | All-forward then all-backward |
| 1F1B | Medium | Low (1 micro) | Alternating forward/backward |
| Interleaved1F1B | Lower | Medium | Multiple stages per rank |
| ZeroBubble | Near zero | Medium | Optimized scheduling |
| DualPipeV | Low | Medium | Bidirectional pipeline |

## 10.9 Collective Communications

```python
import torch.distributed as dist

# Point-to-point
dist.send(tensor, dst=1)
dist.recv(tensor, src=0)

# Collective operations
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.all_gather(output_list, tensor)
dist.reduce_scatter(output, input_list)
dist.broadcast(tensor, src=0)
dist.barrier()

# Process groups
group = dist.new_group(ranks=[0, 1, 2, 3])
dist.all_reduce(tensor, group=group)
```

## 10.10 torchrun & Elastic Training

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (2 nodes, 4 GPUs each)
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
    train.py

# Elastic training (auto-restart on failure)
torchrun --nproc_per_node=4 --nnodes=1:4 \
    --rdzv_backend=c10d --rdzv_endpoint=host:29500 \
    train.py
```

## 10.11 SymmetricMemory

SymmetricMemory provides efficient intra-node communication primitives:

```python
from torch.distributed._symmetric_memory import get_symm_mem_workspace

# Enables peer-to-peer memory access between GPUs on the same node
# Used internally by FSDP2 and tensor parallelism for optimized collectives
# Supports fused all-gather + matmul and FP8 all-gather operations
```

## 10.12 Context Parallel (Experimental)

Context parallel splits long sequences across devices — useful for LLM training with very long contexts:

```python
from torch.distributed.tensor.experimental import context_parallel

# Applies context parallelism to attention layers
# Splits the sequence dimension across the CP mesh dimension
# Uses ring or all-to-all communication patterns
```

## 10.13 Distributed Checkpointing (DCP)

```python
from torch.distributed.checkpoint import save, load, async_save
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict, StateDictOptions
)

# Save distributed state (works with FSDP2, DDP, DTensor)
state = {"model": get_model_state_dict(model), "optim": get_optimizer_state_dict(model, optimizer)}
save(state, checkpoint_id="epoch_5")

# Async save (non-blocking)
f = async_save(state, checkpoint_id="epoch_5")
# ... continue training ...
f.result()  # Wait for completion

# Load
load(state, checkpoint_id="epoch_5")

# HuggingFace safetensors format
from torch.distributed.checkpoint import HuggingFaceStorageReader, HuggingFaceStorageWriter
save(state, storage_writer=HuggingFaceStorageWriter("/path/to/hf_checkpoint"))
```

---

# Part XI: torch.export & Deployment

## 11.1 torch.export API

`torch.export` captures a PyTorch model into a graph representation suitable for deployment:

```python
import torch.export

model = MyModel()
example_inputs = (torch.randn(1, 3, 224, 224),)

# Export with static shapes
exported = torch.export.export(model, example_inputs)

# Export with dynamic shapes
from torch.export import Dim
batch = Dim("batch", min=1, max=128)
exported = torch.export.export(
    model, example_inputs,
    dynamic_shapes={"x": {0: batch}}
)

# Draft export — returns diagnostics on failure instead of raising
from torch.export import draft_export
result = draft_export(model, example_inputs)

# Inspect the exported program
print(exported.graph_module.graph)

# Run the exported model
output = exported.module()(torch.randn(4, 3, 224, 224))

# Save and load (PT2 Archive format)
torch.export.save(exported, "model.pt2")
loaded = torch.export.load("model.pt2")
```

### PT2 Archive Format

The PT2 archive is a zip-based format containing:
- Exported program graph
- Model weights (with deduplication)
- Constants and sample inputs
- Optional AOTInductor artifacts
- Optional ExecuTorch payloads

## 11.2 Export Constraints & Dynamic Shapes

```python
from torch.export import Dim, dims

batch, seq_len = dims("batch", "seq_len", min=1, max=2048)

exported = torch.export.export(
    model,
    (torch.randn(1, 128, 512),),
    dynamic_shapes={"x": {0: batch, 1: seq_len}}
)
```

## 11.3 AOTInductor — C++ Deployment

AOTInductor compiles an exported model ahead-of-time into a shared library:

```python
# Step 1: Export
exported = torch.export.export(model, example_inputs)

# Step 2: Compile to shared library
so_path = torch._inductor.aot_compile(
    exported.module(),
    example_inputs,
    options={"aot_inductor.output_path": "model.so"}
)

# Step 3: Load in C++ for inference
# #include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
# auto runner = AOTIModelContainerRunnerCuda("model.so");
# auto outputs = runner.run(inputs);
```

## 11.4 NativeRT — C++ Inference Engine

NativeRT is a flexible C++ inference engine for torch-exported models, designed as a drop-in replacement for Static Runtime:

```cpp
#include <nativert/core/ModelRunner.h>

int main() {
    auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
        std::make_unique<caffe2::serialize::FileAdapter>("/path/to/model"));

    auto runner = ModelRunner(
        std::move(reader),
        "my_model",
        ExecutorType::INTERPRETER,
        RuntimeConfigs{},
        Placement(torch::Device(torch::kCUDA, 0)));

    const auto [args, kwargs] = runner.loadSampleInputs(reader, placement);
    auto output = runner.run(args, kwargs);
    return 0;
}
```

Key features:
- Integrates with the torch dispatcher (supports all backends)
- Supports AOTInductor-lowered artifacts
- Static dispatch kernels for reduced overhead
- Memory planning and constant folding
- Inter-op parallelism via thread pool

## 11.5 ONNX Export

```python
import torch.onnx

# TorchDynamo-based export (recommended)
torch.onnx.export(
    model,
    example_inputs,
    "model.onnx",
    dynamo=True,              # Use Dynamo-based export
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
```

---

# Part XII: Hardware Acceleration

## 12.1 CUDA Programming

```python
# Device management
device = torch.device('cuda', 0)
torch.cuda.set_device(0)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Streams — concurrent execution
default_stream = torch.cuda.current_stream()
compute_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    y = model(x)

# Synchronize
torch.cuda.synchronize()

# Events — timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
y = model(x)
end.record()
torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end):.2f} ms")

# CUDA Graphs — capture and replay kernels
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = model(static_input)
# Replay
static_input.copy_(new_data)
g.replay()
```

## 12.2 cuDNN Integration

```python
# Enable cuDNN benchmark mode (auto-tune for fixed input sizes)
torch.backends.cudnn.benchmark = True

# Deterministic mode
torch.backends.cudnn.deterministic = True

# Check availability
print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
```

## 12.3 Metal Performance Shaders (MPS) — Apple Silicon

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    x = x.to(device)
    y = model(x)
```

## 12.4 Intel GPU (XPU)

```python
if torch.xpu.is_available():
    device = torch.device("xpu")
    model = model.to(device)
    x = x.to(device)
    y = model(x)
```

## 12.6 torch.accelerator — Device-Agnostic API

```python
import torch.accelerator

# Works across CUDA, XPU, MPS, MTIA, PrivateUse1
if torch.accelerator.is_available():
    device = torch.accelerator.current_device()
    torch.accelerator.synchronize()
```

## 12.7 Custom Backends (PrivateUse1)

```python
from torch.utils.backend_registration import rename_privateuse1_backend

# Register a custom hardware backend
rename_privateuse1_backend("my_accelerator")
# Now torch.device("my_accelerator") works
```

---

# Part XIII: Advanced Features

## 13.1 torch.fx — Graph Transformation

```python
import torch.fx

# Symbolic tracing
def my_fn(x, y):
    return (x + y).relu()

traced = torch.fx.symbolic_trace(my_fn)
print(traced.graph)

# Graph transformation — add logging
for node in traced.graph.nodes:
    if node.op == 'call_function' and node.target == torch.relu:
        with traced.graph.inserting_before(node):
            new_node = traced.graph.call_function(print, args=(node.args[0],))

traced.recompile()
```

## 13.2 Functorch (vmap, grad, etc.)

```python
from torch.func import vmap, grad, jacrev, hessian

# vmap — vectorized map (auto-batching)
def single_loss(x, y):
    return ((x - y) ** 2).sum()

batched_loss = vmap(single_loss)
losses = batched_loss(batch_x, batch_y)

# Per-sample gradients
per_sample_grad = vmap(grad(single_loss))
grads = per_sample_grad(batch_x, batch_y)

# Compose transforms
per_sample_jacobian = vmap(jacrev(model_fn))
```

## 13.3 Sparse Tensors

```python
# COO format
indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

# CSR format
crow_indices = torch.tensor([0, 1, 2, 3])
col_indices = torch.tensor([0, 1, 2])
values = torch.tensor([1.0, 2.0, 3.0])
sparse_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, (3, 3))

# Sparse matrix multiplication
result = torch.sparse.mm(sparse, dense_matrix)
```

## 13.5 Quantization

### Post-Training Quantization (PT2E — recommended)

```python
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.export

# Step 1: Export
exported = torch.export.export(model, example_inputs)

# Step 2: Prepare with quantizer
quantizer = X86InductorQuantizer()
quantizer.set_global(quantizer.get_default_quantization_config())
prepared = prepare_pt2e(exported, quantizer)

# Step 3: Calibrate
for data, _ in calibration_loader:
    prepared(data)

# Step 4: Convert
quantized = convert_pt2e(prepared)

# Step 5: Use with torch.compile
compiled_quantized = torch.compile(quantized)
```

---

# Part XIV: Model Architectures

## 14.1 ResNet Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

def resnet18():  return ResNet(BasicBlock, [2, 2, 2, 2])
def resnet34():  return ResNet(BasicBlock, [3, 4, 6, 3])
def resnet50():  return ResNet(Bottleneck, [3, 4, 6, 3])
def resnet101(): return ResNet(Bottleneck, [3, 4, 23, 3])
def resnet152(): return ResNet(Bottleneck, [3, 8, 36, 3])
```

## 14.2 Transformer with Flash Attention

```python
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None, is_causal=False):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_k)
        Q, K, V = qkv.unbind(0)

        # Uses Flash Attention automatically
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, is_causal=False):
        x = x + self.attn(self.norm1(x), is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x
```

## 14.3 GPT with FlexAttention

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=2048, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying
        self.max_len = max_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x, is_causal=True)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# GPT-2 configurations
def gpt2_small():  return GPT(50257, d_model=768, num_heads=12, num_layers=12)
def gpt2_medium(): return GPT(50257, d_model=1024, num_heads=16, num_layers=24)
def gpt2_large():  return GPT(50257, d_model=1280, num_heads=20, num_layers=36)
def gpt2_xl():     return GPT(50257, d_model=1600, num_heads=25, num_layers=48)
```

## 14.4 Vision Transformer (ViT)

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)

# ViT configurations
def vit_base():  return ViT(embed_dim=768, depth=12, num_heads=12)   # 86M
def vit_large(): return ViT(embed_dim=1024, depth=24, num_heads=16)  # 307M
def vit_huge():  return ViT(embed_dim=1280, depth=32, num_heads=16)  # 632M
```

---

# Part XV: Testing & Debugging

## 15.1 Testing Framework

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestMyOp(TestCase):
    def test_add(self):
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = x + y
        expected = torch.add(x, y)
        self.assertEqual(result, expected)

    def test_grad(self):
        x = torch.randn(3, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        self.assertEqual(x.grad, 2 * x)

    def test_device(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            x = torch.randn(3, device=device)
            self.assertEqual(x.device.type, device)

if __name__ == "__main__":
    run_tests()
```

## 15.3 Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.nn.Linear(1000, 1000).cuda()
x = torch.randn(64, 1000, device='cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_forward"):
        y = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# TensorBoard integration
prof.export_chrome_trace("trace.json")

# Scheduled profiler (for training loops)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
) as prof:
    for step, (data, target) in enumerate(train_loader):
        output = model(data.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
        prof.step()
```

## 15.4 Common Errors & Solutions

| Error | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, use gradient checkpointing, AMP |
| `Expected all tensors on same device` | Mixed CPU/GPU tensors | `.to(device)` all inputs and model |
| `element 0 does not require grad` | Missing `requires_grad` | Set `requires_grad=True` or check model parameters |
| `one of the variables needed has been modified by an inplace operation` | In-place op breaks autograd graph | Avoid in-place ops on tensors that need grad |
| `graph break` in torch.compile | Unsupported Python construct | Use `fullgraph=True` to find breaks, refactor code |

## 15.5 Reproducibility

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)
```

---

# Part XVI: Build System & Contributing

## 16.1 Build from Source

```bash
# Clone
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Install dependencies
pip install -r requirements.txt
pip install cmake ninja

# Build (development mode)
python setup.py develop

# Or faster incremental builds
MAX_JOBS=8 python setup.py develop
```

## 16.2 Code Generation (torchgen)

Operators are defined in YAML and code-generated:

```yaml
# aten/src/ATen/native/native_functions.yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
    MPS: add_mps
    Meta: add_meta
```

Run code generation:
```bash
python torchgen/gen.py
```

## 16.3 Adding New Operators

```python
# Python-based operator registration
import torch
from torch.library import Library, impl

lib = Library("mylib", "DEF")
lib.define("my_op(Tensor x, float scale) -> Tensor")

@impl(lib, "my_op", "CPU")
def my_op_cpu(x, scale):
    return x * scale

@impl(lib, "my_op", "Meta")
def my_op_meta(x, scale):
    return torch.empty_like(x)

# Register autograd
from torch.autograd import Function

class MyOpFunction(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return torch.ops.mylib.my_op(x, scale)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
```

## 16.4 C++ Extensions

```python
from torch.utils.cpp_extension import load

# JIT compilation
module = load(
    name="my_extension",
    sources=["my_extension.cpp", "my_extension_cuda.cu"],
    extra_cuda_cflags=["-O2"]
)

# Use it
output = module.forward(input_tensor)
```

---

# Appendices

## Appendix A: Complete Mathematical Derivations

### A.1 Loss Function Gradients

**Cross-Entropy with Softmax**:
$L = -\log(\text{softmax}(z)_y) = -z_y + \log(\sum_j e^{z_j})$

$\frac{\partial L}{\partial z_i} = \text{softmax}(z)_i - \mathbb{1}[i = y]$

**MSE Loss**:
$L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$

$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$

**Binary Cross-Entropy**:
$L = -[y \log(\sigma(z)) + (1-y)\log(1-\sigma(z))]$

$\frac{\partial L}{\partial z} = \sigma(z) - y$

### A.2 Activation Gradients

| Activation | $f(x)$ | $f'(x)$ |
|---|---|---|
| ReLU | $\max(0, x)$ | $\mathbb{1}[x > 0]$ |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ |
| GELU | $x \cdot \Phi(x)$ | $\Phi(x) + x \cdot \phi(x)$ |
| SiLU | $x \cdot \sigma(x)$ | $\sigma(x)(1 + x(1-\sigma(x)))$ |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $s_i(\delta_{ij} - s_j)$ |

### A.3 Optimizer Update Rules

**SGD with Momentum**:
$v_t = \mu v_{t-1} + g_t$
$\theta_t = \theta_{t-1} - \eta v_t$

**Adam**:
$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
$\hat{m}_t = m_t/(1-\beta_1^t)$
$\hat{v}_t = v_t/(1-\beta_2^t)$
$\theta_t = \theta_{t-1} - \eta\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$

### A.4 Attention Mathematics

**Scaled Dot-Product Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Flash Attention Algorithm** (O(N) memory):
1. Divide Q into blocks of size $B_r$, K/V into blocks of size $B_c$
2. For each Q block, iterate over K/V blocks
3. Use online softmax trick to avoid materializing the N×N attention matrix
4. Correction factor: $c = \exp(m_{old} - m_{new})$

---

## Appendix B: Complete torch.nn Module Reference

### Linear Layers
| Module | Formula | Parameters |
|---|---|---|
| `nn.Linear(in, out)` | $y = xW^T + b$ | W:(out,in), b:(out,) |
| `nn.Bilinear(in1, in2, out)` | $y = x_1^T A x_2 + b$ | A:(out,in1,in2) |
| `nn.LazyLinear(out)` | Same as Linear | Inferred in_features |

### Convolution Layers
| Module | Input | Output |
|---|---|---|
| `nn.Conv1d(C_in, C_out, K)` | (N, C_in, L) | (N, C_out, L_out) |
| `nn.Conv2d(C_in, C_out, K)` | (N, C_in, H, W) | (N, C_out, H_out, W_out) |
| `nn.Conv3d(C_in, C_out, K)` | (N, C_in, D, H, W) | (N, C_out, D_out, H_out, W_out) |
| `nn.ConvTranspose2d(C_in, C_out, K)` | (N, C_in, H, W) | (N, C_out, H_out, W_out) |

### Normalization Layers
| Module | Normalizes Over | Use Case |
|---|---|---|
| `nn.BatchNorm2d(C)` | Batch + Spatial | CNNs |
| `nn.LayerNorm(D)` | Last dim(s) | Transformers |
| `nn.GroupNorm(G, C)` | Channel groups | Small batch CNNs |
| `nn.InstanceNorm2d(C)` | Spatial per instance | Style transfer |
| `nn.RMSNorm(D)` | Last dim (RMS) | LLMs |

### Activation Functions
| Module | Formula | Common Use |
|---|---|---|
| `nn.ReLU()` | $\max(0,x)$ | CNNs (hidden) |
| `nn.GELU()` | $x\Phi(x)$ | Transformers |
| `nn.SiLU()` | $x\sigma(x)$ | LLMs (FFN) |
| `nn.Mish()` | $x\tanh(\text{softplus}(x))$ | Modern CNNs |
| `nn.Softmax(dim)` | $e^{x_i}/\sum e^{x_j}$ | Output (classification) |
| `nn.Sigmoid()` | $1/(1+e^{-x})$ | Output (binary) |

### Recurrent Layers
| Module | Hidden | Output |
|---|---|---|
| `nn.RNN(in, hidden)` | $h_t = \tanh(W_x x_t + W_h h_{t-1})$ | (N, L, H) |
| `nn.LSTM(in, hidden)` | 4 gates: input, forget, cell, output | (N, L, H), (h, c) |
| `nn.GRU(in, hidden)` | 2 gates: reset, update | (N, L, H), h |

---

## Appendix C: torch.linalg Reference

```python
import torch.linalg as LA

A = torch.randn(3, 3)
b = torch.randn(3)

LA.norm(A)                    # Frobenius norm
LA.norm(A, ord=2)             # Spectral norm
LA.det(A)                     # Determinant
LA.inv(A)                     # Inverse
LA.solve(A, b)                # Solve Ax = b
LA.eigh(A @ A.T)              # Eigendecomposition (symmetric)
LA.svd(A)                     # SVD
LA.qr(A)                      # QR decomposition
LA.cholesky(A @ A.T + 0.1*torch.eye(3))  # Cholesky
LA.matrix_rank(A)             # Rank
LA.cond(A)                    # Condition number
LA.pinv(A)                    # Pseudo-inverse
LA.cross(torch.randn(3), torch.randn(3))  # Cross product
```

---

## Appendix D: torch.distributions Reference

```python
from torch.distributions import *

Normal(0, 1).sample((100,))
Bernoulli(0.5).sample((100,))
Categorical(torch.tensor([0.1, 0.3, 0.6])).sample((100,))
Uniform(0, 1).sample((100,))
Exponential(1.0).sample((100,))
Beta(2, 5).sample((100,))
Gamma(2, 1).sample((100,))
Dirichlet(torch.ones(5)).sample((100,))
MultivariateNormal(torch.zeros(3), torch.eye(3)).sample((100,))

# KL divergence between distributions
p = Normal(0, 1)
q = Normal(1, 2)
kl = torch.distributions.kl_divergence(p, q)
```

---

## Appendix E: torch.fft / torch.special Reference

```python
# FFT
x = torch.randn(1024)
X = torch.fft.fft(x)           # Complex DFT
X = torch.fft.rfft(x)          # Real-input DFT
x_back = torch.fft.ifft(X)     # Inverse DFT
X_2d = torch.fft.fft2(img)     # 2D DFT

# Special functions
torch.special.erf(x)           # Error function
torch.special.erfc(x)          # Complementary error function
torch.special.gammaln(x)       # Log gamma
torch.special.digamma(x)       # Digamma
torch.special.softmax(x, dim=0)
torch.special.log_softmax(x, dim=0)
torch.special.expit(x)         # Sigmoid
torch.special.logit(x)         # Logit (inverse sigmoid)
```

---

## Appendix F: Serialization

```python
# Save/Load state dict (recommended)
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', weights_only=True))

# Save/Load full checkpoint
checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'best_acc': best_acc,
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
ckpt = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])

# weights_only=True (secure — only loads tensors, no arbitrary objects)
state = torch.load('model.pt', weights_only=True)

# Distributed checkpointing
from torch.distributed.checkpoint import save, load
save({"model": model.state_dict()}, checkpoint_id="epoch_5")
```

---

## Appendix G: Performance Optimization Checklist

### Data Loading
- [ ] `num_workers > 0` (typically 4-8)
- [ ] `pin_memory=True` for GPU training
- [ ] `persistent_workers=True`
- [ ] `prefetch_factor=2`

### Model
- [ ] Use `torch.compile(model)` for 2x+ speedup
- [ ] Mixed precision (`torch.amp.autocast`)
- [ ] Gradient checkpointing for large models
- [ ] `torch.backends.cudnn.benchmark = True`

### Training Loop
- [ ] `optimizer.zero_grad(set_to_none=True)`
- [ ] `torch.no_grad()` / `torch.inference_mode()` for eval
- [ ] Gradient clipping (`clip_grad_norm_`)
- [ ] Effective batch size via gradient accumulation

### Memory
- [ ] Monitor with `torch.cuda.memory_allocated()`
- [ ] `torch.cuda.empty_cache()` if needed
- [ ] Use `del` for large intermediate tensors
- [ ] Consider CPU offloading for very large models

### Distributed
- [ ] NCCL backend for GPU training
- [ ] FSDP2 (`fully_shard`) for model parallelism
- [ ] `DistributedSampler` for data parallelism
- [ ] Overlap communication and computation

---

## Appendix H: Debugging Guide

### Detecting NaN/Inf

```python
# Anomaly detection (slow but catches NaN sources)
torch.autograd.set_detect_anomaly(True)

# Manual checks
def check_tensor(name, t):
    if torch.isnan(t).any():
        print(f"NaN detected in {name}!")
    if torch.isinf(t).any():
        print(f"Inf detected in {name}!")
```

### Gradient Flow Verification

```python
def check_grad_flow(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"Zero gradient: {name}")
            elif grad_norm > 100:
                print(f"Exploding gradient: {name} (norm={grad_norm:.2f})")
        elif param.requires_grad:
            print(f"No gradient: {name}")
```

### torch.compile Debugging

```python
# See what's happening
import logging
torch._logging.set_logs(dynamo=logging.DEBUG)

# Find graph breaks
torch._dynamo.config.verbose = True

# Explain compilation
explanation = torch._dynamo.explain(fn)(sample_input)
print(explanation)

# Disable compilation temporarily
torch._dynamo.config.suppress_errors = True  # Fall back to eager on error
```

---

## Appendix I: Quick Reference Tables

### Tensor Dimension Conventions

| Domain | Format | Example Shape |
|---|---|---|
| Image (2D) | (N, C, H, W) | (32, 3, 224, 224) |
| Image (3D) | (N, C, D, H, W) | (16, 1, 64, 128, 128) |
| Sequence (NLP) | (N, L, D) | (32, 512, 768) |
| Audio | (N, C, L) | (32, 1, 16000) |
| Point Cloud | (N, P, C) | (16, 1024, 3) |
| Attention | (N, H, L, D) | (32, 12, 512, 64) |

### Common Hyperparameters

| Parameter | Typical Range | Notes |
|---|---|---|
| Learning Rate | 1e-5 to 1e-1 | AdamW: 1e-4 to 3e-4 for transformers |
| Batch Size | 16 - 4096 | Larger = faster but may need LR scaling |
| Weight Decay | 1e-5 to 0.1 | 0.01 - 0.1 for AdamW |
| Dropout | 0.0 - 0.5 | 0.1 for transformers, higher for small data |
| Gradient Clip | 0.5 - 5.0 | 1.0 is common |
| Warmup Steps | 100 - 10000 | ~1-5% of total steps |

### PyTorch Ecosystem

| Library | Description |
|---|---|
| **torchvision** | Computer vision: models, datasets, transforms |
| **torchaudio** | Audio processing and models |
| **torchtext** | NLP utilities |
| **torchao** | Architecture optimization (quantization, sparsity) |
| **TorchServe** | Model serving |
| **TorchTune** | LLM fine-tuning |
| **TorchTitan** | Large-scale training |
| **ExecuTorch** | On-device inference |
| **TorchChat** | Conversational AI |

---

**End of PyTorch: The Definitive Master Reference (2026 Edition)**

*This document covers PyTorch v2.13+ from mathematical foundations through production deployment. It consolidates and updates all previous reference materials with the latest APIs including FlexAttention, FSDP2, NativeRT, Pipeline Parallelism, DeviceMesh, Compiled Autograd, and the full torch.compile stack.*

*Version: June 2026 | PyTorch Main Branch (v2.13.0a0+)*
