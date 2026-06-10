<div align="center">

[🏠 Home](../README.md) | [Next Module →](../02_tensors/)

</div>

---

> **Module 01** of the PyTorch Complete Learning Guide
> **Prerequisites:** None (start here)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `math_with_pytorch.py` | Mathematical foundations with PyTorch |

---

# Module 01: Foundations — PyTorch and the Mathematics of Deep Learning

## Table of Contents
1. [What is PyTorch?](#what-is-pytorch)
2. [Installation](#installation)
3. [Core Philosophy](#core-philosophy)
4. [Architecture Stack](#architecture-stack)
5. [Mathematical Prerequisites](#mathematical-prerequisites)
6. [Linear Algebra](#linear-algebra)
7. [Calculus for Deep Learning](#calculus-for-deep-learning)
8. [Probability and Statistics](#probability-and-statistics)
9. [Optimization Theory](#optimization-theory)
10. [Information Theory](#information-theory)

---

## What is PyTorch?

PyTorch is an open-source machine learning framework developed primarily by Meta AI
(formerly Facebook AI Research). It provides two core capabilities:

1. **N-dimensional tensor computation** — similar to NumPy but with GPU acceleration
2. **Automatic differentiation** — computes gradients of arbitrary computational graphs

### A Brief History

- **2002**: Torch was created in Lua at NYU by Ronan Collobert and others.
- **2016**: PyTorch 0.1 was released by Facebook AI Research (FAIR), bringing the
  Torch tensor library to Python with automatic differentiation built in.
- **2018**: PyTorch 1.0 merged the research-focused PyTorch with the production-focused
  Caffe2, adding TorchScript for model export.
- **2022**: PyTorch 2.0 introduced `torch.compile()`, a compiler-based approach that
  can dramatically speed up models with a single line of code.
- **2023**: PyTorch moved to the Linux Foundation, becoming a truly community-governed project.
- **2024-2025**: Continued evolution with FlexAttention, torch.export improvements,
  and expanded hardware support (Intel XPU, Apple MPS, AMD ROCm).

### PyTorch vs TensorFlow

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Execution** | Eager by default (define-by-run) | Historically graph-based (define-then-run), now eager via tf.function |
| **Debugging** | Standard Python debugger works | Harder to debug graph mode |
| **Research adoption** | Dominant in academia (~80%+ of papers) | Strong in industry/production |
| **Deployment** | TorchServe, ONNX, torch.export | TF Serving, TFLite, TF.js |
| **API Style** | Pythonic, object-oriented | Keras-based high-level API |
| **Compilation** | torch.compile (TorchDynamo + Inductor) | XLA compiler |
| **Community** | Massive open-source ecosystem (HuggingFace, etc.) | Google-driven ecosystem |

**Why PyTorch won in research**: The key reason is *debuggability*. When your model
produces NaN values or wrong outputs, you can insert a `breakpoint()` anywhere in
your forward pass, inspect tensors, and understand what's happening. In graph-based
frameworks, you're debugging a compiled representation, not your original code.

### The PyTorch Ecosystem

PyTorch is not just one library — it's a constellation:

- **torchvision**: Computer vision (datasets, models, transforms)
- **torchaudio**: Audio processing
- **torchtext**: NLP utilities
- **PyTorch Lightning / Fabric**: Training framework that reduces boilerplate
- **HuggingFace Transformers**: Built on PyTorch, the dominant NLP/LLM library
- **TorchServe**: Model serving for production
- **ONNX Runtime**: Export PyTorch models to a portable format
- **torch.export / ExecuTorch**: On-device deployment (mobile, embedded)

---

## Installation

### CPU-Only Installation (Recommended for Learning)

```bash
# Using pip (simplest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Using conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### With CUDA (for NVIDIA GPUs)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Quick test
x = torch.rand(3, 3)
print(f"Random tensor:\n{x}")
```

---

## Core Philosophy

### 1. Eager Execution (Define-by-Run)

In PyTorch, operations execute immediately as Python runs them. There is no separate
"compilation" step before you can see results:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2          # This executes RIGHT NOW — not later in a session
print(y)            # tensor([2., 4., 6.])
```

**Why this matters**: You can use standard Python control flow (`if`, `for`, `while`)
inside your models. The computation graph is built dynamically as your code runs,
which means different inputs can follow different code paths.

### 2. Dynamic Computation Graphs

Unlike static graph frameworks, PyTorch rebuilds the computation graph every time
you run a forward pass. This means:

```python
def dynamic_model(x):
    if x.sum() > 0:         # This condition is evaluated at runtime
        return x * 2
    else:
        return x * 3
```

Each call to `dynamic_model` can follow a different path through the code. The
autograd graph is constructed on the fly and torn down after `.backward()`.

### 3. Python-First Design

PyTorch is designed to feel like an extension of Python and NumPy, not a separate
language embedded in Python. You think in Python, you debug in Python, you profile
in Python. The C++ backend handles performance; you rarely need to think about it.

### 4. torch.compile — The Best of Both Worlds

Starting with PyTorch 2.0, you can optionally compile your models for speed while
keeping the eager-mode development experience:

```python
model = MyModel()
compiled_model = torch.compile(model)  # One line for significant speedup
output = compiled_model(input_data)
```

This uses TorchDynamo to trace your Python code, TorchInductor to generate optimized
kernels, and falls back to eager mode for unsupported patterns.

---

## Architecture Stack

Understanding PyTorch's layered architecture helps you know where to look when
debugging or optimizing. From top to bottom:

```
┌──────────────────────────────────────────────────┐
│  Python Frontend (torch, torch.nn, etc.)         │  ← You write code here
├──────────────────────────────────────────────────┤
│  torch.compile (TorchDynamo + TorchInductor)     │  ← Optional compilation
├──────────────────────────────────────────────────┤
│  Autograd Engine                                 │  ← Automatic differentiation
├──────────────────────────────────────────────────┤
│  ATen (A Tensor Library)                         │  ← Core tensor operations
├──────────────────────────────────────────────────┤
│  C10 (Caffe2 + PyTorch Core)                     │  ← Dispatcher, memory, dtypes
├──────────────────────────────────────────────────┤
│  Hardware Backends (CPU/CUDA/MPS/XPU/ROCm)       │  ← Actual computation
└──────────────────────────────────────────────────┘
```

**Python Frontend**: The `torch` module, `torch.nn`, `torch.optim`, etc. — the
high-level API you interact with daily.

**torch.compile**: TorchDynamo captures your Python code as an FX graph.
TorchInductor generates optimized C++/CUDA/Triton kernels from that graph.

**Autograd**: The engine that tracks operations on tensors with `requires_grad=True`
and computes gradients via reverse-mode automatic differentiation.

**ATen**: "A Tensor Library" — over 2,000 operators written in C++ that implement
the actual math (add, matmul, conv2d, etc.). When you call `torch.add(a, b)`,
this is where the computation happens.

**C10**: The core library providing the dispatcher (routes operations to the right
backend), memory allocators, dtype system, and device abstraction.

**Hardware Backends**: BLAS libraries (MKL for CPU, cuBLAS for GPU), cuDNN for
convolutions, and other hardware-specific optimized libraries.

---

## Mathematical Prerequisites

Deep learning sits at the intersection of linear algebra, calculus, probability,
and optimization. You don't need a PhD in mathematics, but you need working
familiarity with these concepts. This section teaches them through PyTorch code.

---

## Linear Algebra

Linear algebra is the language of deep learning. Neural networks are fundamentally
sequences of matrix multiplications interspersed with non-linear functions.

### Scalars, Vectors, Matrices, and Tensors

```python
import torch

scalar = torch.tensor(3.14)             # 0-D tensor (scalar)
vector = torch.tensor([1.0, 2.0, 3.0])  # 1-D tensor (vector)
matrix = torch.tensor([[1, 2], [3, 4]])  # 2-D tensor (matrix)
tensor_3d = torch.randn(2, 3, 4)        # 3-D tensor

print(f"Scalar shape: {scalar.shape}")   # torch.Size([])
print(f"Vector shape: {vector.shape}")   # torch.Size([3])
print(f"Matrix shape: {matrix.shape}")   # torch.Size([2, 2])
print(f"3D shape: {tensor_3d.shape}")    # torch.Size([2, 3, 4])
```

**Why tensors?** In deep learning, data naturally has multiple dimensions. An image
is a 3D tensor (channels × height × width). A batch of images is 4D (batch ×
channels × height × width). A batch of sequences of word embeddings is 3D
(batch × sequence_length × embedding_dim).

### Norms — Measuring Vector Magnitude

A norm measures the "size" of a vector. Different norms emphasize different properties:

- **L1 norm** (Manhattan): Sum of absolute values. Encourages sparsity in optimization.
  \( \|x\|_1 = \sum_i |x_i| \)

- **L2 norm** (Euclidean): Square root of sum of squares. The "ordinary" distance.
  \( \|x\|_2 = \sqrt{\sum_i x_i^2} \)

- **L∞ norm** (Max): Largest absolute value. Used in adversarial robustness.
  \( \|x\|_\infty = \max_i |x_i| \)

```python
v = torch.tensor([3.0, -4.0])
print(f"L1 norm: {torch.norm(v, p=1)}")      # 7.0
print(f"L2 norm: {torch.norm(v, p=2)}")      # 5.0
print(f"L∞ norm: {torch.norm(v, p=float('inf'))}")  # 4.0
```

**Why norms matter in deep learning**: L2 regularization (weight decay) penalizes
large L2 norms of weight vectors, preventing overfitting. L1 regularization drives
weights to exactly zero, performing feature selection. Gradient clipping uses norms
to prevent exploding gradients.

### Dot Product — Measuring Similarity

The dot product of two vectors measures their alignment:

\( a \cdot b = \sum_i a_i b_i = \|a\| \|b\| \cos\theta \)

```python
a = torch.tensor([1.0, 0.0])
b = torch.tensor([0.0, 1.0])
c = torch.tensor([1.0, 1.0])

print(f"a·b (perpendicular): {torch.dot(a, b)}")    # 0.0
print(f"a·c (45 degrees): {torch.dot(a, c)}")       # 1.0
print(f"a·a (parallel): {torch.dot(a, a)}")          # 1.0
```

**Why dot products matter**: Attention mechanisms in Transformers compute dot
products between query and key vectors to measure relevance. The output of a
linear layer `y = Wx + b` is a batch of dot products between weight rows and
the input vector.

### Matrix Multiplication — The Core Operation

Matrix multiplication is the single most important operation in deep learning.
Every linear layer, every attention head, every convolution can be expressed as
matrix multiplications.

For matrices A (m×n) and B (n×p), the result C = AB is (m×p):

\( C_{ij} = \sum_k A_{ik} B_{kj} \)

```python
A = torch.tensor([[1., 2.], [3., 4.]])  # 2×2
B = torch.tensor([[5., 6.], [7., 8.]])  # 2×2

C = A @ B  # or torch.matmul(A, B) or torch.mm(A, B)
print(f"A @ B =\n{C}")
# tensor([[19., 22.],
#         [43., 50.]])
```

**Key rule**: Inner dimensions must match. (m×**n**) @ (**n**×p) → (m×p).
The **n** dimensions are "consumed" by the multiplication.

### Eigendecomposition

A square matrix A can be decomposed as A = QΛQ⁻¹, where Q contains eigenvectors
and Λ is a diagonal matrix of eigenvalues. An eigenvector v satisfies Av = λv —
the matrix only scales it, doesn't change its direction.

```python
A = torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float)
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
```

**Why eigendecomposition matters**: Principal Component Analysis (PCA) uses
eigendecomposition to find the directions of maximum variance in data. The
condition number (ratio of largest to smallest eigenvalue) tells you how
numerically stable a problem is.

### Singular Value Decomposition (SVD)

SVD generalizes eigendecomposition to non-square matrices: A = UΣVᵀ.
- U: left singular vectors (column space basis)
- Σ: singular values (diagonal, non-negative, sorted)
- Vᵀ: right singular vectors (row space basis)

```python
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float)
U, S, Vh = torch.linalg.svd(A)
print(f"U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}")
```

**Why SVD matters**: Low-rank approximation via SVD is used in LoRA (Low-Rank
Adaptation) for efficient fine-tuning of large language models. It's also the
mathematical foundation of matrix factorization in recommender systems.

---

## Calculus for Deep Learning

### Derivatives and the Chain Rule

A derivative measures how a function's output changes when its input changes:

\( f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \)

The **chain rule** is the single most important calculus concept for deep learning.
If y = f(g(x)), then:

\( \frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} \)

Neural networks are compositions of functions: output = f_n(f_{n-1}(...f_1(x)...)).
Backpropagation is just the chain rule applied repeatedly.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2 + x  # y = x³ + 2x² + x
y.backward()             # dy/dx = 3x² + 4x + 1
print(f"dy/dx at x=2: {x.grad}")  # 3(4) + 4(2) + 1 = 21
```

### Gradients — Multivariable Derivatives

When a function has multiple inputs, the gradient is the vector of all partial
derivatives:

\( \nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots\right] \)

The gradient points in the direction of steepest ascent. To minimize a loss,
we move in the opposite direction: **gradient descent**.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
f = (x**2).sum()  # f = x₁² + x₂² + x₃²
f.backward()
print(f"Gradient: {x.grad}")  # [2, 4, 6] — the gradient ∇f = 2x
```

### Jacobians and Hessians

The **Jacobian** generalizes the gradient to vector-valued functions. If
f: ℝⁿ → ℝᵐ, the Jacobian J is an m×n matrix where J_ij = ∂f_i/∂x_j.

The **Hessian** is the matrix of second derivatives: H_ij = ∂²f/∂x_i∂x_j.
It tells you about the curvature of the loss surface — whether you're at a
minimum, maximum, or saddle point.

```python
from torch.autograd.functional import jacobian, hessian

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)
print(f"Jacobian:\n{J}")
# [[2*x0, 1   ],    = [[2, 1],
#  [x1^2, 2*x0*x1]]    [4, 4]]
```

---

## Probability and Statistics

### Probability Distributions

PyTorch's `torch.distributions` module provides a rich set of probability
distributions. These are essential for:
- Initializing weights (normal, uniform)
- Variational autoencoders (reparameterization trick)
- Reinforcement learning (sampling actions)
- Bayesian neural networks

```python
from torch.distributions import Normal, Bernoulli, Categorical

normal = Normal(loc=0.0, scale=1.0)  # mean=0, std=1
sample = normal.sample((5,))
log_prob = normal.log_prob(torch.tensor(0.0))
print(f"Samples: {sample}")
print(f"Log prob of 0: {log_prob}")  # log(1/√(2π)) ≈ -0.9189
```

### Expectation and Variance

- **Expectation** E[X]: The average value of a random variable.
- **Variance** Var(X) = E[(X - E[X])²]: How spread out the values are.

```python
samples = torch.randn(100000)  # Standard normal
print(f"Mean ≈ {samples.mean():.4f}")  # ≈ 0
print(f"Var ≈ {samples.var():.4f}")    # ≈ 1
```

### Cross-Entropy — The Loss Function of Classification

Cross-entropy measures the difference between two probability distributions p
(true) and q (predicted):

\( H(p, q) = -\sum_i p_i \log(q_i) \)

When p is a one-hot vector (classification), this simplifies to:
\( H(p, q) = -\log(q_{\text{true class}}) \)

This is why we use log-softmax + negative log likelihood, which PyTorch combines
into `nn.CrossEntropyLoss`:

```python
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1]])  # Raw model output
target = torch.tensor([0])                  # True class is 0

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)
print(f"Cross-entropy loss: {loss.item():.4f}")
```

### KL Divergence

KL divergence measures how one probability distribution differs from a reference:

\( D_{KL}(P \| Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)} \)

It's asymmetric: D_KL(P||Q) ≠ D_KL(Q||P). Used in VAEs (variational autoencoders)
to keep the learned latent distribution close to a prior (usually standard normal).

```python
import torch.nn.functional as F

p = torch.tensor([0.4, 0.3, 0.3])  # True distribution
q = torch.tensor([0.33, 0.33, 0.34])  # Predicted distribution

kl = F.kl_div(q.log(), p, reduction='sum')
print(f"KL divergence: {kl.item():.4f}")
```

---

## Optimization Theory

### Gradient Descent — The Foundation

Gradient descent minimizes a function by repeatedly taking steps opposite to the
gradient. The update rule:

\( \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t) \)

where α is the **learning rate** — the most important hyperparameter in deep learning.

- **Too large**: Overshoots the minimum, loss oscillates or diverges
- **Too small**: Converges extremely slowly, may get stuck in local minima
- **Just right**: Smooth convergence to a good minimum

```python
# Minimizing f(x) = (x - 3)² from scratch
x = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(50):
    loss = (x - 3) ** 2
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

print(f"Final x: {x.item():.6f}")  # ≈ 3.0
```

### Stochastic Gradient Descent (SGD)

In practice, computing the gradient over the entire dataset is expensive.
SGD estimates the gradient using a random mini-batch:

\( \theta_{t+1} = \theta_t - \alpha \nabla L_{\text{batch}}(\theta_t) \)

The noise from mini-batch sampling actually helps — it can escape local minima
and leads to solutions that generalize better.

### SGD with Momentum

Plain SGD can oscillate in narrow valleys. Momentum adds a "velocity" term that
accumulates past gradients, smoothing the trajectory:

\( v_t = \beta v_{t-1} + \nabla L(\theta_t) \)
\( \theta_{t+1} = \theta_t - \alpha v_t \)

β (typically 0.9) controls how much history to keep. Think of it like a ball
rolling downhill — it builds up speed in consistent directions.

### Adam — Adaptive Moment Estimation

Adam combines momentum with per-parameter adaptive learning rates. It maintains
two running averages:

- **m** (first moment): exponential moving average of gradients (like momentum)
- **v** (second moment): exponential moving average of squared gradients

\( m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \)
\( v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \)
\( \hat{m}_t = m_t / (1 - \beta_1^t) \)  (bias correction)
\( \hat{v}_t = v_t / (1 - \beta_2^t) \)  (bias correction)
\( \theta_{t+1} = \theta_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) \)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**Why Adam works well**: Parameters that receive sparse, infrequent gradients get
larger effective learning rates (because v is small). Parameters with frequent,
large gradients get smaller effective learning rates (because v is large). This
adaptive behavior is crucial for training models where different parameters need
different learning rates.

**When to use what**:
- **SGD + Momentum**: Often achieves better final performance but requires careful
  learning rate tuning and scheduling. Preferred for vision models (ResNet, etc.).
- **Adam/AdamW**: Faster convergence, less sensitive to learning rate. Preferred
  for Transformers, LLMs, and when you want quick results.
- **AdamW**: Adam with decoupled weight decay — almost always preferred over plain Adam.

---

## Information Theory

### Entropy — Measuring Uncertainty

Entropy quantifies the uncertainty in a probability distribution:

\( H(p) = -\sum_i p_i \log p_i \)

- **Maximum entropy**: uniform distribution (maximum uncertainty)
- **Minimum entropy (0)**: all probability mass on one outcome (certainty)

```python
uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
certain = torch.tensor([1.0, 0.0, 0.0, 0.0])

H_uniform = -(uniform * uniform.log()).sum()
H_certain = -(certain * (certain + 1e-8).log()).sum()

print(f"Entropy of uniform: {H_uniform:.4f}")  # 1.3863 (= ln(4))
print(f"Entropy of certain: {H_certain:.4f}")   # ≈ 0.0
```

### Cross-Entropy Loss Derivation

Why do we use cross-entropy as a loss function for classification?

1. **Maximum likelihood**: We want to find model parameters θ that maximize the
   probability of the observed data: argmax_θ P(data|θ).

2. **Log transformation**: Taking the log converts the product of probabilities
   into a sum (numerically stable, easier to optimize):
   argmax_θ Σ log P(y_i | x_i, θ)

3. **Negation**: Maximizing log-likelihood = minimizing negative log-likelihood:
   argmin_θ -Σ log P(y_i | x_i, θ)

4. **Softmax output**: For classification, P(y=k|x) = softmax(logits)_k, so:
   Loss = -log(softmax(logits)_{true_class})

This is exactly cross-entropy between the one-hot true distribution and the
softmax predicted distribution. The connection is not coincidental — **cross-entropy
loss is the information-theoretically optimal loss for classification**.

```python
logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)
true_class = 0

nll_loss = -torch.log(probs[true_class])
print(f"NLL loss: {nll_loss.item():.4f}")

ce_loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([true_class]))
print(f"CE loss:  {ce_loss.item():.4f}")  # Same value
```

---

## What's Next?

With these mathematical foundations, you're ready to dive into PyTorch's tensor
system in Module 02. The linear algebra you learned here will appear every time
you work with layers (matrix multiplication), losses (norms, cross-entropy), and
optimization (gradients, Adam). The key insight is that **deep learning is applied
linear algebra + calculus, automated by PyTorch's autograd system**.

Run `math_with_pytorch.py` in this directory to see all these concepts in action
with runnable code.

---

<div align="center">

[🏠 Home](../README.md) | [Next Module →](../02_tensors/)

**[📓 Open Notebook](../notebooks/01_tensors_masterclass.ipynb)** — Interactive version of this module

</div>
