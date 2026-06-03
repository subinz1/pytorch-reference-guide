# Module 03: Autograd — Automatic Differentiation

## Table of Contents
1. [What is Automatic Differentiation?](#what-is-automatic-differentiation)
2. [Forward Mode vs Reverse Mode](#forward-mode-vs-reverse-mode)
3. [The Computation Graph](#the-computation-graph)
4. [Leaf Tensors vs Intermediate Tensors](#leaf-tensors-vs-intermediate-tensors)
5. [requires_grad, grad_fn, grad](#requires_grad-grad_fn-grad)
6. [The backward() Function](#the-backward-function)
7. [Gradient Accumulation](#gradient-accumulation)
8. [torch.no_grad() vs torch.inference_mode()](#no_grad-vs-inference_mode)
9. [detach()](#detach)
10. [Custom Autograd Functions](#custom-autograd-functions)
11. [gradcheck and gradgradcheck](#gradcheck-and-gradgradcheck)
12. [Higher-Order Gradients](#higher-order-gradients)
13. [torch.autograd.grad()](#torch-autograd-grad)
14. [Jacobian and Hessian Computation](#jacobian-and-hessian-computation)
15. [Common Pitfalls](#common-pitfalls)
16. [Autograd Hooks](#autograd-hooks)
17. [Compiled Autograd](#compiled-autograd)

---

## What is Automatic Differentiation?

Automatic differentiation (AD) is a technique for computing exact derivatives of
functions expressed as computer programs. It is NOT:

- **Symbolic differentiation** (like Mathematica/Sympy): These manipulate
  mathematical expressions symbolically, which can lead to expression swell
  (the derivative expression becomes exponentially larger than the original).

- **Numerical differentiation** (finite differences): Computing
  (f(x+h) - f(x)) / h is simple but introduces truncation and rounding errors,
  and scales poorly with the number of parameters (requires one function
  evaluation per parameter).

Instead, AD exploits the fact that every computer program, no matter how complex,
is composed of elementary operations (+, *, sin, exp, etc.) whose derivatives
are known. By applying the chain rule systematically, AD computes exact
derivatives at machine precision.

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.sin(x) * torch.exp(x)

# PyTorch knows the derivative of sin, exp, and *.
# It chains them together to get dy/dx exactly.
y.backward()
print(f"dy/dx at x=3: {x.grad.item():.6f}")

# Verify: d/dx[sin(x)*exp(x)] = cos(x)*exp(x) + sin(x)*exp(x)
manual = (torch.cos(torch.tensor(3.0)) + torch.sin(torch.tensor(3.0))) * torch.exp(torch.tensor(3.0))
print(f"Manual:       {manual.item():.6f}")
```

---

## Forward Mode vs Reverse Mode

There are two ways to apply the chain rule through a computation:

### Forward Mode (Tangent Mode)

Propagates derivatives forward through the computation, alongside the primal
(original) computation. For f: R^n → R^m:

- Computes one column of the Jacobian per forward pass
- Cost: O(n) forward passes for full Jacobian
- Efficient when n << m (few inputs, many outputs)

### Reverse Mode (Adjoint Mode) = Backpropagation

Propagates derivatives backward from outputs to inputs. For f: R^n → R^m:

- Computes one row of the Jacobian per backward pass
- Cost: O(m) backward passes for full Jacobian
- Efficient when m << n (many inputs, few outputs)

**Why neural networks use reverse mode**: A neural network has millions of
parameters (inputs to the loss function) but produces a single scalar loss
(one output). Reverse mode computes ALL gradients (∂loss/∂param for every param)
in a single backward pass — regardless of how many parameters there are.
Forward mode would require one pass per parameter, which is millions of times
slower.

This asymmetry is why:
- Training uses reverse mode (backpropagation): 1 scalar output, millions of inputs
- Forward-mode AD is used for Jacobian-vector products in some optimization methods
- PyTorch supports both: `backward()` for reverse mode, `torch.autograd.forward_ad`
  for forward mode

---

## The Computation Graph

When you perform operations on tensors with `requires_grad=True`, PyTorch builds
a directed acyclic graph (DAG) that records every operation. This graph is
essential for computing gradients.

### How the graph is built

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Each operation creates a node in the graph
a = x * y        # MulBackward node
b = a + x        # AddBackward node
c = b.sin()      # SinBackward node

print(f"c.grad_fn: {c.grad_fn}")  # SinBackward
print(f"b.grad_fn: {b.grad_fn}")  # AddBackward
print(f"a.grad_fn: {a.grad_fn}")  # MulBackward
print(f"x.grad_fn: {x.grad_fn}")  # None (leaf tensor)
```

The graph looks like:

```
x ──→ MulBackward(a=x*y) ──→ AddBackward(b=a+x) ──→ SinBackward(c=sin(b))
y ──↗                    x ──↗
```

### Graph lifecycle

The graph is built dynamically during the forward pass and consumed (destroyed)
during backward. This is the "dynamic graph" feature of PyTorch:

1. **Forward pass**: Operations are recorded as nodes in the graph.
2. **backward()**: The graph is traversed in reverse order, computing gradients.
3. **Graph destroyed**: After backward, the graph is freed (by default).
4. **Next forward**: A new graph is built from scratch.

This means every iteration can follow a different code path — `if` statements,
`for` loops, and variable-length sequences all just work.

---

## Leaf Tensors vs Intermediate Tensors

### Leaf tensors

A leaf tensor is one that was created directly by the user (not by an operation).
Leaf tensors are the "starting points" of the computation graph.

```python
x = torch.tensor(1.0, requires_grad=True)   # Leaf
w = torch.randn(3, 4, requires_grad=True)    # Leaf
b = torch.zeros(4, requires_grad=True)        # Leaf

print(f"x.is_leaf: {x.is_leaf}")  # True
print(f"w.is_leaf: {w.is_leaf}")  # True
```

### Intermediate tensors

Tensors created by operations on other tensors are intermediate. They have a
`grad_fn` that records how they were created.

```python
y = x * 2        # Intermediate
z = w @ b        # Intermediate

print(f"y.is_leaf: {y.is_leaf}")      # False
print(f"y.grad_fn: {y.grad_fn}")      # MulBackward0
```

### Key difference: where gradients are stored

By default, PyTorch only stores gradients for leaf tensors. This is a memory
optimization — intermediate gradients are computed during backward but discarded
immediately after use.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 3

z.backward()
print(f"x.grad: {x.grad}")   # 12.0 — stored because x is a leaf
print(f"y.grad: {y.grad}")   # None — not stored because y is intermediate
```

If you need gradients for intermediate tensors, use `retain_grad()`:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.retain_grad()  # Tell PyTorch to keep y's gradient
z = y * 3
z.backward()
print(f"y.grad: {y.grad}")   # 3.0 — now it's stored
```

---

## requires_grad, grad_fn, grad

These three attributes form the autograd trinity:

### requires_grad

A boolean flag indicating whether this tensor participates in gradient computation.

```python
# Tensors that need gradients
w = torch.randn(3, 3, requires_grad=True)   # Set at creation
b = torch.zeros(3)
b.requires_grad_(True)                       # Set after creation (in-place)

# Tensors that don't
data = torch.randn(100, 3)                   # Data doesn't need gradients
labels = torch.randint(0, 2, (100,))          # Labels don't need gradients
```

**Rule**: If ANY input to an operation has `requires_grad=True`, the output will
also have `requires_grad=True` (gradient tracking propagates forward).

### grad_fn

A reference to the backward function (the operation that created this tensor).
Leaf tensors have `grad_fn=None`.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2      # y.grad_fn = PowBackward0
z = y.sum()      # z.grad_fn = SumBackward0
```

You can traverse the graph by following `grad_fn` links:

```python
print(z.grad_fn)                     # SumBackward0
print(z.grad_fn.next_functions)      # Links to y's grad_fn
```

### grad

The accumulated gradient tensor. Only populated after `backward()` is called.
Only leaf tensors have `.grad` by default.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print(f"x.grad: {x.grad}")  # [2., 4., 6.] — the gradient ∂loss/∂x = 2x
```

---

## The backward() Function

`backward()` is the workhorse of training. It computes gradients by traversing
the computation graph in reverse order (topological sort).

### How it works

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3 + 2 * x ** 2 - 5 * x
y.backward()
# Computes dy/dx = 3x² + 4x - 5 = 27 + 12 - 5 = 34
print(f"dy/dx at x=3: {x.grad.item()}")  # 34.0
```

### The gradient argument

When the output is not a scalar, you must provide a `gradient` argument
(also called the "upstream gradient" or "grad_output"):

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y is a vector, not a scalar

# Must provide gradient (Jacobian-vector product)
y.backward(gradient=torch.tensor([1.0, 1.0, 1.0]))
print(f"x.grad: {x.grad}")  # [2., 4., 6.]

# The gradient argument acts as weights in a weighted sum:
# effectively computing d(sum(gradient * y))/dx
```

**Why?** PyTorch's backward pass always computes vector-Jacobian products (VJPs).
For a scalar output, the "vector" is implicitly 1.0. For vector outputs, you
must supply it. In practice, this gradient comes from the downstream loss.

### Graph destruction

By default, `backward()` destroys the computation graph after use:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
# y.backward()  # ERROR: graph already freed

# To keep the graph, use retain_graph=True
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
y.backward()  # OK — graph was retained
```

---

## Gradient Accumulation

**Gradients accumulate by default.** This is the single most common source of
bugs for PyTorch beginners.

```python
x = torch.tensor(1.0, requires_grad=True)

# First backward
y = x * 2
y.backward()
print(f"After first backward: x.grad = {x.grad}")  # 2.0

# Second backward WITHOUT zeroing
y = x * 3
y.backward()
print(f"After second backward: x.grad = {x.grad}")  # 5.0 (= 2.0 + 3.0)
# The gradient ACCUMULATED!
```

### Why accumulation exists

Gradient accumulation is actually a feature, not a bug. It's useful for:

1. **Large effective batch sizes**: Process mini-batches one at a time, accumulate
   gradients, then take one optimizer step. This lets you simulate large batches
   on limited GPU memory.

2. **Multiple losses**: If your model has multiple loss terms, you can backward
   each one separately and the gradients add up correctly.

### How to zero gradients

```python
# Method 1: Manual zeroing
x.grad.zero_()

# Method 2: Optimizer (most common)
optimizer.zero_grad()           # Zeros all parameter gradients

# Method 3: Set to None (slightly faster)
optimizer.zero_grad(set_to_none=True)
# or
x.grad = None
```

### The standard training loop pattern

```python
# for batch in dataloader:
#     optimizer.zero_grad()       # 1. Zero gradients
#     output = model(batch)       # 2. Forward pass
#     loss = criterion(output)    # 3. Compute loss
#     loss.backward()             # 4. Backward pass (compute gradients)
#     optimizer.step()            # 5. Update parameters
```

---

## torch.no_grad() vs torch.inference_mode()

Both disable gradient tracking, but for different purposes:

### torch.no_grad()

Disables gradient computation. Tensors created inside still have `requires_grad`
based on their inputs, but no graph is built.

```python
x = torch.tensor(1.0, requires_grad=True)

with torch.no_grad():
    y = x * 2
    print(f"y.requires_grad: {y.requires_grad}")  # False
    print(f"y.grad_fn: {y.grad_fn}")               # None
```

**Use for**: Validation loops, parameter updates, evaluation metrics.

### torch.inference_mode()

A stricter, more optimized version of `no_grad()`. Disables autograd entirely
and enables additional optimizations.

```python
x = torch.tensor(1.0, requires_grad=True)

with torch.inference_mode():
    y = x * 2
    # Even stricter: y is an InferenceTensor, cannot be used with autograd at all
```

**Use for**: Production inference, deployment, any time you're certain you won't
need gradients.

### When to use which

| Situation | Use |
|-----------|-----|
| Validation loop during training | `torch.no_grad()` |
| Updating parameters manually | `torch.no_grad()` |
| Production inference | `torch.inference_mode()` |
| Need to use result in autograd later | Neither (or `no_grad` carefully) |

---

## detach()

`detach()` creates a new tensor that shares data but is disconnected from the
computation graph.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach()  # z shares data with y but has no grad_fn

print(f"y.requires_grad: {y.requires_grad}")  # True
print(f"z.requires_grad: {z.requires_grad}")  # False
print(f"z.grad_fn: {z.grad_fn}")               # None
print(f"Same data: {y.data_ptr() == z.data_ptr()}")  # True
```

### Common uses

1. **Stopping gradient flow**: In GANs, you detach the generator output when
   training the discriminator, so gradients don't flow back to the generator.

2. **Converting to NumPy**: `tensor.detach().numpy()` — you must detach first
   if the tensor requires gradients.

3. **Target networks**: In reinforcement learning, the target network's output
   is detached so it's treated as a constant.

```python
# GAN-style gradient stopping
# fake = generator(noise)
# d_loss = discriminator(fake.detach())  # Stops gradient to generator
```

---

## Custom Autograd Functions

When PyTorch's built-in operations aren't sufficient, you can define custom
forward and backward passes.

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(f"x: {x}")
print(f"x.grad: {x.grad}")
```

### The ctx object

`ctx` (context) is used to pass information from forward to backward:

- `ctx.save_for_backward(tensor1, tensor2, ...)`: Save tensors needed for backward.
  These are stored efficiently and checked for version consistency.
- `ctx.saved_tensors`: Retrieve saved tensors in backward.
- `ctx.needs_input_grad`: Tuple of booleans indicating which inputs need gradients.
- `ctx.mark_dirty(tensor)`: Mark tensors modified in-place.
- `ctx.mark_non_differentiable(tensor)`: Mark outputs that don't need gradients.

### Rules for custom functions

1. `forward` receives the actual tensor values and returns output tensors.
2. `backward` receives the upstream gradient (`grad_output`) for each output
   and must return one gradient per input (or `None` if that input doesn't
   need gradients).
3. The number of tensors returned by `backward` must match the number of
   inputs to `forward` (excluding `ctx`).

---

## gradcheck and gradgradcheck

These utilities verify that your custom autograd function computes correct
gradients by comparing against numerical finite differences.

```python
from torch.autograd import gradcheck, gradgradcheck

class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result * (1 - result)

# gradcheck requires float64 for numerical precision
x = torch.randn(5, dtype=torch.float64, requires_grad=True)
test = gradcheck(MySigmoid.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {test}")

# gradgradcheck checks second derivatives
test2 = gradgradcheck(MySigmoid.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Double gradient check passed: {test2}")
```

---

## Higher-Order Gradients

By default, `backward()` only computes first-order gradients. To compute
higher-order gradients (gradients of gradients), use `create_graph=True`.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 4  # y = x^4

# First derivative: dy/dx = 4x³
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {grad1.item()}")  # 32.0

# Second derivative: d²y/dx² = 12x²
grad2 = torch.autograd.grad(grad1, x, create_graph=True)[0]
print(f"d²y/dx² = {grad2.item()}")  # 48.0

# Third derivative: d³y/dx³ = 24x
grad3 = torch.autograd.grad(grad2, x)[0]
print(f"d³y/dx³ = {grad3.item()}")  # 48.0
```

### Why create_graph=True

Without `create_graph=True`, the backward pass doesn't build a graph for itself.
This means you can't differentiate through the backward pass. With it enabled,
the gradient computation is itself differentiable.

**Use cases**:
- **Gradient penalty** (WGAN-GP): Penalize the norm of gradients, which requires
  computing the gradient of the gradient norm.
- **MAML (meta-learning)**: Differentiate through the inner optimization loop.
- **Physics-informed neural networks**: Loss functions involving derivatives of
  the network output.

---

## torch.autograd.grad()

`torch.autograd.grad()` computes gradients without storing them in `.grad`
attributes. This is useful for:
- Computing gradients of specific outputs w.r.t. specific inputs
- Higher-order gradients
- Functional-style gradient computation

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()

# Compute gradient without storing in x.grad
grad = torch.autograd.grad(y, x)[0]
print(f"Gradient: {grad}")        # [2., 4., 6.]
print(f"x.grad: {x.grad}")        # None — not stored
```

### Multiple outputs and inputs

```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

out1 = x * y
out2 = x ** 2 + y ** 2

# Gradients of both outputs w.r.t. both inputs
grads = torch.autograd.grad(
    outputs=[out1, out2],
    inputs=[x, y],
    grad_outputs=[torch.tensor(1.0), torch.tensor(1.0)]
)
print(f"d(out1+out2)/dx: {grads[0].item()}")  # y + 2x = 2 + 2 = 4
print(f"d(out1+out2)/dy: {grads[1].item()}")  # x + 2y = 1 + 4 = 5
```

---

## Jacobian and Hessian Computation

### Jacobian

The Jacobian matrix contains all partial derivatives of a vector-valued function.
For f: R^n → R^m, J is m × n where J_ij = ∂f_i/∂x_j.

```python
from torch.autograd.functional import jacobian

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)
print(f"Jacobian:\n{J}")
# [[2*x0, 1   ],     = [[2, 1],
#  [x1^2, 2*x0*x1]]     [4, 4]]
```

### Hessian

The Hessian matrix contains all second partial derivatives. For f: R^n → R,
H is n × n where H_ij = ∂²f/∂x_i∂x_j.

```python
from torch.autograd.functional import hessian

def g(x):
    return x[0]**3 + x[0]*x[1]**2 + x[1]**3

x = torch.tensor([1.0, 2.0])
H = hessian(g, x)
print(f"Hessian:\n{H}")
# [[6*x0,    2*x1 ],     = [[6,  4],
#  [2*x1, 2*x0+6*x1]]      [4, 14]]
```

### Efficient Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs)

Computing the full Jacobian is expensive (O(n) backward passes). Often you only
need the product of the Jacobian with a specific vector:

```python
from torch.autograd.functional import jvp, vjp

def f(x):
    return torch.stack([x[0]**2, x[0]*x[1]])

x = torch.tensor([1.0, 2.0])
v = torch.tensor([1.0, 0.0])  # Direction vector

# JVP: J @ v (forward mode — one forward pass)
_, jvp_result = jvp(f, (x,), (v,))
print(f"JVP (J @ v): {jvp_result}")

# VJP: v^T @ J (reverse mode — one backward pass)
_, vjp_fn = vjp(f, x)
vjp_result = vjp_fn(torch.tensor([1.0, 0.0]))
print(f"VJP (v^T @ J): {vjp_result}")
```

---

## Common Pitfalls

### 1. Forgetting to zero gradients

```python
x = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = x * (i + 1)
    loss.backward()
    print(f"Step {i}: x.grad = {x.grad.item()}")
    # Without zeroing: 1.0, 3.0, 6.0 (accumulated!)
    # x.grad.zero_()  # Uncomment to fix
```

### 2. In-place operations on tensors that need gradients

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

# This WILL cause problems:
# y.add_(1)  # In-place modification of a tensor needed for backward
# y.backward(torch.tensor([1.0, 1.0]))  # RuntimeError!
```

### 3. Gradient not flowing through integer operations

```python
x = torch.tensor(3.0, requires_grad=True)
y = x.int()  # Casting to int breaks gradient flow
# y has no grad_fn — gradient is lost!
```

### 4. NaN gradients

Common causes:
- `log(0)`: Use `log(x + epsilon)` or `torch.clamp(x, min=1e-8)`
- `sqrt(0)`: derivative of sqrt at 0 is infinity. Use `sqrt(x + epsilon)`
- `0/0`: Can occur in normalization layers when variance is zero
- Division by a very small number: use `torch.clamp` on denominators

```python
x = torch.tensor(0.0, requires_grad=True)
# y = torch.log(x)   # Will give -inf, grad is inf
y = torch.log(x + 1e-8)  # Safe
y.backward()
print(f"Safe log grad: {x.grad}")
```

### 5. Modifying parameters without torch.no_grad()

```python
x = torch.tensor(1.0, requires_grad=True)
# x = x - 0.1 * x.grad  # This creates a NEW tensor, breaking the leaf status
# Instead:
# with torch.no_grad():
#     x -= 0.1 * x.grad
```

---

## Autograd Hooks

Hooks let you inspect or modify gradients during the backward pass without
changing the model code.

### Tensor hooks

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def print_grad(grad):
    print(f"  Hook received gradient: {grad}")

x.register_hook(print_grad)
y = (x ** 2).sum()
y.backward()  # Hook fires during backward
```

### Gradient modification hooks

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def clip_grad(grad):
    return torch.clamp(grad, -1.0, 1.0)

x.register_hook(clip_grad)
y = (x ** 3).sum()
y.backward()
print(f"Clipped gradient: {x.grad}")  # Gradients are clipped to [-1, 1]
```

### Module hooks

For `nn.Module`, you can register hooks on the entire module:

```python
import torch.nn as nn

model = nn.Linear(5, 3)

def forward_hook(module, input, output):
    print(f"Forward: input shape={input[0].shape}, output shape={output.shape}")

def backward_hook(module, grad_input, grad_output):
    print(f"Backward: grad_output shape={grad_output[0].shape}")

model.register_forward_hook(forward_hook)
model.register_full_backward_hook(backward_hook)

x = torch.randn(2, 5)
y = model(x)
y.sum().backward()
```

---

## Compiled Autograd

PyTorch 2.x introduced compiled autograd, which applies `torch.compile` to the
backward pass. This can significantly speed up training by:

1. Fusing backward operations into optimized kernels
2. Eliminating Python overhead in the backward pass
3. Enabling whole-graph optimization across forward and backward

```python
# Basic usage (requires torch >= 2.0)
model = torch.nn.Linear(100, 10)
compiled_model = torch.compile(model)

# Both forward AND backward are compiled
x = torch.randn(32, 100)
y = compiled_model(x)
loss = y.sum()
loss.backward()  # This backward is also compiled
```

The compilation happens lazily — the first iteration traces the graph, subsequent
iterations use the compiled version. This means the first iteration is slower
(compilation overhead) but all subsequent iterations are faster.

---

## Summary

Autograd is the engine that makes deep learning practical. Key takeaways:

1. **Reverse-mode AD** is efficient for neural networks (one backward pass for
   all parameters).
2. **The computation graph** is built dynamically during forward and consumed
   during backward.
3. **Always zero gradients** before each optimization step.
4. **Use `torch.no_grad()`** for validation and `inference_mode()` for deployment.
5. **Custom Functions** let you define operations with hand-written gradients.
6. **Hooks** let you inspect and modify gradients without changing model code.

Run the example files to see these concepts in action:
- `gradient_basics.py` — basic gradient computation and the training loop
- `computation_graph.py` — visualizing and understanding the computation graph
- `custom_functions.py` — writing custom autograd functions
- `higher_order_gradients.py` — second derivatives, Jacobians, and Hessians
