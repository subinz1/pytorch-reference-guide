<div align="center">

[← Previous Module](../14_testing/) | [🏠 Home](../README.md) | [Next Module →](../16_activation_checkpointing/)

</div>

---

> **Module 15** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 04 — Neural Networks](../04_neural_networks/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `parametrization.py` | Weight parametrization — enforcing constraints like symmetry, orthogonality, and positivity on parameters |
| `pruning.py` | Model pruning — making neural networks smaller by removing weights |
| `sequence_packing_and_nested.py` | Sequence packing & nested tensors — efficient variable-length processing |
| `conv_bn_fusion.py` | Conv-BN fusion & inference optimization utilities |

---

# Module 15: Practical PyTorch Utilities

## The Hidden Toolkit Most Tutorials Never Teach

PyTorch ships with a rich set of utility modules that most tutorials skip entirely. These are the tools that separate a beginner from a practitioner: weight parametrization, pruning, normalization techniques, sequence packing, nested tensors, and model fusion. This module covers them all.

---

## Table of Contents

1. [torch.nn.utils.parametrize — Weight Constraints](#1-weight-parametrization)
2. [torch.nn.utils.prune — Model Pruning](#2-model-pruning)
3. [Spectral Norm & Weight Norm](#3-spectral-norm--weight-norm)
4. [torch.nn.utils.rnn — Sequence Packing](#4-sequence-packing-for-rnns)
5. [Conv-BN Fusion](#5-conv-bn-fusion)
6. [torch.nested — Nested (Jagged) Tensors](#6-nested-tensors)
7. [torch.nn.utils.clip_grad — Gradient Clipping Internals](#7-gradient-clipping-internals)
8. [parameters_to_vector & skip_init](#8-parameter-utilities)

---

## 1. Weight Parametrization

**What it is:** A framework to apply constraints or transformations to module parameters. Instead of manually enforcing constraints in `forward()`, you register a parametrization that automatically transforms the raw weight into a constrained version.

**Why it matters:** Enforcing constraints like orthogonality, symmetry, or positivity on weights is common in research and production. Without parametrization, you'd need hacky workarounds.

### How It Works

```python
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

class Symmetric(nn.Module):
    """Parametrization that makes a matrix symmetric."""
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

linear = nn.Linear(5, 5)
P.register_parametrization(linear, "weight", Symmetric())

# Now linear.weight is ALWAYS symmetric
print(linear.weight)  # Symmetric!
print(torch.allclose(linear.weight, linear.weight.T))  # True

# The raw unconstrained parameter is stored as:
print(linear.parametrizations.weight.original)
```

### Key Concept: Original vs Parametrized

When you register a parametrization:
- The **original** unconstrained tensor is stored at `module.parametrizations.<name>.original`
- Accessing `module.<name>` runs the parametrization on the original and returns the result
- The optimizer updates the **original** (unconstrained) parameter
- The parametrization is applied on every access (or cached)

### Built-in Parametrizations

```python
from torch.nn.utils import parametrizations

# Orthogonal weight matrix (useful for RNNs, preventing vanishing/exploding gradients)
linear = nn.Linear(5, 5)
parametrizations.orthogonal(linear, "weight")
# linear.weight is now always orthogonal: W^T W = I

# Spectral normalization (stabilize GANs and training)
conv = nn.Conv2d(3, 64, 3)
parametrizations.spectral_norm(conv, "weight")
# Constrains the spectral norm (largest singular value) of the weight to 1

# Weight normalization (decouple magnitude from direction)
linear = nn.Linear(10, 5)
parametrizations.weight_norm(linear, "weight")
# Reparametrizes: w = g * (v / ||v||)
```

### Custom Parametrization Example — Positive Weights

```python
class Positive(nn.Module):
    """Ensures weights are always positive via softplus."""
    def forward(self, X):
        return torch.nn.functional.softplus(X)

linear = nn.Linear(3, 3)
P.register_parametrization(linear, "weight", Positive())
print(linear.weight)  # All positive!
print((linear.weight > 0).all())  # True
```

### Caching for Efficiency

If you use a parametrized weight multiple times in `forward()` (e.g., RNNs sharing the recurrent kernel), use caching to avoid recomputation:

```python
with P.cached():
    output = model(input)  # Parametrizations computed once, cached
```

### Removing Parametrizations

```python
# Remove and keep the parametrized (constrained) value
P.remove_parametrizations(linear, "weight", leave_parametrized=True)

# Remove and go back to unconstrained original
P.remove_parametrizations(linear, "weight", leave_parametrized=False)
```

---

## 2. Model Pruning

**What it is:** Removing (zeroing out) weights from a neural network to make it smaller and faster.

**Why it matters:** Pruned models can be 2-10x smaller with minimal accuracy loss. Critical for edge deployment.

### Pruning Strategies

| Method | What It Does |
|--------|-------------|
| `random_unstructured` | Zero out random individual weights |
| `l1_unstructured` | Zero out weights with smallest L1 magnitude |
| `random_structured` | Zero out entire channels/neurons randomly |
| `ln_structured` | Zero out channels with smallest Ln norm |
| `global_unstructured` | Prune across all layers by global ranking |

### How Pruning Works in PyTorch

1. The original weight is moved to `weight_orig`
2. A binary mask `weight_mask` is created
3. A forward hook computes `weight = weight_orig * weight_mask` before each forward pass

```python
import torch.nn.utils.prune as prune

linear = nn.Linear(10, 5)

# Prune 30% of weights (smallest magnitude)
prune.l1_unstructured(linear, name="weight", amount=0.3)

print(linear.weight)        # Pruned weight (has zeros)
print(linear.weight_mask)   # Binary mask
print(linear.weight_orig)   # Original weight

# Count sparsity
zeros = (linear.weight == 0).sum().item()
total = linear.weight.numel()
print(f"Sparsity: {zeros}/{total} = {zeros/total:.1%}")
```

### Global Pruning (Prune Across All Layers)

```python
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

# Collect all prunable parameters
parameters_to_prune = [
    (model[0], "weight"),
    (model[2], "weight"),
    (model[4], "weight"),
]

# Globally prune 40% of weights (by L1 magnitude across ALL layers)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4,
)
```

### Making Pruning Permanent

```python
# Remove the pruning reparametrization (bake the mask into the weight)
prune.remove(linear, "weight")
# Now linear.weight IS the pruned weight directly (no more mask/orig)
```

---

## 3. Spectral Norm & Weight Norm

### Spectral Normalization

Controls the Lipschitz constant of a layer by normalizing weights by their spectral norm (largest singular value). Essential for stable GAN training.

**Math:** $\bar{W} = W / \sigma(W)$ where $\sigma(W)$ is the largest singular value.

```python
from torch.nn.utils import spectral_norm

# Apply spectral norm
conv = nn.Conv2d(3, 64, 3, padding=1)
conv = spectral_norm(conv, name="weight")

# The spectral norm is estimated via power iteration (efficient)
# No full SVD needed — just one vector update per forward pass
```

### Weight Normalization

Decouples weight magnitude from direction: $w = g \cdot \frac{v}{\|v\|}$

The optimizer can separately learn the magnitude $g$ and direction $v$, which often leads to faster convergence.

```python
from torch.nn.utils import weight_norm

linear = nn.Linear(10, 5)
linear = weight_norm(linear, name="weight")

# Now has: linear.weight_g (magnitude) and linear.weight_v (direction)
print(linear.weight_g.shape)  # (5, 1)
print(linear.weight_v.shape)  # (5, 10)
```

---

## 4. Sequence Packing for RNNs

**Problem:** Sequences in a batch have different lengths. Padding wastes computation — the RNN processes pad tokens unnecessarily.

**Solution:** Pack sequences so the RNN only processes real tokens.

```python
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    pack_sequence,
)

# Variable-length sequences
seqs = [torch.randn(5, 10),   # length 5
        torch.randn(3, 10),   # length 3
        torch.randn(8, 10)]   # length 8

# Step 1: Pad to same length
padded = pad_sequence(seqs, batch_first=True)  # (3, 8, 10)
lengths = torch.tensor([5, 3, 8])

# Step 2: Pack (sorts by length internally)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)

# Step 3: Feed to RNN
rnn = nn.LSTM(10, 20, batch_first=True)
output_packed, (h_n, c_n) = rnn(packed)

# Step 4: Unpack
output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
print(f"Output: {output_padded.shape}")  # (3, 8, 20)
```

### Why Packing Matters

Without packing: RNN processes all 8 timesteps for all 3 sequences (24 steps).
With packing: RNN processes 5+3+8=16 steps total. **33% less computation.**

For long sequences with high variance in length, the savings are much larger.

---

## 5. Conv-BN Fusion

**What it is:** Merging a Conv2d + BatchNorm2d into a single Conv2d for faster inference.

**Why it matters:** During inference, BatchNorm is a fixed affine transform. Fusing it into the convolution eliminates one entire layer with zero accuracy loss.

```python
from torch.nn.utils.fusion import fuse_conv_bn_eval

conv = nn.Conv2d(3, 64, 3, padding=1)
bn = nn.BatchNorm2d(64)

# Train as normal...
# Then for inference:
conv.eval()
bn.eval()

fused_conv = fuse_conv_bn_eval(conv, bn)
# fused_conv is a single Conv2d that produces identical output
# but is faster (one layer instead of two)

x = torch.randn(1, 3, 32, 32)
print(torch.allclose(fused_conv(x), bn(conv(x)), atol=1e-5))  # True
```

This is one of the most common inference optimizations and is done automatically by `torch.compile` and ONNX optimizers.

---

## 6. Nested Tensors

**What it is:** A tensor that can hold sequences of different lengths without padding. Also called "jagged tensors" or "ragged tensors."

**Why it matters:** Eliminates wasted computation on padding tokens in NLP/attention. Flash Attention can directly consume nested tensors for variable-length sequences.

```python
import torch
from torch.nested import nested_tensor, as_nested_tensor

# Create a nested tensor from variable-length sequences
nt = nested_tensor([
    torch.randn(3, 8),   # sequence of length 3, dim 8
    torch.randn(5, 8),   # sequence of length 5, dim 8
    torch.randn(2, 8),   # sequence of length 2, dim 8
])

print(f"Type: {type(nt)}")
print(f"Nested size: {nt.size()}")
# The first dim is the batch, subsequent dims may vary

# Convert to padded tensor when needed
padded = torch.nested.to_padded_tensor(nt, padding=0.0)
print(f"Padded shape: {padded.shape}")  # (3, 5, 8) — padded to max length

# Convert back
nt2 = as_nested_tensor(padded)
```

### Nested Tensors with SDPA

The real power is using nested tensors with `F.scaled_dot_product_attention` — Flash Attention handles the variable lengths natively, avoiding wasted computation on padding:

```python
# Instead of padding and masking:
# attn = F.scaled_dot_product_attention(Q_padded, K_padded, V_padded, attn_mask=mask)

# With nested tensors, no padding needed:
# Q_nested, K_nested, V_nested are NestedTensors
# attn = F.scaled_dot_product_attention(Q_nested, K_nested, V_nested)
# Flash Attention processes only real tokens!
```

---

## 7. Gradient Clipping Internals

We covered gradient clipping in Module 05, but here's the deeper story:

```python
from torch.nn.utils import clip_grad_norm_, clip_grad_value_, get_total_norm

model = nn.Linear(10, 5)
loss = model(torch.randn(3, 10)).sum()
loss.backward()

# Get the total gradient norm BEFORE clipping
total_norm = get_total_norm(model.parameters(), norm_type=2.0)
print(f"Total gradient norm: {total_norm:.4f}")

# Clip by norm (scales all gradients proportionally if norm exceeds max)
clipped_norm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
print(f"Returned (pre-clip) norm: {clipped_norm:.4f}")

# Clip by value (clamps each gradient element independently)
clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Key insight:** `clip_grad_norm_` returns the total norm **before** clipping — useful for monitoring gradient health during training.

---

## 8. Parameter Utilities

### parameters_to_vector / vector_to_parameters

Flatten all model parameters into a single vector (useful for L-BFGS, evolutionary methods, or model comparison):

```python
from torch.nn.utils import parameters_to_vector, vector_to_parameters

model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

# Flatten all parameters
vec = parameters_to_vector(model.parameters())
print(f"All params as vector: {vec.shape}")  # (77,)

# Modify and write back
vec *= 0.5
vector_to_parameters(vec, model.parameters())
```

### skip_init — Create Modules Without Initializing Weights

For very large models, the default weight initialization can be slow and wasteful (especially if you're loading a checkpoint immediately):

```python
from torch.nn.utils import skip_init

# Normal: allocates + initializes weights (slow for large models)
linear = nn.Linear(10000, 10000)

# Skip init: allocates uninitialized memory (fast)
linear = skip_init(nn.Linear, 10000, 10000)
# Then load your checkpoint:
# linear.load_state_dict(torch.load("checkpoint.pt"))
```

---

## Summary

| Utility | What It Does | When to Use |
|---------|-------------|-------------|
| `parametrize` | Constrain weights (orthogonal, symmetric, positive) | Research, stable training |
| `prune` | Zero out weights by magnitude/structure | Model compression, edge deploy |
| `spectral_norm` | Normalize by largest singular value | GAN training stability |
| `weight_norm` | Decouple magnitude and direction | Faster convergence |
| `pack_padded_sequence` | Efficient variable-length RNN processing | Any RNN with variable lengths |
| `fuse_conv_bn_eval` | Merge Conv+BN for inference | Inference optimization |
| `nested_tensor` | Variable-length batches without padding | Attention, NLP, Flash Attention |
| `clip_grad_norm_` | Prevent exploding gradients | Any training with transformers |
| `skip_init` | Skip weight initialization | Loading large pretrained models |

---

## Further Reading

- **`torch.nn.utils.parametrize` docs**: [pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html)
- **Pruning tutorial**: [pytorch.org/tutorials/intermediate/pruning_tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- **NestedTensor**: [pytorch.org/docs/stable/nested.html](https://pytorch.org/docs/stable/nested.html)

---

<div align="center">

[← Previous Module](../14_testing/) | [🏠 Home](../README.md) | [Next Module →](../16_activation_checkpointing/)

**[📓 Open Notebook](../notebooks/14_practical_utilities.ipynb)** — Interactive version

</div>
