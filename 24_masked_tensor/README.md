<div align="center">

[← Previous Module](../23_fx_transforms/) | [🏠 Home](../README.md) | [Next Module →](../25_triton_kernels/)

</div>

---

> **Module 24** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 02 — Tensors](../02_tensors/), [Module 04 — Neural Networks](../04_neural_networks/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — torch.masked API, MaskedTensor, semantics |
| [`masked_tensor_basics.py`](masked_tensor_basics.py) | Masked reductions, softmax, padded sequences, mask propagation |

---

# torch.masked — First-Class Missing Data in PyTorch

## Table of Contents

1. [The Problem: Missing Data & Masking](#1-the-problem-missing-data--masking)
2. [What is MaskedTensor?](#2-what-is-maskedtensor)
3. [Creating MaskedTensors](#3-creating-maskedtensors)
4. [Masked Reductions](#4-masked-reductions)
5. [Masked Softmax](#5-masked-softmax)
6. [Masked Log Softmax and Normalize](#6-masked-log-softmax-and-normalize)
7. [MaskedTensor Semantics](#7-maskedtensor-semantics)
8. [Practical Example: Padded Sequence Mean](#8-practical-example-padded-sequence-mean)
9. [Practical Example: Masked Attention](#9-practical-example-masked-attention)
10. [MaskedTensor vs Manual Masking](#10-maskedtensor-vs-manual-masking)
11. [Current Limitations](#11-current-limitations)
12. [Upstream Updates (June 16, 2026)](#12-upstream-updates-june-16-2026)
13. [Summary](#13-summary)

---

## 1. The Problem: Missing Data & Masking

Missing or invalid data appears in virtually every domain of deep learning:

| Domain | Scenario | What's "Missing" |
|--------|----------|-----------------|
| **NLP** | Padded sequences in a batch | Positions beyond each sequence's true length |
| **Vision** | Irregular shapes, masked regions | Pixels outside the region of interest |
| **Tabular** | Incomplete records | Columns with no observed value |
| **Attention** | Causal masks, padding masks | Future tokens, padding positions |

The standard workarounds all have drawbacks:

### Approach 1: Sentinel Values

```python
# Replace missing values with 0 — but 0 is a valid number!
padded = torch.zeros(batch_size, max_len)
for i, seq in enumerate(sequences):
    padded[i, :len(seq)] = seq
mean = padded.mean(dim=1)  # WRONG — includes padding zeros
```

The mean is diluted by the padding zeros. For a sequence of length 3 padded to length 10, you compute `sum / 10` instead of `sum / 3`.

### Approach 2: masked_fill with -inf

```python
# Common for attention: fill masked positions with -inf before softmax
scores = query @ key.T
scores = scores.masked_fill(mask == 0, float('-inf'))
probs = torch.softmax(scores, dim=-1)
# Works — but produces NaN if an entire row is masked
```

This works for softmax specifically, but is brittle. Different operations need different sentinel values (`-inf` for softmax, `0` for sum, `+inf` for min), and you must remember which to use.

### Approach 3: Manual Boolean Masks

```python
# Correct but verbose
mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
masked_sum = (data * mask.float()).sum(dim=1)
masked_mean = masked_sum / mask.float().sum(dim=1)
```

This is correct, but you carry two tensors (`data` and `mask`) through every operation, manually applying the mask at each step. It's easy to forget, and bugs are subtle.

**The core issue**: PyTorch operations don't natively understand that some elements are "not there." You must manually propagate this information, and every operation needs its own masking logic.

---

## 2. What is MaskedTensor?

`MaskedTensor` is a **tensor subclass** that bundles data and a boolean mask into a single object. The mask is a first-class citizen — operations automatically respect it.

```python
from torch.masked import MaskedTensor

data = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0])
mask = torch.tensor([True, True, True, False, False])

mt = MaskedTensor(data, mask)
# MaskedTensor(
#   [  1.0000,   2.0000,   3.0000,       --,       --]
# )
```

Key properties:

- **`mt.get_data()`** — returns the underlying data tensor
- **`mt.get_mask()`** — returns the boolean mask
- **`True` means valid**, `False` means masked/missing
- Masked elements display as `--` in the repr
- Operations propagate the mask automatically

MaskedTensor is currently a **prototype feature** (as of PyTorch 2.14). Import it with:

```python
from torch.masked import MaskedTensor
```

The `torch.masked` module also provides standalone masked operations that work on regular tensors with explicit mask arguments — useful even without MaskedTensor.

---

## 3. Creating MaskedTensors

### From Data + Mask

The most common pattern: pair a data tensor with a boolean mask tensor of the same shape.

```python
import torch
from torch.masked import MaskedTensor

# 1D
data = torch.tensor([10.0, 20.0, 30.0, 0.0])
mask = torch.tensor([True, True, True, False])
mt = MaskedTensor(data, mask)

# 2D — batch of sequences with padding
data = torch.tensor([
    [1.0, 2.0, 3.0, 0.0, 0.0],
    [4.0, 5.0, 0.0, 0.0, 0.0],
    [6.0, 7.0, 8.0, 9.0, 0.0],
])
lengths = torch.tensor([3, 2, 4])
mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
# mask:
# tensor([[ True,  True,  True, False, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True,  True, False]])

mt = MaskedTensor(data, mask)
```

### From Padded Sequences

When working with `nn.utils.rnn.pad_sequence`, you already have the lengths — just build the mask:

```python
sequences = [torch.randn(3), torch.randn(5), torch.randn(2)]
padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
lengths = torch.tensor([3, 5, 2])
mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

mt = MaskedTensor(padded, mask)
```

### Mask Requirements

- **Shape**: mask must be the same shape as data (broadcastable masks are not supported)
- **Dtype**: must be `torch.bool`
- **Convention**: `True` = valid, `False` = masked

---

## 4. Masked Reductions

The `torch.masked` module provides reduction functions that correctly ignore masked elements. These work with plain tensors + mask arguments — you don't need `MaskedTensor` to use them.

### torch.masked.sum

```python
data = torch.tensor([
    [1.0, 2.0, 3.0, 0.0, 0.0],
    [4.0, 5.0, 0.0, 0.0, 0.0],
])
mask = torch.tensor([
    [True, True, True, False, False],
    [True, True, False, False, False],
])

# Regular sum includes padding zeros (happens to be correct for sum, but misleading)
regular_sum = data.sum(dim=1)        # tensor([6., 9.])

# Masked sum — explicitly only sums valid elements
masked_sum = torch.masked._ops.sum(data, dim=1, mask=mask)
```

### torch.masked.mean

This is where masking matters most — mean divides by the count of **valid** elements, not the total.

```python
# Regular mean includes padding → WRONG
regular_mean = data.mean(dim=1)      # tensor([1.2, 1.8])  (divides by 5)

# Masked mean → CORRECT
# Row 0: (1+2+3)/3 = 2.0
# Row 1: (4+5)/2 = 4.5
masked_mean = torch.masked._ops.mean(data, dim=1, mask=mask)
```

### Other Masked Reductions

| Function | What it does |
|----------|-------------|
| `torch.masked._ops.sum` | Sum of valid elements |
| `torch.masked._ops.mean` | Mean of valid elements (divides by valid count) |
| `torch.masked._ops.amax` | Maximum of valid elements |
| `torch.masked._ops.amin` | Minimum of valid elements |
| `torch.masked._ops.prod` | Product of valid elements |
| `torch.masked._ops.norm` | Norm over valid elements |
| `torch.masked._ops.var` | Variance over valid elements |
| `torch.masked._ops.std` | Standard deviation over valid elements |

All follow the same signature: `func(data, dim, *, mask)`.

---

## 5. Masked Softmax

Softmax over masked data is one of the most common needs in attention mechanisms. The manual approach uses `masked_fill` with `-inf`:

### Manual Approach

```python
scores = torch.randn(2, 5)
mask = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, False],
])

# Step 1: Fill masked positions with -inf
filled = scores.masked_fill(~mask, float('-inf'))
# Step 2: Softmax — exp(-inf) = 0, so masked positions get probability 0
probs = torch.softmax(filled, dim=1)
```

This works, but has a subtle problem: if an **entire row** is masked, `softmax([-inf, -inf, ...])` produces `NaN`.

### torch.masked.softmax

```python
probs = torch.masked.softmax(scores, dim=1, mask=mask)
```

This handles the edge cases correctly and produces zero for masked positions without the `NaN` risk.

```python
# Under the hood (simplified):
# 1. Replace masked positions with -inf
# 2. Compute softmax
# 3. Replace masked positions with 0 in the output
# 4. Handle all-masked rows gracefully
```

### Comparison

```python
scores = torch.tensor([[0.5, 1.2, 0.3, 0.0, 0.0]])
mask = torch.tensor([[True, True, True, False, False]])

# Manual
manual = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=1)
# tensor([[0.2753, 0.5545, 0.2253, 0.0000, 0.0000]])
#   — masked positions are 0 because exp(-inf)=0, but sum of valid = 1.0525 ≠ 1

# torch.masked.softmax
masked = torch.masked.softmax(scores, dim=1, mask=mask)
# tensor([[0.2615, 0.5269, 0.2141, 0.0000, 0.0000]])
#   — valid positions sum to ~1.0 (properly normalized over valid only)
```

The key difference: `torch.masked.softmax` normalizes over **valid elements only**, so the probabilities of valid positions sum to 1.0.

---

## 6. Masked Log Softmax and Normalize

### torch.masked.log_softmax

Log-softmax is used in NLL loss and related computations. The masked version correctly computes `log(softmax(x))` only over valid positions:

```python
log_probs = torch.masked.log_softmax(scores, dim=1, mask=mask)
```

Masked positions in the output are set to `0.0` (or `-inf` depending on implementation), ensuring they don't contribute to downstream loss computations.

### torch.masked.normalize

Normalize a tensor over valid elements only:

```python
# L2 normalize each row, ignoring masked positions
normalized = torch.masked.normalize(data, ord=2.0, dim=1, mask=mask)
```

This computes the norm using only valid elements, then divides each valid element by that norm. Masked positions remain unchanged in the output.

---

## 7. MaskedTensor Semantics

When you use `MaskedTensor` directly, operations follow specific rules for mask propagation.

### Unary Operations — Preserve Mask

Applying a unary function to a MaskedTensor keeps the same mask:

```python
mt = MaskedTensor(
    torch.tensor([1.0, -2.0, 3.0, 0.0]),
    torch.tensor([True, True, True, False])
)

result = mt.abs()
# Data: [1.0, 2.0, 3.0, ???]
# Mask: [True, True, True, False]
# Masked element is still masked — the abs() was applied only to valid elements
```

Other unary ops that preserve masks: `neg()`, `exp()`, `log()`, `sin()`, `cos()`, `sqrt()`, `relu()`, etc.

### Binary Operations — Intersection (AND) of Masks

When combining two MaskedTensors, an element is valid only if it's valid in **both** inputs:

```python
a = MaskedTensor(
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([True, True, False])
)
b = MaskedTensor(
    torch.tensor([10.0, 20.0, 30.0]),
    torch.tensor([True, False, True])
)

result = a + b
# Data: [11.0, ???, ???]
# Mask: [True, False, False]
# Position 0: both valid → valid (1+10=11)
# Position 1: a valid, b masked → masked
# Position 2: a masked → masked
```

This is the conservative (safe) choice: if **either** input is missing, the output is missing.

### Reductions — Collapse Mask

Reductions over a dimension collapse the mask along that dimension. The output position is valid if **any** input along the reduction axis was valid:

```python
mt = MaskedTensor(
    torch.tensor([[1.0, 2.0, 0.0],
                  [4.0, 0.0, 0.0]]),
    torch.tensor([[True, True, False],
                  [True, False, False]])
)

# Sum over dim=1
# Row 0: sum of [1.0, 2.0] = 3.0 (2 valid elements)
# Row 1: sum of [4.0] = 4.0 (1 valid element)
result = mt.sum(dim=1)
```

---

## 8. Practical Example: Padded Sequence Mean

Computing the true mean of variable-length sequences is a common task. Let's compare three approaches.

### Setup

```python
# Three sequences of different lengths, padded to max_len=5
data = torch.tensor([
    [3.0, 1.0, 4.0, 0.0, 0.0],   # length 3, true mean = 2.667
    [2.0, 7.0, 0.0, 0.0, 0.0],   # length 2, true mean = 4.5
    [5.0, 3.0, 2.0, 8.0, 0.0],   # length 4, true mean = 4.5
])
lengths = torch.tensor([3, 2, 4])
```

### Approach 1: Naive Mean (WRONG)

```python
naive_mean = data.mean(dim=1)
# tensor([1.6000, 1.8000, 3.6000])
# All wrong! Divides by 5 instead of the actual lengths.
# Sequence 0: (3+1+4+0+0)/5 = 1.6, should be (3+1+4)/3 = 2.667
```

### Approach 2: Manual Masking (Correct but Verbose)

```python
mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
masked_sum = (data * mask.float()).sum(dim=1)
masked_count = mask.float().sum(dim=1)
manual_mean = masked_sum / masked_count
# tensor([2.6667, 4.5000, 4.5000])  ← Correct!
```

This works but requires 4 lines and careful bookkeeping. In a larger pipeline, you must thread the mask through every operation.

### Approach 3: torch.masked API (Clean)

```python
mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
masked_mean = torch.masked._ops.mean(data, dim=1, mask=mask)
# tensor([2.6667, 4.5000, 4.5000])  ← Correct!
```

One function call, no manual bookkeeping. The division by the valid count is handled internally.

---

## 9. Practical Example: Masked Attention

Attention mechanisms frequently need masking: padding masks (ignore padding tokens), causal masks (prevent attending to future positions), or combined masks.

### Padding Mask in Attention

```python
batch_size, seq_len, d_model = 2, 6, 8
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

lengths = torch.tensor([4, 6])  # sequence 0 has 4 real tokens, sequence 1 has 6

# Build padding mask: [batch, 1, 1, seq_len] for broadcasting
pad_mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

# Attention scores
scores = (query @ key.transpose(-2, -1)) / (d_model ** 0.5)
# scores shape: [batch, seq_len, seq_len]

# Apply padding mask — masked positions get -inf
scores_2d_mask = pad_mask.squeeze(1)  # [batch, 1, seq_len]
scores = scores.masked_fill(~scores_2d_mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
# NaN appears in rows where all positions are masked
attn_weights = attn_weights.nan_to_num(0.0)  # cleanup
```

### Using torch.masked.softmax

```python
mask_2d = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
mask_3d = mask_2d.unsqueeze(1).expand(-1, seq_len, -1)

scores = (query @ key.transpose(-2, -1)) / (d_model ** 0.5)
attn_weights = torch.masked.softmax(scores, dim=-1, mask=mask_3d)
```

No `masked_fill`, no `nan_to_num`. The masked softmax handles everything, including the all-masked-row edge case.

---

## 10. MaskedTensor vs Manual Masking

A side-by-side comparison for common operations:

| Operation | Manual Masking | torch.masked API |
|-----------|---------------|-----------------|
| **Sum** | `(data * mask.float()).sum(dim)` | `torch.masked._ops.sum(data, dim, mask=mask)` |
| **Mean** | `(data * mask.float()).sum(dim) / mask.sum(dim)` | `torch.masked._ops.mean(data, dim, mask=mask)` |
| **Max** | `data.masked_fill(~mask, -inf).max(dim)` | `torch.masked._ops.amax(data, dim, mask=mask)` |
| **Min** | `data.masked_fill(~mask, inf).min(dim)` | `torch.masked._ops.amin(data, dim, mask=mask)` |
| **Softmax** | `softmax(data.masked_fill(~mask, -inf), dim)` | `torch.masked.softmax(data, dim, mask=mask)` |
| **Normalize** | Compute norm manually, divide | `torch.masked.normalize(data, ord, dim, mask=mask)` |

### When to Use Each

**Use `torch.masked.*` functions when:**
- You need masked reductions (sum, mean, amax, etc.)
- You need masked softmax / log_softmax
- You want cleaner, less error-prone code

**Use `MaskedTensor` when:**
- You want automatic mask propagation through a pipeline
- You're prototyping and want to verify your masking logic

**Stick with manual masking when:**
- Performance is critical and you need full control
- You need operations not yet supported by MaskedTensor
- You're working with complex multi-mask scenarios

---

## 11. Current Limitations

MaskedTensor and `torch.masked` are in **prototype** status. Be aware of:

### Not All Ops Are Supported

MaskedTensor works with a subset of PyTorch operations. Unsupported ops will raise errors:

```python
# These work:
mt.sum(), mt.mean(), mt.abs(), mt + mt, mt * 2

# These may not work (as of 2.14):
# torch.nn.functional.linear(mt, weight)  — not all nn.functional ops supported
# mt.view(...)  — some shape ops may not be supported
```

### Performance Overhead

MaskedTensor is a Python tensor subclass, which means:
- Extra Python dispatch overhead on every operation
- Not yet optimized by `torch.compile` in all cases
- For hot loops, manual masking may be faster

### No Gradient Through Mask

The mask itself is not differentiable — it's a fixed boolean tensor. You cannot learn which elements to mask.

### Sparse Mask Support

MaskedTensor supports sparse masks (COO and CSR) for memory efficiency when most elements are masked:

```python
sparse_mask = mask.to_sparse()
mt = MaskedTensor(data, sparse_mask)
```

This can save memory when the mask is mostly `False` (most elements are masked).

### API Stability

The `torch.masked` API may change between releases. Pin your PyTorch version for reproducibility, and check the release notes when upgrading.

---

## 12. Upstream Updates (June 16, 2026)

Recent PyTorch developments relevant to masking and related systems:

### Dynamo O(N²) Decomposition Fix (#177927)

A performance bug was fixed where Dynamo's decomposition of certain ops had quadratic complexity in the number of elements. This affects any workload using `torch.compile` with masked operations, as the decomposed ops could include masking logic.

### TokenSwitch for Distributed Token Routing (#178712)

A new `TokenSwitch` primitive for Mixture-of-Experts models enables efficient token routing across devices. This is relevant to masking because token routing inherently involves masking — tokens are assigned to specific experts, and the routing mask determines which tokens go where.

### Native Itertools Variables in Dynamo (#186973, #186974)

Dynamo now handles Python `itertools` constructs (like `itertools.chain`, `itertools.product`) as native variables, reducing graph breaks. This benefits masked operations that iterate over mask patterns or dynamically construct masks in compiled code.

### NVGEMM Disk Cache (#187013)

NVIDIA's GEMM kernel auto-tuning results are now cached to disk, avoiding re-tuning on subsequent runs. While not directly mask-related, this improves the startup time of any compiled workload, including those using masked operations.

### MPS Metal Kernel Migrations

Ongoing work to migrate MPS (Apple Silicon) kernels from Objective-C++ to native Metal shaders. This improves performance of operations on Apple hardware, including masked operations on MPS devices.

### DTensor Reduction Strategies (#179201)

New reduction strategies for DTensor (Distributed Tensor) improve how reductions are performed across devices. Since masked reductions are a key use case, this work may eventually enable efficient distributed masked operations.

---

## 13. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **The Problem** | Missing data is everywhere: padding, irregular shapes, missing values. Manual masking is verbose and error-prone. |
| **MaskedTensor** | A tensor subclass pairing data + boolean mask. Operations respect the mask automatically. |
| **torch.masked.softmax** | Softmax that correctly ignores masked positions and normalizes over valid elements only. |
| **Masked Reductions** | `torch.masked._ops.sum/mean/amax/amin` — correct reductions that ignore masked elements. |
| **Mask Convention** | `True` = valid, `False` = masked/missing. |
| **Unary Ops** | Preserve the mask. |
| **Binary Ops** | AND the masks (both must be valid). |
| **Prototype Status** | Not all ops supported. API may change. Use for new prototyping, not critical production paths. |

### Quick Reference

```python
from torch.masked import MaskedTensor

# Create
mt = MaskedTensor(data, mask)

# Inspect
mt.get_data()     # underlying data tensor
mt.get_mask()     # boolean mask tensor

# Masked reductions (work on plain tensors too)
torch.masked._ops.sum(data, dim=1, mask=mask)
torch.masked._ops.mean(data, dim=1, mask=mask)
torch.masked._ops.amax(data, dim=1, mask=mask)
torch.masked._ops.amin(data, dim=1, mask=mask)
torch.masked._ops.prod(data, dim=1, mask=mask)
torch.masked._ops.var(data, dim=1, mask=mask)

# Masked softmax / log_softmax / normalize
torch.masked.softmax(data, dim=1, mask=mask)
torch.masked.log_softmax(data, dim=1, mask=mask)
torch.masked.normalize(data, ord=2, dim=1, mask=mask)
```

---

## Further Reading

- [torch.masked Official Docs](https://pytorch.org/docs/stable/masked.html) — API reference
- [MaskedTensor Overview](https://pytorch.org/docs/stable/masked.html#maskedtensor) — creation, semantics, sparsity
- [MaskedTensor RFC](https://github.com/pytorch/rfcs/pull/45) — original design proposal
- [Attention Mechanisms](../09_attention/) — where masked softmax is most commonly used
- [Tensor Subclassing](../19_torch_function_dispatch/) — how MaskedTensor is implemented under the hood

---

<div align="center">

[← Previous Module](../23_fx_transforms/) | [🏠 Home](../README.md) | [Next Module →](../25_triton_kernels/)

**Notebook**: [`24_masked_tensor.ipynb`](../notebooks/24_masked_tensor.ipynb)

</div>
