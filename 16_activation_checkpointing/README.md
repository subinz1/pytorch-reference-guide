# Module 16: Activation Checkpointing — Trading Memory for Compute

*Day 2 of the incremental learning series*

---

## The Problem: Activations Eat Your GPU Memory

When training a neural network, PyTorch stores all intermediate activations (outputs of each layer) during the forward pass. These are needed for the backward pass to compute gradients. For large models, **activations consume far more memory than the model parameters themselves**.

Example memory breakdown for a 7B parameter Transformer:
```
Model parameters:     14 GB  (7B × 2 bytes in BF16)
Optimizer state:      28 GB  (Adam: 2× params in FP32)
Activations:         ~60 GB  (scales with batch_size × seq_len × layers)
Total:              ~102 GB  — doesn't fit on an 80GB GPU!
```

**Activation checkpointing** solves this by **not saving** activations during forward. During backward, it **recomputes** them on the fly. This trades ~33% more compute for ~60% less memory.

---

## Table of Contents

1. [How Activation Checkpointing Works](#1-how-activation-checkpointing-works)
2. [Basic Usage: torch.utils.checkpoint](#2-basic-usage)
3. [checkpoint_sequential for Sequential Models](#3-checkpoint_sequential)
4. [Selective Activation Checkpointing (SAC)](#4-selective-activation-checkpointing)
5. [CheckpointPolicy Options](#5-checkpointpolicy-options)
6. [Integration with torch.compile](#6-integration-with-torchcompile)
7. [Practical Guidelines](#7-practical-guidelines)

---

## 1. How Activation Checkpointing Works

### Normal Training (No Checkpointing)

```
Forward pass:  Input → [Layer 1] → a₁ → [Layer 2] → a₂ → [Layer 3] → a₃ → Loss
                         save a₁      save a₂       save a₃

Backward pass: Uses saved a₁, a₂, a₃ to compute gradients
Memory: O(N) — stores all N layer activations
```

### With Activation Checkpointing

```
Forward pass:  Input → [Layer 1] → [Layer 2] → [Layer 3] → Loss
                        (discard)   (discard)    (discard)

Backward pass: 
  Need a₃ → Recompute: Input → Layer 1 → Layer 2 → Layer 3 → a₃ ✓
  Need a₂ → Recompute: Input → Layer 1 → Layer 2 → a₂ ✓
  Need a₁ → Recompute: Input → Layer 1 → a₁ ✓

Memory: O(1) per checkpointed segment
Compute: ~1.33× (one extra forward pass)
```

In practice, you checkpoint **segments** of the model (e.g., each Transformer layer), not the entire model. This gives a good memory/compute tradeoff.

---

## 2. Basic Usage

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerModel(torch.nn.Module):
    def __init__(self, d_model=512, n_layers=12):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerLayer(d_model) for _ in range(n_layers)
        ])
    
    def forward(self, x, use_checkpoint=False):
        for layer in self.layers:
            if use_checkpoint:
                # Checkpoint each layer — its activations are NOT saved
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
```

### Key Parameter: `use_reentrant=False`

Always use `use_reentrant=False` (the modern, recommended path):
- **`use_reentrant=True`** (legacy): Uses a different autograd mechanism, has subtle bugs with certain ops
- **`use_reentrant=False`** (recommended): Works correctly with all ops, `torch.compile`, and distributed training

---

## 3. checkpoint_sequential

For `nn.Sequential` models, there's a convenience wrapper:

```python
from torch.utils.checkpoint import checkpoint_sequential

model = torch.nn.Sequential(
    TransformerLayer(512),
    TransformerLayer(512),
    TransformerLayer(512),
    TransformerLayer(512),
)

x = torch.randn(8, 32, 512, requires_grad=True)

# Divide into 2 segments — each segment is checkpointed
output = checkpoint_sequential(model, segments=2, input=x, use_reentrant=False)
```

`segments` controls how many checkpointed groups to split the sequential into. More segments = more memory savings but more recomputation.

---

## 4. Selective Activation Checkpointing (SAC)

**The problem with basic checkpointing:** It recomputes EVERYTHING, including expensive operations like matrix multiplies and attention. Ideally, we'd save expensive outputs and only recompute cheap ones (like activations, norms).

**SAC lets you choose per-operation** whether to save or recompute:

```python
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

# Policy function: decides per-op whether to save or recompute
def policy_fn(ctx, op, *args, **kwargs):
    # Save expensive ops (matmul, attention)
    if op in (torch.ops.aten.mm.default, 
              torch.ops.aten.bmm.default,
              torch.ops.aten._scaled_dot_product_flash_attention.default):
        return CheckpointPolicy.MUST_SAVE
    # Recompute cheap ops (relu, add, norm, etc.)
    return CheckpointPolicy.PREFER_RECOMPUTE

# Use with checkpoint's context_fn parameter
context_fn = create_selective_checkpoint_contexts(policy_fn)

x = checkpoint(
    layer, x,
    use_reentrant=False,
    context_fn=context_fn,
)
```

### Shortcut: Pass a List of Ops to Save

```python
# Instead of a policy function, just list the ops you want to save
ops_to_save = [
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
]

context_fn = create_selective_checkpoint_contexts(ops_to_save)

x = checkpoint(layer, x, use_reentrant=False, context_fn=context_fn)
```

---

## 5. CheckpointPolicy Options

| Policy | Behavior |
|--------|----------|
| `MUST_SAVE` | Always save this op's output (never recompute) |
| `PREFER_SAVE` | Save unless `torch.compile` decides otherwise |
| `MUST_RECOMPUTE` | Always recompute (never save) |
| `PREFER_RECOMPUTE` | Recompute unless `torch.compile` decides otherwise |
| `MUST_CPU_OFFLOAD` | Save to CPU during forward, reload to GPU during backward |
| `PREFER_CPU_OFFLOAD` | Offload unless `torch.compile` decides otherwise |

The `PREFER_*` variants allow `torch.compile` to override the decision based on its global optimization analysis. The `MUST_*` variants are strict.

---

## 6. Integration with torch.compile

Activation checkpointing works with `torch.compile`. When compiled, the compiler can make even smarter decisions about what to save vs recompute:

```python
model = TransformerModel(d_model=512, n_layers=12)

# Compile with checkpointing
compiled_model = torch.compile(model)

x = torch.randn(8, 32, 512)
output = compiled_model(x, use_checkpoint=True)
```

When using `PREFER_*` policies with compiled code, the compiler may override your suggestions if it determines a different strategy is more efficient.

---

## 7. Practical Guidelines

### When to Use Activation Checkpointing

| Scenario | Recommendation |
|----------|---------------|
| Model fits in GPU memory | Don't checkpoint (unnecessary overhead) |
| OOM with desired batch size | Checkpoint every Transformer layer |
| Still OOM | Add selective checkpointing (save matmuls, recompute rest) |
| Still OOM | Combine with FSDP2, gradient accumulation, or CPU offload |

### Rules of Thumb

1. **Checkpoint at the Transformer layer granularity** — each `TransformerEncoderLayer` or decoder layer is one checkpoint boundary
2. **Always use `use_reentrant=False`** — the legacy reentrant mode has known issues
3. **With SAC, save matmuls and attention** — these are 10-100x more expensive than activations/norms
4. **Memory savings**: ~50-70% activation memory reduction with basic checkpointing
5. **Compute overhead**: ~30% more training time (one extra forward pass per checkpointed segment)
6. **Stacks with everything**: Works with AMP, DDP, FSDP2, torch.compile

### Memory Estimation

```
Without checkpointing:
  Activation memory ≈ batch_size × seq_len × hidden_dim × num_layers × 2 bytes (BF16)

With checkpointing (per layer):
  Activation memory ≈ batch_size × seq_len × hidden_dim × 2 × 2 bytes
  (Only stores input/output of each checkpointed segment)

Savings = (num_layers - 2) / num_layers ≈ 90%+ for 24+ layer models
```

---

## Further Reading

- [PyTorch Activation Checkpointing Tutorial](https://pytorch.org/docs/stable/checkpoint.html)
- [Selective Activation Checkpointing for torch.compile](https://dev-discuss.pytorch.org/)
- Source code: `torch/utils/checkpoint.py`
