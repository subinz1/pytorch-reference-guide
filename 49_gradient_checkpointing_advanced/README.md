# Module 49: Gradient Checkpointing — Advanced

## Overview

[Module 16](../16_activation_checkpointing/) covers basic activation checkpointing.
This module goes deeper: **selective** policies, **reentrant vs non-reentrant**
implementations, activation **offload** to CPU, and how checkpointing interacts
with `torch.compile` and DDP/FSDP. Use these tools when naive
`checkpoint(layer, x)` is not enough for memory or correctness.

## Key Concepts

### Reentrant vs Non-Reentrant

| Mode | Flag | Notes |
|------|------|-------|
| Reentrant (legacy) | `use_reentrant=True` | Re-enters autograd; needed for some older patterns; more edge cases |
| Non-reentrant (preferred) | `use_reentrant=False` | Cleaner saved-tensor handling; recommended in modern PyTorch |

Prefer `use_reentrant=False` unless you hit a known compatibility issue.

### Selective Activation Checkpointing (SAC)

Instead of recomputing *everything* in a region, apply a **policy** that saves
cheap ops (e.g. pointwise) and recomputes expensive ones (matmul/attention),
or the reverse — depending on memory vs compute priorities.

```python
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
```

Policies typically inspect `ops` / op types during the forward pack phase.

### Offloading Activations

When GPU memory is still tight after SAC, offload saved activations to **CPU
(pinned) memory** during forward and prefetch them for backward. This adds PCIe
traffic but can unlock larger batch sizes.

### Interaction with Distributed & Compile

- **DDP / FSDP**: Checkpoint *inside* each transformer block so recompute stays
  local; avoid wrapping the entire model in one checkpoint.
- **`torch.compile`**: Non-reentrant checkpointing composes better; graph breaks
  can appear at checkpoint boundaries — profile before/after.

## Examples

```python
import torch
from torch.utils.checkpoint import checkpoint

def block(x, weight):
    return torch.nn.functional.gelu(x @ weight)

x = torch.randn(2, 128, requires_grad=True)
w = torch.randn(128, 128, requires_grad=True)

y = checkpoint(block, x, w, use_reentrant=False)
y.sum().backward()
```

## When to Use

- Model OOMs with full activation storage (LLM / ViT training).
- You need a finer memory/compute tradeoff than “checkpoint every layer”.
- GPU memory is scarce but host RAM / PCIe bandwidth is available (offload).
- Skip if the model already fits — checkpointing adds ~20–40% compute.

## Files in This Module

| File | Description |
|------|-------------|
| `selective_checkpoint.py` | Reentrant vs non-reentrant, SAC-style policy, offload sketch |

## References

- [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- [Activation Checkpointing tutorial](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint)
- Module 16: [Activation Checkpointing](../16_activation_checkpointing/)
- Module 10: [Distributed](../10_distributed/) · Module 26: [Memory Profiling](../26_memory_profiling/)
