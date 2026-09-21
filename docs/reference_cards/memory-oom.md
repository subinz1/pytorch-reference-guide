# Out-of-Memory Triage

## Separate peak pressure from a leak

An OOM can be caused by one large activation peak, fragmentation, or tensors
that remain referenced across iterations. First determine which pattern you
have.

```python
import torch

torch.cuda.reset_peak_memory_stats()
loss = train_step(batch)
peak = torch.cuda.max_memory_allocated() / 2**20
current = torch.cuda.memory_allocated() / 2**20
print(f"current={current:.1f} MiB peak={peak:.1f} MiB")
```

## Checklist

- Compare allocated memory after several equal-sized iterations.
- Do not store graph-attached losses, outputs, or metrics in a Python list.
- Use `loss.detach()` or `loss.item()` for scalar logging.
- Reduce activation memory with checkpointing, smaller microbatches, or sequence
  length changes before shrinking the model.
- Use memory snapshots when allocation history—not only peak size—matters.

Calling `empty_cache()` does not free tensors that are still referenced. Fix
ownership and lifetime before treating allocator symptoms.

