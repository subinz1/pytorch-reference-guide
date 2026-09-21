# DataLoader Performance

## Measure the input pipeline

A fast model can still idle while the next batch is prepared. Start by measuring
end-to-end iteration time before changing worker counts or transforms.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

for inputs, targets in loader:
    inputs = inputs.to("cuda", non_blocking=True)
    targets = targets.to("cuda", non_blocking=True)
    train_step(inputs, targets)
```

## Checklist

- Use `pin_memory=True` with CUDA and transfer batches using
  `non_blocking=True`.
- Increase `num_workers` only after measuring throughput and CPU pressure.
- Use `persistent_workers=True` for repeated epochs when workers are enabled.
- Keep expensive random transforms off the main training thread.
- Seed workers deliberately when augmentation must be reproducible.

The best worker count depends on storage, decoding cost, CPU cores, and the
number of distributed ranks. Benchmark it in the real training environment.

