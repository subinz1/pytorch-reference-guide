# Mixed Precision

## Let autocast choose operation dtypes

Automatic mixed precision reduces memory traffic and can improve accelerator
throughput. Keep model parameters in their normal training dtype unless there
is a deliberate lower-precision policy.

```python
import torch

device_type = "cuda"
scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=torch.cuda.is_available()):
    loss = criterion(model(inputs), targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

## Checklist

- Use BF16 when the target hardware supports it and numerical range matters.
- Unscale before inspecting or clipping gradients.
- Keep reductions and numerically sensitive custom operations in a safe dtype.
- Check for non-finite loss before blaming the scaler.
- Benchmark accuracy and throughput separately; faster is not automatically valid.

Autocast is scoped. Put only the forward and loss calculation inside it, then
keep optimizer bookkeeping explicit.

