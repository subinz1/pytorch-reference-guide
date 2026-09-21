# Optimizer Lifecycle

## Preserve the update order

A standard optimizer step has a strict sequence. Keeping it explicit makes
gradient accumulation, clipping, AMP, and schedulers easier to reason about.

```python
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

## Checklist

- Zero gradients before the next accumulation window, not after arbitrary code.
- Use `set_to_none=True` unless code relies on zero-filled `.grad` tensors.
- Clip after backward and before the optimizer step.
- Decide whether the scheduler advances per batch or per epoch; follow its API.
- Save and restore model, optimizer, scheduler, scaler, and step counters together.

For gradient accumulation, divide the loss by the accumulation count, call
`backward()` each microbatch, and step only at the accumulation boundary.

