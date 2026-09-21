# Training and Evaluation Modes

## `train()` and `eval()` change behavior

The mode flag affects modules such as Dropout and BatchNorm. It does not disable
gradient tracking on its own.

```python
import torch

model.train()
loss = criterion(model(train_inputs), train_targets)
loss.backward()

model.eval()
with torch.inference_mode():
    predictions = model(validation_inputs)
```

## Checklist

- Call `model.train()` before each training phase.
- Call `model.eval()` before validation, test, or production inference.
- Pair evaluation with `torch.inference_mode()` when gradients are unnecessary.
- Restore training mode after validation inside a training loop.
- Test evaluation output twice when Dropout should be disabled.

`inference_mode()` is stronger than `no_grad()` and is ideal for pure
inference. Use `no_grad()` instead when code must later re-enable gradients
for tensors created inside the block.

