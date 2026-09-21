# Inference Batching

## Optimize a correct serving path first

Inference should use evaluation mode and disable autograd. Batch requests only
when their shapes, latency budget, and error handling make the grouping valid.

```python
import torch

model.eval()

def predict(batch: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        batch = batch.to(device, non_blocking=True)
        return model(batch).softmax(dim=-1)
```

## Checklist

- Use `model.eval()` and `torch.inference_mode()`.
- Keep preprocessing and postprocessing outside the model timing measurement.
- Define a maximum batch size and a maximum queue delay independently.
- Bucket incompatible sequence or image shapes before batching.
- Return results in the original request order.

Measure p50, p95, and p99 latency as well as throughput. A larger batch may
improve accelerator utilization while making the user-visible tail latency worse.

