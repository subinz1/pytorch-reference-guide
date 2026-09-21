# Profiler Workflow

## Capture a trace with a question in mind

Profile a stable, warmed-up workload. Decide whether you are investigating input
stalls, kernel time, synchronization, or memory before collecting a trace.

```python
import torch
from torch.profiler import ProfilerActivity, profile, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=4, repeat=1),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

## Checklist

- Profile enough steps to exclude initialization and cache warm-up.
- Inspect both CPU and CUDA timelines before optimizing kernels.
- Look for synchronization calls such as `item()`, frequent copies, and tiny kernels.
- Export a Chrome or Perfetto trace for overlap and queueing questions.
- Re-measure the full workload after every optimization.

Profiler output is a map, not a verdict. Confirm the suspected bottleneck with a
small controlled change.

