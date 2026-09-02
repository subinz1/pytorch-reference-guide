# Module 45: PyTorch Profiler Deep Dive

## Overview

The **PyTorch Profiler** (`torch.profiler`) provides fine-grained visibility into
CPU and GPU execution, memory allocation, and operator-level timing. It replaces
the legacy `torch.autograd.profiler` with a unified API that integrates with
Chrome Trace Viewer, TensorBoard, and HTA (Holistic Trace Analysis).

## Key Concepts

### Profiler Context Manager

The primary entry point is `torch.profiler.profile()`, which records events within
a `with` block. Configure it with `activities` (CPU, CUDA), `schedule` for
warm-up/active/repeat cycles, and `on_trace_ready` callbacks for automatic export.

### Scheduling

`torch.profiler.schedule(wait, warmup, active, repeat)` controls when the profiler
records. The **wait** phase skips steps, **warmup** discards initial noisy steps,
and **active** captures the trace. This avoids profiling cold-start overhead.

### Trace Export

- **Chrome Trace**: `prof.export_chrome_trace("trace.json")` produces a JSON file
  viewable at `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/).
- **TensorBoard**: `torch.profiler.tensorboard_trace_handler("./logs")` writes
  traces consumable by the TensorBoard PyTorch Profiler plugin.
- **Stacks**: `prof.export_stacks("stacks.txt")` exports flame-graph-ready data.

### Key Averages

`prof.key_averages()` aggregates events by operator name, returning a table with
`self_cpu_time_total`, `self_cuda_time_total`, `cpu_memory_usage`, and call counts.
Sort by any column to find hotspots.

### Memory Profiling

Enable `profile_memory=True` to track tensor allocations. Combine with
`torch.cuda.memory._record_memory_history()` for allocation-level snapshots
viewable in the [Memory Visualizer](https://pytorch.org/memory_viz).

## Examples

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step(model, data)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## Files in This Module

| File | Description |
|------|-------------|
| `profiler_basics.py` | Context manager usage, scheduling, key averages |
| `trace_analysis.py` | Advanced trace analysis, memory snapshots, CUDA activity |

## References

- [PyTorch Profiler docs](https://pytorch.org/docs/stable/profiler.html)
- [Profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [TensorBoard plugin](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
- [Memory Visualizer](https://pytorch.org/memory_viz)
