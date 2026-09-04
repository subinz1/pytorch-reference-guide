# Reading Chrome / Perfetto Traces from torch.profiler

Companion guide for [Module 45 — PyTorch Profiler](../45_torch_profiler/).

Export a trace with:

```python
prof.export_chrome_trace("trace.json")
```

Open it in **[Perfetto UI](https://ui.perfetto.dev/)** (preferred) or `chrome://tracing`.

## What You Are Looking At

A Chrome trace is a timeline of **slices** (events with start + duration) on
parallel **tracks** (CPU threads, CUDA streams, Python threads).

| Track type | Typical name | Meaning |
|------------|--------------|---------|
| CPU | `python`, `MainThread`, worker threads | Operator scheduling, DataLoader, GIL work |
| CUDA | `stream 7`, `gpu 0` | Kernels actually running on the device |
| Overhead | `Trace`, profiler internals | Ignore unless huge |

### Critical mental model

**CPU `aten::mm` duration ≠ GPU kernel time.** The CPU slice often only
*launches* the kernel; the real work appears later on a CUDA stream track.
Look for **gaps** (idle GPU) and **gaps on CPU** (waiting on device sync).

## Step-by-Step Reading Workflow

1. **Zoom to a steady-state step** — skip profiler warmup / first iteration.
2. **Find one training step** — pattern: forward ops → `Backward` → optimizer.
3. **Check GPU utilization** — long blank regions on CUDA tracks mean the GPU
   is starved (DataLoader, sync, Python overhead).
4. **Find the tallest CUDA kernels** — sort or visually scan for wide slices
   (`gemm`, `cutlass`, `cudnn`, fused attention).
5. **Correlate CPU → GPU** — click a CPU op; Perfetto highlights related flows
   when present. Sync points (`cudaDeviceSynchronize`, `.item()`, `.cpu()`)
   often create bubbles.
6. **Look for memcpy** — `cudaMemcpyAsync` HtoD/DtoH between steps suggests
   missing pinned memory or accidental host transfers.

## Common Patterns & Fixes

| Pattern in trace | Likely cause | Fix direction |
|------------------|--------------|---------------|
| GPU idle between steps | Slow DataLoader / augment | `num_workers`, `pin_memory`, prefetch |
| Many tiny CUDA kernels | Unfused pointwise ops | `torch.compile`, fusion, CUDA Graphs |
| Long `cudaMemcpy` | Host↔device churn | Keep batch on GPU; pinned host buffers |
| CPU spike then GPU burst | Launch overhead | CUDA Graphs / `reduce-overhead` |
| Overlapping memcpy + compute | Good | Keep it; check streams are concurrent |
| Serializer: GPU waits CPU | Python / GIL | Reduce `.item()`, logging, host callbacks |

## Annotating Your Own Regions

```python
with torch.profiler.record_function("my_feature_block"):
    do_work()
```

Custom labels appear as named slices — essential when the default aten names
are not enough to map to your training code.

## Memory + Trace Together

For allocation timelines, prefer:

```python
torch.cuda.memory._record_memory_history()
# ... train ...
torch.cuda.memory._dump_snapshot("mem.pickle")
```

and the [Memory Visualizer](https://pytorch.org/memory_viz). Use Chrome traces
for **time**; memory snapshots for **capacity**.

## Checklist Before You Optimize

- [ ] Trace captured with `schedule(wait, warmup, active)` — not cold start only
- [ ] `activities` include `ProfilerActivity.CUDA` when using GPU
- [ ] You inspected **CUDA tracks**, not only CPU
- [ ] You identified whether the bottleneck is **compute**, **memory transfer**, or **idle/sync**
- [ ] One change at a time, then re-trace

## References

- Module scripts: `profiler_basics.py`, `trace_analysis.py`
- [Perfetto UI](https://ui.perfetto.dev/)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [HTA](https://github.com/facebookresearch/HolisticTraceAnalysis)
