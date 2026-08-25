# Module 44: Performance Case Studies

Five real-world PyTorch optimization stories with before/after code, profiling methodology, and measured speedups. Each case study follows the same structure: identify bottleneck → apply fix → measure improvement.

## Case Studies

| # | Case Study | Bottleneck | Fix | Speedup |
|---|-----------|-----------|-----|---------|
| 1 | Memory-bound DataLoader | CPU→GPU copy on main thread | Pin memory + non-blocking transfers | 2–3× |
| 2 | Naive attention scaling | O(n²) memory, repeated allocation | Flash attention pattern + in-place ops | 4–8× |
| 3 | Training loop overhead | Python dispatch + autograd bookkeeping | torch.compile with graph breaks analysis | 1.5–3× |
| 4 | Inference memory bloat | Gradients + BN running stats retained | Freezing, inference_mode, weight-only quantization | 60–75% memory reduction |
| 5 | Multi-GPU communication | All-reduce blocking compute | Gradient bucketing + overlap comm/compute | 1.3–1.8× at scale |

## Files

| File | Description |
|------|-------------|
| `case1_dataloader.py` | DataLoader pinned memory and prefetch optimization |
| `case2_attention.py` | Attention implementation: naive → memory-efficient → flash |
| `case3_compile.py` | torch.compile graph breaks analysis and fix |
| `case4_inference_memory.py` | Inference memory reduction techniques |
| `case5_distributed.py` | Multi-GPU communication overlap patterns |

## How to Read These

Each script follows the same pattern:

```python
# 1. BEFORE: naive implementation
def slow_version(...): ...

# 2. PROFILING: identify the bottleneck
profile_and_report(slow_version)

# 3. AFTER: optimized implementation
def fast_version(...): ...

# 4. COMPARISON: measure speedup
benchmark_comparison(slow_version, fast_version)
```

## Running

```bash
python case1_dataloader.py
python case2_attention.py
python case3_compile.py
python case4_inference_memory.py
python case5_distributed.py  # requires 2+ GPUs or use NCCL CPU backend
```

## Key Takeaways

- Profile first, optimize second — `torch.profiler` and `torch.cuda.memory_summary()` are your friends
- Memory bandwidth is usually the bottleneck, not compute (especially for inference)
- `torch.compile` wins come from kernel fusion and reduced memory traffic
- Overlapping communication with compute is critical for multi-GPU scaling
- Small changes (pin_memory, non_blocking, inference_mode) compound into large gains
