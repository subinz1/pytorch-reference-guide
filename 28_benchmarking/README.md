<div align="center">

[← Previous Module](../27_multi_gpu_inference/) | [🏠 Home](../README.md) | [Next Module →](../29_mixed_precision/)

</div>

---

> **Module 28** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 07 — Training Pipelines](../07_training/), [Module 08 — torch.compile](../08_torch_compile/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — benchmarking methodology, Timer, Compare, Fuzzer, Callgrind |
| [`benchmark_basics.py`](benchmark_basics.py) | Timer API, blocked_autorange, Measurement objects, Compare tables, num_threads |
| [`benchmark_advanced.py`](benchmark_advanced.py) | torch.compile benchmarking, shape sweeps, Fuzzer, dtype comparison, model comparison |

---

# torch.utils.benchmark Deep Dive — Measuring Performance Correctly

## Table of Contents

1. [Why Proper Benchmarking Matters](#1-why-proper-benchmarking-matters)
2. [torch.utils.benchmark.Timer](#2-torchutilsbenchmarktimer)
3. [blocked_autorange()](#3-blocked_autorange)
4. [Measurement Object](#4-measurement-object)
5. [Compare — Side-by-Side Tables](#5-compare--side-by-side-tables)
6. [Benchmarking torch.compile](#6-benchmarking-torchcompile)
7. [Benchmarking with Different Shapes](#7-benchmarking-with-different-shapes)
8. [num_threads — Controlling CPU Parallelism](#8-num_threads--controlling-cpu-parallelism)
9. [Fuzzer — Random Test Configurations](#9-fuzzer--random-test-configurations)
10. [Callgrind — Instruction Counts](#10-callgrind--instruction-counts)
11. [Common Pitfalls](#11-common-pitfalls)
12. [Practical Recipes](#12-practical-recipes)
13. [Upstream Updates (June 2026)](#13-upstream-updates-june-2026)

---

## 1. Why Proper Benchmarking Matters

Most PyTorch benchmarks you'll find online are **wrong**. Here's why:

### `time.time()` Is Wrong for GPU Code

```python
import time
import torch

x = torch.randn(1000, 1000, device='cuda')
start = time.time()
y = x @ x          # launches kernel but DOESN'T wait for it
elapsed = time.time() - start  # measures launch time (~10μs), NOT compute time
```

CUDA operations are **asynchronous** — `torch.matmul` returns immediately after launching the GPU kernel. The CPU continues while the GPU computes. `time.time()` only captures how long the CPU took to *enqueue* the work.

### `timeit` Doesn't Sync CUDA Either

```python
import timeit
# Still wrong — timeit uses time.perf_counter() internally, no CUDA sync
timeit.timeit(lambda: x @ x, number=100)
```

### What Distorts Benchmark Results

| Factor | Effect | Fix |
|--------|--------|-----|
| No CUDA sync | Measures launch time, not compute time | `torch.cuda.synchronize()` |
| Cold start | First call initializes CUDA context (~1-3s) | Warmup iterations |
| JIT compilation | `torch.compile` first call is slow | Separate warmup phase |
| cuDNN autotuning | First convolution triggers autotuner | `torch.backends.cudnn.benchmark = True` before warmup |
| Garbage collection | GC pauses inject random latency spikes | Disable GC during measurement |
| CPU frequency scaling | Dynamic clocks cause variance | Pin CPU frequency or use instruction counts |
| Memory caching | CUDA caching allocator reuses memory | Consistent allocation patterns |

**`torch.utils.benchmark` handles all of this automatically.**

---

## 2. torch.utils.benchmark.Timer

The core API for all PyTorch benchmarking:

```python
from torch.utils.benchmark import Timer

t = Timer(
    stmt="x @ y",
    setup="x = torch.randn(1000, 1000); y = torch.randn(1000, 1000)",
)
print(t.timeit(100))        # fixed number of runs
print(t.blocked_autorange()) # auto-determine run count
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stmt` | `str` | The code to benchmark (can be multi-line) |
| `setup` | `str` | Code run once before measurement (imports, tensor creation) |
| `globals` | `dict` | Variables accessible in `stmt` and `setup` |
| `num_threads` | `int` | CPU threads to use (controls `torch.set_num_threads`) |
| `label` | `str` | Row label for `Compare` tables |
| `sub_label` | `str` | Sub-row label for `Compare` tables |
| `description` | `str` | Column label for `Compare` tables |
| `env` | `str` | Environment name (for cross-environment comparison) |
| `timer` | `callable` | Custom timer function (default: `timeit.default_timer`) |

### Using `globals` vs `setup`

```python
# Option 1: setup string (self-contained, but limited)
t = Timer(
    stmt="x.mm(y)",
    setup="import torch; x = torch.randn(256, 256); y = torch.randn(256, 256)"
)

# Option 2: globals dict (more flexible — use existing Python objects)
x = torch.randn(256, 256)
y = torch.randn(256, 256)
t = Timer(
    stmt="x.mm(y)",
    globals={"x": x, "y": y}
)
```

Use `globals` when your setup is complex or involves objects that can't be easily expressed as a string.

### Multi-Statement Benchmarks

```python
t = Timer(
    stmt="""
y = model(x)
loss = criterion(y, target)
loss.backward()
""",
    globals={"model": model, "criterion": criterion, "x": x, "target": target}
)
```

### timeit() — Fixed Iteration Count

```python
result = t.timeit(number=100)  # run stmt exactly 100 times
```

Returns a `Measurement` object. Good when you know how many iterations you want, but bad for comparing fast vs slow operations (the fast one may need more iterations for stable results).

---

## 3. blocked_autorange()

The recommended way to benchmark. It automatically determines the right number of iterations:

```python
result = t.blocked_autorange(min_run_time=1.0)
```

### How It Works

1. **Adaptive warmup**: Runs increasing numbers of iterations (1, 2, 4, 8, ...) until a single block takes ≥ `min_run_time` seconds
2. **Measurement**: Runs multiple blocks at the determined iteration count
3. **Aggregation**: Reports **median** of block times (not mean)

### Why Median Over Mean

```
Run times: [1.2ms, 1.1ms, 1.3ms, 1.1ms, 15.2ms, 1.2ms]
Mean:   3.5ms  ← distorted by one GC pause
Median: 1.2ms  ← robust to outliers
```

The mean is distorted by occasional outliers (GC pauses, context switches, thermal throttling). The median gives a more reliable estimate of typical performance.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_run_time` | `2.0` | Minimum total wall time in seconds |
| `callback` | `None` | Called after each block with intermediate results |

---

## 4. Measurement Object

Both `timeit()` and `blocked_autorange()` return a `Measurement` object:

```python
result = t.blocked_autorange()
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.mean` | `float` | Mean time per execution (seconds) |
| `result.median` | `float` | Median time per execution (seconds) |
| `result.times` | `List[float]` | All measured times (per execution) |
| `result.number_per_run` | `int` | Number of `stmt` executions per block |
| `result.raw_times` | `List[float]` | Raw block times (total, not per execution) |
| `result.iqr` | `float` | Interquartile range |
| `result.significant_figures` | `int` | Stable digits across measurements |

### String Representation

```python
print(result)
# Output:
# <torch.utils.benchmark.utils.common.Measurement object at 0x...>
#   x @ y
#   1.23 ms
#   1 measurement, 100 runs, 1 thread
```

The string representation automatically scales the units (ns, μs, ms, s) and includes the IQR to indicate measurement stability.

### Comparing Measurements

```python
# Access raw timing data
for t in result.times:
    print(f"  {t * 1e3:.3f} ms")

# Mean vs median
print(f"Mean:   {result.mean * 1e3:.3f} ms")
print(f"Median: {result.median * 1e3:.3f} ms")
print(f"IQR:    {result.iqr * 1e3:.3f} ms")
```

---

## 5. Compare — Side-by-Side Tables

The `Compare` class renders a formatted table comparing multiple benchmarks:

```python
from torch.utils.benchmark import Timer, Compare

results = []
for n in [64, 256, 1024]:
    for impl in ["mm", "matmul", "einsum"]:
        if impl == "mm":
            stmt = "torch.mm(x, y)"
        elif impl == "matmul":
            stmt = "x @ y"
        else:
            stmt = "torch.einsum('ij,jk->ik', x, y)"

        t = Timer(
            stmt=stmt,
            setup=f"import torch; x = torch.randn({n},{n}); y = torch.randn({n},{n})",
            label="matmul",
            sub_label=f"[{n}x{n}]",
            description=impl,
        )
        results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(results)
compare.print()
```

### Output Format

```
[----------- matmul -----------]
               |   mm   | matmul | einsum
1 threads: ----+---------+--------+-------
  [64x64]     |   5.2  |   5.3  |  12.1
  [256x256]   |  42.0  |  42.1  |  55.3
  [1024x1024] | 850.1  | 851.0  | 870.2

Times are in microseconds (us).
```

### Label Hierarchy

The three label fields control table layout:

- **`label`**: Groups rows into sections (the `[--- label ---]` header)
- **`sub_label`**: Individual rows within a section
- **`description`**: Column headers

### Colorized Output

`compare.colorize()` adds terminal colors highlighting the fastest implementation per row:

```python
compare = Compare(results)
compare.colorize()  # enables green/red coloring
compare.print()     # fastest in green, slowest in red
```

### Trimming Significant Figures

```python
compare.trim_significant_figures()  # reduces digits to match precision
compare.print()
```

---

## 6. Benchmarking torch.compile

`torch.compile` has a critical subtlety: the **first call triggers compilation** (which can take seconds). You must separate warmup from measurement:

### Correct Methodology

```python
import torch
from torch.utils.benchmark import Timer

model = torch.nn.Linear(1024, 1024)
x = torch.randn(32, 1024)

# Eager baseline
eager_timer = Timer(
    stmt="model(x)",
    globals={"model": model, "x": x},
    label="Linear(1024, 1024)",
    sub_label="batch=32",
    description="eager",
)

# Compiled model — warmup OUTSIDE the Timer
compiled_model = torch.compile(model)
for _ in range(3):  # warmup: triggers compilation
    compiled_model(x)

compiled_timer = Timer(
    stmt="compiled_model(x)",
    globals={"compiled_model": compiled_model, "x": x},
    label="Linear(1024, 1024)",
    sub_label="batch=32",
    description="compiled",
)

results = [
    eager_timer.blocked_autorange(),
    compiled_timer.blocked_autorange(),
]

from torch.utils.benchmark import Compare
Compare(results).print()
```

### Comparing Compile Modes

```python
results = []
for mode in [None, "reduce-overhead", "max-autotune"]:
    compiled = torch.compile(model, mode=mode)
    for _ in range(3):
        compiled(x)  # warmup

    desc = mode or "default"
    t = Timer(
        stmt="fn(x)",
        globals={"fn": compiled, "x": x},
        label="compile modes",
        sub_label="Linear(1024,1024)",
        description=desc,
    )
    results.append(t.blocked_autorange())
```

### Common Mistake: Including Compilation Time

```python
# WRONG: compilation happens inside Timer, polluting measurements
t = Timer(
    stmt="torch.compile(model)(x)",
    globals={"model": model, "x": x},
)
```

---

## 7. Benchmarking with Different Shapes

A common task: sweep over input sizes to understand scaling behavior.

```python
import torch
from torch.utils.benchmark import Timer, Compare

results = []
sizes = [128, 256, 512, 1024, 2048, 4096]

for n in sizes:
    x = torch.randn(n, n)
    y = torch.randn(n, n)

    for desc, stmt in [("mm", "x @ y"), ("bmm", "x.unsqueeze(0) @ y.unsqueeze(0)")]:
        t = Timer(
            stmt=stmt,
            globals={"x": x, "y": y},
            label="Matrix multiply",
            sub_label=f"[{n}x{n}]",
            description=desc,
        )
        results.append(t.blocked_autorange(min_run_time=0.5))

Compare(results).print()
```

### Analyzing Scaling

```python
# Check if runtime scales as expected (O(n^3) for matmul)
for i in range(1, len(sizes)):
    ratio = results[i*2].median / results[(i-1)*2].median
    size_ratio = (sizes[i] / sizes[i-1]) ** 3
    print(f"{sizes[i]:4d} vs {sizes[i-1]:4d}: "
          f"time ratio={ratio:.1f}x, theoretical={size_ratio:.1f}x")
```

---

## 8. num_threads — Controlling CPU Parallelism

CPU benchmarks can vary wildly depending on thread count. The `num_threads` parameter pins the thread count:

```python
from torch.utils.benchmark import Timer

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

results = []
for nthreads in [1, 2, 4, 8]:
    t = Timer(
        stmt="x @ y",
        globals={"x": x, "y": y},
        num_threads=nthreads,
        label="matmul",
        sub_label="[1000x1000]",
        description=f"{nthreads} threads",
    )
    results.append(t.blocked_autorange())

from torch.utils.benchmark import Compare
Compare(results).print()
```

### Why This Matters

- **Reproducibility**: Without pinning, thread count may vary between machines
- **Fair comparison**: Two implementations should use the same thread count
- **Scaling analysis**: See how an operation scales with core count
- **Production relevance**: Server may limit threads per process

### Getting Default Thread Count

```python
default_threads = torch.get_num_threads()  # returns current setting
print(f"Default: {default_threads} threads")
```

---

## 9. Fuzzer — Random Test Configurations

For thorough benchmarking, you want to test across a range of random configurations. `torch.utils.benchmark` provides a `Fuzzer` that generates randomized parameters:

```python
from torch.utils.benchmark import Fuzzer, FuzzedParameter, FuzzedTensor

fuzzer = Fuzzer(
    parameters=[
        FuzzedParameter("n", minval=4, maxval=16, distribution="loguniform"),
        FuzzedParameter("m", minval=4, maxval=16, distribution="loguniform"),
    ],
    tensors=[
        FuzzedTensor("x", size=("n", "m"), probability_contiguous=0.6),
        FuzzedTensor("y", size=("m", "n"), probability_contiguous=0.6),
    ],
    seed=42,
)

results = []
for tensors, tensor_params, params in fuzzer.take(10):
    n, m = int(params["n"]), int(params["m"])
    t = Timer(
        stmt="x @ y",
        globals=tensors,
        label="matmul",
        sub_label=f"[{n}x{m}]",
        description="torch.mm",
    )
    results.append(t.blocked_autorange(min_run_time=0.2))
```

### FuzzedParameter Options

| Parameter | Description |
|-----------|-------------|
| `minval`, `maxval` | Range for generated values |
| `distribution` | `"uniform"` or `"loguniform"` |

### FuzzedTensor Options

| Parameter | Description |
|-----------|-------------|
| `size` | Tuple of parameter names or ints |
| `probability_contiguous` | Probability the tensor is contiguous (0.0-1.0) |
| `min_elements` | Minimum total elements |
| `max_elements` | Maximum total elements |
| `dtype` | Tensor dtype |

### Why Use Fuzzer?

- **Avoid cherry-picking**: Testing only powers of 2 can hit cache-aligned fast paths
- **Find edge cases**: Non-contiguous tensors, odd sizes, small inputs
- **Statistical rigor**: Random configurations give a more realistic performance picture

---

## 10. Callgrind — Instruction Counts

Wall-clock time has inherent noise (OS scheduling, thermal throttling, other processes). For micro-benchmarks where you need **deterministic** results, use instruction counting via Valgrind's Callgrind tool:

```python
from torch.utils.benchmark import Timer

t = Timer(
    stmt="x @ y",
    setup="import torch; x = torch.randn(128, 128); y = torch.randn(128, 128)",
)

# Requires Valgrind to be installed
stats = t.collect_callgrind(number=100)
print(stats)
```

### What It Measures

Instead of wall-clock time, Callgrind counts the **number of CPU instructions** executed. This is:

- **Deterministic**: Same input → same count, every time
- **Noise-free**: No interference from other processes, scheduling, or frequency scaling
- **Reproducible**: Results are identical across runs

### CallgrindStats

```python
stats = t.collect_callgrind(number=100)

# Total instruction count
print(f"Total instructions: {stats.counts()}")

# Filter by function pattern
fn_counts = stats.as_standardized().stats(inclusive=True)
```

### FunctionCounts

```python
# Get per-function instruction counts
fn_counts = stats.as_standardized().stats(inclusive=False)
for fn in fn_counts[:10]:
    print(fn)
```

### When to Use Callgrind

| Use Case | Wall Clock | Callgrind |
|----------|:----------:|:---------:|
| Comparing two implementations | ✓ | ✓ |
| Micro-benchmarks (< 1μs) | Noisy | ✓ |
| CI regression testing | Noisy | ✓ |
| Production-like workloads | ✓ | Slow |
| GPU benchmarks | ✓ | ✗ |

**Limitation**: Callgrind only measures CPU instructions. It cannot measure GPU kernel performance.

---

## 11. Common Pitfalls

### Pitfall 1: Not Warming Up

```python
# BAD — first call initializes CUDA context (1-3 seconds)
t = Timer(stmt="x @ y", globals={"x": x_cuda, "y": y_cuda})
result = t.timeit(1)  # includes CUDA init!

# GOOD — blocked_autorange handles warmup automatically
result = t.blocked_autorange()
```

For `torch.compile`, warmup is even more critical:

```python
# BAD — compilation time included
compiled = torch.compile(model)
t = Timer(stmt="fn(x)", globals={"fn": compiled, "x": x})
result = t.blocked_autorange()  # first block includes compilation!

# GOOD — warmup compiled model before benchmarking
compiled = torch.compile(model)
for _ in range(3):
    compiled(x)  # trigger compilation
t = Timer(stmt="fn(x)", globals={"fn": compiled, "x": x})
result = t.blocked_autorange()
```

### Pitfall 2: Not Syncing GPU

```python
# BAD — time.time() doesn't wait for GPU
import time
start = time.time()
y = model(x_cuda)  # async!
print(f"{time.time() - start:.3f}s")  # measures CPU launch time only

# GOOD — torch.utils.benchmark syncs automatically
t = Timer(stmt="model(x)", globals={"model": model, "x": x_cuda})
print(t.blocked_autorange())  # inserts torch.cuda.synchronize()
```

### Pitfall 3: Garbage Collection Interference

```python
# BAD — GC pauses inject random latency
result = t.timeit(100)  # GC may fire mid-measurement

# GOOD — Timer disables GC during measurement by default
result = t.blocked_autorange()  # GC disabled automatically
```

### Pitfall 4: Too Few Iterations

```python
# BAD — single measurement is noisy
result = t.timeit(1)

# GOOD — enough iterations for statistical stability
result = t.blocked_autorange(min_run_time=2.0)  # runs for at least 2 seconds
```

### Pitfall 5: Benchmarking In-Place vs Out-of-Place

```python
# Unfair comparison — in-place doesn't allocate
x = torch.randn(1000, 1000)

# Out-of-place: allocates new tensor each time
t1 = Timer(stmt="x + 1", globals={"x": x}, description="out-of-place")

# In-place: no allocation
t2 = Timer(stmt="x.add_(1)", globals={"x": x}, description="in-place")

# In-place will be faster partly because it avoids allocation overhead
```

### Pitfall 6: Not Controlling CPU Affinity / Frequency Scaling

For reproducible CPU benchmarks:

```python
# Pin thread count
t = Timer(stmt="x @ y", globals={"x": x, "y": y}, num_threads=1)

# On Linux, also consider: taskset, cpufreq-set for pinning CPU core and frequency
```

---

## 12. Practical Recipes

### Recipe 1: Compare Two Model Implementations

```python
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer, Compare

class ModelV1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class ModelV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.linear2(self.silu(self.linear1(x)))

dim = 512
batch = 64
x = torch.randn(batch, dim)
v1, v2 = ModelV1(dim), ModelV2(dim)

results = []
for name, model in [("ReLU-FFN", v1), ("SiLU-FFN", v2)]:
    t = Timer(
        stmt="model(x)",
        globals={"model": model, "x": x},
        label="FFN forward",
        sub_label=f"dim={dim}",
        description=name,
    )
    results.append(t.blocked_autorange())

Compare(results).print()
```

### Recipe 2: Profile Scaling Behavior

```python
import torch
from torch.utils.benchmark import Timer, Compare

results = []
for batch in [1, 4, 16, 64, 256]:
    for seq_len in [128, 512, 2048]:
        x = torch.randn(batch, seq_len, 768)
        w = torch.randn(768, 768)

        t = Timer(
            stmt="x @ w",
            globals={"x": x, "w": w},
            label="Projection",
            sub_label=f"batch={batch}, seq={seq_len}",
            description="matmul",
        )
        results.append(t.blocked_autorange(min_run_time=0.5))

Compare(results).print()
```

### Recipe 3: Benchmark Custom Triton Kernel vs PyTorch Op

```python
import torch
from torch.utils.benchmark import Timer, Compare

# Assuming you have a custom Triton kernel
# from my_kernels import triton_softmax

x = torch.randn(1024, 1024)

results = []
for desc, stmt in [
    ("torch", "torch.softmax(x, dim=-1)"),
    ("manual", "(x - x.max(dim=-1, keepdim=True).values).exp().div_("
     "(x - x.max(dim=-1, keepdim=True).values).exp().sum(dim=-1, keepdim=True))"),
]:
    t = Timer(
        stmt=stmt,
        globals={"x": x},
        label="softmax",
        sub_label="[1024x1024]",
        description=desc,
    )
    results.append(t.blocked_autorange())

Compare(results).print()
```

### Recipe 4: Measure torch.compile Speedup Properly

```python
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer, Compare

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x

model = TransformerBlock()
x = torch.randn(8, 128, 512)

# Eager
eager_timer = Timer(
    stmt="model(x)", globals={"model": model, "x": x},
    label="TransformerBlock", sub_label="[8,128,512]", description="eager",
)

# Compiled — warmup first!
compiled = torch.compile(model)
for _ in range(3):
    compiled(x)

compiled_timer = Timer(
    stmt="fn(x)", globals={"fn": compiled, "x": x},
    label="TransformerBlock", sub_label="[8,128,512]", description="compiled",
)

results = [eager_timer.blocked_autorange(), compiled_timer.blocked_autorange()]
compare = Compare(results)
compare.colorize()
compare.print()

speedup = results[0].median / results[1].median
print(f"\ntorch.compile speedup: {speedup:.2f}x")
```

---

## 13. Upstream Updates (June 2026)

Recent PyTorch changes relevant to benchmarking and performance:

| PR | Feature | Impact |
|----|---------|--------|
| [#187218](https://github.com/pytorch/pytorch/pull/187218) | FlexGEMM BMM support | New batched matmul paths to benchmark |
| [#187605](https://github.com/pytorch/pytorch/pull/187605) | Dynamo RangeVariable symbolic specialization | Changed compile behavior for range-based loops |
| [#187494](https://github.com/pytorch/pytorch/pull/187494) | Distributed backend accessors | Cleaner backend switching for distributed benchmarks |
| [#187602](https://github.com/pytorch/pytorch/pull/187602) | ShapesSpec in non-strict export | Better shape control for exported model benchmarks |
| [#186398](https://github.com/pytorch/pytorch/pull/186398) | DTensor logspace | New distributed tensor op to benchmark |
| N/A | CUPTI profiler refactoring into `_cupti/` package | Cleaner profiling infrastructure, separate from benchmarking |

### FlexGEMM BMM Support

FlexGEMM now supports batched matrix multiplication, providing an alternative to cuBLAS for certain workloads. Benchmark with:

```python
from torch.utils.benchmark import Timer, Compare

results = []
for batch in [1, 8, 32]:
    x = torch.randn(batch, 256, 256)
    y = torch.randn(batch, 256, 256)
    t = Timer(
        stmt="torch.bmm(x, y)",
        globals={"x": x, "y": y},
        label="BMM",
        sub_label=f"batch={batch}",
        description="bmm",
    )
    results.append(t.blocked_autorange())

Compare(results).print()
```

### Dynamo RangeVariable Symbolic Specialization

`torch.compile` now handles `range()` variables differently, specializing on symbolic values. This can affect benchmarks that use range-based iteration in compiled code:

```python
@torch.compile
def loop_fn(x, n):
    for i in range(n):
        x = x + 1
    return x
```

---

## Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `Timer.timeit(N)` | Fixed N iterations | Quick checks, known iteration count |
| `Timer.blocked_autorange()` | Auto iterations, robust stats | Most benchmarks (recommended default) |
| `Compare` | Side-by-side formatted table | Comparing implementations, shapes, configs |
| `Fuzzer` | Random test configurations | Thorough coverage, avoiding bias |
| `Callgrind` | Deterministic instruction counts | Micro-benchmarks, CI regression tests |

### Key Rules

1. **Always use `torch.utils.benchmark`** — never `time.time()` or raw `timeit`
2. **Warmup torch.compile** before measuring — compilation time is not runtime
3. **Use `blocked_autorange()`** — it handles warmup, GC, iteration count
4. **Pin `num_threads`** for CPU benchmarks — reproducibility requires it
5. **Use `Compare` tables** — organized comparison beats ad-hoc prints
6. **Report median, not mean** — outlier resistance matters

---

### Further Resources

- [torch.utils.benchmark documentation](https://pytorch.org/docs/stable/benchmark_utils.html) — official API reference
- [PyTorch Benchmarking Tutorial](https://pytorch.org/tutorials/recipes/benchmark.html) — official tutorial
- [Module 07 — Training Pipelines](../07_training/) — training loops to benchmark
- [Module 08 — torch.compile](../08_torch_compile/) — understanding compilation for proper benchmark methodology
- [Module 14 — Testing & Benchmarking](../14_testing/) — related testing utilities
- [Module 26 — Memory Profiling](../26_memory_profiling/) — complementary profiling tools

---

<div align="center">

[← Previous Module](../27_multi_gpu_inference/) | [🏠 Home](../README.md) | [Next Module →](../29_mixed_precision/)

**Notebook**: [`28_benchmarking.ipynb`](../notebooks/28_benchmarking.ipynb)

</div>
