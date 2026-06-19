"""
Module 28 — torch.utils.benchmark Basics
=========================================
Timer API, blocked_autorange, Measurement objects, Compare tables, num_threads.

All examples run on CPU. No GPU required.

Usage:
    python benchmark_basics.py
"""

import torch
from torch.utils.benchmark import Timer, Compare

print("=" * 70)
print("MODULE 28: torch.utils.benchmark — Basics")
print("=" * 70)

# ============================================================
# 1. Timer with stmt and setup
# ============================================================
print("\n" + "=" * 70)
print("1. Timer — stmt and setup strings")
print("=" * 70)

t = Timer(
    stmt="x @ y",
    setup="import torch; x = torch.randn(256, 256); y = torch.randn(256, 256)",
)
result = t.timeit(100)
print(f"timeit(100) result:\n{result}")
print(f"  Mean:  {result.mean * 1e6:.1f} μs")
print(f"  Times: {len(result.times)} measurement(s)")

# ============================================================
# 2. Timer with globals
# ============================================================
print("\n" + "=" * 70)
print("2. Timer — using globals dict")
print("=" * 70)

x = torch.randn(256, 256)
y = torch.randn(256, 256)

t = Timer(
    stmt="x.mm(y)",
    globals={"x": x, "y": y},
)
result = t.timeit(100)
print(f"x.mm(y) with globals:\n{result}")

# ============================================================
# 3. timeit() vs blocked_autorange()
# ============================================================
print("\n" + "=" * 70)
print("3. timeit() vs blocked_autorange()")
print("=" * 70)

t = Timer(
    stmt="x @ y",
    globals={"x": x, "y": y},
)

fixed = t.timeit(50)
print(f"timeit(50):\n{fixed}")

auto = t.blocked_autorange(min_run_time=1.0)
print(f"\nblocked_autorange(min_run_time=1.0):\n{auto}")
print(f"  Auto-selected {auto.number_per_run} runs per block")
print(f"  Collected {len(auto.raw_times)} block(s)")

# ============================================================
# 4. Measurement object inspection
# ============================================================
print("\n" + "=" * 70)
print("4. Measurement object — detailed inspection")
print("=" * 70)

t = Timer(
    stmt="torch.nn.functional.relu(x)",
    globals={"x": torch.randn(1000, 1000)},
)
m = t.blocked_autorange(min_run_time=1.0)

print(f"Statement:        {m.task_spec.stmt}")
print(f"Mean:             {m.mean * 1e6:.2f} μs")
print(f"Median:           {m.median * 1e6:.2f} μs")
print(f"IQR:              {m.iqr * 1e6:.2f} μs")
print(f"Number per run:   {m.number_per_run}")
print(f"Raw block times:  {len(m.raw_times)} blocks")
print(f"Per-run times:    {len(m.times)} samples")

print("\nAll per-run times (μs):")
for i, t_val in enumerate(m.times):
    print(f"  [{i:2d}] {t_val * 1e6:.2f} μs")

# ============================================================
# 5. Compare table — multiple implementations
# ============================================================
print("\n" + "=" * 70)
print("5. Compare — side-by-side table")
print("=" * 70)

results = []
for n in [64, 256, 1024]:
    a = torch.randn(n, n)
    b = torch.randn(n, n)

    for desc, stmt in [
        ("mm", "a.mm(b)"),
        ("matmul", "a @ b"),
        ("einsum", "torch.einsum('ij,jk->ik', a, b)"),
    ]:
        t = Timer(
            stmt=stmt,
            globals={"a": a, "b": b, "torch": torch},
            label="Matrix multiply",
            sub_label=f"[{n}x{n}]",
            description=desc,
            num_threads=1,
        )
        results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(results)
compare.print()

# ============================================================
# 6. Compare with colorize and trim
# ============================================================
print("\n" + "=" * 70)
print("6. Compare — colorized and trimmed")
print("=" * 70)

compare_color = Compare(results)
compare_color.trim_significant_figures()
compare_color.colorize()
compare_color.print()

# ============================================================
# 7. Benchmarking different input sizes
# ============================================================
print("\n" + "=" * 70)
print("7. Shape sweep — varying matrix size")
print("=" * 70)

sizes = [128, 256, 512, 1024, 2048]
size_results = []

for n in sizes:
    x_n = torch.randn(n, n)
    y_n = torch.randn(n, n)
    t = Timer(
        stmt="x @ y",
        globals={"x": x_n, "y": y_n},
        label="matmul scaling",
        sub_label=f"[{n}x{n}]",
        description="mm",
        num_threads=1,
    )
    size_results.append(t.blocked_autorange(min_run_time=0.5))

Compare(size_results).print()

print("\nScaling analysis (expected O(n^3)):")
for i in range(1, len(sizes)):
    time_ratio = size_results[i].median / size_results[i - 1].median
    theoretical = (sizes[i] / sizes[i - 1]) ** 3
    print(
        f"  {sizes[i]:4d} vs {sizes[i-1]:4d}: "
        f"time={time_ratio:.1f}x, O(n^3)={theoretical:.1f}x"
    )

# ============================================================
# 8. num_threads control
# ============================================================
print("\n" + "=" * 70)
print("8. num_threads — controlling CPU parallelism")
print("=" * 70)

print(f"Default num_threads: {torch.get_num_threads()}")

x_big = torch.randn(1000, 1000)
y_big = torch.randn(1000, 1000)

thread_results = []
max_threads = min(torch.get_num_threads(), 8)
thread_counts = [1] + [t for t in [2, 4, 8] if t <= max_threads]

for nthreads in thread_counts:
    t = Timer(
        stmt="x @ y",
        globals={"x": x_big, "y": y_big},
        num_threads=nthreads,
        label="matmul threading",
        sub_label="[1000x1000]",
        description=f"{nthreads} thread{'s' if nthreads > 1 else ''}",
    )
    thread_results.append(t.blocked_autorange(min_run_time=0.5))

Compare(thread_results).print()

if len(thread_results) > 1:
    base = thread_results[0].median
    for i, nthreads in enumerate(thread_counts):
        speedup = base / thread_results[i].median
        print(f"  {nthreads} thread(s): {speedup:.2f}x speedup vs 1 thread")

# ============================================================
# 9. Multi-statement benchmarks
# ============================================================
print("\n" + "=" * 70)
print("9. Multi-statement benchmarks")
print("=" * 70)

model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
)
criterion = torch.nn.MSELoss()
inp = torch.randn(32, 256)
target = torch.randn(32, 256)

fwd_timer = Timer(
    stmt="model(x)",
    globals={"model": model, "x": inp},
    label="model phases",
    sub_label="[32, 256]",
    description="forward",
    num_threads=1,
)

fwd_bwd_timer = Timer(
    stmt="""
y = model(x)
loss = criterion(y, target)
loss.backward()
""",
    globals={
        "model": model,
        "criterion": criterion,
        "x": inp,
        "target": target,
    },
    label="model phases",
    sub_label="[32, 256]",
    description="fwd+bwd",
    num_threads=1,
)

phase_results = [
    fwd_timer.blocked_autorange(min_run_time=0.5),
    fwd_bwd_timer.blocked_autorange(min_run_time=0.5),
]
Compare(phase_results).print()

ratio = phase_results[1].median / phase_results[0].median
print(f"fwd+bwd / fwd ratio: {ratio:.2f}x")

# ============================================================
# 10. Common pitfall: wrong way to benchmark
# ============================================================
print("\n" + "=" * 70)
print("10. Pitfall demo — time.time() vs Timer")
print("=" * 70)

import time

x_demo = torch.randn(1000, 1000)
y_demo = torch.randn(1000, 1000)

naive_times = []
for _ in range(10):
    start = time.time()
    _ = x_demo @ y_demo
    naive_times.append(time.time() - start)

print("time.time() results (10 runs):")
print(f"  Mean:   {sum(naive_times)/len(naive_times)*1e3:.3f} ms")
print(f"  Min:    {min(naive_times)*1e3:.3f} ms")
print(f"  Max:    {max(naive_times)*1e3:.3f} ms")
print(f"  Spread: {(max(naive_times)-min(naive_times))*1e3:.3f} ms")

proper = Timer(
    stmt="x @ y",
    globals={"x": x_demo, "y": y_demo},
    num_threads=1,
).blocked_autorange(min_run_time=1.0)

print(f"\nTimer.blocked_autorange() result:")
print(f"  Median: {proper.median*1e3:.3f} ms")
print(f"  IQR:    {proper.iqr*1e3:.3f} ms")
print(f"  (much more stable and reliable)")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key takeaways:
  1. Use Timer with stmt/setup or globals — never time.time()
  2. blocked_autorange() > timeit() for most cases
  3. Measurement objects give mean, median, IQR, raw times
  4. Compare tables organize multi-config benchmarks
  5. Pin num_threads for reproducible CPU benchmarks
  6. Use label/sub_label/description for structured comparisons
""")
