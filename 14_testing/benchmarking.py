"""
Benchmarking with torch.utils.benchmark
========================================

Demonstrates proper benchmarking techniques:
1. Basic Timer usage
2. Comparing multiple implementations
3. Effect of input size on performance
4. Common pitfalls
5. Custom benchmarking utilities
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.benchmark import Timer, Compare


# ===========================================================================
# 1. Basic Timer Usage
# ===========================================================================

def demo_basic_timer():
    """Using torch.utils.benchmark.Timer for reliable measurements."""
    print("=" * 70)
    print("BASIC TIMER USAGE")
    print("=" * 70)

    A = torch.randn(256, 256)
    B = torch.randn(256, 256)

    # Create a timer: specify the statement and any global variables it needs
    t = Timer(
        stmt="torch.mm(A, B)",
        globals={"A": A, "B": B, "torch": torch},
        label="Matrix Multiply",
        sub_label="torch.mm",
        description="256x256",
    )

    # blocked_autorange: the recommended method
    # Automatically determines the number of iterations for reliable stats
    result = t.blocked_autorange(min_run_time=0.5)
    print(f"  torch.mm(256x256):")
    print(f"    Median:     {result.median * 1e6:.1f} us")
    print(f"    IQR:        {result.iqr * 1e6:.1f} us")
    print(f"    Num runs:   {result.number_per_run}")

    # timeit: classic approach with fixed number of iterations
    result2 = t.timeit(number=1000)
    print(f"\n  timeit(1000):")
    print(f"    Mean:       {result2.mean * 1e6:.1f} us")
    print(f"    Median:     {result2.median * 1e6:.1f} us")


# ===========================================================================
# 2. Comparing Implementations
# ===========================================================================

def demo_compare_implementations():
    """Compare different ways to implement the same operation."""
    print("\n" + "=" * 70)
    print("COMPARING IMPLEMENTATIONS")
    print("=" * 70)

    results = []

    for size in [64, 128, 256, 512]:
        x = torch.randn(size, size)

        implementations = [
            ("mm", "torch.mm(x, x)", {"x": x, "torch": torch}),
            ("@", "x @ x", {"x": x}),
            ("einsum", "torch.einsum('ij,jk->ik', x, x)", {"x": x, "torch": torch}),
            ("bmm", "torch.bmm(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)", {"x": x, "torch": torch}),
        ]

        for label, stmt, globs in implementations:
            t = Timer(
                stmt=stmt,
                globals=globs,
                label="Matrix Multiply",
                sub_label=label,
                description=f"{size}x{size}",
            )
            results.append(t.blocked_autorange(min_run_time=0.3))

    # Pretty-print comparison table
    compare = Compare(results)
    compare.print()


# ===========================================================================
# 3. Benchmarking Model Operations
# ===========================================================================

def demo_model_benchmarking():
    """Benchmark model forward and backward passes."""
    print("\n" + "=" * 70)
    print("MODEL BENCHMARKING")
    print("=" * 70)

    results = []

    for hidden_dim in [64, 128, 256, 512]:
        model = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )
        model.eval()

        x = torch.randn(64, 784)

        # Forward pass benchmark
        t_fwd = Timer(
            stmt="model(x)",
            globals={"model": model, "x": x},
            label="MLP Forward",
            sub_label=f"hidden={hidden_dim}",
            description="batch=64",
        )
        results.append(t_fwd.blocked_autorange(min_run_time=0.3))

    compare = Compare(results)
    compare.print()

    # Benchmark forward + backward
    print("\n  Forward + Backward:")
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )
    model.train()
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))

    t_fwd_only = Timer(
        stmt="""
with torch.no_grad():
    model(x)
""",
        globals={"model": model, "x": x, "torch": torch},
        label="Forward vs Forward+Backward",
        sub_label="Forward only",
    )

    t_fwd_bwd = Timer(
        stmt="""
pred = model(x)
loss = criterion(pred, y)
loss.backward()
model.zero_grad()
""",
        globals={
            "model": model, "x": x, "y": y,
            "criterion": criterion,
        },
        label="Forward vs Forward+Backward",
        sub_label="Forward + Backward",
    )

    r_fwd = t_fwd_only.blocked_autorange(min_run_time=0.5)
    r_bwd = t_fwd_bwd.blocked_autorange(min_run_time=0.5)
    print(f"    Forward only:      {r_fwd.median * 1e3:.3f} ms")
    print(f"    Forward + Backward: {r_bwd.median * 1e3:.3f} ms")
    print(f"    Backward overhead:  {(r_bwd.median - r_fwd.median) * 1e3:.3f} ms "
          f"({r_bwd.median / r_fwd.median:.1f}x total)")


# ===========================================================================
# 4. Activation Function Comparison
# ===========================================================================

def demo_activation_benchmark():
    """Compare activation function performance."""
    print("\n" + "=" * 70)
    print("ACTIVATION FUNCTION BENCHMARK")
    print("=" * 70)

    x = torch.randn(1024, 1024)
    results = []

    activations = [
        ("ReLU", "F.relu(x)", {"F": F, "x": x}),
        ("GELU", "F.gelu(x)", {"F": F, "x": x}),
        ("SiLU/Swish", "F.silu(x)", {"F": F, "x": x}),
        ("Sigmoid", "torch.sigmoid(x)", {"torch": torch, "x": x}),
        ("Tanh", "torch.tanh(x)", {"torch": torch, "x": x}),
        ("Softplus", "F.softplus(x)", {"F": F, "x": x}),
        ("Mish", "F.mish(x)", {"F": F, "x": x}),
    ]

    for name, stmt, globs in activations:
        t = Timer(
            stmt=stmt,
            globals=globs,
            label="Activations",
            sub_label=name,
            description="1024x1024",
        )
        results.append(t.blocked_autorange(min_run_time=0.3))

    compare = Compare(results)
    compare.print()


# ===========================================================================
# 5. Common Benchmarking Pitfalls
# ===========================================================================

def demo_pitfalls():
    """Show common benchmarking mistakes and how to avoid them."""
    print("\n" + "=" * 70)
    print("COMMON BENCHMARKING PITFALLS")
    print("=" * 70)

    # Pitfall 1: Not warming up
    print("\n  Pitfall 1: Not warming up")
    model = nn.Linear(1000, 1000)
    x = torch.randn(64, 1000)

    # Cold start (includes initialization overhead)
    start = time.perf_counter()
    _ = model(x)
    cold_time = time.perf_counter() - start

    # Warm start
    for _ in range(10):  # warmup
        _ = model(x)
    start = time.perf_counter()
    _ = model(x)
    warm_time = time.perf_counter() - start

    print(f"    Cold start: {cold_time * 1e6:.0f} us")
    print(f"    Warm start: {warm_time * 1e6:.0f} us")
    print(f"    (Timer handles warmup automatically)")

    # Pitfall 2: Single measurement
    print("\n  Pitfall 2: Single measurement vs. proper statistics")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = model(x)
        times.append(time.perf_counter() - start)

    times = sorted(times)
    print(f"    Min:    {times[0] * 1e6:.0f} us")
    print(f"    Median: {times[len(times)//2] * 1e6:.0f} us")
    print(f"    Max:    {times[-1] * 1e6:.0f} us")
    print(f"    Spread: {(times[-1] - times[0]) * 1e6:.0f} us")
    print(f"    (Use Timer.blocked_autorange for robust statistics)")

    # Pitfall 3: Benchmarking in training mode
    print("\n  Pitfall 3: Training vs eval mode")
    model_with_bn = nn.Sequential(
        nn.Linear(100, 100),
        nn.BatchNorm1d(100),
        nn.Dropout(0.5),
        nn.Linear(100, 10),
    )
    x = torch.randn(64, 100)

    t_train = Timer(
        stmt="model(x)",
        globals={"model": model_with_bn, "x": x},
        label="Mode comparison",
        sub_label="train()",
    )
    model_with_bn.train()
    r_train = t_train.blocked_autorange(min_run_time=0.3)

    t_eval = Timer(
        stmt="""
with torch.no_grad():
    model(x)
""",
        globals={"model": model_with_bn, "x": x, "torch": torch},
        label="Mode comparison",
        sub_label="eval() + no_grad",
    )
    model_with_bn.eval()
    r_eval = t_eval.blocked_autorange(min_run_time=0.3)

    print(f"    train():             {r_train.median * 1e6:.0f} us")
    print(f"    eval() + no_grad():  {r_eval.median * 1e6:.0f} us")
    print(f"    Speedup:             {r_train.median / r_eval.median:.2f}x")


# ===========================================================================
# 6. Scaling Analysis
# ===========================================================================

def demo_scaling_analysis():
    """Measure how performance scales with input size."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    print("\n  Matrix multiplication scaling:")
    prev_time = None
    for n in [32, 64, 128, 256, 512, 1024]:
        A = torch.randn(n, n)
        B = torch.randn(n, n)
        t = Timer(
            stmt="A @ B",
            globals={"A": A, "B": B},
        )
        r = t.blocked_autorange(min_run_time=0.2)
        ratio = f"{r.median / prev_time:.2f}x" if prev_time else "  -  "
        prev_time = r.median
        print(f"    n={n:5d}: {r.median * 1e6:10.1f} us  (vs prev: {ratio})")

    print("\n  Expected: ~8x per doubling (O(n^3) for naive matmul)")
    print("  Actual ratio depends on BLAS optimizations and cache effects")

    # Batch size scaling
    print("\n  Batch size scaling (Linear layer):")
    model = nn.Linear(256, 256)
    model.eval()

    prev_time = None
    for bs in [1, 4, 16, 64, 256, 1024]:
        x = torch.randn(bs, 256)
        t = Timer(
            stmt="model(x)",
            globals={"model": model, "x": x},
        )
        r = t.blocked_autorange(min_run_time=0.2)
        throughput = bs / r.median
        print(f"    batch={bs:5d}: {r.median * 1e6:10.1f} us  "
              f"({throughput:.0f} samples/sec)")


if __name__ == "__main__":
    demo_basic_timer()
    demo_compare_implementations()
    demo_model_benchmarking()
    demo_activation_benchmark()
    demo_pitfalls()
    demo_scaling_analysis()
    print("\n" + "=" * 70)
    print("All benchmarking demos completed successfully!")
    print("=" * 70)
