"""
Profiling Deep Dive
===================

Demonstrates PyTorch's profiling tools:
1. Basic CPU profiling with torch.profiler
2. record_function for annotating code regions
3. Profiling a training loop
4. Benchmarking with torch.utils.benchmark
5. Manual timing utilities
"""

import time

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


# ===========================================================================
# A sample model for profiling
# ===========================================================================

class SampleModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        with record_function("features"):
            h = self.features(x)
        with record_function("classifier"):
            return self.classifier(h)


# ===========================================================================
# 1. Basic CPU Profiling
# ===========================================================================

def demo_basic_profiling():
    """Profile a single forward pass and print a summary table."""
    print("=" * 70)
    print("BASIC CPU PROFILING")
    print("=" * 70)

    model = SampleModel()
    model.eval()
    x = torch.randn(64, 784)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)

    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15,
    ))


# ===========================================================================
# 2. record_function for Fine-Grained Annotation
# ===========================================================================

def demo_record_function():
    """Use record_function to annotate specific code regions."""
    print("\n" + "=" * 70)
    print("RECORD_FUNCTION: Fine-Grained Annotation")
    print("=" * 70)

    model = SampleModel()
    model.eval()
    x = torch.randn(32, 784)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("full_pipeline"):
            with record_function("preprocessing"):
                x_norm = (x - x.mean()) / (x.std() + 1e-8)

            with record_function("inference"):
                with torch.no_grad():
                    logits = model(x_norm)

            with record_function("postprocessing"):
                probs = torch.softmax(logits, dim=-1)
                predictions = probs.argmax(dim=-1)

    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10,
    ))
    print(f"  Predictions shape: {list(predictions.shape)}")


# ===========================================================================
# 3. Profiling a Training Loop
# ===========================================================================

def demo_training_profiling():
    """Profile a training loop to find bottlenecks."""
    print("\n" + "=" * 70)
    print("TRAINING LOOP PROFILING")
    print("=" * 70)

    model = SampleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data
    X = torch.randn(256, 784)
    Y = torch.randint(0, 10, (256,))

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(5):
            with record_function(f"step_{step}"):
                with record_function("forward"):
                    logits = model(X)
                    loss = criterion(logits, Y)

                with record_function("backward"):
                    optimizer.zero_grad()
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15,
    ))

    # Group by input shapes to see how data dimensions affect runtime
    print("\nGrouped by input shape:")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total",
        row_limit=10,
    ))


# ===========================================================================
# 4. Benchmarking with torch.utils.benchmark
# ===========================================================================

def demo_benchmarking():
    """Use torch.utils.benchmark for reliable timing comparisons."""
    print("\n" + "=" * 70)
    print("BENCHMARKING: torch.utils.benchmark")
    print("=" * 70)

    # Compare different matrix multiplication approaches
    M, N, K = 256, 256, 256
    A = torch.randn(M, K)
    B = torch.randn(K, N)

    from torch.utils.benchmark import Timer

    # Benchmark torch.mm
    t_mm = Timer(
        stmt="torch.mm(A, B)",
        globals={"A": A, "B": B, "torch": torch},
        label="matmul",
        sub_label="torch.mm",
        description="256x256",
    )

    # Benchmark @ operator
    t_at = Timer(
        stmt="A @ B",
        globals={"A": A, "B": B},
        label="matmul",
        sub_label="A @ B",
        description="256x256",
    )

    # Benchmark torch.einsum
    t_ein = Timer(
        stmt="torch.einsum('ik,kj->ij', A, B)",
        globals={"A": A, "B": B, "torch": torch},
        label="matmul",
        sub_label="einsum",
        description="256x256",
    )

    results = []
    for timer in [t_mm, t_at, t_ein]:
        r = timer.blocked_autorange(min_run_time=0.5)
        results.append(r)
        print(f"  {r.sub_label:15s}: {r.median * 1e6:8.1f} us "
              f"(IQR: {r.iqr * 1e6:.1f} us)")

    # Compare different batch sizes
    print("\n  Matrix multiplication at different sizes:")
    for size in [64, 128, 256, 512]:
        A = torch.randn(size, size)
        B = torch.randn(size, size)
        t = Timer(
            stmt="A @ B",
            globals={"A": A, "B": B},
        )
        r = t.blocked_autorange(min_run_time=0.2)
        print(f"    {size:4d}x{size:4d}: {r.median * 1e6:8.1f} us")


# ===========================================================================
# 5. Manual Timing Utilities
# ===========================================================================

def demo_manual_timing():
    """Simple timing utilities when full profiling is overkill."""
    print("\n" + "=" * 70)
    print("MANUAL TIMING")
    print("=" * 70)

    model = SampleModel()
    model.eval()
    x = torch.randn(128, 784)

    # Warmup (important for accurate timing!)
    for _ in range(10):
        _ = model(x)

    # Time with Python's time module
    num_runs = 100
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_runs * 1000
    print(f"  Average inference time: {avg_ms:.3f} ms")
    print(f"  Throughput: {128 * num_runs / elapsed:.0f} samples/sec")

    # Compare model sizes
    print("\n  Model comparison:")
    for hidden in [64, 128, 256, 512]:
        m = SampleModel(hidden_dim=hidden)
        m.eval()
        params = sum(p.numel() for p in m.parameters())

        # Warmup
        for _ in range(5):
            _ = m(x)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = m(x)
        elapsed = time.perf_counter() - start
        avg_ms = elapsed / 50 * 1000

        print(f"    hidden={hidden:4d}: {params:>8,} params, {avg_ms:.3f} ms/batch")


# ===========================================================================
# 6. Memory Profiling (CPU)
# ===========================================================================

def demo_memory_profiling():
    """Track CPU memory allocations during model operations."""
    print("\n" + "=" * 70)
    print("MEMORY PROFILING (CPU)")
    print("=" * 70)

    model = SampleModel(hidden_dim=512)

    # Profile memory allocations
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
    ) as prof:
        x = torch.randn(256, 784)
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Show events with memory info
    events = prof.key_averages()
    print("  Top memory-allocating operations:")
    table = events.table(sort_by="self_cpu_memory_usage", row_limit=10)
    print(table)


if __name__ == "__main__":
    demo_basic_profiling()
    demo_record_function()
    demo_training_profiling()
    demo_benchmarking()
    demo_manual_timing()
    demo_memory_profiling()
    print("\n" + "=" * 70)
    print("All profiling demos completed successfully!")
    print("=" * 70)
