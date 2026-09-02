"""
Advanced Trace Analysis
=======================

Demonstrates advanced profiling: CUDA activity tracing, memory snapshot
analysis, operator-level breakdowns, and integration with HTA.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule


def create_conv_model():
    """CNN model that exercises both CPU and GPU compute paths."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )


def profile_cuda_activity(model, device="cuda"):
    """Capture both CPU and CUDA activities to see host-device overlap."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU profiling")
        return None

    model = model.to(device)
    x = torch.randn(32, 3, 32, 32, device=device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(5):
            _ = model(x)
            torch.cuda.synchronize()

    print("=== CUDA Activity (sorted by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    flops_table = prof.key_averages()
    print("\n=== FLOP Estimates ===")
    for event in flops_table:
        if event.flops and event.flops > 0:
            gflops = event.flops / 1e9
            print(f"  {event.key:40s} {gflops:.3f} GFLOP")

    return prof


def analyze_cuda_kernels(prof):
    """Extract individual CUDA kernel events from a completed profile."""
    if prof is None:
        return

    print("\n=== Individual CUDA Kernel Events ===")
    events = prof.key_averages(group_by_stack_n=3)
    cuda_events = [e for e in events if e.self_cuda_time_total > 0]
    cuda_events.sort(key=lambda e: e.self_cuda_time_total, reverse=True)

    for event in cuda_events[:10]:
        print(
            f"  {event.key:45s} "
            f"calls={event.count:4d}  "
            f"cuda_time={event.self_cuda_time_total / 1e3:.2f}ms"
        )


def memory_snapshot_analysis(device="cuda"):
    """Record CUDA memory allocation history and analyze patterns."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory snapshot")
        return

    torch.cuda.memory.empty_cache()
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    model = create_conv_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(3):
        x = torch.randn(64, 3, 32, 32, device=device)
        target = torch.randint(0, 10, (64,), device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    print("=== Memory Snapshot Summary ===")
    print(f"  Total segments: {len(snapshot.get('segments', []))}")
    print(f"  Allocator settings: {snapshot.get('allocator_settings', {})}")

    try:
        from pickle import dumps
        snapshot_bytes = dumps(snapshot)
        snapshot_path = "cuda_memory_snapshot.pickle"
        with open(snapshot_path, "wb") as f:
            f.write(snapshot_bytes)
        print(f"  Snapshot saved to: {snapshot_path}")
        print("  Visualize at: https://pytorch.org/memory_viz")
    except Exception as e:
        print(f"  Could not save snapshot: {e}")


def profile_with_tensorboard(model, device="cpu", log_dir="./tb_profiler_logs"):
    """Export traces for the TensorBoard PyTorch Profiler plugin."""
    from torch.profiler import tensorboard_trace_handler

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(8):
            x = torch.randn(32, 3, 32, 32, device=device)
            target = torch.randint(0, 10, (32,), device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            prof.step()

    print(f"TensorBoard traces written to: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")


def profile_data_pipeline():
    """Profile DataLoader to identify I/O bottlenecks."""
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.randn(1000, 3, 32, 32),
        torch.randint(0, 10, (1000,)),
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)

    model = create_conv_model()
    loss_fn = nn.CrossEntropyLoss()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for batch_idx, (x, target) in enumerate(loader):
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            if batch_idx >= 4:
                break

    print("\n=== Data Pipeline Profile ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


if __name__ == "__main__":
    print("--- CPU-only trace analysis ---")
    profile_data_pipeline()

    print("\n--- TensorBoard export (CPU) ---")
    profile_with_tensorboard(create_conv_model(), device="cpu")

    print("\n--- CUDA profiling ---")
    prof = profile_cuda_activity(create_conv_model())
    analyze_cuda_kernels(prof)

    print("\n--- Memory snapshot ---")
    memory_snapshot_analysis()
