"""
PyTorch Profiler Basics
=======================

Demonstrates core profiler usage: context manager, scheduling,
key averages, and trace export.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


def create_sample_model():
    """Build a small feedforward network for profiling demonstrations."""
    return nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def basic_profiling(model, device="cpu"):
    """Profile a forward+backward pass with the simplest API."""
    model = model.to(device)
    x = torch.randn(64, 1024, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    loss_fn = nn.CrossEntropyLoss()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    print("=== Basic Profiling (sorted by CPU time) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    return prof


def profiling_with_schedule(model, device="cpu", num_steps=10):
    """Use a schedule to skip warm-up steps and capture steady-state."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def trace_handler(p):
        print(f"\n=== Trace ready at step {p.step_num} ===")
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    sched = schedule(wait=1, warmup=2, active=3, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=sched,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for step in range(num_steps):
            x = torch.randn(32, 1024, device=device)
            target = torch.randint(0, 10, (32,), device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            prof.step()


def export_chrome_trace(model, output_path="profiler_trace.json"):
    """Capture a trace and export to Chrome Trace format."""
    x = torch.randn(32, 1024)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(5):
            _ = model(x)

    prof.export_chrome_trace(output_path)
    print(f"Chrome trace exported to: {output_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")


def group_by_input_shapes(model):
    """Profile with shape grouping to see how operator cost varies with shape."""
    inputs = [torch.randn(batch, 1024) for batch in [16, 32, 64, 128]]

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for x in inputs:
            _ = model(x)

    print("\n=== Grouped by input shape ===")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=20
        )
    )


def profile_memory_usage(model):
    """Track tensor allocations during forward/backward."""
    x = torch.randn(64, 1024)
    target = torch.randint(0, 10, (64,))
    loss_fn = nn.CrossEntropyLoss()

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    print("\n=== Memory profiling (sorted by self CPU memory) ===")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=15
        )
    )


def export_stacks_for_flamegraph(model, output_path="profiler_stacks.txt"):
    """Export stack traces for flame graph visualization."""
    x = torch.randn(64, 1024)

    with profile(
        activities=[ProfilerActivity.CPU],
        with_stack=True,
    ) as prof:
        for _ in range(10):
            _ = model(x)

    prof.export_stacks(output_path, metric="self_cpu_time_total")
    print(f"Stacks exported to: {output_path}")
    print("Visualize with: flamegraph.pl < profiler_stacks.txt > flamegraph.svg")


if __name__ == "__main__":
    model = create_sample_model()

    basic_profiling(model)
    profiling_with_schedule(create_sample_model())
    export_chrome_trace(model)
    group_by_input_shapes(model)
    profile_memory_usage(create_sample_model())
    export_stacks_for_flamegraph(model)
