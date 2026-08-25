"""
Case Study 3: torch.compile Graph Breaks Analysis

Problem: Python dispatch overhead and autograd bookkeeping add latency
to every operation. torch.compile can fuse kernels and reduce overhead,
but graph breaks fragment the optimization.
Fix: Identify and eliminate graph breaks, then compile effectively.
Expected speedup: 1.5–3× after fixing graph breaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional


# ============================================================
# BEFORE: Model with graph breaks
# ============================================================

class ModelWithBreaks(nn.Module):
    """A model that causes torch.compile graph breaks."""

    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = layer(x)
            x = F.gelu(x)
            x = norm(x)

            # GRAPH BREAK: print triggers a graph break
            if torch.is_grad_enabled():
                print(f"Layer {i}: norm = {x.norm().item():.4f}")  # noqa: T201

            x = x + residual

        return self.output(x)


# ============================================================
# AFTER: Model without graph breaks
# ============================================================

class ModelWithoutBreaks(nn.Module):
    """Same architecture but without operations that cause graph breaks."""

    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x)
            x = F.gelu(x)
            x = norm(x)
            x = x + residual
        return self.output(x)


# ============================================================
# Common graph break causes and fixes
# ============================================================

class GraphBreakExamples:
    """Demonstrations of common graph break patterns and their fixes."""

    @staticmethod
    def break_data_dependent_control_flow(x):
        """BREAKS: condition depends on tensor value."""
        if x.sum() > 0:  # graph break — data-dependent
            return x * 2
        return x * 3

    @staticmethod
    def fix_data_dependent_control_flow(x):
        """FIX: Use torch.where for data-dependent branching."""
        return torch.where(x.sum() > 0, x * 2, x * 3)

    @staticmethod
    def break_python_builtin(x):
        """BREAKS: calling len() on tensor triggers graph break in some contexts."""
        n = x.shape[0]  # This is fine — static shape
        return x[:n // 2]

    @staticmethod
    def break_inplace_on_input(x):
        """BREAKS: in-place modification of function input."""
        x += 1  # graph break if x is a leaf
        return x

    @staticmethod
    def fix_inplace_on_input(x):
        """FIX: operate on a copy."""
        return x + 1


# ============================================================
# BENCHMARK
# ============================================================

def count_graph_breaks(model, sample_input):
    """Compile with fullgraph=True to detect breaks (will raise on break)."""
    try:
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            _ = compiled(sample_input)
        return 0
    except Exception as e:
        error_msg = str(e)
        if "graph break" in error_msg.lower() or "Dynamo" in error_msg:
            return -1  # Has breaks
        return -1


def benchmark_model(model, input_tensor, n_iterations=100, warmup=20, label=""):
    """Benchmark model forward pass."""
    device = input_tensor.device
    model.eval()

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    mean = sum(times) / len(times)

    if label:
        print(f"  {label:30s} | p50={p50:.3f}ms | p95={p95:.3f}ms | mean={mean:.3f}ms")

    return mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("--- Case 3: torch.compile Graph Breaks ---\n")

    batch_size = 16
    d_model = 256
    x = torch.randn(batch_size, 32, d_model, device=device)

    # Model without breaks
    clean_model = ModelWithoutBreaks(d_model=d_model).to(device)

    print("1. Eager vs Compiled (no graph breaks):\n")

    eager_time = benchmark_model(clean_model, x, label="Eager (no compile)")

    compiled_model = torch.compile(clean_model, mode="reduce-overhead")
    compiled_time = benchmark_model(compiled_model, x, label="Compiled (reduce-overhead)")

    compiled_default = torch.compile(clean_model, mode="default")
    default_time = benchmark_model(compiled_default, x, label="Compiled (default)")

    print(f"\n  Speedup (reduce-overhead): {eager_time / compiled_time:.2f}×")
    print(f"  Speedup (default):         {eager_time / default_time:.2f}×")

    # Fullgraph check
    print("\n2. Graph break detection:\n")

    print("  ModelWithoutBreaks: ", end="")
    breaks = count_graph_breaks(ModelWithoutBreaks(d_model=d_model).to(device), x)
    print("✓ No graph breaks" if breaks == 0 else "✗ Has graph breaks")

    # Demonstrate torch.where fix
    print("\n3. Common fix patterns:\n")
    sample = torch.randn(8, 64, device=device)

    print("  Data-dependent control flow:")
    print("    BREAK: if x.sum() > 0: return x*2 else x*3")
    print("    FIX:   torch.where(x.sum() > 0, x*2, x*3)")

    result_break = GraphBreakExamples.break_data_dependent_control_flow(sample)
    result_fix = GraphBreakExamples.fix_data_dependent_control_flow(sample)
    assert torch.allclose(result_break, result_fix)
    print("    ✓ Results match")

    print("\n  In-place on input:")
    print("    BREAK: x += 1")
    print("    FIX:   return x + 1")

    print("\n--- Compile Modes Comparison ---")
    print("  'default'         — balanced compile time vs runtime")
    print("  'reduce-overhead' — minimize kernel launch overhead (best for serving)")
    print("  'max-autotune'    — longest compile, best runtime (best for training)")

    print("\n--- Key Takeaways ---")
    print("  • print(), breakpoint(), pdb all cause graph breaks")
    print("  • Data-dependent if/else → use torch.where()")
    print("  • In-place on inputs → operate on copies")
    print("  • Use fullgraph=True during development to catch breaks early")
    print("  • 'reduce-overhead' is best for inference serving workloads")


if __name__ == "__main__":
    main()
