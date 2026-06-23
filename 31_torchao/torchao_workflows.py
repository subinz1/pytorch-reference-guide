"""
Module 31 — torchao: Complete Quantization Workflows
=====================================================

End-to-end quantization workflows with benchmarking. Demonstrates INT8, INT4,
dynamic quantization, sparsity concepts, and the decision tree for choosing
a quantization strategy.

Usage:
    python torchao_workflows.py

Runs on CPU. GPU examples are clearly marked. torchao is optional — the script
falls back to manual demonstrations if not installed.
"""

import torch
import torch.nn as nn
import time
import sys
from contextlib import contextmanager

# Check torchao availability
try:
    import torchao
    from torchao import quantize_
    from torchao.quantization import (
        int8_weight_only,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
    )
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


# ============================================================================
# Model Definitions
# ============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP block as used in LLaMA-style models."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64  # Round to multiple of 64
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Simplified transformer block for quantization benchmarking."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    """Small transformer for quantization experiments."""
    def __init__(self, dim=512, n_heads=8, n_layers=4, vocab_size=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model):
    """Total memory used by model parameters in bytes."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    # Also count buffers (e.g., RMSNorm weight)
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total


def format_size(size_bytes):
    """Format byte count as human-readable string."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"


@contextmanager
def timer(label=""):
    """Simple context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if label:
        print(f"  [{label}: {elapsed * 1000:.1f} ms]")


def benchmark_model(model, input_fn, warmup=5, iterations=20):
    """Benchmark model inference, return mean time in ms."""
    for _ in range(warmup):
        with torch.no_grad():
            model(input_fn())

    times = []
    for _ in range(iterations):
        inp = input_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


# ============================================================================
# Part 1: Baseline Model Measurement
# ============================================================================

def part1_baseline():
    """Create a model and measure its baseline characteristics."""
    print("=" * 70)
    print("PART 1: Baseline Model Measurement")
    print("=" * 70)

    model = SmallTransformer(dim=512, n_heads=8, n_layers=4, vocab_size=1000)

    n_params = count_parameters(model)
    size = model_size_bytes(model)
    print(f"\n  Model: SmallTransformer (4 layers, dim=512, 8 heads)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Size (FP32): {format_size(size)}")

    tokens = torch.randint(0, 1000, (1, 64))
    latency = benchmark_model(model, lambda: tokens)
    print(f"  Inference latency (CPU, FP32): {latency:.1f} ms")

    # Show per-layer breakdown
    print(f"\n  Per-layer parameter breakdown:")
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            pct = param.numel() / n_params * 100
            print(f"    {name:45s} {str(list(param.shape)):>20s}  ({pct:.1f}%)")

    return model, tokens


# ============================================================================
# Part 2: Quantization with torchao (or manual fallback)
# ============================================================================

def part2_quantize_and_compare(base_model, tokens):
    """Apply different quantization methods and compare results."""
    print("\n" + "=" * 70)
    print("PART 2: Quantization Methods Comparison")
    print("=" * 70)

    import copy

    base_size = model_size_bytes(base_model)
    with torch.no_grad():
        base_output = base_model(tokens)
    base_latency = benchmark_model(base_model, lambda: tokens)

    results = [{
        "method": "FP32 (baseline)",
        "size": base_size,
        "latency": base_latency,
        "max_diff": 0.0,
        "mean_diff": 0.0,
    }]

    if TORCHAO_AVAILABLE:
        print(f"\n  torchao v{torchao.__version__} detected — using real quantization")

        # --- INT8 weight-only ---
        try:
            model_int8 = copy.deepcopy(base_model)
            quantize_(model_int8, int8_weight_only())
            with torch.no_grad():
                out_int8 = model_int8(tokens)
            diff = (base_output - out_int8).abs()
            results.append({
                "method": "INT8 weight-only",
                "size": model_size_bytes(model_int8),
                "latency": benchmark_model(model_int8, lambda: tokens),
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
            })
        except Exception as e:
            print(f"  INT8 weight-only skipped: {e}")

        # --- INT4 weight-only ---
        try:
            model_int4 = copy.deepcopy(base_model)
            quantize_(model_int4, int4_weight_only(group_size=128))
            with torch.no_grad():
                out_int4 = model_int4(tokens)
            diff = (base_output - out_int4).abs()
            results.append({
                "method": "INT4 weight-only (g128)",
                "size": model_size_bytes(model_int4),
                "latency": benchmark_model(model_int4, lambda: tokens),
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
            })
        except Exception as e:
            print(f"  INT4 weight-only skipped: {e}")

        # --- INT8 dynamic ---
        try:
            model_dyn = copy.deepcopy(base_model)
            quantize_(model_dyn, int8_dynamic_activation_int8_weight())
            with torch.no_grad():
                out_dyn = model_dyn(tokens)
            diff = (base_output - out_dyn).abs()
            results.append({
                "method": "INT8 dynamic",
                "size": model_size_bytes(model_dyn),
                "latency": benchmark_model(model_dyn, lambda: tokens),
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
            })
        except Exception as e:
            print(f"  INT8 dynamic skipped: {e}")

    else:
        print("\n  torchao not installed — simulating quantization effects manually")
        print("  Install torchao for real quantization: pip install torchao\n")

        for bits, label in [(8, "INT8 weight-only (sim.)"), (4, "INT4 weight-only (sim.)")]:
            model_sim = copy.deepcopy(base_model)
            with torch.no_grad():
                for name, param in model_sim.named_parameters():
                    if param.ndim >= 2:
                        qmax = 2 ** (bits - 1) - 1
                        qmin = -(2 ** (bits - 1))
                        scale = param.abs().amax() / qmax
                        q = torch.clamp(torch.round(param / scale), qmin, qmax)
                        param.data = q * scale

                out_sim = model_sim(tokens)

            diff = (base_output - out_sim).abs()
            est_size = sum(
                p.numel() * (bits / 8) if p.ndim >= 2 else p.numel() * p.element_size()
                for p in base_model.parameters()
            )
            results.append({
                "method": label,
                "size": int(est_size),
                "latency": benchmark_model(model_sim, lambda: tokens),
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
            })

    # Print comparison table
    print(f"\n  {'Method':>28} | {'Size':>10} | {'Compress':>9} | {'Latency':>10} | {'Max Diff':>10} | {'Mean Diff':>10}")
    print("  " + "-" * 90)
    for r in results:
        compress = base_size / r["size"] if r["size"] > 0 else 0
        print(f"  {r['method']:>28} | {format_size(r['size']):>10} | {compress:>8.2f}× | "
              f"{r['latency']:>8.1f}ms | {r['max_diff']:>10.6f} | {r['mean_diff']:>10.6f}")

    return results


# ============================================================================
# Part 3: torch.compile + Quantization
# ============================================================================

def part3_compile_integration():
    """Demonstrate torch.compile with quantized models."""
    print("\n" + "=" * 70)
    print("PART 3: torch.compile + Quantization")
    print("=" * 70)

    model = SwiGLUMLP(dim=256)
    x = torch.randn(8, 256)

    # Baseline: eager FP32
    eager_time = benchmark_model(model, lambda: x)
    print(f"\n  SwiGLU MLP (dim=256)")
    print(f"  Eager FP32:             {eager_time:.2f} ms")

    # Compiled FP32
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        compiled_time = benchmark_model(compiled_model, lambda: x, warmup=10)
        print(f"  Compiled FP32:          {compiled_time:.2f} ms")
    except Exception as e:
        print(f"  Compiled FP32: skipped ({e})")

    if TORCHAO_AVAILABLE:
        import copy

        # Quantized eager
        model_q = copy.deepcopy(model)
        try:
            quantize_(model_q, int8_weight_only())
            quant_time = benchmark_model(model_q, lambda: x)
            print(f"  INT8 Eager:             {quant_time:.2f} ms")
        except Exception as e:
            print(f"  INT8 Eager: skipped ({e})")

        # Quantized + compiled (the torchao sweet spot)
        try:
            model_qc = torch.compile(model_q, mode="reduce-overhead")
            qc_time = benchmark_model(model_qc, lambda: x, warmup=10)
            print(f"  INT8 + Compiled:        {qc_time:.2f} ms  ← best")
        except Exception as e:
            print(f"  INT8 + Compiled: skipped ({e})")
    else:
        print("\n  [torchao not installed — skipping quantized+compiled benchmarks]")
        print("  Key insight: torch.compile fuses dequantize+matmul into one kernel,")
        print("  eliminating intermediate memory allocations.")

    print(f"\n  The torchao + torch.compile combination works because:")
    print(f"  1. torchao stores weights as quantized tensor subclasses")
    print(f"  2. torch.compile traces through the dequant logic")
    print(f"  3. Inductor fuses dequant + matmul into a single kernel")
    print(f"  4. No intermediate FP16 weight materialization needed")


# ============================================================================
# Part 4: Semi-Structured Sparsity Concepts
# ============================================================================

def part4_sparsity_concepts():
    """Explain and demonstrate 2:4 semi-structured sparsity."""
    print("\n" + "=" * 70)
    print("PART 4: Semi-Structured Sparsity (2:4)")
    print("=" * 70)

    torch.manual_seed(42)
    W = torch.randn(8, 16)

    print(f"\n  Original weight (8×16):")
    for i in range(min(4, W.shape[0])):
        vals = "  ".join(f"{v:+.2f}" for v in W[i, :8].tolist())
        print(f"    row {i}: [{vals} ...]")

    # Apply 2:4 sparsity: keep 2 largest magnitude per group of 4
    W_sparse = W.clone()
    for i in range(W.shape[0]):
        for j in range(0, W.shape[1], 4):
            group = W[i, j:j + 4]
            _, indices = group.abs().topk(2, largest=False)
            for idx in indices:
                W_sparse[i, j + idx] = 0.0

    print(f"\n  After 2:4 sparsification (keep 2 largest per group of 4):")
    for i in range(min(4, W_sparse.shape[0])):
        vals = []
        for v in W_sparse[i, :8].tolist():
            if v == 0.0:
                vals.append("  0.00")
            else:
                vals.append(f"{v:+.2f}")
        print(f"    row {i}: [{('  '.join(vals))} ...]")

    # Measure sparsity and accuracy
    total_elements = W.numel()
    zero_elements = (W_sparse == 0).sum().item()
    sparsity = zero_elements / total_elements
    weight_error = (W - W_sparse).abs().mean().item()

    print(f"\n  Sparsity: {sparsity:.0%} ({zero_elements}/{total_elements} zeros)")
    print(f"  Weight MAE: {weight_error:.6f}")

    # Show matmul accuracy
    x = torch.randn(4, 16)
    ref_out = x @ W.T
    sparse_out = x @ W_sparse.T
    matmul_error = (ref_out - sparse_out).abs().mean().item()
    print(f"  Matmul output MAE: {matmul_error:.6f}")

    # Memory savings
    dense_bytes = total_elements * 2  # FP16
    sparse_data_bytes = (total_elements // 2) * 2  # Only non-zeros, FP16
    sparse_meta_bytes = total_elements // 4  # 2-bit index per group of 4
    sparse_total = sparse_data_bytes + sparse_meta_bytes

    print(f"\n  Memory comparison (FP16):")
    print(f"    Dense:  {dense_bytes} bytes")
    print(f"    Sparse: {sparse_total} bytes (data: {sparse_data_bytes}, metadata: {sparse_meta_bytes})")
    print(f"    Savings: {dense_bytes / sparse_total:.2f}×")

    print(f"\n  Hardware support: NVIDIA A100, H100 (Sparse Tensor Cores)")
    print(f"  Kernel speedup: ~2× for large matmuls on supported hardware")


# ============================================================================
# Part 5: PT2E Quantization Flow Explanation
# ============================================================================

def part5_pt2e_flow():
    """Explain the PT2E quantization pipeline."""
    print("\n" + "=" * 70)
    print("PART 5: PT2E Quantization Flow")
    print("=" * 70)

    print("""
  PT2E = PyTorch 2 Export-based quantization.
  Unlike torchao's quantize_() which works on eager models, PT2E:

  1. Exports the model with torch.export() → ATen IR graph
  2. Inserts observers (prepare_pt2e) to collect activation statistics
  3. Calibrates with representative data → determines quantization params
  4. Converts (convert_pt2e) → replaces observers with quantize/dequantize ops
  5. Optimizes for a specific backend (XNNPack, x86, ARM)

  Pipeline:

    model → torch.export() → prepare_pt2e(quantizer) → calibrate → convert_pt2e()
            ─────────────   ──────────────────────     ────────   ──────────────
            Capture graph   Insert observers           Run data   Apply quantization

  When to use PT2E:
    ✓ Mobile/edge deployment (Android, iOS)
    ✓ Static quantization (need calibration for best accuracy)
    ✓ Backend-specific optimizations (XNNPack, x86 AVX, ARM NEON)

  When to use torchao quantize_() instead:
    ✓ GPU inference (CUDA kernels)
    ✓ LLM serving
    ✓ Dynamic quantization (no calibration needed)
    ✓ Composing with torch.compile
""")

    # Demonstrate torch.export (which PT2E builds on)
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32)
            self.out = nn.Linear(32, 10)

        def forward(self, x):
            return self.out(torch.relu(self.fc(x)))

    model = SimpleNet()
    x = torch.randn(1, 64)

    try:
        exported = torch.export.export(model, (x,))
        print(f"  torch.export graph (basis for PT2E):")
        graph_str = str(exported.graph)
        for line in graph_str.split("\n")[:15]:
            print(f"    {line}")
        if graph_str.count("\n") > 15:
            print(f"    ... ({graph_str.count(chr(10)) - 15} more lines)")
    except Exception as e:
        print(f"  torch.export demo skipped: {e}")


# ============================================================================
# Part 6: Decision Tree — Choosing a Strategy
# ============================================================================

def recommend_quantization(
    task: str = "inference",
    batch_size: int = 1,
    gpu: str = "A100",
    model_size_b: float = 7.0,
    accuracy_priority: str = "medium",
    target: str = "gpu",
) -> str:
    """Recommend a quantization strategy based on use case parameters.

    Args:
        task: "inference" or "training"
        batch_size: typical batch size
        gpu: GPU model ("H100", "A100", "T4", "CPU", etc.)
        model_size_b: model size in billions of parameters
        accuracy_priority: "low", "medium", or "high"
        target: "gpu", "mobile", or "edge"
    """
    reasons = []

    if task == "training":
        if gpu in ("H100", "MI300"):
            reasons.append("H100/MI300 detected — FP8 training recommended")
            return f"FP8 training (Float8Linear) — {'; '.join(reasons)}"
        else:
            reasons.append("Standard GPU — BF16 mixed precision recommended")
            return f"BF16 mixed precision (AMP) — {'; '.join(reasons)}"

    # Inference
    if target in ("mobile", "edge"):
        reasons.append("Mobile/edge target — PT2E with XNNPack quantizer")
        return f"PT2E + XNNPack INT8 — {'; '.join(reasons)}"

    if gpu in ("H100", "L40S", "MI300"):
        if accuracy_priority == "high":
            reasons.append(f"{gpu} with high accuracy priority — FP8 preserves precision")
            return f"float8_dynamic_activation_float8_weight — {'; '.join(reasons)}"

    is_memory_bound = batch_size <= 2
    is_large_model = model_size_b >= 7.0

    if is_memory_bound or is_large_model:
        if accuracy_priority == "high":
            reasons.append("Memory-bound + accuracy priority — INT8 weight-only")
            return f"int8_weight_only() — {'; '.join(reasons)}"
        else:
            reasons.append("Memory-bound — INT4 weight-only for maximum memory savings")
            return f"int4_weight_only(group_size=128) — {'; '.join(reasons)}"
    else:
        reasons.append(f"Compute-bound (batch={batch_size}) — dynamic quantization")
        return f"int8_dynamic_activation_int8_weight() — {'; '.join(reasons)}"


def part6_decision_tree():
    """Demonstrate the quantization strategy decision tree."""
    print("\n" + "=" * 70)
    print("PART 6: Quantization Strategy Decision Tree")
    print("=" * 70)

    scenarios = [
        {"task": "inference", "batch_size": 1, "gpu": "A100", "model_size_b": 70.0,
         "accuracy_priority": "medium", "target": "gpu",
         "description": "LLM serving, single request, A100"},
        {"task": "inference", "batch_size": 32, "gpu": "A100", "model_size_b": 1.0,
         "accuracy_priority": "medium", "target": "gpu",
         "description": "Batch inference, medium model, A100"},
        {"task": "inference", "batch_size": 1, "gpu": "H100", "model_size_b": 7.0,
         "accuracy_priority": "high", "target": "gpu",
         "description": "Quality-critical LLM serving, H100"},
        {"task": "training", "batch_size": 64, "gpu": "H100", "model_size_b": 13.0,
         "accuracy_priority": "medium", "target": "gpu",
         "description": "Training 13B model on H100"},
        {"task": "inference", "batch_size": 1, "gpu": "CPU", "model_size_b": 0.1,
         "accuracy_priority": "medium", "target": "mobile",
         "description": "Mobile deployment, small model"},
        {"task": "training", "batch_size": 16, "gpu": "A100", "model_size_b": 7.0,
         "accuracy_priority": "medium", "target": "gpu",
         "description": "Training 7B model on A100"},
    ]

    for s in scenarios:
        desc = s.pop("description")
        recommendation = recommend_quantization(**s)
        print(f"\n  Scenario: {desc}")
        print(f"  → {recommendation}")

    print(f"\n  Decision summary:")
    print(f"    Memory-bound inference → INT4/INT8 weight-only")
    print(f"    Compute-bound inference → INT8 dynamic")
    print(f"    H100 inference → FP8")
    print(f"    H100 training → FP8 (Float8Linear)")
    print(f"    Other training → BF16 (AMP)")
    print(f"    Mobile/edge → PT2E + XNNPack")


# ============================================================================
# Part 7: Complete Workflow Summary
# ============================================================================

def part7_workflow_summary():
    """Print the complete recommended workflow."""
    print("\n" + "=" * 70)
    print("PART 7: Complete Quantization Workflow")
    print("=" * 70)

    print("""
  Step 1: Choose your method
  ─────────────────────────
    from torchao import quantize_
    from torchao.quantization import int8_weight_only  # or int4, dynamic, fp8

  Step 2: Quantize in-place
  ─────────────────────────
    quantize_(model, int8_weight_only())

  Step 3: Compile for fused kernels
  ─────────────────────────────────
    model = torch.compile(model, mode="max-autotune")

  Step 4: Benchmark
  ─────────────────
    # Warm up (first call compiles)
    output = model(sample_input)
    # Benchmark subsequent calls
    latency = benchmark(model, sample_input)

  Step 5: Validate accuracy
  ─────────────────────────
    # Compare against FP32/FP16 baseline on your evaluation set
    accuracy_quantized = evaluate(model_quantized, eval_data)
    accuracy_baseline = evaluate(model_baseline, eval_data)
    print(f"Accuracy drop: {accuracy_baseline - accuracy_quantized:.2f}%")

  Step 6: Deploy
  ──────────────
    # Option A: Serve with torch.compile (GPU)
    # Option B: Export with torch.export → NativeRT (C++ serving)
    # Option C: PT2E → XNNPack (mobile)
""")


# ============================================================================
# Main
# ============================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + "Module 31 — torchao Quantization Workflows".center(68) + "║")
    print("║" + "Complete quantization pipelines with benchmarking".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if TORCHAO_AVAILABLE:
        print(f"\n  torchao {torchao.__version__} detected ✓")
    else:
        print(f"\n  torchao not installed — running with manual fallbacks")
        print(f"  Install: pip install torchao")

    model, tokens = part1_baseline()
    part2_quantize_and_compare(model, tokens)
    part3_compile_integration()
    part4_sparsity_concepts()
    part5_pt2e_flow()
    part6_decision_tree()
    part7_workflow_summary()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  torchao makes model optimization a one-liner:
    quantize_(model, int8_weight_only())   # + torch.compile for best results

  Key methods:
    int8_weight_only()                        → ~2x memory savings
    int4_weight_only(group_size=128)          → ~4x memory savings
    int8_dynamic_activation_int8_weight()     → best for batch inference
    float8_dynamic_activation_float8_weight() → best on H100

  The magic: torchao + torch.compile = fused quantized kernels

  See: quantization_basics.py for the underlying math
  See: Module 29 (Mixed Precision) for FP16/BF16/FP8 training details
  See: Module 08 (torch.compile) for compilation deep dive
""")


if __name__ == "__main__":
    main()
