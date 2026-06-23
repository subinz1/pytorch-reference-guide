"""
Module 31 — torchao: Quantization Basics
=========================================

Demonstrates quantization fundamentals using pure PyTorch (no torchao required).
If torchao is installed, also shows the quantize_() API in action.

Usage:
    python quantization_basics.py

No GPU required — all examples run on CPU.
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# Part 1: What is Quantization?
# ============================================================================

def demonstrate_quantization_math():
    """Show the core math behind INT8 quantization."""
    print("=" * 70)
    print("PART 1: Quantization Math")
    print("=" * 70)

    torch.manual_seed(42)
    weights = torch.randn(4, 8) * 0.5  # Typical small weight values

    print(f"\nOriginal FP32 weights (4×8):")
    print(weights)
    print(f"  dtype: {weights.dtype}")
    print(f"  range: [{weights.min():.4f}, {weights.max():.4f}]")

    # --- Symmetric quantization (zero_point = 0) ---
    print("\n--- Symmetric INT8 Quantization ---")
    abs_max = weights.abs().max()
    scale = abs_max / 127.0  # INT8 range: [-128, 127]

    quantized = torch.clamp(torch.round(weights / scale), -128, 127).to(torch.int8)
    dequantized = quantized.float() * scale

    error = (weights - dequantized).abs()
    print(f"  scale: {scale:.6f}")
    print(f"  quantized (int8):\n{quantized}")
    print(f"  dequantized:\n{dequantized}")
    print(f"  max absolute error: {error.max():.6f}")
    print(f"  mean absolute error: {error.mean():.6f}")

    # --- Asymmetric quantization (with zero_point) ---
    print("\n--- Asymmetric INT8 Quantization ---")
    w_min, w_max = weights.min(), weights.max()
    scale_asym = (w_max - w_min) / 255.0  # UINT8 range: [0, 255]
    zero_point = torch.clamp(torch.round(-w_min / scale_asym), 0, 255).to(torch.int32)

    quantized_asym = torch.clamp(
        torch.round(weights / scale_asym) + zero_point, 0, 255
    ).to(torch.uint8)
    dequantized_asym = (quantized_asym.float() - zero_point.float()) * scale_asym

    error_asym = (weights - dequantized_asym).abs()
    print(f"  scale: {scale_asym:.6f}, zero_point: {zero_point.item()}")
    print(f"  quantized (uint8):\n{quantized_asym}")
    print(f"  max absolute error: {error_asym.max():.6f}")
    print(f"  mean absolute error: {error_asym.mean():.6f}")

    return scale, quantized, dequantized


# ============================================================================
# Part 2: Precision Loss — Round-Trip Error
# ============================================================================

def demonstrate_precision_loss():
    """Show how quantization introduces error, and how it varies with data distribution."""
    print("\n" + "=" * 70)
    print("PART 2: Precision Loss (FP32 → INT8 → FP32 Round-Trip)")
    print("=" * 70)

    torch.manual_seed(42)
    sizes = [64, 256, 1024, 4096]

    print(f"\n{'Size':>8} | {'Max Error':>12} | {'Mean Error':>12} | {'Relative Error':>15}")
    print("-" * 60)

    for size in sizes:
        w = torch.randn(size, size) * 0.02  # Typical init scale

        abs_max = w.abs().max()
        scale = abs_max / 127.0
        w_int8 = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        w_roundtrip = w_int8.float() * scale

        max_err = (w - w_roundtrip).abs().max()
        mean_err = (w - w_roundtrip).abs().mean()
        rel_err = mean_err / w.abs().mean()

        print(f"{size:>8} | {max_err:>12.8f} | {mean_err:>12.8f} | {rel_err:>14.4%}")

    # Show that outliers hurt quantization
    print("\n--- Effect of Outliers ---")
    w_normal = torch.randn(1024) * 0.1
    w_outlier = w_normal.clone()
    w_outlier[0] = 10.0  # Single outlier

    for name, tensor in [("No outliers", w_normal), ("With outlier", w_outlier)]:
        abs_max = tensor.abs().max()
        scale = abs_max / 127.0
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        dq = q.float() * scale
        mean_err = (tensor - dq).abs().mean()
        print(f"  {name:15s}: scale={scale:.6f}, mean_error={mean_err:.6f}")

    print("\n  → Outliers increase scale, reducing precision for all other values.")
    print("  → This is why per-group quantization (group_size=128) helps!")


# ============================================================================
# Part 3: Group-Wise Quantization
# ============================================================================

def demonstrate_group_quantization():
    """Implement and compare per-tensor vs per-channel vs per-group quantization."""
    print("\n" + "=" * 70)
    print("PART 3: Group-Wise Quantization")
    print("=" * 70)

    torch.manual_seed(42)
    out_features, in_features = 256, 512
    W = torch.randn(out_features, in_features) * 0.02
    # Add some per-channel variance (realistic for trained models)
    channel_scales = torch.rand(out_features, 1) * 2 + 0.5
    W = W * channel_scales

    def quantize_per_tensor(w):
        scale = w.abs().max() / 127.0
        q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        return q.float() * scale

    def quantize_per_channel(w):
        scales = w.abs().amax(dim=1, keepdim=True) / 127.0
        scales = scales.clamp(min=1e-8)
        q = torch.clamp(torch.round(w / scales), -128, 127).to(torch.int8)
        return q.float() * scales

    def quantize_per_group(w, group_size=128):
        orig_shape = w.shape
        assert w.shape[1] % group_size == 0
        w_grouped = w.reshape(-1, group_size)
        scales = w_grouped.abs().amax(dim=1, keepdim=True) / 127.0
        scales = scales.clamp(min=1e-8)
        q = torch.clamp(torch.round(w_grouped / scales), -128, 127).to(torch.int8)
        dq = q.float() * scales
        return dq.reshape(orig_shape)

    results = {}
    for name, fn in [
        ("Per-tensor", quantize_per_tensor),
        ("Per-channel", quantize_per_channel),
        ("Per-group (128)", lambda w: quantize_per_group(w, 128)),
        ("Per-group (32)", lambda w: quantize_per_group(w, 32)),
    ]:
        dq = fn(W)
        err = (W - dq).abs()
        results[name] = {"max": err.max().item(), "mean": err.mean().item()}

    print(f"\nWeight shape: {W.shape}")
    print(f"\n{'Method':>20} | {'Max Error':>12} | {'Mean Error':>12} | {'Improvement':>12}")
    print("-" * 65)
    baseline = results["Per-tensor"]["mean"]
    for name, r in results.items():
        improvement = baseline / r["mean"]
        print(f"{name:>20} | {r['max']:>12.8f} | {r['mean']:>12.8f} | {improvement:>11.2f}×")

    print("\n  → Per-group quantization significantly reduces error")
    print("  → Smaller group_size = better accuracy, but more scale storage overhead")


# ============================================================================
# Part 4: Memory Savings Calculation
# ============================================================================

def demonstrate_memory_savings():
    """Calculate and compare memory usage across dtypes and quantization levels."""
    print("\n" + "=" * 70)
    print("PART 4: Memory Savings Comparison")
    print("=" * 70)

    param_counts = {
        "Small model (100M)": 100_000_000,
        "Medium model (1B)": 1_000_000_000,
        "Large model (7B)": 7_000_000_000,
        "XL model (70B)": 70_000_000_000,
    }

    dtypes = {
        "FP32": 4,
        "FP16/BF16": 2,
        "INT8": 1,
        "INT4": 0.5,
    }

    print(f"\n{'Model':>25} | ", end="")
    for dtype in dtypes:
        print(f"{dtype:>12} | ", end="")
    print()
    print("-" * 85)

    for model_name, params in param_counts.items():
        print(f"{model_name:>25} | ", end="")
        for dtype_name, bytes_per_param in dtypes.items():
            size_gb = params * bytes_per_param / (1024 ** 3)
            if size_gb >= 1:
                print(f"{size_gb:>9.1f} GB | ", end="")
            else:
                size_mb = size_gb * 1024
                print(f"{size_mb:>8.0f} MB | ", end="")
        print()

    # Show quantization overhead (scales storage)
    print("\n--- Quantization Overhead (Scale Storage) ---")
    n_params = 7_000_000_000
    for group_size in [None, 128, 32]:
        weight_bytes = n_params  # INT8 = 1 byte per param
        if group_size is None:
            label = "Per-channel"
            n_scales = n_params // 512  # Approximate: one scale per output channel
        else:
            label = f"Group size {group_size}"
            n_scales = n_params // group_size

        scale_bytes = n_scales * 2  # FP16 scales
        total_gb = (weight_bytes + scale_bytes) / (1024 ** 3)
        overhead_pct = scale_bytes / weight_bytes * 100
        print(f"  {label:>15}: {total_gb:.2f} GB total ({overhead_pct:.1f}% scale overhead)")


# ============================================================================
# Part 5: Symmetric vs Asymmetric Quantization
# ============================================================================

def demonstrate_symmetric_vs_asymmetric():
    """Compare symmetric and asymmetric quantization approaches."""
    print("\n" + "=" * 70)
    print("PART 5: Symmetric vs Asymmetric Quantization")
    print("=" * 70)

    torch.manual_seed(42)

    distributions = {
        "Centered (normal)": torch.randn(10000) * 0.5,
        "Positive-skewed (ReLU output)": torch.relu(torch.randn(10000)) * 0.5,
        "Biased (mean=2.0)": torch.randn(10000) * 0.5 + 2.0,
    }

    print(f"\n{'Distribution':>30} | {'Symm. Error':>13} | {'Asymm. Error':>14} | {'Winner':>10}")
    print("-" * 80)

    for name, data in distributions.items():
        # Symmetric: maps to [-128, 127], zero stays at zero
        abs_max = data.abs().max()
        sym_scale = abs_max / 127.0
        sym_q = torch.clamp(torch.round(data / sym_scale), -128, 127).to(torch.int8)
        sym_dq = sym_q.float() * sym_scale
        sym_err = (data - sym_dq).abs().mean()

        # Asymmetric: maps to [0, 255], uses zero_point
        d_min, d_max = data.min(), data.max()
        asym_scale = (d_max - d_min) / 255.0
        asym_zp = torch.round(-d_min / asym_scale).clamp(0, 255)
        asym_q = torch.clamp(torch.round(data / asym_scale) + asym_zp, 0, 255).to(torch.uint8)
        asym_dq = (asym_q.float() - asym_zp) * asym_scale
        asym_err = (data - asym_dq).abs().mean()

        winner = "Symmetric" if sym_err <= asym_err else "Asymmetric"
        print(f"{name:>30} | {sym_err:>13.8f} | {asym_err:>14.8f} | {winner:>10}")

    print("\n  → Symmetric: simpler (no zero_point), faster computation")
    print("  → Asymmetric: better for non-centered distributions (e.g., after ReLU)")
    print("  → Most modern frameworks (torchao, GPTQ, AWQ) use symmetric by default")


# ============================================================================
# Part 6: torchao quantize_() Demo (if available)
# ============================================================================

def demonstrate_torchao_quantize():
    """Demo torchao's quantize_() API if installed, skip gracefully otherwise."""
    print("\n" + "=" * 70)
    print("PART 6: torchao quantize_() API")
    print("=" * 70)

    try:
        import torchao
        from torchao import quantize_
        from torchao.quantization import int8_weight_only
        TORCHAO_AVAILABLE = True
        print(f"\n  torchao version: {torchao.__version__}")
    except ImportError:
        TORCHAO_AVAILABLE = False
        print("\n  torchao not installed — showing conceptual demo instead.")
        print("  Install with: pip install torchao")

    class SimpleModel(nn.Module):
        def __init__(self, in_features=512, hidden=1024, out_features=256):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, out_features)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleModel()
    x = torch.randn(4, 512)

    def model_size_mb(m):
        return sum(p.nelement() * p.element_size() for p in m.parameters()) / (1024 * 1024)

    # Baseline
    baseline_out = model(x)
    baseline_size = model_size_mb(model)
    print(f"\n  Baseline model size: {baseline_size:.2f} MB")
    print(f"  Weight dtype: {model.fc1.weight.dtype}")
    print(f"  Output shape: {baseline_out.shape}")

    if TORCHAO_AVAILABLE:
        quantize_(model, int8_weight_only())

        quantized_out = model(x)
        quantized_size = model_size_mb(model)

        print(f"\n  After int8_weight_only():")
        print(f"  Model size: {quantized_size:.2f} MB")
        print(f"  Weight type: {type(model.fc1.weight).__name__}")
        print(f"  Compression ratio: {baseline_size / quantized_size:.2f}×")

        diff = (baseline_out - quantized_out).abs()
        print(f"  Max output difference: {diff.max():.6f}")
        print(f"  Mean output difference: {diff.mean():.6f}")
    else:
        # Manual simulation of what torchao does
        print("\n  [Simulating INT8 weight-only quantization manually]")
        total_orig = 0
        total_quant = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                orig_bytes = param.nelement() * param.element_size()
                quant_bytes = param.nelement() * 1  # INT8 = 1 byte
                scale_bytes = param.shape[0] * 2  # FP16 scale per channel
                total_orig += orig_bytes
                total_quant += quant_bytes + scale_bytes
                print(f"  {name}: {list(param.shape)} → "
                      f"{orig_bytes / 1024:.1f} KB → {(quant_bytes + scale_bytes) / 1024:.1f} KB")
            else:
                total_orig += param.nelement() * param.element_size()
                total_quant += param.nelement() * param.element_size()

        print(f"\n  Estimated compression: {total_orig / total_quant:.2f}×")
        print(f"  Original: {total_orig / (1024 * 1024):.2f} MB → Quantized: ~{total_quant / (1024 * 1024):.2f} MB")


# ============================================================================
# Part 7: End-to-End — Accuracy vs Compression
# ============================================================================

def demonstrate_accuracy_tradeoff():
    """Show how different quantization bit-widths affect a simple computation."""
    print("\n" + "=" * 70)
    print("PART 7: Accuracy vs Compression Tradeoff")
    print("=" * 70)

    torch.manual_seed(42)
    W = torch.randn(256, 256) * 0.02  # Weight matrix
    x = torch.randn(32, 256) * 0.1  # Input batch

    reference = x @ W.T  # FP32 reference output

    results = []
    for bits, label in [(8, "INT8"), (4, "INT4"), (2, "INT2")]:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        scale = W.abs().max() / qmax
        W_q = torch.clamp(torch.round(W / scale), qmin, qmax)
        W_dq = W_q * scale

        output = x @ W_dq.T
        output_err = (reference - output).abs()

        results.append({
            "bits": bits,
            "label": label,
            "weight_error": (W - W_dq).abs().mean().item(),
            "output_max_error": output_err.max().item(),
            "output_mean_error": output_err.mean().item(),
            "compression": 32.0 / bits,
        })

    print(f"\n{'Method':>8} | {'Compression':>12} | {'Weight Err':>12} | {'Out Max Err':>12} | {'Out Mean Err':>13}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:>8} | {r['compression']:>11.1f}× | {r['weight_error']:>12.8f} | "
              f"{r['output_max_error']:>12.8f} | {r['output_mean_error']:>13.8f}")

    print("\n  → INT8: Excellent accuracy with 4× compression")
    print("  → INT4: Good accuracy with 8× compression (standard for LLM serving)")
    print("  → INT2: Significant degradation — rarely used in practice")


# ============================================================================
# Main
# ============================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + "Module 31 — Quantization Basics".center(68) + "║")
    print("║" + "Pure PyTorch quantization fundamentals".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    demonstrate_quantization_math()
    demonstrate_precision_loss()
    demonstrate_group_quantization()
    demonstrate_memory_savings()
    demonstrate_symmetric_vs_asymmetric()
    demonstrate_torchao_quantize()
    demonstrate_accuracy_tradeoff()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key takeaways:
  1. Quantization maps floating-point values to integers using scale + zero_point
  2. Symmetric quantization is simpler and preferred for weights
  3. Group-wise quantization reduces error by using per-group scales
  4. INT8 gives ~4× compression (vs FP32) with minimal accuracy loss
  5. INT4 gives ~8× compression — the standard for LLM inference
  6. torchao's quantize_() makes this a one-liner in practice
  7. Outliers in weights hurt quantization — group-wise quantization mitigates this

Next: torchao_workflows.py — complete quantization pipelines with benchmarking
""")


if __name__ == "__main__":
    main()
