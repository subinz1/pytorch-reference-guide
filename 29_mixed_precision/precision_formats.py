"""
Precision Formats — Exploring PyTorch's Numerical Dtypes

This script demonstrates all floating-point formats available in PyTorch:
FP32, FP16, BF16, FP8 E4M3, FP8 E5M2. All examples run on CPU.

Topics covered:
- Creating tensors in every precision format
- Range and precision properties (max, min, eps)
- Memory usage comparison
- Precision loss in FP32 → FP16 → FP32 round-trips
- BF16 vs FP16 range comparison (overflow behavior)
- Float8 creation and properties
"""

import torch
import struct


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# 1. Creating Tensors in Every Dtype
# ============================================================================

section("1. Creating Tensors in Every Dtype")

dtypes_info = [
    ("FP32", torch.float32, 32),
    ("FP16", torch.float16, 16),
    ("BF16", torch.bfloat16, 16),
    ("FP8 E4M3", torch.float8_e4m3fn, 8),
    ("FP8 E5M2", torch.float8_e5m2, 8),
]

x_base = torch.tensor([1.0, 0.1, 3.14159, 100.0, 0.0001])

print(f"Source FP32 tensor: {x_base}")
print(f"{'Format':<12} {'Bits':<6} {'Tensor values':<50} {'Bytes'}")
print("-" * 80)

for name, dtype, bits in dtypes_info:
    x = x_base.to(dtype)
    x_back = x.to(torch.float32)
    nbytes = x.element_size() * x.numel()
    print(f"{name:<12} {bits:<6} {str(x_back.tolist()):<50} {nbytes}")


# ============================================================================
# 2. Range and Precision Properties
# ============================================================================

section("2. Range and Precision for Each Format")

print(f"{'Format':<12} {'Max Value':<15} {'Min Pos Normal':<18} {'Eps (at 1.0)':<15} {'Exponent':<10} {'Mantissa'}")
print("-" * 90)

fp_formats = [
    ("FP32", torch.float32, 8, 23),
    ("FP16", torch.float16, 5, 10),
    ("BF16", torch.bfloat16, 8, 7),
]

for name, dtype, exp_bits, mant_bits in fp_formats:
    info = torch.finfo(dtype)
    print(f"{name:<12} {info.max:<15.6g} {info.tiny:<18.6g} {info.eps:<15.6g} {exp_bits:<10} {mant_bits}")

# FP8 formats don't have finfo, show known values
fp8_formats = [
    ("FP8 E4M3", torch.float8_e4m3fn, 4, 3, 448.0, 2**-6),
    ("FP8 E5M2", torch.float8_e5m2, 5, 2, 57344.0, 2**-2),
]

for name, dtype, exp_bits, mant_bits, max_val, min_normal in fp8_formats:
    # Verify max by creating tensor
    try:
        test = torch.tensor(max_val, dtype=torch.float32).to(dtype).to(torch.float32)
        actual_max = test.item()
    except Exception:
        actual_max = max_val
    print(f"{name:<12} {max_val:<15.6g} {min_normal:<18.6g} {'N/A':<15} {exp_bits:<10} {mant_bits}")


# ============================================================================
# 3. Memory Comparison
# ============================================================================

section("3. Memory Usage Comparison")

num_elements = 1_000_000  # 1M elements (simulating model parameters)

print(f"Storing {num_elements:,} elements (like a 1M-parameter layer):\n")
print(f"{'Format':<12} {'Bytes/Element':<15} {'Total Memory':<15} {'vs FP32'}")
print("-" * 60)

fp32_size = num_elements * 4
for name, dtype, bits in dtypes_info:
    x = torch.zeros(num_elements, dtype=dtype)
    total = x.element_size() * x.numel()
    ratio = total / fp32_size
    print(f"{name:<12} {x.element_size():<15} {total/1024/1024:.2f} MB{'':<8} {ratio:.2f}×")

print(f"\nFor a 7B parameter model:")
for name, dtype, bits in dtypes_info:
    size_gb = 7e9 * (bits / 8) / 1e9
    print(f"  {name:<12}: {size_gb:.1f} GB")


# ============================================================================
# 4. Precision Loss Demonstration
# ============================================================================

section("4. Precision Loss: FP32 → FP16 → FP32 Round-Trip")

test_values = torch.tensor([
    1.0,           # Exactly representable
    0.1,           # Not exactly representable in binary
    3.14159265,    # Pi — shows precision loss
    1e-4,          # Small value — near FP16 limits
    1e-7,          # Very small — below FP16 min normal
    12345.678,     # Large with decimals
    0.333333333,   # 1/3
])

print(f"{'Original FP32':<20} {'→ FP16 → FP32':<20} {'Abs Error':<15} {'Rel Error'}")
print("-" * 75)

for val in test_values:
    original = val.item()
    roundtrip = val.to(torch.float16).to(torch.float32).item()
    abs_err = abs(original - roundtrip)
    rel_err = abs_err / abs(original) if original != 0 else 0
    print(f"{original:<20.10f} {roundtrip:<20.10f} {abs_err:<15.2e} {rel_err:.2e}")

# Same for BF16
print(f"\n{'Original FP32':<20} {'→ BF16 → FP32':<20} {'Abs Error':<15} {'Rel Error'}")
print("-" * 75)

for val in test_values:
    original = val.item()
    roundtrip = val.to(torch.bfloat16).to(torch.float32).item()
    abs_err = abs(original - roundtrip)
    rel_err = abs_err / abs(original) if original != 0 else 0
    print(f"{original:<20.10f} {roundtrip:<20.10f} {abs_err:<15.2e} {rel_err:.2e}")


# ============================================================================
# 5. Accumulated Precision Error
# ============================================================================

section("5. Accumulated Error: Summing 10,000 Small Values")

n = 10_000
small_val = 0.0001

# FP32 accumulation
acc_fp32 = torch.tensor(0.0, dtype=torch.float32)
for _ in range(n):
    acc_fp32 += small_val

# FP16 accumulation
acc_fp16 = torch.tensor(0.0, dtype=torch.float16)
for _ in range(n):
    acc_fp16 += torch.tensor(small_val, dtype=torch.float16)

# BF16 accumulation
acc_bf16 = torch.tensor(0.0, dtype=torch.bfloat16)
for _ in range(n):
    acc_bf16 += torch.tensor(small_val, dtype=torch.bfloat16)

expected = n * small_val
print(f"Expected result: {expected}")
print(f"FP32 result:     {acc_fp32.item():.6f}  (error: {abs(acc_fp32.item() - expected):.6f})")
print(f"FP16 result:     {acc_fp16.float().item():.6f}  (error: {abs(acc_fp16.float().item() - expected):.6f})")
print(f"BF16 result:     {acc_bf16.float().item():.6f}  (error: {abs(acc_bf16.float().item() - expected):.6f})")
print(f"\nBF16 has LESS precision than FP16 (7 vs 10 mantissa bits)")
print(f"But BF16 has MORE range — which matters more for training stability")


# ============================================================================
# 6. BF16 vs FP16 Range Comparison
# ============================================================================

section("6. BF16 vs FP16: Range Comparison (Overflow Behavior)")

large_values = [100.0, 1000.0, 10000.0, 50000.0, 65504.0, 70000.0, 100000.0, 1e10, 1e20, 1e38]

print(f"{'FP32 Value':<15} {'FP16':<15} {'BF16':<15} {'FP16 OK?':<10} {'BF16 OK?'}")
print("-" * 70)

for val in large_values:
    t_fp32 = torch.tensor(val, dtype=torch.float32)
    t_fp16 = t_fp32.to(torch.float16).to(torch.float32)
    t_bf16 = t_fp32.to(torch.bfloat16).to(torch.float32)

    fp16_ok = "✓" if not torch.isinf(t_fp16) else "OVERFLOW"
    bf16_ok = "✓" if not torch.isinf(t_bf16) else "OVERFLOW"

    print(f"{val:<15.4g} {t_fp16.item():<15.4g} {t_bf16.item():<15.4g} {fp16_ok:<10} {bf16_ok}")

print("\nKey insight: BF16 shares FP32's exponent (8 bits) → same range (3.4e38)")
print("FP16 has only 5 exponent bits → max value is 65504")
print("This is why BF16 doesn't need GradScaler — no overflow risk!")


# ============================================================================
# 7. Simulating Gradient Underflow
# ============================================================================

section("7. Gradient Underflow in FP16")

gradient_magnitudes = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

print("Simulating small gradients (common in deep networks):")
print(f"\n{'Gradient (FP32)':<18} {'In FP16':<18} {'In BF16':<18} {'FP16 Zero?':<12} {'BF16 Zero?'}")
print("-" * 80)

for mag in gradient_magnitudes:
    g_fp32 = torch.tensor(mag, dtype=torch.float32)
    g_fp16 = g_fp32.to(torch.float16)
    g_bf16 = g_fp32.to(torch.bfloat16)

    fp16_zero = "ZERO!" if g_fp16.item() == 0.0 else "ok"
    bf16_zero = "ZERO!" if g_bf16.item() == 0.0 else "ok"

    print(f"{mag:<18.1e} {g_fp16.float().item():<18.1e} {g_bf16.float().item():<18.1e} {fp16_zero:<12} {bf16_zero}")

print("\nWith GradScaler (scale=65536), the same gradients after scaling:")
scale = 65536.0
print(f"\n{'Original':<15} {'Scaled (FP32)':<15} {'Scaled in FP16':<18} {'Survives?'}")
print("-" * 65)
for mag in gradient_magnitudes:
    scaled = mag * scale
    g_scaled_fp16 = torch.tensor(scaled, dtype=torch.float16)
    survives = "YES" if g_scaled_fp16.item() != 0.0 and not torch.isinf(g_scaled_fp16) else "NO"
    print(f"{mag:<15.1e} {scaled:<15.1e} {g_scaled_fp16.float().item():<18.6g} {survives}")


# ============================================================================
# 8. Float8 Creation and Properties
# ============================================================================

section("8. Float8 Formats: E4M3 and E5M2")

# Create test values
test_vals = torch.tensor([0.5, 1.0, 2.0, 10.0, 100.0, 200.0, 400.0, 448.0, 500.0])

print("E4M3 (4 exponent, 3 mantissa) — range ±448, better precision:")
print(f"{'FP32 Input':<12} {'E4M3 Output':<15} {'Representable?'}")
print("-" * 45)
for val in test_vals:
    fp8 = val.unsqueeze(0).to(torch.float8_e4m3fn)
    back = fp8.to(torch.float32).item()
    ok = "YES" if abs(back) <= 448 and not (val.item() > 448 and back == 448) else "CLIPPED"
    if val.item() > 448:
        ok = "SATURATED"
    print(f"{val.item():<12.1f} {back:<15.4f} {ok}")

print(f"\nE5M2 (5 exponent, 2 mantissa) — range ±57344, less precision:")
test_vals_e5m2 = torch.tensor([0.5, 1.0, 2.0, 100.0, 1000.0, 10000.0, 50000.0, 57344.0, 60000.0])
print(f"{'FP32 Input':<12} {'E5M2 Output':<15} {'Representable?'}")
print("-" * 45)
for val in test_vals_e5m2:
    fp8 = val.unsqueeze(0).to(torch.float8_e5m2)
    back = fp8.to(torch.float32).item()
    ok = "YES" if not torch.isinf(torch.tensor(back)) else "OVERFLOW"
    print(f"{val.item():<12.1f} {back:<15.4f} {ok}")


# ============================================================================
# 9. Quantization Error Distribution
# ============================================================================

section("9. Quantization Error Distribution Across Formats")

torch.manual_seed(42)
x = torch.randn(10000)  # Standard normal — typical activation distribution

formats = [
    ("FP16", torch.float16),
    ("BF16", torch.bfloat16),
    ("FP8 E4M3", torch.float8_e4m3fn),
    ("FP8 E5M2", torch.float8_e5m2),
]

print(f"Input: 10,000 values from N(0,1) (typical activation distribution)")
print(f"\n{'Format':<12} {'Mean Abs Err':<15} {'Max Abs Err':<15} {'RMSE':<15} {'% Exact Zeros'}")
print("-" * 75)

for name, dtype in formats:
    x_cast = x.to(dtype).to(torch.float32)
    error = (x - x_cast).abs()
    mean_err = error.mean().item()
    max_err = error.max().item()
    rmse = ((x - x_cast) ** 2).mean().sqrt().item()
    pct_zero = ((x_cast == 0) & (x != 0)).float().mean().item() * 100
    print(f"{name:<12} {mean_err:<15.6f} {max_err:<15.6f} {rmse:<15.6f} {pct_zero:.2f}%")


# ============================================================================
# 10. Practical Scaling for FP8
# ============================================================================

section("10. Per-Tensor Scaling for FP8")

torch.manual_seed(42)
activations = torch.randn(64, 256) * 5.0  # Simulated layer activations

print(f"Activations shape: {activations.shape}")
print(f"Activations range: [{activations.min():.3f}, {activations.max():.3f}]")
print(f"Activations absmax: {activations.abs().max():.3f}")

# Without scaling — direct cast
direct_fp8 = activations.to(torch.float8_e4m3fn).to(torch.float32)
direct_error = (activations - direct_fp8).abs().mean()
print(f"\nDirect cast to E4M3 (no scaling):")
print(f"  Mean absolute error: {direct_error:.6f}")
print(f"  Values clipped to ±448: {(activations.abs() > 448).sum().item()}")

# With per-tensor scaling
absmax = activations.abs().max()
e4m3_max = 448.0
scale = e4m3_max / absmax
scaled_activations = activations * scale
scaled_fp8 = scaled_activations.to(torch.float8_e4m3fn).to(torch.float32)
unscaled_result = scaled_fp8 / scale

scaled_error = (activations - unscaled_result).abs().mean()
print(f"\nWith per-tensor scaling (scale={scale.item():.4f}):")
print(f"  Mean absolute error: {scaled_error:.6f}")
print(f"  Improvement: {direct_error / scaled_error:.2f}×")


# ============================================================================
# 11. Summary
# ============================================================================

section("Summary: Choosing the Right Precision")

print("""
┌─────────────────────────────────────────────────────────────────────┐
│  Format    │ Memory │ Speed   │ Accuracy  │ Best For               │
├────────────┼────────┼─────────┼───────────┼────────────────────────┤
│  FP32      │ 4B     │ 1×      │ Highest   │ Debugging, reference   │
│  BF16      │ 2B     │ 2-3×    │ Good      │ LLM training (default) │
│  FP16      │ 2B     │ 2-3×    │ Good+     │ Inference, older GPUs  │
│  FP8 E4M3  │ 1B     │ 4-6×    │ Moderate  │ Forward pass (H100+)   │
│  FP8 E5M2  │ 1B     │ 4-6×    │ Lower     │ Backward pass (H100+)  │
└─────────────────────────────────────────────────────────────────────┘

Key rules:
  • BF16 for training (Ampere+ GPUs) — no GradScaler needed
  • FP16 for inference on older GPUs — bounded inputs prevent overflow
  • FP8 for large-scale training on H100+ — requires per-tensor scaling
  • Always keep optimizer states and master weights in FP32
""")

if __name__ == "__main__":
    pass
