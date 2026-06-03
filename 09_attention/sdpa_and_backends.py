"""
SDPA and Backend Control — PyTorch's Optimized Attention
=========================================================
Demonstrates F.scaled_dot_product_attention and its backend selection.

Run: python sdpa_and_backends.py
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import time

# =============================================================================
# 1. Basic SDPA usage
# =============================================================================

print("=" * 60)
print("F.scaled_dot_product_attention (SDPA)")
print("=" * 60)

torch.manual_seed(42)

batch, heads, seq_len, dim = 2, 8, 64, 32

Q = torch.randn(batch, heads, seq_len, dim)
K = torch.randn(batch, heads, seq_len, dim)
V = torch.randn(batch, heads, seq_len, dim)

print(f"\n  Input shapes:")
print(f"    Q: {list(Q.shape)} [batch, heads, seq_q, dim]")
print(f"    K: {list(K.shape)} [batch, heads, seq_kv, dim]")
print(f"    V: {list(V.shape)} [batch, heads, seq_kv, dim]")

# Basic attention (no mask)
output = F.scaled_dot_product_attention(Q, K, V)
print(f"\n  Output shape: {list(output.shape)} [batch, heads, seq_q, dim]")

# =============================================================================
# 2. Causal attention with is_causal=True
# =============================================================================

print("\n" + "=" * 60)
print("CAUSAL ATTENTION")
print("=" * 60 + "\n")

# Method 1: is_causal flag (recommended — most efficient)
output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(f"  is_causal=True output shape: {list(output_causal.shape)}")

# Method 2: Explicit mask (flexible but less optimized)
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
output_mask = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask)
print(f"  Explicit mask output shape: {list(output_mask.shape)}")

# Verify they produce the same result
print(f"  Match: {torch.allclose(output_causal, output_mask, atol=1e-6)}")

# =============================================================================
# 3. Custom attention mask
# =============================================================================

print("\n" + "=" * 60)
print("CUSTOM ATTENTION MASKS")
print("=" * 60 + "\n")

# Boolean mask: True = attend, False = block
# Example: sliding window attention (each position attends to ±3 positions)
window_size = 3
positions = torch.arange(seq_len)
# |q_pos - k_pos| <= window_size
sliding_mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= window_size

print(f"  Sliding window mask (window={window_size}):")
print(f"    Shape: {list(sliding_mask.shape)}")
print(f"    First 8x8 block:")
for i in range(min(8, seq_len)):
    row = ''.join(['1' if sliding_mask[i, j] else '.' for j in range(min(8, seq_len))])
    print(f"      {row}")

output_window = F.scaled_dot_product_attention(Q, K, V, attn_mask=sliding_mask)
print(f"\n  Sliding window output shape: {list(output_window.shape)}")

# Float mask (additive): 0 = attend normally, -inf = block
# Useful for soft masking (e.g., adding position biases)
float_mask = torch.zeros(seq_len, seq_len)
float_mask[~sliding_mask] = float('-inf')
output_float_mask = F.scaled_dot_product_attention(Q, K, V, attn_mask=float_mask)
print(f"  Float mask matches bool mask: "
      f"{torch.allclose(output_window, output_float_mask, atol=1e-5)}")

# =============================================================================
# 4. Attention with dropout
# =============================================================================

print("\n" + "=" * 60)
print("SDPA WITH DROPOUT")
print("=" * 60 + "\n")

# Dropout is only applied during training
model_in_training = True

if model_in_training:
    # dropout_p > 0 randomly zeros attention weights
    output_drop = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.1)
    output_nodrop = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
    print(f"  With dropout=0.1: output differs from no-dropout")
    print(f"  Max diff: {(output_drop - output_nodrop).abs().max().item():.4f}")
    print(f"  (Dropout adds noise during training for regularization)")

# =============================================================================
# 5. Custom scale factor
# =============================================================================

print("\n" + "=" * 60)
print("CUSTOM SCALE FACTOR")
print("=" * 60 + "\n")

# Default scale: 1 / sqrt(d_k)
default_scale = 1.0 / (dim ** 0.5)
print(f"  Default scale (1/sqrt({dim})): {default_scale:.4f}")

# You can override it
output_default = F.scaled_dot_product_attention(Q, K, V)
output_custom = F.scaled_dot_product_attention(Q, K, V, scale=1.0 / (dim ** 0.5))
print(f"  Default and explicit scale match: "
      f"{torch.allclose(output_default, output_custom, atol=1e-6)}")

# Different scale changes attention sharpness
output_sharp = F.scaled_dot_product_attention(Q, K, V, scale=2.0 / (dim ** 0.5))
output_soft = F.scaled_dot_product_attention(Q, K, V, scale=0.5 / (dim ** 0.5))
print(f"  Sharper attention (2x scale): output range = "
      f"[{output_sharp.min():.3f}, {output_sharp.max():.3f}]")
print(f"  Softer attention (0.5x scale): output range = "
      f"[{output_soft.min():.3f}, {output_soft.max():.3f}]")

# =============================================================================
# 6. Backend selection with sdpa_kernel
# =============================================================================

print("\n" + "=" * 60)
print("SDPA BACKEND SELECTION")
print("=" * 60 + "\n")

print("  Available backends:")
print(f"    - SDPBackend.MATH: Always available, O(N^2) memory")
print(f"    - SDPBackend.FLASH_ATTENTION: GPU only, O(N) memory")
print(f"    - SDPBackend.EFFICIENT_ATTENTION: GPU only, O(N) memory")
print(f"    - SDPBackend.CUDNN_ATTENTION: GPU + cuDNN 8.9+")

# Force MATH backend (works on CPU, useful for debugging)
print("\n  Using MATH backend (CPU compatible):")
with sdpa_kernel(SDPBackend.MATH):
    output_math = F.scaled_dot_product_attention(Q, K, V)
print(f"    Output shape: {list(output_math.shape)}")

# On CPU, only MATH backend is typically available
# On GPU, Flash Attention and Memory-Efficient would also be options
print("\n  On CPU: only MATH backend runs")
print("  On GPU: Flash Attention or Memory-Efficient would be selected automatically")

# =============================================================================
# 7. Comparing backends (behavior should be identical)
# =============================================================================

print("\n" + "=" * 60)
print("BACKEND COMPARISON (Numerical Equivalence)")
print("=" * 60 + "\n")

# All backends should produce the same output (within floating point tolerance)
with sdpa_kernel(SDPBackend.MATH):
    out_math = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Compare with default (which may select a different backend on GPU)
out_default = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

print(f"  MATH vs Default backend:")
print(f"    Max absolute difference: {(out_math - out_default).abs().max().item():.2e}")
print(f"    Match: {torch.allclose(out_math, out_default, atol=1e-5)}")

# =============================================================================
# 8. Performance comparison (CPU)
# =============================================================================

print("\n" + "=" * 60)
print("PERFORMANCE BENCHMARK (CPU)")
print("=" * 60 + "\n")

# Larger tensors for meaningful timing
batch, heads, seq_len, dim = 4, 8, 128, 64
Q_big = torch.randn(batch, heads, seq_len, dim)
K_big = torch.randn(batch, heads, seq_len, dim)
V_big = torch.randn(batch, heads, seq_len, dim)

# Warmup
for _ in range(5):
    F.scaled_dot_product_attention(Q_big, K_big, V_big)

# Benchmark
runs = 50
start = time.time()
for _ in range(runs):
    F.scaled_dot_product_attention(Q_big, K_big, V_big)
sdpa_time = (time.time() - start) / runs * 1000

# Compare with manual implementation
import math

def manual_attention(q, k, v):
    scale = math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

for _ in range(5):
    manual_attention(Q_big, K_big, V_big)

start = time.time()
for _ in range(runs):
    manual_attention(Q_big, K_big, V_big)
manual_time = (time.time() - start) / runs * 1000

print(f"  Shape: [{batch}, {heads}, {seq_len}, {dim}]")
print(f"  SDPA time:   {sdpa_time:.2f} ms")
print(f"  Manual time: {manual_time:.2f} ms")
print(f"  Ratio: {manual_time/sdpa_time:.2f}x")
print(f"  (On GPU, SDPA is dramatically faster due to Flash Attention)")

# =============================================================================
# 9. Cross-attention (Q and KV have different sequence lengths)
# =============================================================================

print("\n" + "=" * 60)
print("CROSS-ATTENTION (Different Q and KV lengths)")
print("=" * 60 + "\n")

# Example: decoder attends to encoder output
# Decoder query: shorter sequence
# Encoder key/value: longer sequence
batch, heads = 2, 4
seq_q = 16   # Decoder length (generated so far)
seq_kv = 64  # Encoder length (full input)
dim = 32

Q_cross = torch.randn(batch, heads, seq_q, dim)
K_cross = torch.randn(batch, heads, seq_kv, dim)
V_cross = torch.randn(batch, heads, seq_kv, dim)

output_cross = F.scaled_dot_product_attention(Q_cross, K_cross, V_cross)
print(f"  Query shape (decoder):    {list(Q_cross.shape)}")
print(f"  Key shape (encoder):      {list(K_cross.shape)}")
print(f"  Value shape (encoder):    {list(V_cross.shape)}")
print(f"  Output shape:             {list(output_cross.shape)}")
print(f"  (Output matches query sequence length: {seq_q})")

# =============================================================================
# 10. GQA — Grouped Query Attention (fewer KV heads)
# =============================================================================

print("\n" + "=" * 60)
print("GROUPED QUERY ATTENTION (GQA)")
print("=" * 60 + "\n")

# In GQA, multiple query heads share the same KV heads
# This saves memory (smaller KV cache) at minimal quality loss
num_q_heads = 8
num_kv_heads = 2  # Each KV head serves 4 query heads
head_dim = 32

Q_gqa = torch.randn(batch, num_q_heads, seq_len, head_dim)
K_gqa = torch.randn(batch, num_kv_heads, seq_len, head_dim)
V_gqa = torch.randn(batch, num_kv_heads, seq_len, head_dim)

# Expand KV heads to match Q heads (repeat interleave)
num_groups = num_q_heads // num_kv_heads  # 4 groups
K_expanded = K_gqa.repeat_interleave(num_groups, dim=1)  # [batch, 8, seq, dim]
V_expanded = V_gqa.repeat_interleave(num_groups, dim=1)  # [batch, 8, seq, dim]

output_gqa = F.scaled_dot_product_attention(Q_gqa, K_expanded, V_expanded)

print(f"  Q heads: {num_q_heads}, KV heads: {num_kv_heads}, Groups: {num_groups}")
print(f"  Q shape: {list(Q_gqa.shape)}")
print(f"  K shape (original): {list(K_gqa.shape)}")
print(f"  K shape (expanded): {list(K_expanded.shape)}")
print(f"  Output shape: {list(output_gqa.shape)}")
print(f"\n  Memory saving: KV cache is {num_kv_heads}/{num_q_heads} = "
      f"{num_kv_heads/num_q_heads:.0%} of full MHA")

print("\nSDPA and backends demonstration complete!")
