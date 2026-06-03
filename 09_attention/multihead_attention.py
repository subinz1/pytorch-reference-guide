"""
Multi-Head Attention — Complete Implementation
================================================
Implements Multi-Head Attention from scratch with detailed shape tracking,
then compares against PyTorch's nn.MultiheadAttention.

Run: python multihead_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. Multi-Head Attention from scratch
# =============================================================================

print("=" * 60)
print("MULTI-HEAD ATTENTION FROM SCRATCH")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.

    The key insight: instead of one attention with d_model dimensions,
    we use h parallel attention heads each with d_model/h dimensions.
    This lets different heads attend to different things.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V (one big matrix for all heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection (combines all heads back to d_model)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, is_causal=False):
        """
        Args:
            query: [batch, seq_q, d_model]
            key:   [batch, seq_k, d_model]
            value: [batch, seq_k, d_model]
            mask:  [batch, seq_q, seq_k] or [seq_q, seq_k] (optional)
            is_causal: If True, apply causal mask

        Returns:
            output: [batch, seq_q, d_model]
            attn_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # Step 1: Project inputs to Q, K, V
        # Each: [batch, seq, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Reshape to [batch, num_heads, seq, head_dim]
        # From [batch, seq, d_model] to [batch, seq, num_heads, head_dim]
        # Then transpose to [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention scores
        # Q @ K^T: [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
        #        = [batch, heads, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 4: Apply mask
        if is_causal:
            causal_mask = torch.tril(torch.ones(seq_q, seq_k, device=query.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float('-inf'))
        elif mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_q, seq_k]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_q, seq_k]
            scores = scores.masked_fill(~mask, float('-inf'))

        # Step 5: Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 6: Weighted sum of values
        # weights @ V: [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
        #            = [batch, heads, seq_q, head_dim]
        context = torch.matmul(attn_weights, V)

        # Step 7: Concatenate heads
        # From [batch, heads, seq_q, head_dim] to [batch, seq_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)

        # Step 8: Final output projection
        output = self.W_o(context)

        return output, attn_weights


# =============================================================================
# 2. Demonstrate with shapes
# =============================================================================

print("\n--- Shape walkthrough ---\n")

d_model = 64
num_heads = 4
head_dim = d_model // num_heads  # 16
batch = 2
seq_len = 8

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch, seq_len, d_model)

output, weights = mha(x, x, x)  # Self-attention

print(f"  Configuration: d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}")
print(f"  Input:          {list(x.shape)} [batch, seq, d_model]")
print(f"  Output:         {list(output.shape)} [batch, seq, d_model]")
print(f"  Attn weights:   {list(weights.shape)} [batch, heads, seq_q, seq_k]")

# Verify each head has valid attention (sums to 1)
for h in range(num_heads):
    row_sums = weights[0, h, 0, :].sum().item()
    print(f"  Head {h} attention sum (first query): {row_sums:.6f}")

# =============================================================================
# 3. Causal self-attention
# =============================================================================

print("\n" + "=" * 60)
print("CAUSAL SELF-ATTENTION")
print("=" * 60 + "\n")

output_causal, weights_causal = mha(x, x, x, is_causal=True)

print(f"  Causal attention weights (head 0, batch 0):")
print(f"  (Each row shows what that query position attends to)")
w = weights_causal[0, 0]
for i in range(min(6, seq_len)):
    row = [f"{w[i, j].item():.2f}" for j in range(min(6, seq_len))]
    print(f"    Position {i}: [{', '.join(row)}, ...]")
print("  (Upper triangle is 0 — no attending to future)")

# =============================================================================
# 4. Cross-attention (query from one source, key/value from another)
# =============================================================================

print("\n" + "=" * 60)
print("CROSS-ATTENTION")
print("=" * 60 + "\n")

# Decoder queries attend to encoder key/values
encoder_output = torch.randn(batch, 20, d_model)  # Encoder: 20 tokens
decoder_input = torch.randn(batch, 5, d_model)    # Decoder: 5 tokens

# Query from decoder, Key and Value from encoder
cross_output, cross_weights = mha(
    query=decoder_input,
    key=encoder_output,
    value=encoder_output,
)

print(f"  Encoder output: {list(encoder_output.shape)}")
print(f"  Decoder input:  {list(decoder_input.shape)}")
print(f"  Cross-attention output: {list(cross_output.shape)}")
print(f"  Cross-attention weights: {list(cross_weights.shape)}")
print(f"  (5 decoder queries, each attending over 20 encoder positions)")

# =============================================================================
# 5. Comparing with PyTorch's nn.MultiheadAttention
# =============================================================================

print("\n" + "=" * 60)
print("COMPARING WITH nn.MultiheadAttention")
print("=" * 60 + "\n")

# PyTorch's implementation
pytorch_mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    dropout=0.0,
    batch_first=True,  # Important: use [batch, seq, dim] format
)

x = torch.randn(batch, seq_len, d_model)

# PyTorch's MHA
pytorch_out, pytorch_weights = pytorch_mha(x, x, x)

print(f"  nn.MultiheadAttention output:  {list(pytorch_out.shape)}")
print(f"  nn.MultiheadAttention weights: {list(pytorch_weights.shape)}")
print(f"  (Note: PyTorch averages weights over heads by default)")

# With causal mask
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
pytorch_causal_out, _ = pytorch_mha(x, x, x, attn_mask=causal_mask)
print(f"  Causal output shape: {list(pytorch_causal_out.shape)}")

# =============================================================================
# 6. Head analysis — what different heads learn
# =============================================================================

print("\n" + "=" * 60)
print("ANALYZING ATTENTION HEADS")
print("=" * 60 + "\n")

# Create input where we know what patterns exist
seq_len = 16
x_patterned = torch.zeros(1, seq_len, d_model)

# Embed some structure: first half and second half are related
x_patterned[0, :8, :32] = torch.randn(8, 32)  # First half uses dims 0-31
x_patterned[0, 8:, 32:] = torch.randn(8, 32)  # Second half uses dims 32-63

mha_analyze = MultiHeadAttention(d_model, num_heads)
_, head_weights = mha_analyze(x_patterned, x_patterned, x_patterned)

print(f"  Input: {seq_len} tokens, first 8 use one feature set, last 8 another")
print(f"\n  Per-head attention patterns:")
for h in range(num_heads):
    # Measure: does this head prefer within-group or cross-group attention?
    within_first = head_weights[0, h, :8, :8].mean().item()
    within_second = head_weights[0, h, 8:, 8:].mean().item()
    cross_group = head_weights[0, h, :8, 8:].mean().item()
    print(f"    Head {h}: within-first={within_first:.3f}, "
          f"within-second={within_second:.3f}, cross={cross_group:.3f}")

# =============================================================================
# 7. Efficient implementation using F.scaled_dot_product_attention
# =============================================================================

print("\n" + "=" * 60)
print("EFFICIENT MHA USING F.scaled_dot_product_attention")
print("=" * 60 + "\n")

class EfficientMHA(nn.Module):
    """MHA that uses PyTorch's fused SDPA for the attention computation."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # Fused QKV projection (single matrix multiply instead of 3)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, is_causal=False):
        batch, seq, _ = x.shape

        # Single projection for Q, K, V together
        qkv = self.qkv(x)  # [batch, seq, 3 * d_model]
        qkv = qkv.view(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv.unbind(0)

        # Use PyTorch's optimized SDPA
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq, self.d_model)
        return self.out_proj(attn_out)


efficient_mha = EfficientMHA(d_model=64, num_heads=4)
x = torch.randn(2, 32, 64)
out = efficient_mha(x, is_causal=True)
print(f"  Input: {list(x.shape)}")
print(f"  Output: {list(out.shape)}")

# Benchmark comparison
import time

x_bench = torch.randn(4, 128, 64)
mha_slow = MultiHeadAttention(64, 4)
mha_fast = EfficientMHA(64, 4)

# Warmup
for _ in range(10):
    mha_slow(x_bench, x_bench, x_bench, is_causal=True)
    mha_fast(x_bench, is_causal=True)

runs = 100
start = time.time()
with torch.no_grad():
    for _ in range(runs):
        mha_slow(x_bench, x_bench, x_bench, is_causal=True)
slow_time = (time.time() - start) / runs * 1000

start = time.time()
with torch.no_grad():
    for _ in range(runs):
        mha_fast(x_bench, is_causal=True)
fast_time = (time.time() - start) / runs * 1000

print(f"\n  Benchmark (batch=4, seq=128, dim=64):")
print(f"    Manual attention: {slow_time:.2f} ms")
print(f"    SDPA-based:       {fast_time:.2f} ms")
print(f"    Speedup: {slow_time/fast_time:.2f}x")

print("\nMulti-head attention implementation complete!")
