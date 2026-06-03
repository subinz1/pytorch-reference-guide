"""
Transformer Block — Full Implementation
=========================================
Builds a complete transformer block with:
- RMSNorm / LayerNorm
- Multi-head self-attention
- Feed-forward network (SwiGLU)
- Positional encoding (sinusoidal and RoPE)
- KV cache for inference

Run: python transformer_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. RMSNorm (used in modern LLMs like LLaMA)
# =============================================================================

print("=" * 60)
print("TRANSFORMER BLOCK COMPONENTS")
print("=" * 60)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler than LayerNorm: no mean subtraction, no bias.
    Used in LLaMA, Mistral, and most modern LLMs.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


print("\n--- RMSNorm vs LayerNorm ---")
dim = 64
x = torch.randn(2, 8, dim)

rms_norm = RMSNorm(dim)
layer_norm = nn.LayerNorm(dim)

rms_out = rms_norm(x)
ln_out = layer_norm(x)

print(f"  Input shape: {list(x.shape)}")
print(f"  RMSNorm output mean: {rms_out.mean():.4f}, std: {rms_out.std():.4f}")
print(f"  LayerNorm output mean: {ln_out.mean():.4f}, std: {ln_out.std():.4f}")
print(f"  RMSNorm has {sum(p.numel() for p in rms_norm.parameters())} params (no bias)")
print(f"  LayerNorm has {sum(p.numel() for p in layer_norm.parameters())} params")

# =============================================================================
# 2. SwiGLU Feed-Forward Network
# =============================================================================

print("\n--- SwiGLU Feed-Forward Network ---")

class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit.

    Used in LLaMA, PaLM. More expressive than simple ReLU/GELU FFN.
    SwiGLU(x) = (W1 @ x * SiLU(W_gate @ x)) @ W2
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to multiple of 8 for efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)      # Up projection
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)   # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)       # Down projection

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


ffn = SwiGLU(dim=64)
ffn_out = ffn(x)
print(f"  SwiGLU input:  {list(x.shape)}")
print(f"  SwiGLU output: {list(ffn_out.shape)}")
print(f"  SwiGLU parameters: {sum(p.numel() for p in ffn.parameters()):,}")

# =============================================================================
# 3. Rotary Position Embeddings (RoPE)
# =============================================================================

print("\n--- Rotary Position Embeddings (RoPE) ---")

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings.

    Encodes position by rotating Q and K vectors, so their dot product
    naturally depends on relative position.
    """

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        # Compute frequencies for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for all positions
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # [max_seq, dim//2]
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, x, seq_len):
        """Returns cos and sin for positions [0, seq_len)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to a tensor.

    x: [batch, heads, seq, dim]
    cos, sin: [seq, dim//2]
    """
    # Split into pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]  # Even and odd dimensions

    # Reshape cos/sin for broadcasting: [1, 1, seq, dim//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    return torch.stack([out1, out2], dim=-1).flatten(-2)


rope = RotaryEmbedding(dim=32, max_seq_len=512)
q = torch.randn(2, 4, 16, 32)  # [batch, heads, seq, dim]
k = torch.randn(2, 4, 16, 32)

cos, sin = rope(q, seq_len=16)
q_rot = apply_rotary_emb(q, cos, sin)
k_rot = apply_rotary_emb(k, cos, sin)

print(f"  RoPE dim: 32, max_seq: 512")
print(f"  Q before RoPE: shape={list(q.shape)}, norm={q.norm():.2f}")
print(f"  Q after RoPE:  shape={list(q_rot.shape)}, norm={q_rot.norm():.2f}")
print(f"  (RoPE preserves norm — it's a rotation)")

# Verify: relative position property
# dot(q_rot[pos_i], k_rot[pos_j]) depends only on (i - j)
dot_01 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum().item()
dot_12 = (q_rot[0, 0, 1] * k_rot[0, 0, 2]).sum().item()
print(f"  Relative position check (different absolute, same relative offset):")
print(f"    dot(q[0], k[1]): {dot_01:.4f}")
print(f"    dot(q[1], k[2]): {dot_12:.4f}")
print(f"    (Would be equal with identical q[0]=q[1], k[1]=k[2])")

# =============================================================================
# 4. Complete Transformer Block
# =============================================================================

print("\n" + "=" * 60)
print("COMPLETE TRANSFORMER BLOCK")
print("=" * 60 + "\n")

class TransformerBlock(nn.Module):
    """Complete transformer block with pre-norm, RoPE, and SwiGLU.

    Architecture follows LLaMA/Mistral style:
    - Pre-norm (norm before attention/FFN)
    - RMSNorm (instead of LayerNorm)
    - RoPE (instead of absolute position embeddings)
    - SwiGLU (instead of ReLU/GELU FFN)
    """

    def __init__(self, dim, num_heads, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Attention projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # FFN
        self.ffn = SwiGLU(dim)

    def forward(self, x, is_causal=True):
        """
        Args:
            x: [batch, seq_len, dim]
            is_causal: apply causal mask

        Returns:
            [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape

        # Pre-norm + Attention
        normed = self.norm1(x)

        # QKV projections
        q = self.wq(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(q, seq_len)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.dim)
        attn_out = self.wo(attn_out)

        # Residual connection
        x = x + attn_out

        # Pre-norm + FFN + Residual
        x = x + self.ffn(self.norm2(x))

        return x


# Test the block
block = TransformerBlock(dim=64, num_heads=4)
x = torch.randn(2, 16, 64)
out = block(x)

print(f"  TransformerBlock(dim=64, heads=4)")
print(f"  Input:  {list(x.shape)}")
print(f"  Output: {list(out.shape)}")
print(f"  Parameters: {sum(p.numel() for p in block.parameters()):,}")

# =============================================================================
# 5. KV Cache for Inference
# =============================================================================

print("\n" + "=" * 60)
print("KV CACHE FOR INFERENCE")
print("=" * 60 + "\n")

class CachedTransformerBlock(nn.Module):
    """Transformer block with KV cache for fast autoregressive generation."""

    def __init__(self, dim, num_heads, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.ffn = SwiGLU(dim)

    def forward(self, x, start_pos, kv_cache=None):
        """
        Args:
            x: [batch, seq_len, dim] — full sequence or just new tokens
            start_pos: position offset (0 for prefill, >0 for generation)
            kv_cache: tuple of (cached_k, cached_v) or None

        Returns:
            output, new_kv_cache
        """
        batch, seq_len, _ = x.shape

        normed = self.norm1(x)

        q = self.wq(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with correct position offset
        cos, sin = self.rope(q, start_pos + seq_len)
        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV Cache: append new KV to existing cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # Extend along seq dimension
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        # Attention (causal only needed during prefill; during generation
        # the query is length 1, so causal is implicit)
        is_causal = seq_len > 1
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.dim)
        attn_out = self.wo(attn_out)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, new_kv_cache


# Simulate autoregressive generation
print("  Simulating autoregressive generation with KV cache:\n")

cached_block = CachedTransformerBlock(dim=64, num_heads=4)
cached_block.eval()

# Phase 1: Prefill (process full prompt at once)
prompt = torch.randn(1, 8, 64)  # 8-token prompt
with torch.no_grad():
    output, kv_cache = cached_block(prompt, start_pos=0, kv_cache=None)

print(f"  Prefill: processed {prompt.shape[1]} tokens")
print(f"    KV cache K shape: {list(kv_cache[0].shape)} [batch, heads, cached_seq, dim]")

# Phase 2: Generate tokens one at a time
generated_tokens = []
current_pos = 8

for step in range(4):
    # In practice, this would be the embedding of the last predicted token
    new_token = torch.randn(1, 1, 64)

    with torch.no_grad():
        output, kv_cache = cached_block(new_token, start_pos=current_pos, kv_cache=kv_cache)

    current_pos += 1
    generated_tokens.append(output)

    print(f"  Generate step {step+1}: KV cache size = {kv_cache[0].shape[2]} "
          f"(Q attends to {kv_cache[0].shape[2]} positions)")

print(f"\n  Total generated: {len(generated_tokens)} tokens")
print(f"  Final KV cache size: {kv_cache[0].shape[2]} = 8 (prompt) + 4 (generated)")

# Memory comparison
print(f"\n  Memory comparison (single layer):")
kv_memory = kv_cache[0].numel() + kv_cache[1].numel()
print(f"    KV cache memory: {kv_memory * 4 / 1024:.1f} KB")
print(f"    Without cache: would recompute all KV at every step")

# =============================================================================
# 6. Sinusoidal Positional Encoding (original Transformer)
# =============================================================================

print("\n" + "=" * 60)
print("SINUSOIDAL POSITIONAL ENCODING")
print("=" * 60 + "\n")

class SinusoidalPositionalEncoding(nn.Module):
    """Original Transformer positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq, d_model]

    def forward(self, x):
        """Add positional encoding to input. x: [batch, seq, dim]"""
        return x + self.pe[:, :x.shape[1]]


pos_enc = SinusoidalPositionalEncoding(d_model=64, max_seq_len=100)
x = torch.randn(2, 20, 64)
x_with_pos = pos_enc(x)

print(f"  Input: {list(x.shape)}")
print(f"  Output: {list(x_with_pos.shape)} (position added)")
print(f"  PE magnitude (pos 0): {pos_enc.pe[0, 0].norm():.4f}")
print(f"  PE magnitude (pos 10): {pos_enc.pe[0, 10].norm():.4f}")

# Verify: nearby positions are more similar
cos_sim_01 = F.cosine_similarity(pos_enc.pe[0, 0:1], pos_enc.pe[0, 1:2]).item()
cos_sim_010 = F.cosine_similarity(pos_enc.pe[0, 0:1], pos_enc.pe[0, 10:11]).item()
cos_sim_050 = F.cosine_similarity(pos_enc.pe[0, 0:1], pos_enc.pe[0, 50:51]).item()
print(f"\n  Cosine similarity (closer = more similar):")
print(f"    pos 0 vs pos 1:  {cos_sim_01:.4f}")
print(f"    pos 0 vs pos 10: {cos_sim_010:.4f}")
print(f"    pos 0 vs pos 50: {cos_sim_050:.4f}")
print(f"    (Nearby positions are more similar — encodes locality)")

print("\nTransformer block implementation complete!")
