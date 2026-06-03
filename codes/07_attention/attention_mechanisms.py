"""
Attention Mechanisms — SDPA, Multi-Head Attention, and FlexAttention
=====================================================================
Covers: scaled dot-product attention, multi-head attention, FlexAttention API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("1. SCALED DOT-PRODUCT ATTENTION (Manual)")
print("=" * 60)

def manual_attention(Q, K, V, mask=None):
    """Standard attention: softmax(QK^T / sqrt(d_k)) V"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V, attn_weights

B, H, N, D = 2, 4, 16, 64
Q = torch.randn(B, H, N, D)
K = torch.randn(B, H, N, D)
V = torch.randn(B, H, N, D)

output, weights = manual_attention(Q, K, V)
print(f"Q,K,V shape: ({B}, {H}, {N}, {D})")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

print("\n" + "=" * 60)
print("2. PyTorch SDPA (F.scaled_dot_product_attention)")
print("=" * 60)

# Automatically selects best backend (Flash, Efficient, cuDNN, Math)
output_sdpa = F.scaled_dot_product_attention(Q, K, V)
print(f"SDPA output shape: {output_sdpa.shape}")

# Causal attention (autoregressive)
output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(f"Causal SDPA output shape: {output_causal.shape}")

# Verify causal matches manual with mask
causal_mask = torch.tril(torch.ones(N, N)).unsqueeze(0).unsqueeze(0)
output_manual_causal, _ = manual_attention(Q, K, V, mask=causal_mask)
print(f"Manual causal matches SDPA: {torch.allclose(output_causal, output_manual_causal, atol=1e-5)}")

print("\n" + "=" * 60)
print("3. SDPA BACKEND SELECTION")
print("=" * 60)

from torch.nn.attention import sdpa_kernel, SDPBackend

# List available backends
backends = [b.name for b in SDPBackend]
print(f"Available backends: {backends}")

# Force math backend (always available)
with sdpa_kernel(SDPBackend.MATH):
    output_math = F.scaled_dot_product_attention(Q, K, V)
    print(f"Math backend output: {output_math.shape}")

print("\n" + "=" * 60)
print("4. MULTI-HEAD ATTENTION MODULE")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, is_causal=False):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_k)
        Q, K, V = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 32, 512)
output = mha(x, is_causal=True)
print(f"Input:  {x.shape}")
print(f"Output: {output.shape}")

print("\n" + "=" * 60)
print("5. BUILT-IN nn.MultiheadAttention")
print("=" * 60)

mha_builtin = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(2, 32, 512)
attn_output, attn_weights = mha_builtin(x, x, x)
print(f"Built-in MHA output: {attn_output.shape}")
print(f"Attention weights:   {attn_weights.shape}")

print("\n" + "=" * 60)
print("6. TRANSFORMER BLOCK")
print("=" * 60)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, is_causal=False):
        x = x + self.attn(self.norm1(x), is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x

block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(2, 32, 512)
output = block(x, is_causal=True)
print(f"Transformer block: {x.shape} -> {output.shape}")

total_params = sum(p.numel() for p in block.parameters())
print(f"Block parameters: {total_params:,}")

print("\n" + "=" * 60)
print("7. FlexAttention OVERVIEW")
print("=" * 60)

print("""
FlexAttention (torch.nn.attention.flex_attention) allows custom
attention patterns through score_mod and mask_mod functions:

  from torch.nn.attention.flex_attention import (
      flex_attention, create_block_mask
  )

  # Causal mask
  def causal(b, h, q_idx, kv_idx):
      return q_idx >= kv_idx

  block_mask = create_block_mask(causal, B=2, H=8, Q_LEN=1024, KV_LEN=1024)
  output = flex_attention(Q, K, V, block_mask=block_mask)

  # Score modification (e.g., ALiBi)
  def alibi(score, b, h, q_idx, kv_idx):
      return score - abs(q_idx - kv_idx) * slope

  output = flex_attention(Q, K, V, score_mod=alibi, block_mask=block_mask)

NOTE: FlexAttention requires torch.compile and GPU (Triton kernels).
      See the reference guide for full API documentation.
""")

print("Done!")
