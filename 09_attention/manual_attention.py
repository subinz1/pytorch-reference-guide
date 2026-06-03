"""
Manual Attention — Implementing Attention from Scratch
=======================================================
Builds scaled dot-product attention step-by-step with shape annotations.
Covers: basic attention, causal masking, and padding masks.

Run: python manual_attention.py
"""

import torch
import torch.nn.functional as F
import math

# =============================================================================
# 1. Scaled Dot-Product Attention — Step by Step
# =============================================================================

print("=" * 60)
print("SCALED DOT-PRODUCT ATTENTION FROM SCRATCH")
print("=" * 60)

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention manually.

    Args:
        query: [batch, seq_q, d_k]
        key:   [batch, seq_k, d_k]
        value: [batch, seq_k, d_v]
        mask:  [batch, seq_q, seq_k] or broadcastable, True = attend

    Returns:
        output: [batch, seq_q, d_v]
        weights: [batch, seq_q, seq_k]
    """
    d_k = query.shape[-1]

    # Step 1: Compute raw attention scores
    # query @ key^T: [batch, seq_q, d_k] @ [batch, d_k, seq_k] = [batch, seq_q, seq_k]
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k) to prevent softmax saturation
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    # Step 4: Softmax to get attention weights (sum to 1 per query)
    weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    # weights @ value: [batch, seq_q, seq_k] @ [batch, seq_k, d_v] = [batch, seq_q, d_v]
    output = torch.matmul(weights, value)

    return output, weights


# Demonstrate with concrete shapes
print("\n--- Basic Attention (no mask) ---\n")

batch_size = 2
seq_len = 4
d_k = 8
d_v = 8

torch.manual_seed(42)
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"  Query shape:   {list(Q.shape)}  [batch, seq_q, d_k]")
print(f"  Key shape:     {list(K.shape)}  [batch, seq_k, d_k]")
print(f"  Value shape:   {list(V.shape)}  [batch, seq_k, d_v]")
print(f"  Output shape:  {list(output.shape)}  [batch, seq_q, d_v]")
print(f"  Weights shape: {list(weights.shape)}  [batch, seq_q, seq_k]")

# Verify weights sum to 1 along key dimension
print(f"\n  Attention weights (first batch, first query):")
print(f"  {weights[0, 0].tolist()}")
print(f"  Sum: {weights[0, 0].sum().item():.6f} (should be 1.0)")

# =============================================================================
# 2. Causal (Autoregressive) Attention
# =============================================================================

print("\n" + "=" * 60)
print("CAUSAL (AUTOREGRESSIVE) ATTENTION")
print("=" * 60)
print("\nEach position can only attend to itself and previous positions.\n")

def create_causal_mask(seq_len):
    """Create lower-triangular causal mask."""
    # True = can attend, False = blocked
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

causal_mask = create_causal_mask(seq_len)
print(f"  Causal mask (seq_len={seq_len}):")
for i in range(seq_len):
    row = ['1' if causal_mask[i, j] else '0' for j in range(seq_len)]
    print(f"    Position {i} attends to: [{', '.join(row)}]")

# Apply causal attention
output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

print(f"\n  Causal attention weights (first batch):")
for i in range(seq_len):
    w = [f"{weights_causal[0, i, j]:.3f}" for j in range(seq_len)]
    print(f"    Query {i}: [{', '.join(w)}]")
print("  (Notice: weights are 0 for future positions)")

# =============================================================================
# 3. Padding Mask
# =============================================================================

print("\n" + "=" * 60)
print("PADDING MASK")
print("=" * 60)
print("\nIgnore padding tokens in attention.\n")

# Suppose we have sequences of different lengths, padded to seq_len=6
seq_len_padded = 6
actual_lengths = [4, 6]  # First sequence has 4 real tokens, second has 6
batch_size = 2

Q_pad = torch.randn(batch_size, seq_len_padded, d_k)
K_pad = torch.randn(batch_size, seq_len_padded, d_k)
V_pad = torch.randn(batch_size, seq_len_padded, d_v)

# Create padding mask: True for real tokens, False for padding
padding_mask = torch.zeros(batch_size, seq_len_padded, dtype=torch.bool)
for b, length in enumerate(actual_lengths):
    padding_mask[b, :length] = True

print(f"  Padding mask:")
print(f"    Batch 0 (len=4): {padding_mask[0].int().tolist()}")
print(f"    Batch 1 (len=6): {padding_mask[1].int().tolist()}")

# For attention: query at position i can attend to key at position j
# only if position j is NOT padding
# Shape needs to be [batch, seq_q, seq_k]
attn_mask = padding_mask.unsqueeze(1).expand(-1, seq_len_padded, -1)

output_pad, weights_pad = scaled_dot_product_attention(Q_pad, K_pad, V_pad, mask=attn_mask)

print(f"\n  Attention weights for batch 0 (padded), query position 0:")
print(f"    {[f'{w:.3f}' for w in weights_pad[0, 0].tolist()]}")
print(f"    (Positions 4,5 are padding — weight is 0.000)")

# =============================================================================
# 4. Combined: Causal + Padding
# =============================================================================

print("\n" + "=" * 60)
print("COMBINED CAUSAL + PADDING MASK")
print("=" * 60 + "\n")

def create_combined_mask(seq_len, padding_mask):
    """Combine causal mask with padding mask."""
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    # Expand: padding_mask is [batch, seq] -> [batch, 1, seq] for broadcasting
    pad = padding_mask.unsqueeze(1)  # [batch, 1, seq_k]
    # Combined: must satisfy BOTH causal AND padding
    combined = causal.unsqueeze(0) & pad  # [batch, seq_q, seq_k]
    return combined

combined_mask = create_combined_mask(seq_len_padded, padding_mask)

output_combined, weights_combined = scaled_dot_product_attention(
    Q_pad, K_pad, V_pad, mask=combined_mask
)

print(f"  Combined mask for batch 0 (causal + padding at len=4):")
for i in range(seq_len_padded):
    row = [f"{int(combined_mask[0, i, j])}" for j in range(seq_len_padded)]
    print(f"    Query {i}: [{', '.join(row)}]")

# =============================================================================
# 5. Attention with dropout (training)
# =============================================================================

print("\n" + "=" * 60)
print("ATTENTION WITH DROPOUT")
print("=" * 60 + "\n")

def attention_with_dropout(query, key, value, mask=None, dropout_p=0.1, training=True):
    """Attention with dropout on weights (for regularization during training)."""
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    # Apply dropout to attention weights (randomly zero some connections)
    if training and dropout_p > 0:
        weights = F.dropout(weights, p=dropout_p, training=True)

    output = torch.matmul(weights, value)
    return output, weights

# Compare: with and without dropout
torch.manual_seed(42)
out_nodrop, w_nodrop = attention_with_dropout(Q, K, V, dropout_p=0.0)
out_drop, w_drop = attention_with_dropout(Q, K, V, dropout_p=0.3)

print(f"  Without dropout — weights (query 0):")
print(f"    {[f'{w:.3f}' for w in w_nodrop[0, 0].tolist()]}")
print(f"  With dropout=0.3 — weights (query 0):")
print(f"    {[f'{w:.3f}' for w in w_drop[0, 0].tolist()]}")
print(f"  (Some weights randomly zeroed, others scaled up)")

# =============================================================================
# 6. Verifying against PyTorch's SDPA
# =============================================================================

print("\n" + "=" * 60)
print("VERIFYING AGAINST F.scaled_dot_product_attention")
print("=" * 60 + "\n")

torch.manual_seed(42)
Q = torch.randn(2, 4, 16)
K = torch.randn(2, 4, 16)
V = torch.randn(2, 4, 16)

# Our implementation
our_output, _ = scaled_dot_product_attention(Q, K, V)

# PyTorch's SDPA (expects [batch, heads, seq, dim] but works with [batch, seq, dim] via unsqueeze)
# Actually SDPA expects 4D, so let's add a head dimension
Q4d = Q.unsqueeze(1)  # [2, 1, 4, 16]
K4d = K.unsqueeze(1)
V4d = V.unsqueeze(1)
pytorch_output = F.scaled_dot_product_attention(Q4d, K4d, V4d).squeeze(1)

print(f"  Our output shape: {list(our_output.shape)}")
print(f"  PyTorch SDPA shape: {list(pytorch_output.shape)}")
print(f"  Max difference: {(our_output - pytorch_output).abs().max().item():.2e}")
print(f"  Outputs match: {torch.allclose(our_output, pytorch_output, atol=1e-6)}")

# Causal comparison
our_causal, _ = scaled_dot_product_attention(Q, K, V, mask=create_causal_mask(4))
pytorch_causal = F.scaled_dot_product_attention(Q4d, K4d, V4d, is_causal=True).squeeze(1)

print(f"\n  Causal - max difference: {(our_causal - pytorch_causal).abs().max().item():.2e}")
print(f"  Causal outputs match: {torch.allclose(our_causal, pytorch_causal, atol=1e-6)}")

# =============================================================================
# 7. Visualizing attention patterns
# =============================================================================

print("\n" + "=" * 60)
print("ATTENTION PATTERN VISUALIZATION (ASCII)")
print("=" * 60 + "\n")

# Create a clear example: tokens that are similar should attend to each other
tokens = torch.tensor([
    [1.0, 0.0],  # Token A (category 1)
    [0.9, 0.1],  # Token B (similar to A)
    [0.0, 1.0],  # Token C (category 2)
    [0.1, 0.9],  # Token D (similar to C)
])

# Use tokens as Q, K, V (self-attention)
_, attention = scaled_dot_product_attention(
    tokens.unsqueeze(0), tokens.unsqueeze(0), tokens.unsqueeze(0)
)

print("  Self-attention between 4 tokens (A≈B, C≈D):")
print("  " + "".join(f"    {c}   " for c in "ABCD"))
for i, name in enumerate("ABCD"):
    row = ""
    for j in range(4):
        w = attention[0, i, j].item()
        bar = "#" * int(w * 10)
        row += f"  {w:.2f} "
    print(f"  {name}: {row}")
print("\n  (A attends mostly to B (similar), C attends mostly to D)")

print("\nManual attention implementation complete!")
