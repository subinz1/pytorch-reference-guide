"""
FlexAttention Patterns — Custom Attention with Compiled Kernels
================================================================
Demonstrates the FlexAttention API and common attention patterns.

NOTE: FlexAttention requires torch.compile and typically a GPU for full
performance. This file demonstrates the API patterns and runs what it can
on CPU. Some examples are shown as documentation even if they require GPU.

Run: python flex_attention_patterns.py
"""

import torch
import torch.nn.functional as F
import math

# =============================================================================
# 1. What is FlexAttention?
# =============================================================================

print("=" * 60)
print("FLEXATTENTION — Custom Attention Patterns")
print("=" * 60)
print("""
FlexAttention lets you define custom attention patterns using simple
Python functions (score_mod / mask_mod), which PyTorch then compiles
into efficient fused kernels.

Key concepts:
  - score_mod(score, b, h, q_idx, kv_idx): modifies attention scores
  - mask_mod(b, h, q_idx, kv_idx): returns True/False for valid positions
  - BlockMask: sparse mask that skips computation for blocked regions
  - create_block_mask: builds a BlockMask from a mask_mod function
""")

# =============================================================================
# 2. Try to import FlexAttention (available in PyTorch 2.5+)
# =============================================================================

try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
    )
    FLEX_AVAILABLE = True
    print("  FlexAttention is available!")
except ImportError:
    FLEX_AVAILABLE = False
    print("  FlexAttention not available in this PyTorch version.")
    print("  Showing patterns as documentation.\n")

# =============================================================================
# 3. Common mask patterns (pure functions — always demonstrable)
# =============================================================================

print("\n" + "=" * 60)
print("COMMON ATTENTION MASK PATTERNS")
print("=" * 60)

# These are the mask_mod functions you'd pass to FlexAttention
# They take (b, h, q_idx, kv_idx) and return True if attention is allowed

def causal_mask(b, h, q_idx, kv_idx):
    """Standard causal (autoregressive) mask."""
    return q_idx >= kv_idx

def sliding_window_mask(window_size):
    """Sliding window: each position attends to nearby positions only."""
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= window_size
    return mask_fn

def causal_sliding_window(window_size):
    """Causal + sliding window (used in Mistral)."""
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= window_size)
    return mask_fn

def prefix_lm_mask(prefix_length):
    """Bidirectional within prefix, causal after prefix."""
    def mask_fn(b, h, q_idx, kv_idx):
        # If key is in prefix, always attend to it
        # If key is after prefix, only attend if causal
        return (kv_idx < prefix_length) | (q_idx >= kv_idx)
    return mask_fn

def block_diagonal_mask(block_size):
    """Block diagonal: attention only within blocks."""
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx // block_size) == (kv_idx // block_size)
    return mask_fn

# Visualize these masks
print("\n--- Visualizing Mask Patterns (8x8) ---\n")

seq_len = 8

def visualize_mask(mask_fn, name, seq_len=8):
    """Visualize a mask pattern as ASCII art."""
    print(f"  {name}:")
    for q in range(seq_len):
        row = ""
        for kv in range(seq_len):
            # Call with dummy batch/head indices
            if mask_fn(0, 0, torch.tensor(q), torch.tensor(kv)):
                row += "# "
            else:
                row += ". "
        print(f"    {row}")
    print()

visualize_mask(causal_mask, "Causal")
visualize_mask(sliding_window_mask(2), "Sliding Window (size=2)")
visualize_mask(causal_sliding_window(3), "Causal + Sliding Window (size=3)")
visualize_mask(prefix_lm_mask(3), "Prefix LM (prefix=3)")
visualize_mask(block_diagonal_mask(4), "Block Diagonal (block=4)")

# =============================================================================
# 4. Score modification patterns
# =============================================================================

print("=" * 60)
print("SCORE MODIFICATION PATTERNS")
print("=" * 60)
print("""
score_mod functions modify attention scores BEFORE softmax.
They receive: (score, batch, head, q_idx, kv_idx) and return modified score.
""")

# ALiBi (Attention with Linear Biases) — used in BLOOM
def alibi_score_mod(num_heads):
    """ALiBi: adds position-dependent linear bias to attention scores."""
    def score_fn(score, b, h, q_idx, kv_idx):
        # Each head gets a different slope (geometric sequence)
        slope = 2.0 ** (-(h + 1) * 8.0 / num_heads)
        bias = -slope * (q_idx - kv_idx).abs().float()
        return score + bias
    return score_fn

# Relative position bias (like in T5)
def relative_position_bias(max_distance=16):
    """Adds a learnable bias based on relative position."""
    def score_fn(score, b, h, q_idx, kv_idx):
        rel_pos = (q_idx - kv_idx).clamp(-max_distance, max_distance)
        # In practice, this would index into a learned bias table
        # Here we use a simple decay for demonstration
        bias = -0.1 * rel_pos.abs().float()
        return score + bias
    return score_fn

# Soft causal (instead of hard -inf, use a steep slope)
def soft_causal_score_mod(score, b, h, q_idx, kv_idx):
    """Soft causal: strongly discourage (but don't block) future tokens."""
    penalty = torch.where(q_idx < kv_idx, torch.tensor(-100.0), torch.tensor(0.0))
    return score + penalty

# Show ALiBi biases for different heads
print("\n--- ALiBi Bias Values (4 heads, positions 0-7) ---\n")
num_heads = 4
alibi = alibi_score_mod(num_heads)
print(f"  Position offset:  ", end="")
for offset in range(8):
    print(f"{offset:6d}", end="")
print()

for h in range(num_heads):
    print(f"  Head {h} bias:     ", end="")
    for offset in range(8):
        bias = alibi(torch.tensor(0.0), 0, h, torch.tensor(offset), torch.tensor(0))
        print(f"{bias.item():6.2f}", end="")
    print()

print("\n  (Closer positions get smaller penalty, farther get larger)")
print("  (Different heads have different slopes — multi-scale locality)")

# =============================================================================
# 5. Document masking pattern
# =============================================================================

print("\n" + "=" * 60)
print("DOCUMENT MASKING")
print("=" * 60)
print("""
When packing multiple documents into one sequence for efficient training,
we need to prevent attention across document boundaries.
""")

# Example: sequence of 12 tokens from 3 documents
# Doc 0: tokens 0-3, Doc 1: tokens 4-7, Doc 2: tokens 8-11
document_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

def document_mask_fn(doc_ids):
    """Attention only within the same document."""
    def mask_fn(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]
    return mask_fn

print("\n  Document IDs: ", document_ids.tolist())
print("\n  Document mask (12 tokens, 3 documents):")
doc_mask = document_mask_fn(document_ids)
for q in range(12):
    row = ""
    for kv in range(12):
        if doc_mask(0, 0, torch.tensor(q), torch.tensor(kv)):
            row += "# "
        else:
            row += ". "
    if q in [0, 4, 8]:
        print(f"    {row}  <- Doc {document_ids[q].item()} start")
    else:
        print(f"    {row}")

# Combined: causal + document masking
def causal_document_mask(doc_ids):
    """Causal attention within documents only."""
    def mask_fn(b, h, q_idx, kv_idx):
        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
        causal = q_idx >= kv_idx
        return same_doc & causal
    return mask_fn

print("\n  Causal + Document mask:")
causal_doc = causal_document_mask(document_ids)
for q in range(12):
    row = ""
    for kv in range(12):
        if causal_doc(0, 0, torch.tensor(q), torch.tensor(kv)):
            row += "# "
        else:
            row += ". "
    print(f"    {row}")

# =============================================================================
# 6. FlexAttention API usage (if available)
# =============================================================================

print("\n" + "=" * 60)
print("FLEXATTENTION API USAGE")
print("=" * 60 + "\n")

if FLEX_AVAILABLE:
    print("  Running FlexAttention examples...\n")

    batch, heads, seq_len, dim = 1, 4, 16, 32

    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, dim)

    # Create a BlockMask for causal attention
    def simple_causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    try:
        block_mask = create_block_mask(
            simple_causal, B=batch, H=heads, Q_LEN=seq_len, KV_LEN=seq_len
        )
        print(f"  Created causal BlockMask: {block_mask}")

        # Use flex_attention (requires torch.compile for performance)
        # On CPU without compile, falls back to math backend
        compiled_flex = torch.compile(flex_attention)
        output = compiled_flex(q, k, v, block_mask=block_mask)
        print(f"  FlexAttention output shape: {list(output.shape)}")

        # Compare with standard SDPA
        output_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = (output - output_sdpa).abs().max().item()
        print(f"  Max diff vs SDPA causal: {diff:.6f}")
    except Exception as e:
        print(f"  FlexAttention execution note: {str(e)[:100]}")
        print("  (Some features require GPU or specific PyTorch build)")
else:
    print("  FlexAttention requires PyTorch 2.5+")
    print("  The patterns above show the API that would be used.")

# =============================================================================
# 7. Building attention masks manually (CPU fallback approach)
# =============================================================================

print("\n" + "=" * 60)
print("MANUAL MASK-BASED ATTENTION (CPU Compatible)")
print("=" * 60 + "\n")

def apply_attention_with_pattern(q, k, v, mask_fn, seq_len):
    """Apply attention with a custom mask pattern (non-fused, for demonstration)."""
    # Build the mask matrix
    q_pos = torch.arange(seq_len).unsqueeze(1)   # [seq_q, 1]
    kv_pos = torch.arange(seq_len).unsqueeze(0)  # [1, seq_kv]

    mask = mask_fn(0, 0, q_pos, kv_pos)  # [seq_q, seq_kv]

    # Apply via SDPA with explicit mask
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

batch, heads, seq_len, dim = 2, 4, 32, 16
q = torch.randn(batch, heads, seq_len, dim)
k = torch.randn(batch, heads, seq_len, dim)
v = torch.randn(batch, heads, seq_len, dim)

# Test different patterns
patterns = [
    ("Causal", causal_mask),
    ("Sliding Window (w=4)", sliding_window_mask(4)),
    ("Causal + Window (w=8)", causal_sliding_window(8)),
    ("Block Diagonal (bs=8)", block_diagonal_mask(8)),
]

for name, mask_fn in patterns:
    output = apply_attention_with_pattern(q, k, v, mask_fn, seq_len)
    print(f"  {name:30s}: output shape = {list(output.shape)}")

# =============================================================================
# 8. Performance considerations
# =============================================================================

print("\n" + "=" * 60)
print("PERFORMANCE CONSIDERATIONS")
print("=" * 60)
print("""
  FlexAttention Performance Tips:

  1. BlockMask sparsity: The more zeros in your mask, the faster
     FlexAttention runs (skips entire blocks of computation).

  2. torch.compile is REQUIRED for performance. Without it,
     FlexAttention falls back to the math kernel.

  3. score_mod should be simple arithmetic — complex Python
     logic may cause graph breaks.

  4. create_block_mask precomputes which blocks are all-zeros,
     allowing the kernel to skip them entirely.

  5. For standard patterns (causal, sliding window), the built-in
     is_causal=True in SDPA may be just as fast as FlexAttention.
     FlexAttention shines for CUSTOM patterns.

  When to use FlexAttention:
    - Custom attention patterns (document masking, ALiBi, etc.)
    - Combining multiple patterns (causal + sliding + document)
    - When you need score modifications (not just masking)

  When to use standard SDPA:
    - Simple causal or no mask
    - When you don't need custom patterns
    - Maximum portability (works everywhere)
""")

# =============================================================================
# 9. Summary of attention patterns and their use cases
# =============================================================================

print("=" * 60)
print("ATTENTION PATTERN SUMMARY")
print("=" * 60)
print("""
  Pattern              | Use Case                    | Models
  ---------------------+-----------------------------+-----------------
  Causal               | Autoregressive generation   | GPT, LLaMA
  Sliding Window       | Long context efficiency     | Mistral, Longformer
  Prefix LM           | Encode prompt, decode after | T5, PaLM
  Document Masking    | Multi-doc training          | Training efficiency
  ALiBi               | Position without embeddings | BLOOM
  Block Diagonal      | Sparse local attention      | BigBird
  Relative Pos Bias   | Position-aware scoring      | T5, Swin
  Dilated             | Skip-n attention            | Sparse Transformer
""")

print("FlexAttention patterns demonstration complete!")
