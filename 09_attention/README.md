<div align="center">

[← Previous Module](../08_torch_compile/) | [🏠 Home](../README.md) | [Next Module →](../10_distributed/)

</div>

---

> **Module 09** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 04 — Neural Networks](../04_neural_networks/), [Module 07 — Training](../07_training/)
> **Time to complete:** ~3 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `manual_attention.py` | Manual attention — implementing attention from scratch with shape annotations |
| `sdpa_and_backends.py` | SDPA and backend control — PyTorch's optimized attention |
| `multihead_attention.py` | Multi-head attention — complete implementation with shape tracking |
| `transformer_block.py` | Transformer block — full implementation with RMSNorm/LayerNorm |
| `flex_attention_patterns.py` | FlexAttention patterns — custom attention with compiled kernels |

---

# Module 09: Attention Mechanisms — From Scratch to FlexAttention

## Overview

Attention is the core mechanism behind transformers — the architecture powering
GPT, BERT, LLaMA, and virtually all modern AI models. This module builds
attention from first principles and works up to PyTorch's most advanced APIs.

---

## 1. What is Attention? (Intuitive Explanation)

### The Analogy

Imagine you're reading a sentence: "The cat sat on the mat because **it** was tired."

What does "it" refer to? You (unconsciously) **attend** to earlier words and
determine "it" = "cat". Attention is this mechanism: for each position in a
sequence, it looks at ALL other positions and decides which ones are relevant.

### The Core Idea

Given a query ("what am I looking for?"), attention:
1. Compares the query against all keys ("what information is available?")
2. Produces attention weights (how relevant each key is)
3. Uses those weights to aggregate values ("what to return?")

---

## 2. Scaled Dot-Product Attention

### The Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- Q (Query): [batch, seq_len, d_k] — what each position is looking for
- K (Key): [batch, seq_len, d_k] — what each position offers for matching
- V (Value): [batch, seq_len, d_v] — what each position actually provides
- d_k: dimension of keys/queries

### Why Scale by sqrt(d_k)?

Without scaling, dot products grow with d_k. Large dot products push softmax
into regions where it has extremely small gradients (saturation). Dividing by
sqrt(d_k) keeps variance ~1 regardless of dimension.

Example: if Q and K entries are independent with mean 0, variance 1:
- `Q @ K^T` has variance = d_k
- After scaling: variance = d_k / d_k = 1

### Step-by-Step with Shapes

```python
# Input: Q, K, V each with shape [batch, seq_len, d_model]
# For simplicity, assume d_k = d_v = d_model

# Step 1: Compute attention scores
scores = Q @ K.transpose(-2, -1)    # [batch, seq_len, seq_len]

# Step 2: Scale
scores = scores / math.sqrt(d_k)    # [batch, seq_len, seq_len]

# Step 3: (Optional) Apply mask
scores = scores.masked_fill(mask == 0, float('-inf'))

# Step 4: Softmax to get attention weights
weights = softmax(scores, dim=-1)   # [batch, seq_len, seq_len]

# Step 5: Weighted sum of values
output = weights @ V                # [batch, seq_len, d_v]
```

---

## 3. Causal (Autoregressive) Attention

### What It Is

In language models that generate text left-to-right, position i should only
attend to positions 0, 1, ..., i (not future positions). This is enforced
with a causal mask:

```
mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
```

### Implementation

```python
# Create causal mask (lower triangular)
seq_len = Q.shape[1]
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

# Apply: set future positions to -inf before softmax
scores = scores.masked_fill(causal_mask == 0, float('-inf'))
# After softmax, -inf becomes 0 (no attention to future)
```

---

## 4. Multi-Head Attention

### Why Multiple Heads?

A single attention head can only focus on one pattern (e.g., subject-verb
agreement). Multiple heads let the model attend to different patterns
simultaneously:
- Head 1 might attend to syntactic relationships
- Head 2 might attend to semantic similarity
- Head 3 might attend to positional proximity

### How It Works

Instead of one attention with d_model dimensions, use h heads each with
d_k = d_model / h dimensions:

```python
# Input: x with shape [batch, seq_len, d_model]
# Project to Q, K, V for each head
Q = W_q(x)  # [batch, seq_len, d_model]
K = W_k(x)  # [batch, seq_len, d_model]
V = W_v(x)  # [batch, seq_len, d_model]

# Reshape to [batch, num_heads, seq_len, head_dim]
Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
V = V.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

# Apply attention independently per head
# output: [batch, num_heads, seq_len, head_dim]
output = scaled_dot_product_attention(Q, K, V)

# Concatenate heads and project
output = output.transpose(1, 2).reshape(batch, seq_len, d_model)
output = W_o(output)  # Final linear projection
```

---

## 5. F.scaled_dot_product_attention (SDPA)

PyTorch provides an optimized implementation:

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query,      # [batch, heads, seq_q, dim]
    key,        # [batch, heads, seq_kv, dim]
    value,      # [batch, heads, seq_kv, dim_v]
    attn_mask=None,     # Optional mask
    dropout_p=0.0,      # Dropout probability
    is_causal=False,    # If True, applies causal mask automatically
    scale=None,         # Custom scale factor (default: 1/sqrt(dim))
)
```

### Benefits

- Automatically selects the fastest available backend
- Memory-efficient (doesn't materialize the full attention matrix)
- Fused implementation (fewer memory reads/writes)

---

## 6. SDPA Backends

PyTorch has multiple backends for SDPA, each optimized for different cases:

### Flash Attention
- **Memory**: O(N) instead of O(N^2)
- **Speed**: Fastest for long sequences
- **How**: Tiles the computation, never materializes full attention matrix
- **Requires**: GPU with compute capability >= 8.0 (A100+)

### Memory-Efficient Attention
- **Memory**: O(N) via chunked computation
- **Speed**: Good general-purpose
- **How**: Processes attention in chunks
- **Requires**: Any GPU

### cuDNN Attention
- **Speed**: Can be fastest for standard shapes
- **Requires**: cuDNN 8.9+

### Math (Fallback)
- **Memory**: O(N^2) — materializes full attention matrix
- **Speed**: Slowest for long sequences
- **Works**: Everywhere (CPU and GPU)
- **Use**: When other backends aren't available or for debugging

### Backend Selection

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force a specific backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(q, k, v)

# Use math backend for debugging (can inspect attention weights)
with sdpa_kernel(SDPBackend.MATH):
    output = F.scaled_dot_product_attention(q, k, v)
```

---

## 7. Flash Attention Explained Simply

### The Problem

Standard attention computes:
```
S = Q @ K^T    # [N, N] — this is HUGE for long sequences
P = softmax(S) # [N, N] — stored in memory
O = P @ V      # output
```

For N = 16384 (16K tokens): S alone is 16384^2 * 4 bytes = 1 GB!

### The Flash Attention Trick

Instead of computing the full N×N matrix:
1. **Tile** Q, K, V into small blocks that fit in GPU fast memory (SRAM)
2. Compute attention **block by block**
3. Use the **online softmax trick** to accumulate correct softmax results
   across blocks without needing the full row

Result: O(N) memory instead of O(N^2), and faster due to fewer memory accesses.

### Online Softmax

The key insight: you can compute softmax incrementally. As you process each
block of keys, maintain a running maximum and running sum, then correct at
the end. This avoids needing all scores simultaneously.

---

## 8. FlexAttention

### What It Is

FlexAttention (torch.nn.attention.flex_attention) lets you define custom
attention patterns using simple Python functions, while still getting the
performance benefits of Flash Attention-like fused kernels.

### Why It Exists

Previously, custom attention patterns (sliding window, ALiBi, document
masking) required:
- Writing custom CUDA kernels (hard)
- Materializing masks (memory expensive)
- Giving up on fused implementations (slow)

FlexAttention compiles your pattern into an efficient kernel.

### Core API

```python
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

# score_mod: modifies attention scores before softmax
def causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float('-inf'))

# mask_mod: defines which positions can attend to which (for BlockMask)
def causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Create a BlockMask for efficiency
block_mask = create_block_mask(causal_mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len)

# Apply flex attention
output = flex_attention(query, key, value, block_mask=block_mask)
```

### score_mod vs mask_mod

- **score_mod(score, b, h, q_idx, kv_idx)**: Modifies the attention score
  for a specific (query, key) pair. Can add biases, apply causal masking, etc.
  Returns the modified score.

- **mask_mod(b, h, q_idx, kv_idx)**: Returns True/False for whether this
  (query, key) pair should be allowed. Used by BlockMask to skip entire blocks
  of computation.

---

## 9. FlexAttention Patterns

### Causal Attention

```python
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
```

### Sliding Window

```python
def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= window_size
```

### Causal + Sliding Window

```python
def causal_sliding(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= window_size)
```

### ALiBi (Attention with Linear Biases)

```python
def alibi_score_mod(score, b, h, q_idx, kv_idx):
    slope = 2 ** (-(h + 1) * 8 / num_heads)
    bias = -slope * (q_idx - kv_idx).abs()
    return score + bias
```

### Document Masking (Multiple Documents in One Sequence)

```python
# document_id[i] tells which document position i belongs to
def document_mask(b, h, q_idx, kv_idx):
    return document_id[q_idx] == document_id[kv_idx]
```

### Prefix LM (Bidirectional prefix + Causal suffix)

```python
def prefix_lm(b, h, q_idx, kv_idx):
    # Allow bidirectional attention within prefix
    # Causal attention after prefix
    return (kv_idx < prefix_length) | (q_idx >= kv_idx)
```

---

## 10. Building a Transformer Block

### Pre-Norm vs Post-Norm

```python
# Post-norm (original Transformer paper):
x = x + attention(x)
x = layer_norm(x)

# Pre-norm (GPT-2, modern models — more stable training):
x = x + attention(layer_norm(x))
```

### RMSNorm vs LayerNorm

```python
# LayerNorm: normalize, then scale + shift
# Subtracts mean AND divides by std
y = (x - mean(x)) / std(x) * gamma + beta

# RMSNorm: only divides by RMS (no mean subtraction)
# Faster, often works just as well
y = x / rms(x) * gamma
```

### Activation Functions

- **GELU**: Smooth approximation of ReLU, used in BERT/GPT
- **SiLU (Swish)**: x * sigmoid(x), used in LLaMA, PaLM
- **SwiGLU**: Gated variant, used in modern LLMs

### Complete Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 11. Positional Encoding

Attention is permutation-invariant — it doesn't know token order. We add
positional information explicitly.

### Sinusoidal (Original Transformer)

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension oscillates at a different frequency. The model can learn
to attend to relative positions via linear combinations.

### Learned Positional Embeddings

```python
self.pos_embed = nn.Embedding(max_seq_len, d_model)
x = x + self.pos_embed(positions)
```

Simple but limited to max_seq_len seen during training.

### RoPE (Rotary Position Embeddings)

Used in LLaMA, Mistral, and most modern LLMs.

Key idea: encode position by **rotating** Q and K vectors in 2D subspaces.
The dot product Q·K then naturally depends on relative position.

```python
# For each pair of dimensions (2i, 2i+1):
q_rotated[2i]   = q[2i] * cos(theta) - q[2i+1] * sin(theta)
q_rotated[2i+1] = q[2i] * sin(theta) + q[2i+1] * cos(theta)
# Where theta = position * base_freq^(-2i/d)
```

Benefits:
- Relative position awareness
- Can extrapolate to longer sequences than training
- No additional parameters

---

## 12. KV Cache for Inference

### The Problem

During autoregressive generation, each new token needs to attend to ALL
previous tokens. Without caching, you recompute K and V for all previous
tokens at every step.

### The Solution: KV Cache

Cache the K and V projections from previous steps:

```python
# Step 1: Process prompt
K_cache = key_projection(prompt)    # [batch, num_heads, prompt_len, head_dim]
V_cache = value_projection(prompt)  # [batch, num_heads, prompt_len, head_dim]

# Step 2+: Generate each new token
for _ in range(max_new_tokens):
    # Only project the NEW token
    new_k = key_projection(new_token)   # [batch, num_heads, 1, head_dim]
    new_v = value_projection(new_token) # [batch, num_heads, 1, head_dim]

    # Append to cache
    K_cache = torch.cat([K_cache, new_k], dim=2)
    V_cache = torch.cat([V_cache, new_v], dim=2)

    # Query only needs the new token, but attends to full cache
    new_q = query_projection(new_token) # [batch, num_heads, 1, head_dim]
    output = attention(new_q, K_cache, V_cache)
```

### Memory vs Speed Tradeoff

- Without cache: O(n) compute per step, O(1) extra memory
- With cache: O(1) amortized compute per step, O(n) extra memory

The KV cache makes generation ~10x faster for long sequences.

---

## Summary

| Concept              | Purpose                              | Key Shape               |
|----------------------|--------------------------------------|-------------------------|
| Scaled dot-product   | Core attention computation           | [B, N, N] scores       |
| Causal mask          | Prevent attending to future          | Lower triangular        |
| Multi-head           | Multiple attention patterns          | h heads, d_k = d/h     |
| SDPA                 | Optimized PyTorch attention          | Same as manual          |
| Flash Attention      | O(N) memory attention                | Tiled computation       |
| FlexAttention        | Custom patterns, fused               | score_mod/mask_mod      |
| RoPE                 | Relative position encoding           | 2D rotations            |
| KV Cache             | Fast autoregressive generation       | Cached K, V tensors     |

---

<div align="center">

[← Previous Module](../08_torch_compile/) | [🏠 Home](../README.md) | [Next Module →](../10_distributed/)

**[📓 Open Notebook](../notebooks/07_attention_and_transformers.ipynb)** — Interactive version of this module

</div>
