<div align="center">

[← Previous Module](../21_cuda_graphs/) | [🏠 Home](../README.md) | Next Module →

</div>

---

> **Module 22** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 04 — Neural Networks](../04_neural_networks/), [Module 07 — Training](../07_training/), [Module 08 — torch.compile](../08_torch_compile/), [Module 09 — Attention](../09_attention/)
> **Time to complete:** ~3 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `rope_embeddings.py` | RoPE — precompute freqs, apply rotary embeddings, position encoding visualization |
| `kv_cache.py` | KV Cache — pre-allocated cache, prefill/decode phases, GQA repeat_kv, benchmarking |
| `llm_training_loop.py` | Complete mini-LLM with RoPE, GQA, SwiGLU, RMSNorm, KV cache, bf16, gradient accumulation |

---

# Module 22: LLM Training Recipes — Building Blocks of Modern Language Models

*Day 8 of the incremental learning series*

---

## Table of Contents

1. [RoPE (Rotary Position Embeddings)](#1-rope-rotary-position-embeddings)
2. [KV Cache](#2-kv-cache)
3. [Grouped-Query Attention (GQA)](#3-grouped-query-attention-gqa)
4. [Sliding Window Attention](#4-sliding-window-attention)
5. [RMSNorm](#5-rmsnorm)
6. [SwiGLU / SiLU FFN](#6-swiglu--silu-ffn)
7. [Weight Tying](#7-weight-tying)
8. [BFloat16 Training](#8-bfloat16-training)
9. [Gradient Accumulation for Large Batch](#9-gradient-accumulation-for-large-batch)
10. [Complete Mini-LLM Training Setup](#10-complete-mini-llm-training-setup)
11. [Upstream Updates (June 12–15, 2026)](#11-upstream-updates-june-1215-2026)

---

## 1. RoPE (Rotary Position Embeddings)

### Why RoPE?

Traditional positional encodings (sinusoidal or learned) are added to token embeddings before attention. This has downsides:
- Learned embeddings have a fixed maximum length
- Sinusoidal embeddings don't interact with attention scores directly
- Neither encodes *relative* position naturally

RoPE (Su et al., 2021) applies position information as a **rotation** to the query and key vectors *inside* the attention computation. The result: attention scores naturally depend on the *relative* distance between tokens.

### The Math

Given a head dimension `d`, RoPE defines frequency bands:

```
θ_i = 10000^(-2i/d)    for i = 0, 1, ..., d/2 - 1
```

For position `m`, the rotation angles are:

```
angles_m = [m·θ_0, m·θ_1, ..., m·θ_{d/2-1}]
```

The rotation is applied by treating consecutive pairs of dimensions as 2D vectors and rotating them:

```
[x_{2i}, x_{2i+1}] → [x_{2i}·cos(m·θ_i) - x_{2i+1}·sin(m·θ_i),
                        x_{2i}·sin(m·θ_i) + x_{2i+1}·cos(m·θ_i)]
```

Equivalently, using complex numbers: view each pair as a complex number `z = x_{2i} + j·x_{2i+1}`, then multiply by `e^{j·m·θ_i}`.

### Implementation

```python
import torch

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*freq)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to Q and K tensors."""
    # Reshape to complex: (batch, seq, heads, dim) -> (batch, seq, heads, dim/2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs for broadcasting: (seq, dim/2) -> (1, seq, 1, dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Rotate
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### Why RoPE Enables Length Generalization

Because RoPE encodes *relative* position through rotation differences, models trained at one sequence length can often extrapolate to longer sequences (especially with techniques like NTK-aware scaling or YaRN). The rotation dot-product `Re[q_m · conj(k_n)]` depends only on `m - n`.

---

## 2. KV Cache

### The Problem

During autoregressive generation, token `t` needs to attend to all previous tokens `0..t-1`. Naively, this means recomputing K and V for all prior tokens at every step — O(n²) total work for n tokens.

### The Solution

Cache the K and V projections from all previous steps. At each decode step, only compute Q, K, V for the *new* token, append K and V to the cache, then attend over the full cached sequence.

### How It Works Step by Step

1. **Prefill (prompt processing):** Process the full prompt in one forward pass. Store all K, V in the cache.
2. **Decode (token generation):** For each new token:
   - Compute Q, K, V for just that token
   - Append new K, V to cache → cache grows by 1
   - Compute attention: Q (1 token) × K^T (all cached) → scores → softmax → × V

### Implementation

```python
class KVCache:
    def __init__(self, max_batch, max_seq_len, n_kv_heads, head_dim, dtype=torch.float16):
        shape = (max_batch, max_seq_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype)
        self.v_cache = torch.zeros(shape, dtype=dtype)
        self.seq_len = 0

    def update(self, k, v):
        """Append new K, V to cache. k, v shape: (batch, new_len, heads, dim)"""
        new_len = k.shape[1]
        self.k_cache[:, self.seq_len:self.seq_len + new_len] = k
        self.v_cache[:, self.seq_len:self.seq_len + new_len] = v
        self.seq_len += new_len
        return self.k_cache[:, :self.seq_len], self.v_cache[:, :self.seq_len]
```

### Memory Calculation

```
cache_size = 2 × n_layers × seq_len × n_kv_heads × head_dim × dtype_size

Example: Llama 2 7B (32 layers, 32 KV heads, dim=128, fp16, seq=4096)
= 2 × 32 × 4096 × 32 × 128 × 2 bytes = 2 GB
```

With GQA (8 KV heads instead of 32): only 512 MB — a 4× reduction.

### Impact on Generation Speed

Without cache: generating n tokens takes O(n²) total computation.
With cache: generating n tokens takes O(n) total computation (each step is O(1) for the new token's QKV, O(seq_so_far) for attention).

For a 2048-token generation, KV cache provides ~1000× speedup in total compute.

---

## 3. Grouped-Query Attention (GQA)

### What Is GQA?

Standard multi-head attention uses the same number of Q, K, and V heads. GQA uses fewer K/V heads than Q heads. Multiple Q heads share the same K/V head.

```
MHA:  32 Q heads, 32 KV heads  (standard)
GQA:  32 Q heads,  8 KV heads  (Llama 2 70B, Llama 3)
MQA:  32 Q heads,  1 KV head   (extreme)
```

### Why GQA?

1. **Smaller KV cache** — proportional reduction (4× with 8 KV heads for 32 Q heads)
2. **Less memory bandwidth** — KV cache read is often the bottleneck during decode
3. **Minimal quality loss** — GQA with 8 heads achieves near-MHA quality

### Implementation: repeat_kv

To use GQA with standard attention, expand the KV heads to match Q heads:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q heads. x: (batch, seq, n_kv_heads, head_dim)"""
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(3).expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
```

---

## 4. Sliding Window Attention

### Concept

Instead of attending to all previous tokens, only attend to the last `W` tokens (the "window"). Tokens beyond the window cannot be directly attended to.

```
Standard causal:   token t attends to tokens 0..t
Sliding window:    token t attends to tokens max(0, t-W)..t
```

### Why?

- Memory: O(n × W) instead of O(n²)
- Compute: O(n × W) instead of O(n²)
- Information still propagates through layers: after L layers, token at position t has indirect access to tokens at position t - L×W

Used in Mistral 7B (W=4096) and Mixtral.

### Implementation

```python
def make_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create a sliding window causal mask."""
    mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i + 1] = 0.0
    return mask
```

Or more efficiently:

```python
def make_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    row_idx = torch.arange(seq_len).unsqueeze(1)
    col_idx = torch.arange(seq_len).unsqueeze(0)
    # Causal: col <= row; Window: row - col < window_size
    valid = (col_idx <= row_idx) & (row_idx - col_idx < window_size)
    mask = torch.where(valid, 0.0, float('-inf'))
    return mask
```

---

## 5. RMSNorm

### Why Not LayerNorm?

LayerNorm computes:
```
y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
```

RMSNorm removes the mean subtraction and bias — just normalizes by the root-mean-square:
```
y = x / RMS(x) * γ
where RMS(x) = sqrt(mean(x²) + ε)
```

### Why Faster?

- No mean computation (one less reduction)
- No bias parameter
- Empirically, centering doesn't help much for Transformer layers

Used in: Llama, Llama 2, Llama 3, Mistral, Gemma, and most modern LLMs.

### PyTorch Implementation

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

PyTorch also provides `torch.nn.RMSNorm` as a built-in (since 2.4+):

```python
norm = torch.nn.RMSNorm(dim, eps=1e-6)
```

---

## 6. SwiGLU / SiLU FFN

### The Standard FFN

```
FFN(x) = ReLU(x @ W1) @ W2
```

Two weight matrices, ReLU activation. Simple.

### The Modern FFN (SwiGLU)

```
FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W1)) @ W2
```

Three weight matrices:
- `W_gate` (dim → hidden): produces the gating signal
- `W1` (dim → hidden): produces the value
- `W2` (hidden → dim): projects back

`⊙` is element-wise multiplication. `SiLU(x) = x * sigmoid(x)`.

### Why Three Matrices?

The gating mechanism (GLU = Gated Linear Unit) allows the network to learn which dimensions to activate. SiLU provides smooth gradients (unlike ReLU). The combination of gating + smooth activation empirically trains better for LLMs.

### Implementation

```python
class SwiGLU(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w_gate = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w_gate(x)) * self.w1(x))
```

### Hidden Dimension Convention

In Llama models: `hidden_dim = int(2/3 * 4 * dim)` rounded to a multiple of 256. The 2/3 factor compensates for the extra gate projection (3 matrices of size 2/3 ≈ 2 matrices of size 1).

---

## 7. Weight Tying

### Concept

The embedding layer (vocab_size × dim) and the output projection (dim → vocab_size) are transposes of each other. By sharing the same weight matrix, you save one copy.

```python
class LLM(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights
        self.output.weight = self.embedding.weight
```

### Impact

For a model with vocab_size=32000 and dim=4096:
- Embedding matrix: 32000 × 4096 × 2 bytes (fp16) = 250 MB
- Without tying: 500 MB for embedding + output
- With tying: 250 MB total — 50% savings on these layers

For vocab-heavy models (large vocabulary relative to model dimension), this can save ~30% of total parameters.

---

## 8. BFloat16 Training

### Why bf16 Over fp16?

| Property | fp16 | bf16 |
|----------|------|------|
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Max value | ~65504 | ~3.4×10³⁸ |
| Min normal | ~6×10⁻⁵ | ~1.2×10⁻³⁸ |
| Precision | Higher | Lower |
| Overflow risk | **High** | Very low |
| Loss scaling needed | **Yes** | No |

For LLM training, bf16 wins because:
1. **No loss scaler needed** — bf16 has the same dynamic range as fp32
2. **Simpler code** — no GradScaler, no inf checks
3. **Better stability** — gradients rarely overflow

### Usage

```python
# bf16 autocast — no GradScaler needed
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

Compare with fp16 which requires:
```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast('cuda', dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 9. Gradient Accumulation for Large Batch

### The Problem

LLMs benefit from large effective batch sizes (often 1M+ tokens). But a single GPU can only fit a small micro-batch (e.g., 4 sequences of 2048 tokens = 8K tokens).

### The Solution

Accumulate gradients over multiple micro-batches before stepping:

```python
accumulation_steps = 128  # 128 × 8K = 1M tokens effective batch

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model(batch) / accumulation_steps  # Normalize loss

    loss.backward()  # Gradients accumulate

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

Key details:
- **Divide loss** by `accumulation_steps` to average gradients (equivalent to a large batch mean)
- **Gradient clipping** after accumulation, before step
- **Memory**: only one micro-batch is active at a time (constant memory regardless of effective batch)

---

## 10. Complete Mini-LLM Training Setup

Putting all techniques together into a minimal but complete LLM:

```python
class MiniLLM(nn.Module):
    """Minimal LLM with all modern techniques."""
    def __init__(self, vocab_size=32000, dim=512, n_layers=6,
                 n_heads=8, n_kv_heads=4, max_seq_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(vocab_size, dim, bias=False)
        self.output.weight = self.embedding.weight  # Weight tying

        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.attention = GQAAttention(dim, n_heads, n_kv_heads)
        self.ffn = SwiGLU(dim, int(2/3 * 4 * dim))
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x, freqs_cis, mask=None, cache=None):
        x = x + self.attention(self.norm1(x), freqs_cis, mask, cache)
        x = x + self.ffn(self.norm2(x))
        return x
```

Training loop:
```python
model = torch.compile(MiniLLM())
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for step, batch in enumerate(dataloader):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(batch['input_ids'])
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch['labels'].view(-1)
        ) / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

Generation with KV cache:
```python
@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8, top_k=50):
    caches = [KVCache(...) for _ in range(n_layers)]

    # Prefill
    logits = model.forward_with_cache(prompt_tokens, caches)

    # Decode
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1] / temperature
        # Top-k filtering
        topk_vals, topk_idx = next_logits.topk(top_k)
        next_logits = torch.full_like(next_logits, float('-inf'))
        next_logits.scatter_(1, topk_idx, topk_vals)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        logits = model.forward_with_cache(next_token, caches)
        yield next_token
```

See `llm_training_loop.py` for the complete runnable implementation.

---

## 11. Upstream Updates (June 12–15, 2026)

Recent PyTorch commits and features relevant to LLM training:

### DTensor: `_StridedShard` to `Shard` via all-to-all (#170915)

Converts `_StridedShard` placements to `Shard` using all-to-all collective, enabling more efficient tensor parallelism redistribution. Previously, strided-shard tensors required falling back to full gather + reshard.

### Symmetric Memory NCCL EP Support

Symmetric memory allocator now supports NCCL Extensible Parallelism (EP) endpoints, enabling custom collective algorithms that bypass the NCCL ring/tree topology for specific communication patterns (e.g., expert parallelism in MoE models).

### Version 2.14.0a0

The development version has been bumped to 2.14.0a0. Key features in flight:
- FlexGEMM epilogue templates for fused post-GEMM operations
- Extended Dynamo polyfills (`itertools.permutations` now supported)
- Bug fixes for `argmin`/`argmax` on boolean tensors

### FlexGEMM Epilogue Templates

Inductor now supports FlexGEMM epilogue templates, allowing users to fuse post-matmul operations (bias add, activation, scaling) into a single CUTLASS kernel call. Reduces memory traffic for FFN layers.

### Dynamo Polyfills: `itertools.permutations`

`torch._dynamo` can now trace through `itertools.permutations`, converting it to a constant at trace time. This enables more Python patterns in `torch.compile`-d code without graph breaks.

### argmin/argmax Boolean Fix

Fixed incorrect results from `torch.argmin` and `torch.argmax` on boolean tensors, which previously could return wrong indices due to an optimization that assumed numeric ordering.

### MPS: `log_sigmoid` Metal Migration

The `log_sigmoid` operation on Apple Silicon (MPS backend) has been migrated from the CPU fallback to a native Metal shader, providing significant speedup for MPS users training models with sigmoid-based losses.

---

## Key Takeaways

| Technique | What It Does | Memory Impact | Speed Impact |
|-----------|-------------|---------------|--------------|
| RoPE | Relative position via rotation | None | Slight compute |
| KV Cache | Cache K/V for generation | +cache memory | ~1000× faster decode |
| GQA | Fewer KV heads | 4× less KV cache | Faster decode |
| Sliding Window | Limit attention span | O(n·W) vs O(n²) | Faster for long seq |
| RMSNorm | Faster normalization | Same | ~10% faster norm |
| SwiGLU | Gated FFN with SiLU | +50% FFN params | Better convergence |
| Weight Tying | Share embed/output | -30% for vocab-heavy | Same |
| bf16 | Wider exponent range | Half vs fp32 | 2× throughput |
| Grad Accumulation | Large effective batch | Constant | Linear in steps |

---

## Further Reading

- [Llama 2 Paper](https://arxiv.org/abs/2307.09288) — GQA, RoPE, SwiGLU, RMSNorm in practice
- [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — Original RoPE paper
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Sliding window attention
- [GLU Variants (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) — SwiGLU and friends
- [torchtune](https://github.com/pytorch/torchtune) — PyTorch's official LLM fine-tuning library

---

<div align="center">

[← Previous Module](../21_cuda_graphs/) | [🏠 Home](../README.md) | [Next Module →](../23_fx_transforms/)

**Notebook**: [`22_llm_recipes.ipynb`](../notebooks/22_llm_recipes.ipynb)

</div>
