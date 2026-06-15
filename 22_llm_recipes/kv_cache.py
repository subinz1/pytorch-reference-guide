"""
KV Cache — Implementation and Benchmarking

Demonstrates:
- KVCache class with pre-allocated tensors
- Prefill (full prompt) and decode (token-by-token) phases
- GQA repeat_kv function for grouped-query attention
- Benchmark: generation with vs without cache
"""

import torch
import torch.nn.functional as F
import time
import math


# =============================================================================
# KV Cache Implementation
# =============================================================================

class KVCache:
    """
    Pre-allocated KV cache for autoregressive generation.

    Pre-allocates memory for the maximum sequence length upfront to avoid
    repeated allocations during generation. Tracks current position to
    know where to write new K/V entries.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        cache_shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.seq_pos = 0

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V to cache and return full cached K/V.

        Args:
            k: New key tensor (batch, new_seq_len, n_kv_heads, head_dim)
            v: New value tensor (batch, new_seq_len, n_kv_heads, head_dim)

        Returns:
            (cached_k, cached_v) containing all entries up to current position
        """
        new_len = k.shape[1]
        self.k_cache[:, self.seq_pos : self.seq_pos + new_len] = k
        self.v_cache[:, self.seq_pos : self.seq_pos + new_len] = v
        self.seq_pos += new_len
        return self.k_cache[:, : self.seq_pos], self.v_cache[:, : self.seq_pos]

    def reset(self):
        """Reset cache position (no need to zero memory)."""
        self.seq_pos = 0

    @property
    def current_seq_len(self) -> int:
        return self.seq_pos

    def memory_bytes(self) -> int:
        """Total memory used by this cache layer."""
        return self.k_cache.nelement() * self.k_cache.element_size() * 2


# =============================================================================
# GQA: repeat_kv
# =============================================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match the number of Q heads for grouped-query attention.

    If n_rep=1 (MHA), returns x unchanged.
    If n_rep=4 and n_kv_heads=8, each KV head is repeated 4 times -> 32 heads.

    Args:
        x: (batch, seq_len, n_kv_heads, head_dim)
        n_rep: Number of times to repeat each KV head

    Returns:
        (batch, seq_len, n_kv_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(3).expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


# =============================================================================
# Attention with KV Cache
# =============================================================================

class CachedAttention(torch.nn.Module):
    """Multi-head attention with GQA and KV cache support."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = torch.nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        if cache is not None:
            k, v = cache.update(k, v)

        # GQA: expand KV heads to match Q heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Attention: (batch, heads, seq_q, seq_kv)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.wo(out)


# =============================================================================
# Benchmark: With vs Without Cache
# =============================================================================

def attention_without_cache(
    x_full: torch.Tensor, attention: CachedAttention, seq_len: int
) -> torch.Tensor:
    """Naive generation: recompute everything at each step."""
    outputs = []
    for t in range(1, seq_len + 1):
        # Recompute attention over all tokens up to t
        x_slice = x_full[:, :t, :]
        mask = torch.triu(
            torch.full((t, t), float("-inf")), diagonal=1
        )
        out = attention(x_slice, mask=mask)
        outputs.append(out[:, -1:, :])
    return torch.cat(outputs, dim=1)


def attention_with_cache(
    x_full: torch.Tensor, attention: CachedAttention, seq_len: int
) -> torch.Tensor:
    """Cached generation: only compute new token at each step."""
    batch = x_full.shape[0]
    cache = KVCache(
        max_batch_size=batch,
        max_seq_len=seq_len,
        n_kv_heads=attention.n_kv_heads,
        head_dim=attention.head_dim,
    )

    outputs = []
    for t in range(seq_len):
        x_t = x_full[:, t : t + 1, :]
        # During decode, no mask needed (single query attends to all cached)
        out = attention(x_t, mask=None, cache=cache)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def benchmark_cache():
    """Compare generation speed with and without KV cache."""
    print("=" * 70)
    print("KV Cache Benchmark: With vs Without Cache")
    print("=" * 70)

    dim = 256
    n_heads = 8
    n_kv_heads = 4
    batch_size = 1
    seq_len = 64

    torch.manual_seed(42)
    attention = CachedAttention(dim, n_heads, n_kv_heads)
    attention.eval()

    x = torch.randn(batch_size, seq_len, dim)

    # Warmup
    with torch.no_grad():
        _ = attention_with_cache(x, attention, seq_len)
        _ = attention_without_cache(x, attention, seq_len)

    # Benchmark without cache
    n_runs = 5
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            out_no_cache = attention_without_cache(x, attention, seq_len)
    time_no_cache = (time.perf_counter() - start) / n_runs

    # Benchmark with cache
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            out_with_cache = attention_with_cache(x, attention, seq_len)
    time_with_cache = (time.perf_counter() - start) / n_runs

    print(f"\nConfig: dim={dim}, heads={n_heads}, kv_heads={n_kv_heads}, seq={seq_len}")
    print(f"\nWithout cache: {time_no_cache*1000:.2f} ms")
    print(f"With cache:    {time_with_cache*1000:.2f} ms")
    print(f"Speedup:       {time_no_cache/time_with_cache:.2f}x")

    # Memory calculation
    cache = KVCache(batch_size, seq_len, n_kv_heads, dim // n_heads)
    print(f"\nCache memory per layer: {cache.memory_bytes() / 1024:.1f} KB")
    print(f"  = 2 * {batch_size} * {seq_len} * {n_kv_heads} * {dim // n_heads} * 4 bytes")


# =============================================================================
# Memory Analysis
# =============================================================================

def analyze_cache_memory():
    """Show KV cache memory for various model configurations."""
    print("\n" + "=" * 70)
    print("KV Cache Memory Analysis")
    print("=" * 70)

    configs = [
        ("Llama 2 7B", 32, 32, 128, 4096, 2),
        ("Llama 2 7B GQA", 32, 8, 128, 4096, 2),
        ("Llama 2 70B", 64, 8, 128, 4096, 2),
        ("Llama 3 8B", 32, 8, 128, 8192, 2),
        ("Mistral 7B", 32, 8, 128, 32768, 2),
        ("GPT-3 175B", 96, 96, 128, 2048, 2),
    ]

    print(f"\n{'Model':<20} {'Layers':<8} {'KV Heads':<10} {'Seq Len':<10} {'Cache (GB)':<12}")
    print("-" * 65)

    for name, n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes in configs:
        # cache_size = 2 (K+V) * n_layers * batch * seq * heads * dim * bytes
        cache_bytes = 2 * n_layers * 1 * seq_len * n_kv_heads * head_dim * dtype_bytes
        cache_gb = cache_bytes / (1024 ** 3)
        print(f"{name:<20} {n_layers:<8} {n_kv_heads:<10} {seq_len:<10} {cache_gb:<12.2f}")

    print("\nGQA (fewer KV heads) dramatically reduces cache memory.")
    print("Longer sequences (Mistral 32K) need proportionally more cache.")


# =============================================================================
# Prefill + Decode Demo
# =============================================================================

def demo_prefill_decode():
    """Demonstrate the two-phase generation pattern."""
    print("\n" + "=" * 70)
    print("Prefill + Decode Phases")
    print("=" * 70)

    dim = 128
    n_heads = 4
    n_kv_heads = 2
    batch_size = 1
    prompt_len = 8
    gen_len = 4
    total_len = prompt_len + gen_len

    torch.manual_seed(0)
    attention = CachedAttention(dim, n_heads, n_kv_heads)
    attention.eval()

    cache = KVCache(batch_size, total_len, n_kv_heads, dim // n_heads)

    prompt = torch.randn(batch_size, prompt_len, dim)
    print(f"\nPrompt length: {prompt_len} tokens")
    print(f"Tokens to generate: {gen_len}")

    # Phase 1: Prefill — process entire prompt at once
    print("\n--- Phase 1: Prefill ---")
    causal_mask = torch.triu(
        torch.full((prompt_len, prompt_len), float("-inf")), diagonal=1
    )
    with torch.no_grad():
        out = attention(prompt, mask=causal_mask, cache=cache)
    print(f"  Processed {prompt_len} tokens in one pass")
    print(f"  Cache position after prefill: {cache.current_seq_len}")

    # Phase 2: Decode — generate one token at a time
    print("\n--- Phase 2: Decode ---")
    last_hidden = out[:, -1:, :]  # Use last output as input to next step

    for step in range(gen_len):
        with torch.no_grad():
            # Single token: no mask needed (attends to all cached)
            out = attention(last_hidden, mask=None, cache=cache)
        last_hidden = out  # In real model, would go through full block
        print(f"  Step {step + 1}: generated token, cache pos = {cache.current_seq_len}")

    print(f"\nFinal cache position: {cache.current_seq_len}")
    print(f"Total tokens processed: {prompt_len} (prefill) + {gen_len} (decode) = {total_len}")


# =============================================================================
# GQA Memory Comparison
# =============================================================================

def demo_gqa_savings():
    """Compare memory between MHA, GQA, and MQA."""
    print("\n" + "=" * 70)
    print("GQA Memory Savings (repeat_kv)")
    print("=" * 70)

    batch, seq_len, n_q_heads, head_dim = 1, 512, 32, 128

    configs = [
        ("MHA (32 KV heads)", 32),
        ("GQA (8 KV heads)", 8),
        ("GQA (4 KV heads)", 4),
        ("MQA (1 KV head)", 1),
    ]

    print(f"\nQ heads: {n_q_heads}, head_dim: {head_dim}, seq_len: {seq_len}")
    print(f"\n{'Config':<25} {'KV Cache (KB)':<15} {'After repeat_kv':<20} {'Savings'}")
    print("-" * 75)

    mha_size = None
    for name, n_kv_heads in configs:
        n_rep = n_q_heads // n_kv_heads
        kv_size = batch * seq_len * n_kv_heads * head_dim * 4 * 2  # K+V, float32
        kv_kb = kv_size / 1024

        if mha_size is None:
            mha_size = kv_size
            savings = "baseline"
        else:
            savings = f"{(1 - kv_size / mha_size) * 100:.0f}% less"

        # Demonstrate repeat_kv
        k = torch.randn(batch, seq_len, n_kv_heads, head_dim)
        k_expanded = repeat_kv(k, n_rep)
        assert k_expanded.shape == (batch, seq_len, n_q_heads, head_dim)

        print(f"{name:<25} {kv_kb:<15.1f} ({n_kv_heads}→{n_q_heads} heads)     {savings}")

    print("\nrepeat_kv expands KV heads at compute time — storage stays small.")


if __name__ == "__main__":
    benchmark_cache()
    analyze_cache_memory()
    demo_prefill_decode()
    demo_gqa_savings()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
KV Cache:
  - Pre-allocate K/V storage for max sequence length
  - Prefill: process prompt in one pass, fill cache
  - Decode: one token at a time, append to cache, attend over full history
  - Speedup: O(n) total vs O(n^2) without cache

GQA (Grouped-Query Attention):
  - Fewer KV heads (e.g., 8) shared across more Q heads (e.g., 32)
  - repeat_kv expands at compute time
  - 4x cache memory reduction (32->8 KV heads)
  - Minimal quality loss vs full MHA
""")
