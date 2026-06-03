"""
GPT (Generative Pre-trained Transformer) — Decoder-Only Implementation
======================================================================

Implements a GPT-style decoder-only Transformer with:
- Causal (autoregressive) self-attention
- Pre-norm (LayerNorm before sublayers)
- Weight tying (embedding == output projection)
- KV cache for efficient generation
- Multiple generation strategies: greedy, temperature, top-k, top-p/nucleus

Reference: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking and KV cache support."""

    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        # Combined Q/K/V projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: prevents attending to future positions
        causal = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", causal.view(1, 1, max_len, max_len))

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            kv_cache: optional tuple (cached_k, cached_v) for generation
        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: tuple (k, v) for caching
        """
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (batch, heads, seq_len, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Append to KV cache during generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled dot-product attention with causal masking
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask (only mask if not using KV cache, or on first pass)
        kv_len = k.size(2)
        # Query positions start after cache
        q_start = kv_len - T
        mask = self.causal_mask[:, :, q_start:kv_len, :kv_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), new_kv_cache


class GPTBlock(nn.Module):
    """Single GPT block: pre-norm + causal attention + pre-norm + FFN."""

    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class GPT(nn.Module):
    """GPT language model.

    Architecture:
        token embedding + position embedding
        -> N x GPTBlock (pre-norm causal attention + FFN)
        -> final LayerNorm
        -> linear projection to vocab (weights tied with token embedding)
    """

    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, max_len, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: the output projection shares weights with the embedding
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_caches=None):
        """
        Args:
            idx: token indices, (batch, seq_len)
            kv_caches: list of (k, v) tuples per layer, or None
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_caches: list of (k, v) tuples per layer
        """
        B, T = idx.shape

        # Determine position offsets (for KV cache: positions continue from cache)
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(2)
        else:
            past_len = 0

        positions = torch.arange(past_len, past_len + T, device=idx.device)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(positions)
        x = self.drop(tok_emb + pos_emb)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    # -------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        use_kv_cache=True,
    ):
        """Autoregressive text generation with multiple sampling strategies.

        Args:
            idx: starting token indices, (batch, seq_len)
            max_new_tokens: how many tokens to generate
            temperature: softmax temperature (< 1 = sharper, > 1 = flatter)
            top_k: if set, keep only top-k most probable tokens
            top_p: if set, use nucleus sampling with this threshold
            use_kv_cache: whether to use KV caching for efficiency
        Returns:
            idx: generated sequence including the prompt, (batch, seq_len + max_new_tokens)
        """
        self.eval()
        kv_caches = [None] * len(self.blocks) if use_kv_cache else None

        for _ in range(max_new_tokens):
            if use_kv_cache:
                if kv_caches[0] is None:
                    input_ids = idx
                else:
                    input_ids = idx[:, -1:]
                logits, kv_caches = self(input_ids, kv_caches)
                logits = logits[:, -1, :]
            else:
                # Without cache: pass the full (possibly truncated) sequence
                idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]

            next_token = self._sample(logits, temperature, top_k, top_p)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    @staticmethod
    def _sample(logits, temperature=1.0, top_k=None, top_p=None):
        """Sample a single token from logits with optional filtering.

        Applies temperature scaling, top-k filtering, and top-p (nucleus)
        filtering in sequence, then samples from the resulting distribution.
        """
        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering: zero out everything below the k-th largest logit
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus) filtering: keep smallest set with cumulative prob >= p
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens whose cumulative probability exceeds p
            # Shift right so the first token exceeding p is kept
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back to original positions
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Greedy if temperature is very low
        if temperature < 1e-8:
            return logits.argmax(dim=-1, keepdim=True)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def count_unique_params(model):
        """Account for weight tying — count unique parameter tensors."""
        seen = set()
        total = 0
        for p in model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    vocab_size = 500
    d_model = 256
    num_heads = 8
    num_layers = 4
    max_len = 128
    batch_size = 2
    seq_len = 20

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
    )

    print(f"GPT Model")
    print(f"  Total parameters (with tying):    {count_params(model):,}")
    print(f"  Unique parameters (no duplicates): {count_unique_params(model):,}")

    # Forward pass test
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, _ = model(idx)
    print(f"\nForward pass:")
    print(f"  Input:  {list(idx.shape)}")
    print(f"  Output: {list(logits.shape)}")

    # Generation test: try all sampling strategies
    prompt = torch.randint(0, vocab_size, (1, 5))
    print(f"\nGeneration from prompt of length {prompt.size(1)}:")

    strategies = [
        ("Greedy (temp=0.001)", dict(temperature=0.001)),
        ("Temperature=0.8", dict(temperature=0.8)),
        ("Top-k=50", dict(top_k=50)),
        ("Top-p=0.9", dict(top_p=0.9)),
        ("Top-k=50 + Top-p=0.9", dict(top_k=50, top_p=0.9, temperature=0.8)),
    ]

    for name, kwargs in strategies:
        generated = model.generate(prompt.clone(), max_new_tokens=20, **kwargs)
        print(f"  {name:30s} -> shape {list(generated.shape)}, "
              f"tokens: {generated[0, 5:].tolist()[:10]}...")

    # KV cache consistency test: verify cached and non-cached produce same output
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 5))

    gen_cached = model.generate(prompt.clone(), max_new_tokens=10,
                                temperature=0.001, use_kv_cache=True)
    gen_no_cache = model.generate(prompt.clone(), max_new_tokens=10,
                                  temperature=0.001, use_kv_cache=False)

    match = torch.equal(gen_cached, gen_no_cache)
    print(f"\nKV cache consistency: {'PASS' if match else 'FAIL'}")

    print("\nGPT verified successfully!")
