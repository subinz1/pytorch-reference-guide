"""
Complete Mini-LLM Training Setup

A minimal but complete language model combining all modern techniques:
- RoPE (Rotary Position Embeddings)
- GQA (Grouped-Query Attention)
- SwiGLU FFN
- RMSNorm
- Weight Tying
- BFloat16 autocast (falls back to float32 on CPU)
- Gradient Accumulation
- torch.compile (optional)
- KV Cache for generation
- Temperature / Top-k sampling

Runs on CPU by default. Set device='cuda' for GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class LLMConfig:
    vocab_size: int = 1024
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    dropout: float = 0.0


# =============================================================================
# Building Blocks
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 / 3 * 4 * dim)
            # Round to nearest multiple of 64 for efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


# =============================================================================
# RoPE
# =============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (seq, dim/2) -> broadcast with (batch, seq, heads, dim/2)
    ndim = xq_c.ndim
    shape = [1] * ndim
    shape[1] = freqs_cis.shape[0]
    shape[-1] = freqs_cis.shape[1]
    freqs = freqs_cis.view(*shape)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# KV Cache
# =============================================================================

class KVCache:
    def __init__(self, batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype):
        shape = (batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(self, k, v):
        new_len = k.shape[1]
        self.k_cache[:, self.pos:self.pos + new_len] = k
        self.v_cache[:, self.pos:self.pos + new_len] = v
        self.pos += new_len
        return self.k_cache[:, :self.pos], self.v_cache[:, :self.pos]

    def reset(self):
        self.pos = 0


# =============================================================================
# GQA Attention
# =============================================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x.unsqueeze(3).expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


class GQAAttention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x, freqs_cis, mask=None, cache=None):
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Update KV cache if in generation mode
        if cache is not None:
            k, v = cache.update(k, v)

        # Expand KV heads for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.wo(out)


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attention = GQAAttention(config)
        self.ffn = SwiGLU(config.dim)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

    def forward(self, x, freqs_cis, mask=None, cache=None):
        x = x + self.attention(self.norm1(x), freqs_cis, mask, cache)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Complete Mini-LLM
# =============================================================================

class MiniLLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.output.weight = self.embedding.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, caches: list[KVCache] | None = None):
        """
        Forward pass for training (caches=None) or generation (caches provided).

        Args:
            tokens: (batch, seq_len) token indices
            caches: List of KVCache (one per layer) for generation mode
        """
        batch, seq_len = tokens.shape
        x = self.embedding(tokens)

        # Get the appropriate slice of freqs_cis
        if caches is not None and caches[0].pos > 0:
            start_pos = caches[0].pos
            freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        else:
            start_pos = 0
            freqs_cis = self.freqs_cis[:seq_len]

        # Causal mask (only needed during training or prefill)
        mask = None
        if seq_len > 1:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
                diagonal=1,
            )

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, freqs_cis, mask, cache)

        x = self.norm(x)
        logits = self.output(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Generation with KV Cache
# =============================================================================

@torch.no_grad()
def generate(
    model: MiniLLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[int]:
    """
    Generate tokens autoregressively using KV cache.

    Args:
        model: The MiniLLM model
        prompt_tokens: (1, prompt_len) tensor of token ids
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Only sample from top-k logits
    """
    config = model.config
    device = prompt_tokens.device

    # Create KV caches for each layer
    caches = [
        KVCache(
            batch_size=1,
            max_seq_len=config.max_seq_len,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.dim // config.n_heads,
            device=device,
            dtype=torch.float32,
        )
        for _ in range(config.n_layers)
    ]

    # Prefill: process the prompt
    logits = model(prompt_tokens, caches=caches)
    next_logits = logits[:, -1, :]

    generated = []
    for _ in range(max_new_tokens):
        # Temperature scaling
        scaled_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            topk_values, topk_indices = scaled_logits.topk(
                min(top_k, scaled_logits.size(-1))
            )
            filtered = torch.full_like(scaled_logits, float("-inf"))
            filtered.scatter_(1, topk_indices, topk_values)
            scaled_logits = filtered

        probs = F.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())

        # Decode step: only process new token
        next_logits = model(next_token, caches=caches)[:, -1, :]

    return generated


# =============================================================================
# Training Loop with Gradient Accumulation + bf16
# =============================================================================

def train_mini_llm(
    config: LLMConfig | None = None,
    num_steps: int = 100,
    accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    use_compile: bool = False,
    device: str = "cpu",
):
    """
    Train the mini-LLM with all modern techniques.

    Uses bf16 autocast on CUDA, falls back to float32 on CPU.
    """
    if config is None:
        config = LLMConfig()

    print("=" * 70)
    print("Mini-LLM Training")
    print("=" * 70)

    model = MiniLLM(config).to(device)
    print(f"\nModel config: {config}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print(f"Gradient accumulation: {accumulation_steps} steps")

    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Determine autocast dtype
    use_amp = device == "cuda" and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Synthetic data (random tokens for demonstration)
    def get_batch(batch_size: int = 4):
        data = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len + 1))
        x = data[:, :-1].to(device)
        y = data[:, 1:].to(device)
        return x, y

    print(f"\nTraining for {num_steps} optimizer steps "
          f"({num_steps * accumulation_steps} forward passes)...")
    print(f"AMP dtype: {amp_dtype}\n")

    model.train()
    total_loss = 0.0
    start_time = time.perf_counter()

    micro_step = 0
    for step in range(num_steps * accumulation_steps):
        x, y = get_batch()

        with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                y.view(-1),
            ) / accumulation_steps

        loss.backward()
        micro_step += 1

        if micro_step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            opt_step = micro_step // accumulation_steps
            total_loss += loss.item() * accumulation_steps

            if opt_step % 20 == 0 or opt_step == num_steps:
                avg_loss = total_loss / min(opt_step, 20)
                elapsed = time.perf_counter() - start_time
                tokens_per_sec = (
                    opt_step * accumulation_steps * 4 * config.max_seq_len / elapsed
                )
                print(
                    f"  Step {opt_step:>4}/{num_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )
                total_loss = 0.0

    elapsed = time.perf_counter() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    return model


# =============================================================================
# Demo: Full Pipeline
# =============================================================================

def demo_full_pipeline():
    """Run training + generation to demonstrate the full pipeline."""
    config = LLMConfig(
        vocab_size=256,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
    )

    # Train
    model = train_mini_llm(config=config, num_steps=50, accumulation_steps=2)
    model.eval()

    # Generate
    print("\n" + "=" * 70)
    print("Generation with KV Cache")
    print("=" * 70)

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    print(f"\nPrompt tokens: {prompt[0].tolist()}")

    start = time.perf_counter()
    generated = generate(model, prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    gen_time = time.perf_counter() - start

    print(f"Generated tokens: {generated}")
    print(f"Generation time: {gen_time*1000:.1f} ms ({len(generated)/gen_time:.0f} tokens/s)")

    # Compare speed: with cache vs without cache (re-encoding full sequence)
    print("\n--- Speed comparison: cached vs uncached ---")
    prompt_long = torch.randint(0, config.vocab_size, (1, 32))

    start = time.perf_counter()
    _ = generate(model, prompt_long, max_new_tokens=32)
    cached_time = time.perf_counter() - start

    # Uncached: recompute everything each step (simulated)
    start = time.perf_counter()
    tokens_so_far = prompt_long.clone()
    for _ in range(32):
        logits = model(tokens_so_far)
        next_logits = logits[:, -1, :] / 0.8
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens_so_far = torch.cat([tokens_so_far, next_token], dim=1)
    uncached_time = time.perf_counter() - start

    print(f"  With KV cache:    {cached_time*1000:.1f} ms")
    print(f"  Without KV cache: {uncached_time*1000:.1f} ms")
    print(f"  Speedup:          {uncached_time/cached_time:.2f}x")


# =============================================================================
# Model Architecture Summary
# =============================================================================

def print_architecture_summary():
    """Print a summary of the model architecture."""
    config = LLMConfig()
    model = MiniLLM(config)

    print("=" * 70)
    print("Mini-LLM Architecture Summary")
    print("=" * 70)
    print(f"""
Config:
  vocab_size:   {config.vocab_size}
  dim:          {config.dim}
  n_layers:     {config.n_layers}
  n_heads:      {config.n_heads} (Q heads)
  n_kv_heads:   {config.n_kv_heads} (KV heads, GQA ratio = {config.n_heads // config.n_kv_heads}:1)
  head_dim:     {config.dim // config.n_heads}
  max_seq_len:  {config.max_seq_len}
  rope_theta:   {config.rope_theta}

Architecture:
  Embedding       -> ({config.vocab_size}, {config.dim})
  {config.n_layers}x TransformerBlock:
    RMSNorm       -> ({config.dim},)
    GQA Attention -> Q: {config.n_heads} heads, KV: {config.n_kv_heads} heads
    RoPE          -> applied to Q, K
    RMSNorm       -> ({config.dim},)
    SwiGLU FFN    -> dim={config.dim}, hidden~{int(2/3*4*config.dim)}
  RMSNorm         -> ({config.dim},)
  Output Linear   -> ({config.dim}, {config.vocab_size}) [tied with embedding]

Total parameters: {model.count_parameters():,}
  (weight tying saves {config.vocab_size * config.dim:,} parameters)

Techniques used:
  [x] RoPE (Rotary Position Embeddings)
  [x] GQA (Grouped-Query Attention)
  [x] SwiGLU FFN (Gated Linear Unit with SiLU)
  [x] RMSNorm (Root Mean Square Normalization)
  [x] Weight Tying (embedding == output projection)
  [x] BFloat16 autocast (on CUDA)
  [x] Gradient Accumulation
  [x] torch.compile compatible
  [x] KV Cache for generation
  [x] Temperature + Top-k sampling
""")


if __name__ == "__main__":
    print_architecture_summary()
    demo_full_pipeline()
