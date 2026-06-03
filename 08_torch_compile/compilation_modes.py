"""
Compilation Modes — default, reduce-overhead, max-autotune
===========================================================
Demonstrates the different torch.compile modes and their trade-offs.

Run: python compilation_modes.py
"""

import torch
import torch.nn as nn
import time

# =============================================================================
# 1. Model for benchmarking
# =============================================================================

class TransformerBlock(nn.Module):
    """A simplified transformer block — good target for torch.compile."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    """Stack of transformer blocks."""

    def __init__(self, dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# 2. Benchmark helper
# =============================================================================

def benchmark(fn, x, warmup=5, runs=50):
    """Benchmark a function, returns average time in ms."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)

    # Timed runs
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            fn(x)
    elapsed = (time.time() - start) / runs * 1000  # ms
    return elapsed

# =============================================================================
# 3. Compare compilation modes
# =============================================================================

print("=" * 60)
print("COMPILATION MODES COMPARISON")
print("=" * 60)

torch._dynamo.reset()

x = torch.randn(16, 32, 256)  # [batch, seq_len, dim]

# Eager (no compilation)
print("\n--- Eager (no compilation) ---")
model_eager = SmallTransformer()
model_eager.eval()
eager_time = benchmark(model_eager, x)
print(f"  Average time: {eager_time:.2f} ms")

# Default mode
print("\n--- torch.compile(mode='default') ---")
print("  Balanced between compile time and runtime.")
model_default = SmallTransformer()
model_default.eval()
compiled_default = torch.compile(model_default, mode="default")
# Trigger compilation
with torch.no_grad():
    compiled_default(x)
default_time = benchmark(compiled_default, x)
print(f"  Average time: {default_time:.2f} ms")
print(f"  Speedup vs eager: {eager_time / default_time:.2f}x")

# Reduce-overhead mode
print("\n--- torch.compile(mode='reduce-overhead') ---")
print("  Minimizes framework overhead (CUDA graphs on GPU).")
print("  On CPU, behavior is similar to default.")
torch._dynamo.reset()
model_reduce = SmallTransformer()
model_reduce.eval()
compiled_reduce = torch.compile(model_reduce, mode="reduce-overhead")
with torch.no_grad():
    compiled_reduce(x)
reduce_time = benchmark(compiled_reduce, x)
print(f"  Average time: {reduce_time:.2f} ms")
print(f"  Speedup vs eager: {eager_time / reduce_time:.2f}x")

# Max-autotune mode
print("\n--- torch.compile(mode='max-autotune') ---")
print("  Tries many kernel variants, picks the fastest.")
print("  Longer compilation, best runtime.")
torch._dynamo.reset()
model_autotune = SmallTransformer()
model_autotune.eval()

start_compile = time.time()
compiled_autotune = torch.compile(model_autotune, mode="max-autotune")
with torch.no_grad():
    compiled_autotune(x)
compile_time = time.time() - start_compile

autotune_time = benchmark(compiled_autotune, x)
print(f"  Compilation time: {compile_time:.2f}s")
print(f"  Average time: {autotune_time:.2f} ms")
print(f"  Speedup vs eager: {eager_time / autotune_time:.2f}x")

# =============================================================================
# 4. Summary table
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n{'Mode':<20} {'Time (ms)':<12} {'Speedup':<10}")
print(f"{'-'*42}")
print(f"{'Eager':<20} {eager_time:<12.2f} {'1.00x':<10}")
print(f"{'default':<20} {default_time:<12.2f} {eager_time/default_time:<10.2f}x")
print(f"{'reduce-overhead':<20} {reduce_time:<12.2f} {eager_time/reduce_time:<10.2f}x")
print(f"{'max-autotune':<20} {autotune_time:<12.2f} {eager_time/autotune_time:<10.2f}x")

# =============================================================================
# 5. Mode selection guidance
# =============================================================================

print("\n" + "=" * 60)
print("WHEN TO USE EACH MODE")
print("=" * 60)
print("""
Mode Selection Guide:

  'default':
    - During development and debugging
    - When compile time matters (iterating quickly)
    - When you're not sure which mode to use

  'reduce-overhead':
    - GPU workloads with many small kernels
    - When Python/framework overhead dominates
    - Inference serving with low latency requirements
    - Note: Uses more GPU memory due to CUDA graphs

  'max-autotune':
    - Production deployment (compile once, run many times)
    - When you've already fixed all graph breaks
    - Benchmarking to find max possible speed
    - Willing to wait minutes for compilation

  Tips:
    - Start with 'default'
    - Move to 'max-autotune' for production
    - Use 'reduce-overhead' for GPU serving
    - Profile to verify actual speedup!
""")

# =============================================================================
# 6. Combining options
# =============================================================================

print("=" * 60)
print("COMBINING OPTIONS")
print("=" * 60)

torch._dynamo.reset()
model = SmallTransformer()
model.eval()

# Full optimization: max-autotune + fullgraph + dynamic shapes
compiled_full = torch.compile(
    model,
    mode="max-autotune",
    fullgraph=True,     # Error if there are graph breaks
    dynamic=True,       # Handle varying batch sizes
)

# Works with different batch sizes thanks to dynamic=True
with torch.no_grad():
    out1 = compiled_full(torch.randn(8, 32, 256))
    out2 = compiled_full(torch.randn(16, 32, 256))
    out3 = compiled_full(torch.randn(4, 32, 256))

print(f"\n  Batch 8:  output shape = {out1.shape}")
print(f"  Batch 16: output shape = {out2.shape}")
print(f"  Batch 4:  output shape = {out3.shape}")
print(f"  All work without recompilation thanks to dynamic=True!")

print("\nCompilation modes demonstration complete!")
