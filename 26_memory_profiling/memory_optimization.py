"""
Memory Optimization Techniques — Every Trick in the Book
=========================================================

Demonstrates memory optimization techniques, with all core concepts
runnable on CPU. GPU-specific measurements are included when available.

Techniques covered:
  1. Gradient checkpointing (trade compute for memory)
  2. Mixed precision (fp32 vs bf16 parameter memory)
  3. Gradient accumulation (effective batch without memory cost)
  4. In-place operations (and autograd warnings)
  5. del + gc.collect pattern
  6. GPU measurements before/after each technique (if available)

Run:
    python memory_optimization.py
"""

import gc
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_CUDA = torch.cuda.is_available()


def gpu_mem_mb() -> float:
    if not HAS_CUDA:
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


@contextmanager
def track_gpu_memory(label: str):
    """Context manager that reports GPU memory delta."""
    if not HAS_CUDA:
        yield
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    yield
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{label}] before={before/1e6:.1f}MB  after={after/1e6:.1f}MB  "
          f"peak={peak/1e6:.1f}MB  delta={+(after-before)/1e6:.1f}MB")


# ─────────────────────────────────────────────────────────────────────
# Shared model definition
# ─────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.up = nn.Linear(dim, dim * mult, bias=False)
        self.down = nn.Linear(dim * mult, dim, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, dim=256, num_heads=8, num_layers=6, vocab_size=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def make_batch(batch_size=8, seq_len=128, vocab_size=1000, device="cpu"):
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


# ─────────────────────────────────────────────────────────────────────
# 1. Gradient Checkpointing
# ─────────────────────────────────────────────────────────────────────

def demo_gradient_checkpointing():
    print("=" * 70)
    print("1. GRADIENT CHECKPOINTING — Trade Compute for Memory")
    print("=" * 70)

    device = "cuda" if HAS_CUDA else "cpu"
    dim, layers = (256, 12) if not HAS_CUDA else (512, 12)
    model = SmallTransformer(dim=dim, num_heads=8, num_layers=layers).to(device)
    batch = make_batch(batch_size=16, seq_len=128, device=device)

    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"\nModel: {param_count:,} params ({param_mb:.1f} MB)")

    # --- Without checkpointing ---
    print("\n  Without gradient checkpointing:")
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    output = model(batch)
    loss = output.sum()
    loss.backward()
    t_no_ckpt = time.perf_counter() - t0
    peak_no_ckpt = torch.cuda.max_memory_allocated() / 1e6 if HAS_CUDA else 0

    model.zero_grad(set_to_none=True)
    if HAS_CUDA:
        torch.cuda.empty_cache()

    print(f"    Time: {t_no_ckpt*1000:.1f} ms")
    if HAS_CUDA:
        print(f"    Peak GPU memory: {peak_no_ckpt:.1f} MB")

    # --- With checkpointing ---
    print("\n  With gradient checkpointing:")
    from torch.utils.checkpoint import checkpoint

    class CheckpointedTransformer(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, x):
            x = self.base.embed(x)
            for layer in self.base.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            return self.base.head(self.base.norm(x))

    ckpt_model = CheckpointedTransformer(model)

    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    output = ckpt_model(batch)
    loss = output.sum()
    loss.backward()
    t_ckpt = time.perf_counter() - t0
    peak_ckpt = torch.cuda.max_memory_allocated() / 1e6 if HAS_CUDA else 0

    print(f"    Time: {t_ckpt*1000:.1f} ms")
    if HAS_CUDA:
        print(f"    Peak GPU memory: {peak_ckpt:.1f} MB")
        savings_pct = (1 - peak_ckpt / peak_no_ckpt) * 100 if peak_no_ckpt > 0 else 0
        print(f"    Memory savings: {savings_pct:.1f}%")
    print(f"    Compute overhead: {((t_ckpt / t_no_ckpt) - 1) * 100:.0f}% slower")

    model.zero_grad(set_to_none=True)
    del ckpt_model
    if HAS_CUDA:
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────
# 2. Mixed Precision
# ─────────────────────────────────────────────────────────────────────

def demo_mixed_precision():
    print("\n" + "=" * 70)
    print("2. MIXED PRECISION — Halve Memory with bf16/fp16")
    print("=" * 70)

    configs = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]

    for label, dtype in configs:
        with torch.device("meta"):
            model = SmallTransformer(dim=512, num_heads=8, num_layers=12)

        total_bytes = sum(p.numel() * torch.tensor([], dtype=dtype).element_size()
                         for p in model.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}: {total_params:,} params × {torch.tensor([], dtype=dtype).element_size()} bytes = {total_bytes/1e6:.1f} MB")

    if HAS_CUDA:
        print("\n  GPU measurements:")
        for label, dtype in configs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = SmallTransformer(dim=512, num_heads=8, num_layers=12).to(dtype).cuda()
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"    {label} model on GPU: {mem:.1f} MB")
            del model
            torch.cuda.empty_cache()

    # autocast demo
    print("\n  torch.autocast context manager:")
    model_fp32 = SmallTransformer(dim=256, num_heads=8, num_layers=6)
    batch = make_batch(batch_size=4, seq_len=64)

    print(f"    Model dtype: {next(model_fp32.parameters()).dtype}")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model_fp32(batch)
        print(f"    Output dtype under autocast: {output.dtype}")
    print(f"    Output dtype without autocast: {model_fp32(batch).dtype}")

    del model_fp32


# ─────────────────────────────────────────────────────────────────────
# 3. Gradient Accumulation
# ─────────────────────────────────────────────────────────────────────

def demo_gradient_accumulation():
    print("\n" + "=" * 70)
    print("3. GRADIENT ACCUMULATION — Large Effective Batch, Small Memory")
    print("=" * 70)

    device = "cuda" if HAS_CUDA else "cpu"
    model = SmallTransformer(dim=256, num_heads=8, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    micro_batch_size = 4
    accumulation_steps = 8
    effective_batch_size = micro_batch_size * accumulation_steps

    print(f"\n  Micro batch size:    {micro_batch_size}")
    print(f"  Accumulation steps:  {accumulation_steps}")
    print(f"  Effective batch:     {effective_batch_size}")

    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad()
    for step in range(accumulation_steps):
        batch = make_batch(batch_size=micro_batch_size, seq_len=64, device=device)
        output = model(batch)
        loss = output.sum() / accumulation_steps
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if HAS_CUDA:
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  Peak memory with accumulation (micro_bs={micro_batch_size}): {peak:.1f} MB")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    big_batch = make_batch(batch_size=effective_batch_size, seq_len=64, device=device)
    output = model(big_batch)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if HAS_CUDA:
        peak_big = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak memory without accumulation (bs={effective_batch_size}): {peak_big:.1f} MB")
        print(f"  Memory ratio: {peak_big / peak:.1f}x")
    else:
        print(f"\n  (GPU not available — memory comparison skipped)")
        print(f"  Concept: {accumulation_steps} forward/backward passes with micro_bs={micro_batch_size}")
        print(f"  mathematically equivalent to single pass with bs={effective_batch_size}")
        print(f"  but uses {accumulation_steps}x less activation memory")

    del model, optimizer


# ─────────────────────────────────────────────────────────────────────
# 4. In-Place Operations
# ─────────────────────────────────────────────────────────────────────

def demo_inplace_operations():
    print("\n" + "=" * 70)
    print("4. IN-PLACE OPERATIONS — Avoid Allocating New Tensors")
    print("=" * 70)

    # Out-of-place vs in-place on non-leaf tensors
    print("\n  Out-of-place vs in-place (non-leaf, no grad):")
    x = torch.randn(1000, 1000)

    y = F.relu(x)
    print(f"    F.relu(x):            new tensor at {id(y)}, x unchanged at {id(x)}")
    assert id(x) != id(y)

    z = F.relu(x, inplace=True)
    print(f"    F.relu(x, inplace):   same tensor, id(z)==id(x): {id(z) == id(x)}")

    # In-place on leaf with requires_grad
    print("\n  In-place on leaf with requires_grad (autograd warning):")
    leaf = torch.randn(10, requires_grad=True)
    try:
        leaf.add_(1)
        print("    leaf.add_(1): succeeded (shouldn't for leaf requiring grad)")
    except RuntimeError as e:
        print(f"    leaf.add_(1): RuntimeError — {e}")

    # Safe in-place on intermediate (non-leaf) requiring grad
    print("\n  Safe in-place patterns:")
    a = torch.randn(10, requires_grad=True)
    b = a * 2  # b is non-leaf
    # b.relu_() would fail if b is needed for backward of a*2
    # but for standalone non-leaf nodes not saved for backward, it can work
    c = b.clone()
    c.relu_()
    print(f"    clone + relu_(): OK, avoids modifying tensors saved for backward")

    # Common safe in-place: zero_grad
    print("\n  Common safe in-place pattern: optimizer.zero_grad(set_to_none=True)")
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    out = model(torch.randn(5, 10)).sum()
    out.backward()
    print(f"    Before zero_grad: grad norm = {model.weight.grad.norm():.4f}")
    optimizer.zero_grad(set_to_none=True)
    print(f"    After zero_grad(set_to_none=True): grad is None = {model.weight.grad is None}")
    print(f"    set_to_none=True saves memory by not keeping zero tensors")


# ─────────────────────────────────────────────────────────────────────
# 5. del + gc.collect Pattern
# ─────────────────────────────────────────────────────────────────────

def demo_del_gc_pattern():
    print("\n" + "=" * 70)
    print("5. del + gc.collect() — Explicit Cleanup of Large Intermediates")
    print("=" * 70)

    device = "cuda" if HAS_CUDA else "cpu"

    if HAS_CUDA:
        torch.cuda.empty_cache()

    print(f"\n  Allocating large tensors on {device}...")
    large_a = torch.randn(5000, 5000, device=device)
    large_b = torch.randn(5000, 5000, device=device)
    result = large_a @ large_b

    mem_before = gpu_mem_mb() if HAS_CUDA else 0
    tensor_mb = large_a.numel() * large_a.element_size() / 1e6
    print(f"  Each tensor: {tensor_mb:.0f} MB ({device})")
    if HAS_CUDA:
        print(f"  GPU allocated: {mem_before:.0f} MB")

    print("\n  Deleting intermediates...")
    del large_a, large_b
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        mem_after_del = gpu_mem_mb()
        print(f"  GPU allocated after del+gc+empty_cache: {mem_after_del:.0f} MB")
        print(f"  Freed: {mem_before - mem_after_del:.0f} MB")

    del result
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        print(f"  GPU allocated after all cleanup: {gpu_mem_mb():.0f} MB")

    # Python reference counting subtlety
    print("\n  Reference counting subtlety:")
    x = torch.randn(100, 100, device=device)
    y = x  # y is an alias, not a copy
    del x
    gc.collect()
    print(f"  After 'del x' with alias y: tensor still alive = {y is not None}")
    del y
    gc.collect()
    print(f"  After 'del y': tensor now eligible for GC")


# ─────────────────────────────────────────────────────────────────────
# 6. GPU Memory Measurements (before/after each technique)
# ─────────────────────────────────────────────────────────────────────

def demo_gpu_before_after():
    if not HAS_CUDA:
        print("\n" + "=" * 70)
        print("6. GPU BEFORE/AFTER MEASUREMENTS — Skipped (no CUDA)")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("6. GPU BEFORE/AFTER MEASUREMENTS")
    print("=" * 70)

    dim, num_layers = 512, 8
    batch = make_batch(batch_size=16, seq_len=128, device="cuda")

    # Baseline: fp32, no checkpointing
    torch.cuda.empty_cache()
    model = SmallTransformer(dim=dim, num_heads=8, num_layers=num_layers).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    with track_gpu_memory("fp32 baseline"):
        out = model(batch)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    model.zero_grad(set_to_none=True)
    del model, optimizer
    torch.cuda.empty_cache()

    # bf16
    model = SmallTransformer(dim=dim, num_heads=8, num_layers=num_layers).cuda().bfloat16()
    optimizer = torch.optim.Adam(model.parameters())
    batch_bf16 = batch

    with track_gpu_memory("bf16"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch_bf16)
            loss = out.sum()
        loss.backward()
        optimizer.step()

    model.zero_grad(set_to_none=True)
    del model, optimizer
    torch.cuda.empty_cache()

    # Gradient checkpointing
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    model = SmallTransformer(dim=dim, num_heads=8, num_layers=num_layers).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    def forward_with_ckpt(model, x):
        x = model.embed(x)
        for layer in model.layers:
            x = ckpt_fn(layer, x, use_reentrant=False)
        return model.head(model.norm(x))

    with track_gpu_memory("gradient checkpointing"):
        out = forward_with_ckpt(model, batch)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    model.zero_grad(set_to_none=True)
    del model, optimizer
    torch.cuda.empty_cache()

    # SGD (less optimizer state)
    model = SmallTransformer(dim=dim, num_heads=8, num_layers=num_layers).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    with track_gpu_memory("SGD (less optimizer state)"):
        out = model(batch)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    model.zero_grad(set_to_none=True)
    del model, optimizer
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────
# 7. Memory leak detection demo
# ─────────────────────────────────────────────────────────────────────

def demo_memory_leak_detection():
    print("\n" + "=" * 70)
    print("7. MEMORY LEAK DETECTION PATTERNS")
    print("=" * 70)

    device = "cuda" if HAS_CUDA else "cpu"
    model = SmallTransformer(dim=128, num_heads=4, num_layers=2).to(device)

    # BAD: accumulating loss tensors (with computation graph)
    print("\n  BAD pattern: accumulating loss tensors with graph references")
    all_losses_bad = []
    for i in range(5):
        batch = make_batch(batch_size=4, seq_len=32, device=device)
        output = model(batch)
        loss = output.sum()
        all_losses_bad.append(loss)  # keeps computation graph alive!

    live_grads = sum(1 for l in all_losses_bad if l.grad_fn is not None)
    print(f"    Stored {len(all_losses_bad)} losses, {live_grads} still have grad_fn (graph reference)")

    del all_losses_bad
    gc.collect()

    # GOOD: storing only scalar values
    print("\n  GOOD pattern: store .item() or .detach()")
    all_losses_good = []
    for i in range(5):
        batch = make_batch(batch_size=4, seq_len=32, device=device)
        output = model(batch)
        loss = output.sum()
        all_losses_good.append(loss.item())  # scalar, no graph

    print(f"    Stored {len(all_losses_good)} scalar values: {all_losses_good}")
    print(f"    No graph references held")

    # Monitoring pattern
    print("\n  Monitoring pattern for leak detection:")
    if HAS_CUDA:
        torch.cuda.empty_cache()

    readings = []
    for step in range(10):
        batch = make_batch(batch_size=4, seq_len=32, device=device)
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
        if HAS_CUDA:
            torch.cuda.synchronize()
            readings.append(torch.cuda.memory_allocated())
        else:
            readings.append(0)

    if HAS_CUDA:
        growth = readings[-1] - readings[0]
        status = "LEAK DETECTED" if growth > readings[0] * 0.1 else "No leak"
        print(f"    Memory start: {readings[0]/1e6:.1f} MB, end: {readings[-1]/1e6:.1f} MB → {status}")
    else:
        print(f"    (Run on GPU to see memory readings; leak detection pattern demonstrated)")

    del model


# ─────────────────────────────────────────────────────────────────────
# 8. Comprehensive comparison table
# ─────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 70)
    print("MEMORY OPTIMIZATION SUMMARY")
    print("=" * 70)
    print("""
┌───────────────────────────┬──────────────────────┬─────────────────────┬───────────────────┐
│ Technique                 │ What it saves        │ Cost                │ Typical saving    │
├───────────────────────────┼──────────────────────┼─────────────────────┼───────────────────┤
│ Gradient checkpointing    │ Activations          │ ~33% more compute   │ 60-70% act. mem.  │
│ Mixed precision (bf16)    │ Params + activations │ None (often better) │ ~50%              │
│ Gradient accumulation     │ Activations          │ None                │ ∝ accum steps     │
│ In-place operations       │ Temporaries          │ Autograd limits     │ 5-15%             │
│ del + gc.collect          │ Named intermediates  │ Manual effort       │ Variable          │
│ empty_cache (between)     │ Cached blocks        │ Realloc overhead    │ Variable          │
│ CPU offloading (FSDP2)    │ Params + optimizer   │ Transfer overhead   │ Up to 90% GPU     │
│ 8-bit optimizers          │ Optimizer state      │ Accuracy impact     │ 75% optim mem.    │
│ Flash Attention           │ Attention scores     │ None                │ O(N) vs O(N²)     │
│ LoRA / adapter tuning     │ Everything except    │ Frozen backbone     │ 90%+ total        │
│                           │ adapter weights      │                     │                   │
└───────────────────────────┴──────────────────────┴─────────────────────┴───────────────────┘
""")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      Memory Optimization Techniques — Every Trick in the Book      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\nCUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    demo_gradient_checkpointing()
    demo_mixed_precision()
    demo_gradient_accumulation()
    demo_inplace_operations()
    demo_del_gc_pattern()
    demo_gpu_before_after()
    demo_memory_leak_detection()
    print_summary()

    print("\nDone! All optimization techniques demonstrated.")
