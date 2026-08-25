"""
Case Study 2: Attention Implementation Scaling

Problem: Naive dot-product attention allocates O(n²) memory for the attention
matrix, making long sequences infeasible and slow.
Fix: Memory-efficient attention (chunked) and flash attention patterns.
Expected speedup: 4–8× for seq_len >= 512, with O(n) memory.
"""

import torch
import torch.nn.functional as F
import time
import math


# ============================================================
# BEFORE: Naive attention (O(n²) memory)
# ============================================================

def naive_attention(query, key, value):
    """Standard scaled dot-product attention with full materialization.

    Memory: O(batch * heads * seq_len²) for the attention matrix.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)


# ============================================================
# AFTER: Memory-efficient chunked attention
# ============================================================

def chunked_attention(query, key, value, chunk_size=64):
    """Process attention in chunks to reduce peak memory.

    Instead of materializing the full (seq_len × seq_len) attention matrix,
    compute it in chunks of size chunk_size.
    Memory: O(batch * heads * seq_len * chunk_size) — linear in seq_len.
    """
    batch, heads, seq_len, d_k = query.shape
    scale = 1.0 / math.sqrt(d_k)
    output = torch.zeros_like(query)

    for i in range(0, seq_len, chunk_size):
        q_chunk = query[:, :, i : i + chunk_size]
        chunk_len = q_chunk.size(2)

        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        output[:, :, i : i + chunk_len] = torch.matmul(attn_weights, value)

    return output


# ============================================================
# AFTER: Using PyTorch's built-in scaled_dot_product_attention
# ============================================================

def sdpa_attention(query, key, value):
    """PyTorch 2.0+ scaled_dot_product_attention with automatic backend selection.

    Automatically uses FlashAttention or Memory-Efficient Attention
    depending on hardware and input shapes.
    """
    return F.scaled_dot_product_attention(query, key, value)


# ============================================================
# AFTER: In-place softmax optimization
# ============================================================

def inplace_attention(query, key, value):
    """Attention with in-place operations to reduce memory allocations."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores.mul_(1.0 / math.sqrt(d_k))
    scores = F.softmax(scores, dim=-1)
    return torch.matmul(scores, value)


# ============================================================
# PROFILING & BENCHMARK
# ============================================================

def measure_memory_and_time(fn, query, key, value, n_runs=20, warmup=5):
    """Measure both peak memory and execution time."""
    device = query.device

    # Warmup
    for _ in range(warmup):
        _ = fn(query, key, value)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000  # ms

    if device.type == "cuda":
        peak_mem = (torch.cuda.max_memory_allocated() - baseline_mem) / 1024 / 1024  # MB
    else:
        peak_mem = 0.0

    return avg_time, peak_mem


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("--- Case 2: Attention Scaling ---\n")

    configs = [
        {"seq_len": 128, "batch": 4, "heads": 8, "d_k": 64},
        {"seq_len": 512, "batch": 4, "heads": 8, "d_k": 64},
        {"seq_len": 1024, "batch": 2, "heads": 8, "d_k": 64},
    ]

    implementations = [
        ("Naive (O(n²))", naive_attention),
        ("Chunked (chunk=64)", lambda q, k, v: chunked_attention(q, k, v, chunk_size=64)),
        ("In-place softmax", inplace_attention),
        ("SDPA (auto backend)", sdpa_attention),
    ]

    for cfg in configs:
        seq_len = cfg["seq_len"]
        batch = cfg["batch"]
        heads = cfg["heads"]
        d_k = cfg["d_k"]

        print(f"seq_len={seq_len}, batch={batch}, heads={heads}, d_k={d_k}")
        print(f"  Theoretical attn matrix: {batch * heads * seq_len * seq_len * 4 / 1024 / 1024:.1f} MB")
        print()

        query = torch.randn(batch, heads, seq_len, d_k, device=device)
        key = torch.randn(batch, heads, seq_len, d_k, device=device)
        value = torch.randn(batch, heads, seq_len, d_k, device=device)

        for name, fn in implementations:
            try:
                avg_time, peak_mem = measure_memory_and_time(fn, query, key, value)
                mem_str = f"{peak_mem:.1f} MB" if device.type == "cuda" else "N/A"
                print(f"  {name:25s} | {avg_time:7.3f} ms | mem: {mem_str}")
            except Exception as e:
                print(f"  {name:25s} | FAILED: {e}")

        print()

    # Correctness check
    print("--- Correctness Verification ---")
    q = torch.randn(1, 2, 32, 16, device=device)
    k = torch.randn(1, 2, 32, 16, device=device)
    v = torch.randn(1, 2, 32, 16, device=device)

    ref = naive_attention(q, k, v)
    for name, fn in implementations[1:]:
        out = fn(q, k, v)
        max_diff = (ref - out).abs().max().item()
        status = "✓" if max_diff < 1e-5 else f"✗ (diff={max_diff:.2e})"
        print(f"  {name:25s} vs naive: {status}")

    print("\n--- Key Takeaways ---")
    print("  • Naive attention: simple but O(n²) memory kills long sequences")
    print("  • Chunked: linear memory, moderate speedup, good for CPU/older GPUs")
    print("  • SDPA: best of all worlds — auto-selects FlashAttention when available")
    print("  • In-place: marginal gains but reduces allocator pressure")


if __name__ == "__main__":
    main()
