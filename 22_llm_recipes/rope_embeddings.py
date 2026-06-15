"""
RoPE (Rotary Position Embeddings) — Complete Implementation

Demonstrates:
- Precomputing frequency-based complex exponentials (freqs_cis)
- Applying rotary embeddings to Q and K tensors
- How position information is encoded via rotation
- Comparison of attention scores with and without RoPE
"""

import torch
import torch.nn.functional as F
import math


# =============================================================================
# Core RoPE Implementation
# =============================================================================

def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Precompute complex exponentials for RoPE.

    For each position m and frequency band i:
        freqs_cis[m, i] = exp(j * m * theta_i)
    where theta_i = 10000^(-2i/dim)

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        theta: Base for geometric frequency series
        device: Target device

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    assert dim % 2 == 0, "RoPE requires even head dimension"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis for broadcasting with x: (batch, seq, heads, dim/2)."""
    ndim = x.ndim
    assert ndim >= 2
    shape = [1] * ndim
    shape[1] = freqs_cis.shape[0]  # seq dim
    shape[-1] = freqs_cis.shape[1]  # freq dim
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    The rotation is applied by:
    1. Viewing consecutive dimension pairs as complex numbers
    2. Multiplying by the precomputed complex exponentials
    3. Converting back to real representation

    Args:
        xq: Query tensor (batch, seq_len, n_heads, head_dim)
        xk: Key tensor (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: Precomputed complex exponentials (seq_len, head_dim // 2)

    Returns:
        Rotated (xq, xk) with same shapes and dtypes as inputs
    """
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis_q = reshape_for_broadcast(freqs_cis, xq_complex)
    freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_complex)

    xq_out = torch.view_as_real(xq_complex * freqs_cis_q).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis_k).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# Alternative: Real-valued implementation (no complex numbers)
# =============================================================================

def apply_rotary_emb_real(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings using real-valued cos/sin (no complex ops).

    This is mathematically equivalent to the complex version but works on
    platforms that don't support complex tensors well (e.g., some backends).

    x: (batch, seq, heads, dim)
    freqs_cos, freqs_sin: (seq, dim/2)
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, d)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def precompute_freqs_real(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Return (cos, sin) tensors for real-valued RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


# =============================================================================
# NTK-Aware Scaling for Length Extension
# =============================================================================

def precompute_freqs_cis_ntk(
    dim: int,
    max_seq_len: int,
    original_max_len: int = 2048,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    NTK-aware RoPE scaling for extending context length beyond training length.

    Instead of linearly interpolating positions (which compresses high-freq info),
    NTK scaling increases the base theta, effectively spreading the frequencies
    more evenly across the extended range.
    """
    if max_seq_len <= original_max_len:
        return precompute_freqs_cis(dim, max_seq_len, theta)

    scale = max_seq_len / original_max_len
    theta_scaled = theta * (scale ** (dim / (dim - 2)))

    return precompute_freqs_cis(dim, max_seq_len, theta_scaled)


# =============================================================================
# Demonstration and Testing
# =============================================================================

def demo_position_encoding():
    """Show how RoPE encodes position information into attention scores."""
    print("=" * 70)
    print("RoPE (Rotary Position Embeddings) Demonstration")
    print("=" * 70)

    batch_size = 1
    seq_len = 16
    n_heads = 4
    head_dim = 64

    torch.manual_seed(42)

    # Precompute frequencies
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    print(f"\nFreqs shape: {freqs_cis.shape}")
    print(f"  = (max_seq_len={seq_len}, head_dim//2={head_dim // 2})")

    # Create random Q and K
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)

    print(f"\nQ shape: {q.shape}")
    print(f"K shape: {k.shape}")

    # Apply RoPE
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

    print(f"\nQ after RoPE shape: {q_rope.shape} (same)")
    print(f"K after RoPE shape: {k_rope.shape} (same)")

    # Show that RoPE preserves norms
    q_norms_before = q.norm(dim=-1).mean()
    q_norms_after = q_rope.norm(dim=-1).mean()
    print(f"\nNorm preservation: before={q_norms_before:.4f}, after={q_norms_after:.4f}")
    print("  (RoPE is a rotation — norms are exactly preserved)")

    return q, k, q_rope, k_rope


def demo_relative_position():
    """Demonstrate that RoPE attention scores depend on relative position."""
    print("\n" + "=" * 70)
    print("Relative Position Property")
    print("=" * 70)

    head_dim = 32
    seq_len = 32
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    torch.manual_seed(123)
    q_vec = torch.randn(1, 1, 1, head_dim)  # Single query vector
    k_vec = torch.randn(1, 1, 1, head_dim)  # Single key vector

    # Place the same q at position 5, same k at positions 6, 7, 8, ...
    # The attention score q_5 · k_n should depend only on (n - 5)
    scores_by_distance = []
    q_pos = 5

    for distance in range(1, 10):
        k_pos = q_pos + distance

        q_at_pos = q_vec.clone()
        k_at_pos = k_vec.clone()

        # Apply RoPE for specific positions
        q_rotated, _ = apply_rotary_emb(q_at_pos, q_at_pos, freqs_cis[q_pos:q_pos+1])
        _, k_rotated = apply_rotary_emb(k_at_pos, k_at_pos, freqs_cis[k_pos:k_pos+1])

        score = (q_rotated * k_rotated).sum().item()
        scores_by_distance.append((distance, score))

    # Now shift both positions by 10 and verify scores are the same
    print("\nScores q·k at different relative distances:")
    print(f"{'Distance':<10} {'Score (pos 5)':<18} {'Score (pos 15)':<18} {'Match?'}")
    print("-" * 60)

    q_pos_shifted = 15
    for distance, original_score in scores_by_distance:
        k_pos_shifted = q_pos_shifted + distance

        q_rotated, _ = apply_rotary_emb(
            q_vec.clone(), q_vec.clone(), freqs_cis[q_pos_shifted:q_pos_shifted+1]
        )
        _, k_rotated = apply_rotary_emb(
            k_vec.clone(), k_vec.clone(), freqs_cis[k_pos_shifted:k_pos_shifted+1]
        )
        shifted_score = (q_rotated * k_rotated).sum().item()
        match = abs(original_score - shifted_score) < 1e-5
        print(f"{distance:<10} {original_score:<18.6f} {shifted_score:<18.6f} {'✓' if match else '✗'}")

    print("\nScores depend ONLY on relative distance — position-shift invariant.")


def demo_attention_with_without_rope():
    """Compare attention patterns with and without RoPE."""
    print("\n" + "=" * 70)
    print("Attention Scores: With vs Without RoPE")
    print("=" * 70)

    batch, seq_len, n_heads, head_dim = 1, 8, 1, 32
    torch.manual_seed(0)

    q = torch.randn(batch, seq_len, n_heads, head_dim)
    k = torch.randn(batch, seq_len, n_heads, head_dim)

    # Without RoPE
    scale = 1.0 / math.sqrt(head_dim)
    scores_no_rope = torch.einsum("bshd,bthd->bhst", q, k) * scale
    attn_no_rope = F.softmax(scores_no_rope, dim=-1)

    # With RoPE
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)
    scores_rope = torch.einsum("bshd,bthd->bhst", q_rope, k_rope) * scale
    attn_rope = F.softmax(scores_rope, dim=-1)

    print("\nAttention weights WITHOUT RoPE (first head, first 8 positions):")
    print("  (No position information — patterns are random)")
    for i in range(min(8, seq_len)):
        row = attn_no_rope[0, 0, i, :8].tolist()
        print(f"  pos {i}: [{', '.join(f'{v:.3f}' for v in row)}]")

    print("\nAttention weights WITH RoPE (first head, first 8 positions):")
    print("  (Position-aware — nearby tokens often attend more strongly)")
    for i in range(min(8, seq_len)):
        row = attn_rope[0, 0, i, :8].tolist()
        print(f"  pos {i}: [{', '.join(f'{v:.3f}' for v in row)}]")


def demo_frequency_bands():
    """Visualize the frequency bands used by RoPE."""
    print("\n" + "=" * 70)
    print("RoPE Frequency Bands")
    print("=" * 70)

    head_dim = 64
    theta = 10000.0

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    wavelengths = 2 * math.pi / freqs

    print(f"\nHead dim: {head_dim}, Base theta: {theta}")
    print(f"Number of frequency bands: {head_dim // 2}")
    print(f"\n{'Band':<6} {'Frequency':<15} {'Wavelength (tokens)':<22} {'Role'}")
    print("-" * 65)
    for i in range(0, head_dim // 2, 4):
        role = "high-freq (local)" if i < 8 else "mid-freq" if i < 24 else "low-freq (global)"
        print(f"{i:<6} {freqs[i]:<15.6f} {wavelengths[i]:<22.1f} {role}")

    print(f"\nLowest freq band: wavelength = {wavelengths[-1]:.0f} tokens")
    print(f"Highest freq band: wavelength = {wavelengths[0]:.1f} tokens")
    print("High-freq bands capture local patterns; low-freq bands capture global structure.")


if __name__ == "__main__":
    demo_position_encoding()
    demo_relative_position()
    demo_attention_with_without_rope()
    demo_frequency_bands()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
RoPE applies position information as rotations to Q and K vectors:
  1. Precompute frequency-dependent rotation angles: theta_i = 10000^(-2i/d)
  2. For position m: rotate dimension pair i by angle m * theta_i
  3. Attention score q_m · k_n depends only on relative position (m - n)

Advantages:
  - Encodes relative position naturally
  - No extra parameters (unlike learned embeddings)
  - Enables length generalization via frequency scaling (NTK, YaRN)
  - Works with KV cache (just apply correct position index)
""")
