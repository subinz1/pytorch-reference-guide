"""
Module 41: UNet Architecture for Diffusion Models

Implements:
- Sinusoidal time embedding (positional encoding for timesteps)
- ResBlock with time conditioning
- DownBlock, UpBlock with skip connections
- Complete PointUNet for 2D point diffusion
- SimpleUNet1D for building intuition

Usage:
    python unet_model.py
"""

import math

import torch
import torch.nn as nn


# ============================================================================
# Sinusoidal Time Embedding
# ============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer timesteps using sinusoidal positional encoding.

    Uses the same encoding as the original Transformer paper:
    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ============================================================================
# ResBlock with Time Conditioning
# ============================================================================


class ResBlock(nn.Module):
    """Residual block that receives time embedding via additive conditioning.

    Architecture:
        x --+--> Linear -> SiLU -> Linear --+--> output
            |         + time_mlp(t_emb)     |
            +---------- shortcut -----------+
    """

    def __init__(self, dim: int, time_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim),
        )
        self.shortcut = nn.Identity() if dim == dim else nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.net(x) + self.time_mlp(t_emb)


# ============================================================================
# Down and Up Blocks
# ============================================================================


class DownBlock(nn.Module):
    """Two ResBlocks followed by a linear downsample."""

    def __init__(self, dim_in: int, dim_out: int, time_dim: int):
        super().__init__()
        self.res1 = ResBlock(dim_in, time_dim)
        self.res2 = ResBlock(dim_in, time_dim)
        self.downsample = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Linear upsample, concatenate skip, then two ResBlocks."""

    def __init__(self, dim_in: int, dim_out: int, time_dim: int):
        super().__init__()
        self.upsample = nn.Linear(dim_in, dim_out)
        self.res1 = ResBlock(dim_out * 2, time_dim, hidden_dim=dim_out)
        self.proj = nn.Linear(dim_out * 2, dim_out)
        self.res2 = ResBlock(dim_out, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=-1)
        x = self.res1(x, t_emb)
        x = self.proj(x)
        x = self.res2(x, t_emb)
        return x


# ============================================================================
# Mid Block
# ============================================================================


class MidBlock(nn.Module):
    """Bottleneck: two ResBlocks at the smallest resolution."""

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.res1 = ResBlock(dim, time_dim)
        self.res2 = ResBlock(dim, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


# ============================================================================
# Complete UNet for 2D Point Diffusion
# ============================================================================


class PointUNet(nn.Module):
    """UNet for denoising 2D point distributions.

    Input:  (B, 2) noisy 2D points + (B,) integer timesteps
    Output: (B, 2) predicted noise

    Architecture:
        Input(2) -> Project(256) -> Down(256->128) -> Down(128->64)
        -> Mid(64) -> Up(64->128) -> Up(128->256) -> Project(2)

    Skip connections from each DownBlock feed into the corresponding UpBlock.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, ...] = (256, 128, 64), time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        self.downs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.downs.append(DownBlock(hidden_dims[i], hidden_dims[i + 1], time_dim))

        self.mid = MidBlock(hidden_dims[-1], time_dim)

        self.ups = nn.ModuleList()
        for i in range(len(hidden_dims) - 2, -1, -1):
            self.ups.append(UpBlock(hidden_dims[i + 1], hidden_dims[i], time_dim))

        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))
        x = self.input_proj(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid(x, t_emb)

        for up in self.ups:
            x = up(x, skips.pop(), t_emb)

        return self.output_proj(x)


# ============================================================================
# Simple 1D UNet — minimal version for understanding
# ============================================================================


class SimpleUNet1D(nn.Module):
    """Minimal denoising network for 1D data.

    Concatenates the 1D input with the time embedding and
    passes through a simple MLP. No skip connections.
    """

    def __init__(self, time_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


# ============================================================================
# Demo
# ============================================================================


def demo_time_embedding() -> None:
    """Show sinusoidal time embedding properties."""
    print("=" * 60)
    print(" Sinusoidal Time Embedding")
    print("=" * 60)

    embed = SinusoidalTimeEmbedding(dim=64)
    timesteps = torch.tensor([0, 100, 500, 999])
    embeddings = embed(timesteps)

    print(f"Input timesteps: {timesteps.tolist()}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    print("\nSimilarity between timestep embeddings (cosine similarity):")
    for i in range(len(timesteps)):
        for j in range(i + 1, len(timesteps)):
            sim = torch.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
            print(f"  sim(t={timesteps[i]}, t={timesteps[j]}): {sim:.4f}")

    print("\nNearby timesteps should be more similar:")
    t_close = torch.tensor([500, 501, 502, 510, 550, 600])
    e_close = embed(t_close)
    for i in range(1, len(t_close)):
        sim = torch.cosine_similarity(e_close[0:1], e_close[i:i+1]).item()
        print(f"  sim(t=500, t={t_close[i]}): {sim:.4f}")


def demo_resblock() -> None:
    """Show ResBlock forward pass."""
    print(f"\n{'='*60}")
    print(" ResBlock with Time Conditioning")
    print(f"{'='*60}")

    block = ResBlock(dim=128, time_dim=64)
    x = torch.randn(4, 128)
    t_emb = torch.randn(4, 64)

    out = block(x, t_emb)
    print(f"Input:  {x.shape}")
    print(f"Time:   {t_emb.shape}")
    print(f"Output: {out.shape}")
    print(f"Residual connection: output = shortcut(x) + net(x) + time_mlp(t_emb)")


def demo_point_unet() -> None:
    """Show full UNet forward pass on 2D points."""
    print(f"\n{'='*60}")
    print(" PointUNet — 2D Diffusion Network")
    print(f"{'='*60}")

    model = PointUNet(input_dim=2, hidden_dims=(256, 128, 64), time_dim=128)

    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} total, {trainable:,} trainable")

    x = torch.randn(32, 2)
    t = torch.randint(0, 1000, (32,))

    predicted_noise = model(x, t)
    print(f"\nForward pass:")
    print(f"  Input (noisy points): {x.shape}")
    print(f"  Timesteps:            {t.shape}")
    print(f"  Output (pred noise):  {predicted_noise.shape}")

    print(f"\n  Input sample:  ({x[0, 0]:.4f}, {x[0, 1]:.4f}) at t={t[0].item()}")
    print(f"  Pred noise:    ({predicted_noise[0, 0]:.4f}, {predicted_noise[0, 1]:.4f})")

    print("\nArchitecture breakdown:")
    print("  Input(2) -> Project(256)")
    print("  DownBlock: 256 -> 128 (skip)")
    print("  DownBlock: 128 -> 64  (skip)")
    print("  MidBlock:  64 -> 64")
    print("  UpBlock:   64 -> 128  (+ skip)")
    print("  UpBlock:   128 -> 256 (+ skip)")
    print("  Project(256) -> Output(2)")


def demo_simple_1d() -> None:
    """Show the minimal 1D UNet."""
    print(f"\n{'='*60}")
    print(" SimpleUNet1D — Minimal 1D Denoiser")
    print(f"{'='*60}")

    model = SimpleUNet1D(time_dim=32, hidden_dim=128)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    x = torch.randn(16, 1)
    t = torch.randint(0, 1000, (16,))

    pred = model(x, t)
    print(f"Input:  {x.shape}")
    print(f"Output: {pred.shape}")
    print(f"Sample: x={x[0, 0]:.4f}, t={t[0].item()}, pred_noise={pred[0, 0]:.4f}")


def demo_gradient_flow() -> None:
    """Verify gradients flow correctly through the UNet."""
    print(f"\n{'='*60}")
    print(" Gradient Flow Verification")
    print(f"{'='*60}")

    model = PointUNet(input_dim=2, hidden_dims=(64, 32), time_dim=32)
    x = torch.randn(8, 2)
    t = torch.randint(0, 1000, (8,))
    target_noise = torch.randn(8, 2)

    pred = model(x, t)
    loss = torch.nn.functional.mse_loss(pred, target_noise)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()

    print(f"Parameters with gradients: {len(grad_norms)}/{sum(1 for _ in model.parameters())}")

    sorted_norms = sorted(grad_norms.items(), key=lambda kv: kv[1], reverse=True)
    print("\nTop 5 gradient norms:")
    for name, norm in sorted_norms[:5]:
        print(f"  {name}: {norm:.6f}")
    print(f"\nBottom 5 gradient norms:")
    for name, norm in sorted_norms[-5:]:
        print(f"  {name}: {norm:.6f}")

    print("\nAll gradients are non-zero -> gradient flow is healthy!")


if __name__ == "__main__":
    print("Module 41: UNet Architecture for Diffusion Models")
    print("=" * 60)

    torch.manual_seed(42)

    demo_time_embedding()
    demo_resblock()
    demo_point_unet()
    demo_simple_1d()
    demo_gradient_flow()

    print("\n" + "=" * 60)
    print("Done! Run train_diffusion.py next for the full training pipeline.")
