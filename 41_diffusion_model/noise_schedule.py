"""
Module 41: Noise Schedules for Diffusion Models

Implements linear and cosine beta schedules, the forward diffusion process
q(x_t | x_0), and visualization of noise levels across timesteps.

Usage:
    python noise_schedule.py
"""

import math

import torch
import torch.nn.functional as F


# ============================================================================
# Beta Schedules
# ============================================================================


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear interpolation from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal 2021).

    Designs alpha_bar_t to follow a cosine curve, distributing
    information destruction more evenly across timesteps.
    """
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# ============================================================================
# Diffusion Schedule — precomputes all derived quantities
# ============================================================================


class DiffusionSchedule:
    """Precomputes and stores all quantities needed for training and sampling.

    Attributes:
        betas: noise variance at each timestep
        alphas: 1 - betas
        alphas_cumprod: cumulative product of alphas (alpha_bar)
        sqrt_alphas_cumprod: sqrt(alpha_bar) for forward process mean
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) for forward process std
        sqrt_recip_alphas: 1/sqrt(alpha) for reverse process
        posterior_variance: variance of q(x_{t-1} | x_t, x_0)
    """

    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "cosine"):
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            self.betas = linear_beta_schedule(num_timesteps)
        elif schedule_type == "cosine":
            self.betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: sample x_t from q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def to(self, device: torch.device) -> "DiffusionSchedule":
        """Move all tensors to specified device."""
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


# ============================================================================
# Visualization helpers
# ============================================================================


def print_schedule_summary(schedule: DiffusionSchedule, name: str = "Schedule") -> None:
    """Print key values at representative timesteps."""
    print(f"\n{'='*60}")
    print(f" {name} Summary (T={schedule.num_timesteps})")
    print(f"{'='*60}")
    print(f"{'Timestep':>10} {'beta_t':>10} {'alpha_bar':>12} {'sqrt(a_bar)':>12} {'sqrt(1-a_bar)':>14}")
    print("-" * 60)

    check_points = [0, 50, 100, 250, 500, 750, 900, 999]
    check_points = [t for t in check_points if t < schedule.num_timesteps]

    for t in check_points:
        print(
            f"{t:>10d} {schedule.betas[t]:>10.6f} "
            f"{schedule.alphas_cumprod[t]:>12.6f} "
            f"{schedule.sqrt_alphas_cumprod[t]:>12.6f} "
            f"{schedule.sqrt_one_minus_alphas_cumprod[t]:>14.6f}"
        )

    print(f"\nbeta range: [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
    print(f"alpha_bar range: [{schedule.alphas_cumprod[-1]:.6f}, {schedule.alphas_cumprod[0]:.6f}]")


def visualize_forward_process(schedule: DiffusionSchedule, x_0: torch.Tensor) -> None:
    """Show how noise corrupts data at different timesteps."""
    print(f"\n{'='*60}")
    print(" Forward Process Visualization")
    print(f"{'='*60}")
    print(f"Data shape: {x_0.shape}, range: [{x_0.min():.2f}, {x_0.max():.2f}]")

    timesteps = [0, 100, 250, 500, 750, 999]
    timesteps = [t for t in timesteps if t < schedule.num_timesteps]

    noise = torch.randn_like(x_0)

    for t_val in timesteps:
        t = torch.tensor([t_val])
        x_t = schedule.q_sample(x_0, t, noise)
        signal_ratio = schedule.sqrt_alphas_cumprod[t_val].item()
        noise_ratio = schedule.sqrt_one_minus_alphas_cumprod[t_val].item()

        print(
            f"\nt={t_val:>4d}: signal={signal_ratio:.4f}, noise={noise_ratio:.4f}, "
            f"SNR={signal_ratio/max(noise_ratio, 1e-8):.4f}"
        )
        if x_0.dim() == 2 and x_0.shape[-1] == 2:
            for i in range(min(3, x_0.shape[0])):
                print(f"  point {i}: ({x_t[i, 0]:.3f}, {x_t[i, 1]:.3f})")


def compare_schedules() -> None:
    """Compare linear vs cosine schedules side by side."""
    print(f"\n{'='*60}")
    print(" Linear vs Cosine Schedule Comparison")
    print(f"{'='*60}")

    linear_sched = DiffusionSchedule(1000, "linear")
    cosine_sched = DiffusionSchedule(1000, "cosine")

    print(f"\n{'Timestep':>10} {'Linear a_bar':>14} {'Cosine a_bar':>14} {'Diff':>10}")
    print("-" * 50)

    for t in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]:
        l_val = linear_sched.alphas_cumprod[t].item()
        c_val = cosine_sched.alphas_cumprod[t].item()
        print(f"{t:>10d} {l_val:>14.6f} {c_val:>14.6f} {c_val - l_val:>10.6f}")

    t_half_linear = (linear_sched.alphas_cumprod - 0.5).abs().argmin().item()
    t_half_cosine = (cosine_sched.alphas_cumprod - 0.5).abs().argmin().item()
    print(f"\nTimestep where alpha_bar = 0.5:")
    print(f"  Linear: t={t_half_linear}")
    print(f"  Cosine: t={t_half_cosine}")
    print("  Cosine preserves signal longer, distributing destruction more evenly.")


# ============================================================================
# Demo: forward process on 2D data
# ============================================================================


def demo_forward_on_2d() -> None:
    """Demonstrate the forward process on a simple 2D distribution."""
    print(f"\n{'='*60}")
    print(" Forward Process on 2D Points")
    print(f"{'='*60}")

    torch.manual_seed(42)
    t_vals = 1.5 * math.pi * (1 + 2 * torch.rand(200))
    x = t_vals * torch.cos(t_vals)
    y = t_vals * torch.sin(t_vals)
    data = torch.stack([x, y], dim=-1)
    data = (data - data.mean(0)) / data.std()

    print(f"Swiss roll data: {data.shape}")
    print(f"  Mean: ({data[:, 0].mean():.4f}, {data[:, 1].mean():.4f})")
    print(f"  Std:  ({data[:, 0].std():.4f}, {data[:, 1].std():.4f})")

    schedule = DiffusionSchedule(1000, "cosine")

    print("\nForward noising at different timesteps:")
    for t_val in [0, 250, 500, 750, 999]:
        t = torch.full((data.shape[0],), t_val, dtype=torch.long)
        noise = torch.randn_like(data)
        x_t = schedule.q_sample(data, t, noise)
        print(f"  t={t_val:>4d}: mean=({x_t[:, 0].mean():.3f}, {x_t[:, 1].mean():.3f}), "
              f"std=({x_t[:, 0].std():.3f}, {x_t[:, 1].std():.3f})")

    print("\nAt t=999, data should look like standard Gaussian noise:")
    t = torch.full((data.shape[0],), 999, dtype=torch.long)
    x_t = schedule.q_sample(data, t, torch.randn_like(data))
    print(f"  Mean: ({x_t[:, 0].mean():.4f}, {x_t[:, 1].mean():.4f})  (should be ~0)")
    print(f"  Std:  ({x_t[:, 0].std():.4f}, {x_t[:, 1].std():.4f})  (should be ~1)")

    print("\nSample coordinates at t=500:")
    t = torch.full((data.shape[0],), 500, dtype=torch.long)
    x_t = schedule.q_sample(data, t, torch.randn_like(data))
    for i in range(5):
        print(f"  Original: ({data[i, 0]:.3f}, {data[i, 1]:.3f}) -> "
              f"Noised: ({x_t[i, 0]:.3f}, {x_t[i, 1]:.3f})")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("Module 41: Noise Schedules for Diffusion Models")
    print("=" * 60)

    linear_schedule = DiffusionSchedule(1000, "linear")
    print_schedule_summary(linear_schedule, "Linear Schedule")

    cosine_schedule = DiffusionSchedule(1000, "cosine")
    print_schedule_summary(cosine_schedule, "Cosine Schedule")

    compare_schedules()

    demo_forward_on_2d()

    print("\n" + "=" * 60)
    print(" Custom Schedule Lengths")
    print("=" * 60)
    for T in [100, 500, 1000, 2000]:
        sched = DiffusionSchedule(T, "cosine")
        print(f"  T={T:>5d}: alpha_bar[0]={sched.alphas_cumprod[0]:.6f}, "
              f"alpha_bar[-1]={sched.alphas_cumprod[-1]:.6f}")

    print("\nDone! Run unet_model.py next to build the denoising network.")
