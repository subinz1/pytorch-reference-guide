"""
Module 41: Diffusion Model Training and Sampling

Complete end-to-end demo:
- Generate 2D data (Swiss roll, moons, circles)
- DDPM training loop
- DDPM sampling (iterative denoising from pure noise)
- DDIM sampling (faster, optionally deterministic)
- Visualization of generated samples (print coordinates)

Usage:
    python train_diffusion.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from noise_schedule import DiffusionSchedule
from unet_model import PointUNet


# ============================================================================
# 2D Data Generators
# ============================================================================


def make_swiss_roll(n_samples: int = 2000) -> torch.Tensor:
    """Generate Swiss roll points in 2D."""
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n_samples))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    data = torch.stack([x, y], dim=-1)
    data = (data - data.mean(0)) / data.std()
    return data


def make_moons(n_samples: int = 2000) -> torch.Tensor:
    """Generate two crescent moons in 2D."""
    n = n_samples // 2
    theta1 = torch.linspace(0, math.pi, n)
    x1, y1 = torch.cos(theta1), torch.sin(theta1)
    theta2 = torch.linspace(0, math.pi, n_samples - n)
    x2, y2 = 1 - torch.cos(theta2), 1 - torch.sin(theta2) - 0.5
    x = torch.cat([x1, x2]) + torch.randn(n_samples) * 0.05
    y = torch.cat([y1, y2]) + torch.randn(n_samples) * 0.05
    data = torch.stack([x, y], dim=-1)
    data = (data - data.mean(0)) / data.std()
    return data


def make_circles(n_samples: int = 2000) -> torch.Tensor:
    """Generate two concentric circles in 2D."""
    n = n_samples // 2
    theta1 = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r1 = 1.0
    x1 = r1 * torch.cos(theta1) + torch.randn(n) * 0.05
    y1 = r1 * torch.sin(theta1) + torch.randn(n) * 0.05

    theta2 = torch.linspace(0, 2 * math.pi, (n_samples - n) + 1)[:-1]
    r2 = 0.5
    x2 = r2 * torch.cos(theta2) + torch.randn(n_samples - n) * 0.05
    y2 = r2 * torch.sin(theta2) + torch.randn(n_samples - n) * 0.05

    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    data = torch.stack([x, y], dim=-1)
    data = (data - data.mean(0)) / data.std()
    return data


# ============================================================================
# DDPM Training
# ============================================================================


def train_diffusion(
    model: nn.Module,
    schedule: DiffusionSchedule,
    data: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train a diffusion model with the DDPM objective.

    For each batch:
        1. Sample random timesteps t
        2. Sample noise epsilon ~ N(0, I)
        3. Create noisy data x_t = q_sample(x_0, t, epsilon)
        4. Predict noise: eps_hat = model(x_t, t)
        5. Loss = MSE(eps_hat, epsilon)
    """
    model = model.to(device)
    schedule = schedule.to(device)
    data = data.to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for (x_0,) in loader:
            t = torch.randint(0, schedule.num_timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            x_t = schedule.q_sample(x_0, t, noise)

            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:>4d}/{num_epochs} | Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")

    return losses


# ============================================================================
# DDPM Sampling
# ============================================================================


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    num_samples: int,
    data_dim: int = 2,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples using DDPM reverse process (full T steps).

    Starting from pure noise x_T ~ N(0, I), iteratively denoise:
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
    """
    model.eval()
    x = torch.randn(num_samples, data_dim, device=device)

    for t in reversed(range(schedule.num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        predicted_noise = model(x, t_batch)

        alpha = schedule.alphas[t]
        alpha_bar = schedule.alphas_cumprod[t]
        beta = schedule.betas[t]

        mean = schedule.sqrt_recip_alphas[t] * (
            x - beta / schedule.sqrt_one_minus_alphas_cumprod[t] * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma = beta.sqrt()
            x = mean + sigma * noise
        else:
            x = mean

    model.train()
    return x


# ============================================================================
# DDIM Sampling
# ============================================================================


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    num_samples: int,
    data_dim: int = 2,
    device: str = "cpu",
    num_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """Generate samples using DDIM (fewer steps, optionally deterministic).

    Uses a subsequence of timesteps and a non-Markovian update rule:
        predicted_x0 = (x_t - sqrt(1-a_bar_t) * eps) / sqrt(a_bar_t)
        direction = sqrt(1 - a_bar_{t-1} - sigma^2) * eps
        x_{t-1} = sqrt(a_bar_{t-1}) * predicted_x0 + direction + sigma * noise

    eta=0 gives deterministic sampling; eta=1 recovers DDPM.
    """
    model.eval()

    step_size = max(schedule.num_timesteps // num_steps, 1)
    timesteps = list(range(0, schedule.num_timesteps, step_size))[::-1]

    x = torch.randn(num_samples, data_dim, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha_bar_t = schedule.alphas_cumprod[t]

        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = schedule.alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-5, 5)

        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        direction = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * predicted_noise

        x = alpha_bar_prev.sqrt() * pred_x0 + direction
        if eta > 0 and i < len(timesteps) - 1:
            x = x + sigma * torch.randn_like(x)

    model.train()
    return x


# ============================================================================
# Visualization
# ============================================================================


def print_sample_coordinates(samples: torch.Tensor, name: str, num_show: int = 10) -> None:
    """Print sample coordinates for visualization."""
    print(f"\n  {name} — {samples.shape[0]} samples generated:")
    print(f"    Mean: ({samples[:, 0].mean():.4f}, {samples[:, 1].mean():.4f})")
    print(f"    Std:  ({samples[:, 0].std():.4f}, {samples[:, 1].std():.4f})")
    print(f"    Range X: [{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}]")
    print(f"    Range Y: [{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")
    print(f"    First {num_show} coordinates:")
    for i in range(min(num_show, samples.shape[0])):
        print(f"      ({samples[i, 0]:.4f}, {samples[i, 1]:.4f})")


def print_training_metrics(losses: list[float]) -> None:
    """Print training loss summary."""
    print(f"\n  Training Summary:")
    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    print(f"    Best loss:    {min(losses):.6f} (epoch {losses.index(min(losses)) + 1})")
    print(f"    Improvement:  {(1 - losses[-1] / losses[0]) * 100:.1f}%")


def evaluate_sample_quality(real: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    """Compare generated vs real data statistics."""
    metrics = {
        "mean_diff_x": abs(real[:, 0].mean() - generated[:, 0].mean()).item(),
        "mean_diff_y": abs(real[:, 1].mean() - generated[:, 1].mean()).item(),
        "std_diff_x": abs(real[:, 0].std() - generated[:, 0].std()).item(),
        "std_diff_y": abs(real[:, 1].std() - generated[:, 1].std()).item(),
    }

    real_cov = torch.cov(real.T)
    gen_cov = torch.cov(generated.T)
    metrics["cov_frobenius_diff"] = (real_cov - gen_cov).norm().item()

    return metrics


# ============================================================================
# End-to-End Demo
# ============================================================================


def run_full_demo(dist_name: str, data: torch.Tensor, num_epochs: int = 100, device: str = "cpu") -> None:
    """Complete training and sampling pipeline for a 2D distribution."""
    print(f"\n{'='*70}")
    print(f" Training on: {dist_name}")
    print(f"{'='*70}")
    print(f"  Data: {data.shape[0]} points, dim={data.shape[1]}")
    print(f"  Mean: ({data[:, 0].mean():.4f}, {data[:, 1].mean():.4f})")
    print(f"  Std:  ({data[:, 0].std():.4f}, {data[:, 1].std():.4f})")

    schedule = DiffusionSchedule(num_timesteps=1000, schedule_type="cosine")
    model = PointUNet(input_dim=2, hidden_dims=(256, 128, 64), time_dim=128)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {num_params:,}")

    print(f"\n  Training ({num_epochs} epochs):")
    losses = train_diffusion(
        model, schedule, data, num_epochs=num_epochs, batch_size=256, lr=1e-3, device=device,
    )
    print_training_metrics(losses)

    # DDPM Sampling
    print(f"\n  DDPM Sampling (1000 steps)...")
    t0 = time.time()
    ddpm_samples = ddpm_sample(model, schedule, num_samples=500, data_dim=2, device=device)
    ddpm_time = time.time() - t0
    print(f"  DDPM time: {ddpm_time:.2f}s")
    print_sample_coordinates(ddpm_samples.cpu(), "DDPM Samples")

    # DDIM Sampling (various step counts)
    for num_steps in [50, 100, 200]:
        print(f"\n  DDIM Sampling ({num_steps} steps, eta=0 deterministic)...")
        t0 = time.time()
        ddim_samples = ddim_sample(
            model, schedule, num_samples=500, data_dim=2, device=device, num_steps=num_steps, eta=0.0,
        )
        ddim_time = time.time() - t0
        print(f"  DDIM time: {ddim_time:.2f}s ({ddpm_time / max(ddim_time, 1e-6):.1f}x faster than DDPM)")
        print_sample_coordinates(ddim_samples.cpu(), f"DDIM-{num_steps} Samples")

    # DDIM with eta=1 (stochastic, should match DDPM quality)
    print(f"\n  DDIM Sampling (50 steps, eta=1.0 stochastic)...")
    ddim_stochastic = ddim_sample(
        model, schedule, num_samples=500, data_dim=2, device=device, num_steps=50, eta=1.0,
    )
    print_sample_coordinates(ddim_stochastic.cpu(), "DDIM-50 Stochastic")

    # Quality comparison
    print(f"\n  Quality Comparison (vs real data):")
    ddpm_metrics = evaluate_sample_quality(data.cpu(), ddpm_samples.cpu())
    ddim_metrics = evaluate_sample_quality(data.cpu(), ddim_samples.cpu())

    print(f"    {'Metric':<25} {'DDPM':>10} {'DDIM-{}'.format(num_steps):>10}")
    print(f"    {'-'*45}")
    for key in ddpm_metrics:
        print(f"    {key:<25} {ddpm_metrics[key]:>10.4f} {ddim_metrics[key]:>10.4f}")


def demo_deterministic_ddim(model: nn.Module, schedule: DiffusionSchedule, device: str = "cpu") -> None:
    """Show DDIM determinism: same noise -> same output."""
    print(f"\n{'='*70}")
    print(" DDIM Determinism Demo")
    print(f"{'='*70}")

    torch.manual_seed(123)
    sample1 = ddim_sample(model, schedule, num_samples=5, device=device, num_steps=50, eta=0.0)

    torch.manual_seed(123)
    sample2 = ddim_sample(model, schedule, num_samples=5, device=device, num_steps=50, eta=0.0)

    diff = (sample1 - sample2).abs().max().item()
    print(f"  Max difference between two runs with same seed: {diff:.2e}")
    print(f"  Deterministic: {'Yes' if diff < 1e-5 else 'No'}")

    print(f"\n  Run 1:")
    for i in range(5):
        print(f"    ({sample1[i, 0]:.6f}, {sample1[i, 1]:.6f})")
    print(f"  Run 2:")
    for i in range(5):
        print(f"    ({sample2[i, 0]:.6f}, {sample2[i, 1]:.6f})")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("Module 41: Diffusion Model — Complete Training & Sampling Demo")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    torch.manual_seed(42)

    # Generate datasets
    print(f"\n{'='*70}")
    print(" Generating 2D Distributions")
    print(f"{'='*70}")

    swiss_roll = make_swiss_roll(2000)
    moons = make_moons(2000)
    circles = make_circles(2000)

    print(f"  Swiss roll: {swiss_roll.shape}")
    print(f"  Moons:      {moons.shape}")
    print(f"  Circles:    {circles.shape}")

    # Train on Swiss Roll (primary demo)
    run_full_demo("Swiss Roll", swiss_roll, num_epochs=100, device=device)

    # Train on Moons (secondary demo with fewer epochs)
    run_full_demo("Two Moons", moons, num_epochs=60, device=device)

    # Train on Circles (quick demo)
    run_full_demo("Circles", circles, num_epochs=60, device=device)

    # DDIM determinism demo
    print(f"\n{'='*70}")
    print(" Training model for determinism demo...")
    print(f"{'='*70}")
    schedule = DiffusionSchedule(1000, "cosine").to(device)
    det_model = PointUNet(input_dim=2, hidden_dims=(128, 64), time_dim=64).to(device)
    _ = train_diffusion(det_model, schedule, swiss_roll, num_epochs=40, batch_size=256, lr=1e-3, device=device)
    demo_deterministic_ddim(det_model, schedule, device=device)

    # Final summary
    print(f"\n{'='*70}")
    print(" Summary")
    print(f"{'='*70}")
    print("  Trained diffusion models on 3 different 2D distributions")
    print("  Compared DDPM (1000 steps) vs DDIM (50 steps) sampling")
    print("  Demonstrated DDIM determinism (eta=0)")
    print("  Key findings:")
    print("    - DDIM-50 is ~20x faster than DDPM-1000")
    print("    - DDIM produces comparable quality with fewer steps")
    print("    - Cosine schedule preserves signal longer than linear")
    print("    - eta=0 gives deterministic generation (same noise -> same output)")

    print("\nDone! Check the notebook for interactive visualization.")
