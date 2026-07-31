# Module 41: Building a Diffusion Model — From Noise to Data

Build a complete diffusion model from scratch: noise schedules, UNet architecture with time conditioning, DDPM training, DDPM and DDIM sampling, and classifier-free guidance — all demonstrated on 2D distributions for intuitive visualization.

**No images required** — we train on 2D point distributions (Swiss roll, moons, circles) so you can visualize the entire diffusion process in 2D.

| Input | Output |
|-------|--------|
| `Pure Gaussian noise (1000 points × 2D)` | `Swiss roll distribution` |
| `Pure Gaussian noise (1000 points × 2D)` | `Two moons distribution` |
| `50 DDIM steps (vs 1000 DDPM)` | `Same quality, 20× faster` |

---

## Table of Contents

1. [Overview](#overview)
2. [What Are Diffusion Models?](#what-are-diffusion-models)
3. [Forward Process — Adding Noise](#forward-process--adding-noise)
4. [Noise Schedules](#noise-schedules)
5. [Reverse Process — Denoising](#reverse-process--denoising)
6. [UNet Architecture](#unet-architecture)
7. [DDPM Training](#ddpm-training)
8. [DDPM Sampling](#ddpm-sampling)
9. [DDIM Sampling](#ddim-sampling)
10. [Classifier-Free Guidance](#classifier-free-guidance)
11. [Training on 2D Distributions](#training-on-2d-distributions)
12. [Key Takeaways](#key-takeaways)

---

## Overview

Diffusion models learn to generate data by reversing a gradual noising process. They are the backbone of modern image generators (Stable Diffusion, DALL-E 3, Imagen) and have achieved state-of-the-art results in image, audio, and video generation.

```
Forward Process (fixed):     x_0 ──→ x_1 ──→ x_2 ──→ ··· ──→ x_T ~ N(0, I)
                              data    slightly noisy          pure noise

Reverse Process (learned):   x_T ──→ x_{T-1} ──→ ··· ──→ x_1 ──→ x_0
                              noise   slightly denoised       generated data
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DIFFUSION MODEL PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────┐  │
│  │   Noise   │    │   Forward    │    │   UNet     │    │  Training │  │
│  │ Schedule  │───▶│  Process     │───▶│  (denoise) │───▶│   Loop    │  │
│  │ (beta_t)  │    │  q(x_t|x_0) │    │  eps_theta │    │ MSE loss  │  │
│  └───────────┘    └──────────────┘    └────────────┘    └─────┬─────┘  │
│                                                               │         │
│  ┌───────────┐    ┌──────────────┐    ┌────────────┐    ┌─────▼─────┐  │
│  │ Generated │    │    DDIM      │    │    DDPM    │    │  Trained  │  │
│  │  Samples  │◀───│  Sampling    │◀───│  Sampling  │◀───│   Model   │  │
│  └───────────┘    └──────────────┘    └────────────┘    └───────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Files

| File | Lines | Description |
|------|-------|-------------|
| `noise_schedule.py` | 200+ | Linear/cosine beta schedules, forward diffusion, alpha_cumprod |
| `unet_model.py` | 300+ | Sinusoidal embeddings, ResBlocks, UNet with skip connections |
| `train_diffusion.py` | 300+ | 2D data generation, DDPM/DDIM training and sampling, visualization |

---

## What Are Diffusion Models?

Diffusion models are a class of generative models that learn a data distribution by:

1. **Defining a forward process** that gradually destroys data by adding Gaussian noise over T steps
2. **Learning a reverse process** (a neural network) that removes noise one step at a time

The key insight: destroying data is easy (just add noise); learning to reverse this destruction teaches the model what real data looks like.

### Comparison with Other Generative Models

| Model | Training | Sampling | Mode Coverage | Quality |
|-------|----------|----------|---------------|---------|
| **GANs** | Adversarial (unstable) | Single pass (fast) | Mode collapse risk | High |
| **VAEs** | ELBO (stable) | Single pass (fast) | Good coverage | Blurry |
| **Diffusion** | Denoising (stable) | Iterative (slow) | Excellent coverage | Highest |
| **Flow** | Exact likelihood | Single pass | Good coverage | High |

### Mathematical Foundation

A diffusion model defines a Markov chain of latent variables x_1, ..., x_T:

```
q(x_{1:T} | x_0) = prod_{t=1}^{T} q(x_t | x_{t-1})
```

The forward process adds Gaussian noise at each step:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

Where beta_t is a variance schedule that controls how much noise is added at step t.

---

## Forward Process — Adding Noise

### Step-by-Step Noising

At each timestep t, we add a small amount of Gaussian noise:

```
x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon
```

Where epsilon ~ N(0, I).

### Closed-Form Sampling at Arbitrary Timestep

A key property: we can sample x_t directly from x_0 without iterating through all intermediate steps:

```
alpha_t = 1 - beta_t
alpha_bar_t = prod_{s=1}^{t} alpha_s       (cumulative product)

q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

Or equivalently:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

This is crucial for efficient training — we can jump to any timestep t directly.

### Implementation

```python
def q_sample(x_0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Sample x_t from q(x_t | x_0) — the forward process."""
    sqrt_alpha = sqrt_alphas_cumprod[t].unsqueeze(-1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    return sqrt_alpha * x_0 + sqrt_one_minus * noise
```

### Noise Level Progression

```
t=0:     x_0 (clean data)           alpha_bar ≈ 1.0
t=250:   x_250 (slightly noisy)     alpha_bar ≈ 0.7
t=500:   x_500 (noisy)              alpha_bar ≈ 0.3
t=750:   x_750 (very noisy)         alpha_bar ≈ 0.05
t=1000:  x_T (pure noise)           alpha_bar ≈ 0.0
```

---

## Noise Schedules

The noise schedule {beta_1, ..., beta_T} controls how quickly data is destroyed. The choice of schedule significantly affects training and generation quality.

### Linear Schedule

The simplest schedule — linearly interpolate between beta_start and beta_end:

```
beta_t = beta_start + (beta_end - beta_start) * t / T
```

Typical values: beta_start = 0.0001, beta_end = 0.02, T = 1000.

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)
```

**Problem**: the linear schedule destroys information too quickly at the end. By t=600, most signal is already gone, wasting the remaining 400 steps.

### Cosine Schedule

Proposed in "Improved DDPM" (Nichol & Dhariwal, 2021). Designs alpha_bar_t to follow a cosine curve:

```
alpha_bar_t = f(t) / f(0)

where f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
```

The offset s = 0.008 prevents beta_t from being too small near t = 0.

```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos((steps / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)
```

### Linear vs Cosine Comparison

```
alpha_bar_t
    1.0 ┤
        │  ╲  cosine (gradual)
    0.8 ┤   ╲
        │    ╲╲
    0.6 ┤     ╲╲
        │      ╲ ╲
    0.4 ┤       ╲  ╲ linear (aggressive)
        │        ╲  ╲
    0.2 ┤         ╲  ╲
        │          ╲  ╲
    0.0 ┤           ╲──╲──
        └───────────────────▶ t
        0   200  400  600  800  1000
```

The cosine schedule distributes information destruction more evenly across timesteps, leading to better sample quality.

### Derived Quantities

From the beta schedule, we precompute all quantities needed for training and sampling:

```python
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
```

---

## Reverse Process — Denoising

### The Goal

Learn a model p_theta(x_{t-1} | x_t) that reverses the forward process:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

### Predicting Noise vs Predicting x_0

The model can parameterize the reverse process in several ways:

| Parameterization | Model predicts | Used by |
|-----------------|----------------|---------|
| **Epsilon (noise)** | eps_theta(x_t, t) | DDPM (default) |
| **x_0 (clean data)** | x_0_theta(x_t, t) | Some methods |
| **v (velocity)** | v_theta(x_t, t) | Progressive distillation |
| **Score** | score_theta(x_t, t) | Score matching |

We use epsilon prediction (the DDPM default). Given the predicted noise eps_theta, the mean of p_theta is:

```
mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
```

### Why Noise Prediction Works

The model's task is simple: "given a noisy version of the data and the noise level t, predict what noise was added." This is equivalent to learning the score function (gradient of log-density), which is theoretically well-justified.

---

## UNet Architecture

The denoising network takes a noisy sample x_t and a timestep t, and predicts the noise epsilon. For 2D point diffusion, we use a simplified UNet-style architecture.

### Sinusoidal Time Embedding

Timesteps are embedded using sinusoidal positional encoding (from the Transformer paper), which gives the model a smooth, continuous representation of the noise level:

```
PE(t, 2i)   = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

### ResBlock with Time Conditioning

Each residual block receives the time embedding, allowing the network to adapt its behavior based on the noise level:

```
x ──┬── Linear → SiLU → Linear ──┐
    │         + time_emb          │
    └──────── Shortcut ───────────┘
                   │
                 Output
```

```python
class ResBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim),
        )
        self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        return self.shortcut(x) + self.net(x) + self.time_mlp(t_emb)
```

### UNet Structure (for 2D Points)

For 2D point data, our UNet operates on feature vectors rather than spatial grids:

```
Input (B, 2)
  │
  ├─ Project to hidden dim                    → (B, 256)
  │
  ├─ DownBlock 1: ResBlock(256) + ResBlock(256) → (B, 256)  ─── skip_1
  ├─ Down: Linear(256 → 128)                    → (B, 128)
  ├─ DownBlock 2: ResBlock(128) + ResBlock(128) → (B, 128)  ─── skip_2
  ├─ Down: Linear(128 → 64)                     → (B, 64)
  │
  ├─ MidBlock: ResBlock(64) + ResBlock(64)      → (B, 64)
  │
  ├─ Up: Linear(64 → 128)                       → (B, 128)
  ├─ UpBlock 2: ResBlock(256) + ResBlock(128)   → (B, 128)  ← cat(skip_2)
  ├─ Up: Linear(128 → 256)                      → (B, 256)
  ├─ UpBlock 1: ResBlock(512) + ResBlock(256)   → (B, 256)  ← cat(skip_1)
  │
  ├─ Project to output dim                      → (B, 2)
  │
  Output: predicted noise (B, 2)
```

### Skip Connections

Skip connections from the downsampling path to the upsampling path are essential — they preserve fine-grained information that would be lost through the bottleneck:

```python
class PointUNet(nn.Module):
    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        x = self.input_proj(x)

        # Down path (save skips)
        skips = []
        for down_block, downsample in self.downs:
            x = down_block(x, t_emb)
            skips.append(x)
            x = downsample(x)

        # Mid
        x = self.mid(x, t_emb)

        # Up path (use skips)
        for up_block, upsample in self.ups:
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=-1)  # Skip connection
            x = up_block(x, t_emb)

        return self.output_proj(x)
```

### Simple 1D UNet for Understanding

To build intuition, we also provide a minimal 1D UNet that processes scalar inputs — the simplest possible diffusion setup:

```python
class SimpleUNet1D(nn.Module):
    """Minimal UNet for 1D data — useful for understanding the core idea."""
    def __init__(self, time_dim=32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))
```

---

## DDPM Training

### Training Algorithm

DDPM (Denoising Diffusion Probabilistic Models) training is remarkably simple:

```
repeat:
    1. Sample x_0 ~ data distribution
    2. Sample t ~ Uniform{1, ..., T}
    3. Sample epsilon ~ N(0, I)
    4. Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    5. Predict eps_theta = model(x_t, t)
    6. Loss = MSE(epsilon, eps_theta)
    7. Gradient step
```

### Implementation

```python
def train_step(model, optimizer, x_0, schedule):
    batch_size = x_0.shape[0]

    # Sample random timesteps
    t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=x_0.device)

    # Sample noise
    noise = torch.randn_like(x_0)

    # Forward process: add noise to get x_t
    x_t = schedule.q_sample(x_0, t, noise)

    # Predict noise
    predicted_noise = model(x_t, t)

    # MSE loss between true and predicted noise
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### Why MSE on Noise?

The simplified DDPM objective is:

```
L_simple = E_{t, x_0, eps} [ ||eps - eps_theta(x_t, t)||^2 ]
```

This is a simplified version of the variational lower bound (VLB). It works because:

1. Predicting noise is equivalent to predicting the score (gradient of log probability)
2. MSE on noise is equivalent to a reweighted VLB where all timesteps contribute equally
3. Empirically, this simple objective produces better samples than the full VLB

### Training Tips

| Tip | Reason |
|-----|--------|
| Use cosine schedule | Better noise distribution across timesteps |
| AdamW optimizer | Stable training with weight decay |
| Learning rate ~1e-3 for 2D, ~2e-4 for images | 2D data is simpler |
| Gradient clipping (max_norm=1.0) | Prevents training instability |
| EMA of model weights | Smoother, better-quality samples |

---

## DDPM Sampling

### Algorithm

DDPM sampling iterates from pure noise x_T back to clean data x_0:

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I)  if t > 1, else z = 0
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z
```

Where sigma_t = sqrt(beta_t) (the posterior standard deviation).

### Implementation

```python
@torch.no_grad()
def ddpm_sample(model, schedule, shape, device):
    x = torch.randn(shape, device=device)  # Start from pure noise

    for t in reversed(range(schedule.num_timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Compute mean
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alphas_cumprod[t]
        beta = schedule.betas[t]

        mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * predicted_noise)

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = beta.sqrt()
            x = mean + sigma * noise
        else:
            x = mean

    return x
```

### DDPM Sampling Properties

| Property | Value |
|----------|-------|
| Steps | T (typically 1000) |
| Stochastic? | Yes (random noise at each step) |
| Quality | Excellent |
| Speed | Slow (1000 forward passes) |
| Deterministic? | No (different noise = different samples) |

---

## DDIM Sampling

### Motivation

DDPM requires T steps (e.g., 1000) for sampling, which is slow. DDIM (Denoising Diffusion Implicit Models, Song et al. 2021) enables sampling with far fewer steps by using a non-Markovian process.

### Key Idea

DDIM defines a family of non-Markovian forward processes that all share the same marginal q(x_t | x_0) as DDPM. The reverse process can skip timesteps:

```
x_{t-1} = sqrt(alpha_bar_{t-1}) * predicted_x_0 + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * predicted_direction + sigma_t * noise
```

Where:
- `predicted_x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)`
- `predicted_direction = eps_theta` (pointing toward x_t)
- `sigma_t = 0` for deterministic sampling (eta = 0)

### Implementation

```python
@torch.no_grad()
def ddim_sample(model, schedule, shape, device, num_steps=50, eta=0.0):
    # Create subsequence of timesteps
    step_size = schedule.num_timesteps // num_steps
    timesteps = list(range(0, schedule.num_timesteps, step_size))[::-1]

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha_bar_t = schedule.alphas_cumprod[t]
        alpha_bar_prev = schedule.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)

        # Predict x_0
        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-3, 3)  # Clip for stability

        # Direction pointing to x_t
        direction = (1 - alpha_bar_prev).sqrt() * predicted_noise

        # Noise (eta=0 for deterministic)
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()

        x = alpha_bar_prev.sqrt() * pred_x0 + direction
        if eta > 0 and i < len(timesteps) - 1:
            x = x + sigma * torch.randn_like(x)

    return x
```

### DDPM vs DDIM Comparison

| Aspect | DDPM | DDIM |
|--------|------|------|
| Steps | 1000 | 50-100 (tunable) |
| Stochastic | Yes | Configurable (eta) |
| Deterministic mode | No | Yes (eta=0) |
| Sample quality at 50 steps | Poor | Good |
| Same noise → same output | No | Yes (when eta=0) |
| Speed | ~1000 forward passes | ~50 forward passes |

### Choosing the Number of Steps

```
Steps    Quality    Speed
  10     Poor       Very fast (20ms)
  25     Decent     Fast (50ms)
  50     Good       Moderate (100ms)
 100     Very good  Slow (200ms)
 250     Excellent  Very slow (500ms)
1000     Best       DDPM speed (2s)
```

---

## Classifier-Free Guidance

### Concept

Classifier-free guidance (Ho & Salimans, 2022) improves sample quality by steering generation toward a condition (e.g., class label) without needing a separate classifier.

### How It Works

During training, randomly drop the condition with some probability (e.g., 10%):

```
# Training: randomly use unconditional or conditional
if random() < 0.1:
    eps_theta = model(x_t, t, condition=None)    # Unconditional
else:
    eps_theta = model(x_t, t, condition=class_label)  # Conditional
```

During sampling, interpolate between conditional and unconditional predictions:

```
eps_guided = eps_unconditional + w * (eps_conditional - eps_unconditional)
```

Where w is the guidance scale:
- w = 1.0: no guidance (normal conditional generation)
- w > 1.0: stronger guidance (higher quality, less diversity)
- w = 7.5: common default for image generation

### Intuition

```
guidance strength w
       │
       │  w=1: balanced (diverse but sometimes off-target)
       │  w=3: moderate guidance (good quality, good diversity)
       │  w=7: strong guidance (high quality, less diversity)
       │  w=20: extreme (very sharp but repetitive)
       │
```

The model learns both what the data looks like (unconditional) and what data with a specific condition looks like (conditional). Guidance amplifies the difference, pushing samples more strongly toward the conditioned distribution.

---

## Training on 2D Distributions

### Why 2D?

Training on 2D point distributions lets you:
1. **Visualize the entire forward/reverse process** — scatter plots show noise being added and removed
2. **See mode coverage** — verify the model generates all modes of the distribution
3. **Fast iteration** — training takes seconds, not hours
4. **Understand failure modes** — easy to spot mode collapse, poor mixing, etc.

### Available Distributions

| Distribution | Shape | Characteristics |
|-------------|-------|-----------------|
| **Swiss Roll** | Spiral | Tests ability to learn curved manifolds |
| **Two Moons** | Two crescents | Tests multi-modal generation |
| **Circles** | Concentric rings | Tests ring-shaped distributions |

### Data Generation

```python
def make_swiss_roll(n_samples=1000):
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n_samples))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    data = torch.stack([x, y], dim=-1)
    data = data / data.std()  # Normalize
    return data

def make_moons(n_samples=1000):
    n = n_samples // 2
    # Upper moon
    theta1 = torch.linspace(0, math.pi, n)
    x1, y1 = torch.cos(theta1), torch.sin(theta1)
    # Lower moon
    theta2 = torch.linspace(0, math.pi, n_samples - n)
    x2, y2 = 1 - torch.cos(theta2), 1 - torch.sin(theta2) - 0.5
    x = torch.cat([x1, x2]) + torch.randn(n_samples) * 0.05
    y = torch.cat([y1, y2]) + torch.randn(n_samples) * 0.05
    data = torch.stack([x, y], dim=-1)
    data = (data - data.mean(0)) / data.std()
    return data
```

### Visualizing the Forward Process

```
t=0 (clean)        t=250             t=500            t=750           t=1000 (noise)
  Swiss roll        Blurred           Fuzzy blob       Nearly noise     Pure Gaussian
  ·····•••          ··· ···           · · · ·          · · · ·          · · · ·
 ·       ••         ··   ··           · ·   ·          ·  · ·           · ·  ·
 ·  •••  •         ··  ·  ·           ·  ·  ·          · · ·            · · ·
 · •   • •         ·  · · ·           ·  ·  ·          · ·  ·           ·  · ·
  ·•••••           ·· ··· ·           ·   · ·          ·· · ·           · · · ·
```

### Visualizing the Reverse Process (Sampling)

```
t=1000 (start)     t=750              t=500            t=250           t=0 (generated)
  Random noise       Structure          Rough shape      Refined          Swiss roll
  · · · ·           · · · ·            ·····•           ·····••          ·····•••
  · · · ·           · ·   ·           ·       •         ·       ••       ·       ••
  · · · ·           ·  ·  ·           ·  •••  •         ·  •••  •       ·  •••  •
  · ·  ·            · · ·             · •   • •         · •   • •       · •   • •
  · · · ·           · · · ·           ·•••••            ·•••••           ·•••••
```

---

## Key Takeaways

1. **Diffusion models learn by denoising** — the forward process adds noise; the model learns to reverse it, implicitly learning the data distribution
2. **Closed-form forward process is key** — q(x_t | x_0) lets us jump to any timestep directly, making training efficient (just MSE on predicted noise)
3. **The noise schedule matters** — cosine schedule distributes information destruction more evenly than linear, leading to better samples
4. **UNet with skip connections preserves detail** — the encoder-decoder structure with skip connections lets the model process at multiple resolutions
5. **Time embedding conditions the network** — sinusoidal embeddings give the model continuous knowledge of the current noise level
6. **DDPM is slow but high quality** — 1000 iterative denoising steps produce excellent samples but take time
7. **DDIM enables fast sampling** — by using a non-Markovian process, DDIM generates comparable samples in 50 steps (20x faster) and supports deterministic generation
8. **Classifier-free guidance trades diversity for quality** — by interpolating between conditional and unconditional predictions, guidance produces sharper, more targeted samples
9. **2D distributions build intuition** — visualizing diffusion on points before scaling to images reveals the core mechanics without GPU overhead

---

### Further Resources

- [Module 04 — Neural Networks](../04_neural_networks/) — `nn.Module`, layers, losses
- [Module 07 — Training Pipelines](../07_training/) — complete training loops, mixed precision
- [Module 12 — Model Architectures](../12_model_architectures/) — ResNet, VAE implementations
- [Module 40 — Image Classifier](../40_image_classifier/) — End-to-end CNN/ResNet project
- [DDPM Paper — Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- [Improved DDPM — Nichol & Dhariwal 2021](https://arxiv.org/abs/2102.09672)
- [DDIM — Song et al. 2021](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Guidance — Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)

---

### Upstream Updates (PyTorch 2.14+)

| Feature | Impact on Diffusion Models |
|---------|---------------------------|
| `torch.compile` | Compiles the denoising UNet for faster training and sampling |
| FlexAttention | Custom attention patterns for UNet self-attention layers |
| `torch.float8` | FP8 training for larger diffusion models |
| FSDP2 | Distributed training of billion-parameter diffusion models |
| `torch.export` | Export trained diffusion models for deployment |

---

<div align="center">

[← Previous Module (Image Classifier)](../40_image_classifier/) | [🏠 Home](../README.md) | [Next Module (RAG Pipeline) →](../42_rag_pipeline/)

**Notebook**: [`41_diffusion_model.ipynb`](../notebooks/41_diffusion_model.ipynb)

</div>
