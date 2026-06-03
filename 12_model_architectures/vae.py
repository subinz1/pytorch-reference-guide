"""
Variational Autoencoder (VAE) — Complete Implementation
=======================================================

Implements a VAE with:
- Convolutional encoder (image -> latent distribution)
- Reparameterization trick (differentiable sampling)
- Convolutional decoder (latent sample -> reconstructed image)
- ELBO loss (reconstruction + KL divergence)

Reference: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder that maps images to latent distribution parameters.

    Progressively downsamples the spatial dimensions while increasing channels,
    then projects to mean (mu) and log-variance (log_var) of the latent space.
    """

    def __init__(self, in_channels=1, hidden_dims=None, latent_dim=128):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*layers)

        # After encoding a 28x28 image with 4 stride-2 convs: 28->14->7->4->2
        # Flatten: 256 * 2 * 2 = 1024
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Convolutional decoder that maps latent samples back to images.

    Projects from the latent space to a spatial feature map, then progressively
    upsamples using transposed convolutions.
    """

    def __init__(self, out_channels=1, hidden_dims=None, latent_dim=128):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        # Project from latent space to spatial feature map
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4)
        self.initial_channels = hidden_dims[0]

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
            ))

        self.decoder = nn.Sequential(*layers)

        # Final layer: map to image channels with sigmoid for [0, 1] output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], hidden_dims[-1],
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.initial_channels, 2, 2)
        h = self.decoder(h)
        return self.final(h)


class VAE(nn.Module):
    """Variational Autoencoder combining encoder, reparameterization, and decoder.

    Training objective: maximize the Evidence Lower Bound (ELBO), which is
    equivalent to minimizing: reconstruction_loss + KL_divergence.
    """

    def __init__(self, in_channels=1, hidden_dims=None, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        decoder_hidden = list(reversed(hidden_dims)) if hidden_dims else None
        self.decoder = Decoder(in_channels, decoder_hidden, latent_dim)

    def reparameterize(self, mu, log_var):
        """The reparameterization trick: sample z = mu + sigma * epsilon.

        Instead of sampling z ~ N(mu, sigma^2) (which is non-differentiable),
        we sample epsilon ~ N(0, 1) and compute z = mu + sigma * epsilon.
        This makes z a deterministic, differentiable function of mu and sigma.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Args:
            x: input images, (batch, channels, height, width)
        Returns:
            x_recon: reconstructed images
            mu: mean of the latent distribution
            log_var: log-variance of the latent distribution
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def sample(self, num_samples, device="cpu"):
        """Generate new images by sampling from the prior N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)

    @staticmethod
    def loss_function(x_recon, x, mu, log_var, kl_weight=1.0):
        """ELBO loss = reconstruction loss + KL divergence.

        Reconstruction loss: binary cross-entropy between input and output.
        For continuous data, MSE can be used instead.

        KL divergence: closed-form for two Gaussians.
        KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Args:
            x_recon: reconstructed images from the decoder
            x: original input images
            mu: latent mean from encoder
            log_var: latent log-variance from encoder
            kl_weight: weight for KL term (beta-VAE uses values != 1)
        """
        # Crop reconstruction to match input size if needed
        x_recon = x_recon[:, :, :x.size(2), :x.size(3)]

        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

        # KL divergence (analytical formula for Gaussian)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    batch_size = 4
    in_channels = 1
    img_size = 28   # MNIST-like
    latent_dim = 32

    model = VAE(
        in_channels=in_channels,
        hidden_dims=[32, 64, 128, 256],
        latent_dim=latent_dim,
    )
    model.eval()

    print(f"VAE Model")
    print(f"  Parameters: {count_params(model):,}")
    print(f"  Latent dim: {latent_dim}")

    # Forward pass
    x = torch.rand(batch_size, in_channels, img_size, img_size)
    with torch.no_grad():
        x_recon, mu, log_var = model(x)

    print(f"\nForward pass:")
    print(f"  Input:            {list(x.shape)}")
    print(f"  Reconstruction:   {list(x_recon.shape)}")
    print(f"  Latent mu:        {list(mu.shape)}")
    print(f"  Latent log_var:   {list(log_var.shape)}")

    # Crop reconstruction to match input and compute loss
    total_loss, recon_loss, kl_loss = VAE.loss_function(x_recon, x, mu, log_var)
    print(f"\nLoss breakdown:")
    print(f"  Reconstruction:   {recon_loss.item():.2f}")
    print(f"  KL divergence:    {kl_loss.item():.2f}")
    print(f"  Total ELBO loss:  {total_loss.item():.2f}")

    # Sampling from the prior
    with torch.no_grad():
        samples = model.sample(8)
    print(f"\nSampled images: {list(samples.shape)}")
    print(f"  Pixel range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")

    # Reparameterization trick verification:
    # z should be differentiable w.r.t. mu and log_var
    mu_test = torch.randn(2, latent_dim, requires_grad=True)
    log_var_test = torch.randn(2, latent_dim, requires_grad=True)
    z = model.reparameterize(mu_test, log_var_test)
    z.sum().backward()
    print(f"\nReparameterization gradient check:")
    print(f"  mu.grad exists:      {mu_test.grad is not None}")
    print(f"  log_var.grad exists: {log_var_test.grad is not None}")

    # Beta-VAE: increasing kl_weight disentangles the latent space
    print("\nBeta-VAE loss comparison (different KL weights):")
    with torch.no_grad():
        for beta in [0.1, 1.0, 5.0, 10.0]:
            total, recon, kl = VAE.loss_function(x_recon, x, mu, log_var, kl_weight=beta)
            print(f"  beta={beta:5.1f}: total={total.item():8.1f}, "
                  f"recon={recon.item():8.1f}, kl={kl.item():8.1f}")

    print("\nVAE verified successfully!")
