"""
common/cvae.py
Class-Conditional Variational Autoencoder (CVAE) for 64x64 PPE safety crops.
Supports:
  - Class conditioning on encoder (spatial tile concatenation) and decoder (latent concatenation).
  - True ELBO calculation with sigma-VAE (calibrated decoder variance) and KL annealing.
  - Active units tracking (Var[mu] > 0.01) and per-dimension KL.
  - Spherical linear interpolation (SLERP) in latent space and continuous class morphing.
  - Temperature-controlled sampling (tau in [0.6, 1.0]).
  - Checkerboard-free decoder (Upsample + Conv3x3 + Residual + GroupNorm).
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import CLASS_NAMES


# ==============================================================================
# Building Blocks: Residual Blocks with GroupNorm
# ==============================================================================

class CVAEResBlock(nn.Module):
    """Residual Block with GroupNorm and SiLU activations for stable generative modeling."""
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.norm1(x))
        out = self.conv1(out)
        out = self.act(self.norm2(out))
        out = self.conv2(out)
        return out + res


# ==============================================================================
# Spherical Linear Interpolation (SLERP)
# ==============================================================================

def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two latent vectors z0 and z1.
    Traverses the high-probability Gaussian shell radius sqrt(K) rather than
    cutting through the zero-probability origin.
    """
    norm_z0 = z0 / (torch.norm(z0, dim=-1, keepdim=True) + 1e-8)
    norm_z1 = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-8)
    dot = torch.sum(norm_z0 * norm_z1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    scale0 = torch.sin((1.0 - alpha) * omega) / (sin_omega + 1e-8)
    scale1 = torch.sin(alpha * omega) / (sin_omega + 1e-8)
    return scale0 * z0 + scale1 * z1


# ==============================================================================
# Conditional VAE Architecture
# ==============================================================================

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for 64x64 PPE crops.
    
    Conditioning:
      - Encoder: Class embedding (16-d) spatially tiled to (16, 64, 64) and concatenated to RGB (19, 64, 64).
      - Decoder: Class embedding (32-d) concatenated directly to latent vector z (K + 32).
    
    Decoder:
      - Nearest upsampling + Conv3x3 + Residual Blocks with GroupNorm (no checkerboard).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        latent_dim: int = 128,
        enc_class_emb_dim: int = 16,
        dec_class_emb_dim: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.enc_class_emb_dim = enc_class_emb_dim
        self.dec_class_emb_dim = dec_class_emb_dim

        # 1. Conditioning Embeddings
        self.enc_class_emb = nn.Embedding(num_classes, enc_class_emb_dim)
        self.dec_class_emb = nn.Embedding(num_classes, dec_class_emb_dim)

        # 2. Encoder: 64x64 -> 8x8
        enc_in = in_channels + enc_class_emb_dim
        self.enc_stem = nn.Sequential(
            nn.Conv2d(enc_in, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.enc_stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 64 -> 32
            CVAEResBlock(64, num_groups=8)
        )
        self.enc_stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 32 -> 16
            CVAEResBlock(128, num_groups=8)
        )
        self.enc_stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),# 16 -> 8
            CVAEResBlock(256, num_groups=8)
        )
        self.enc_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Latent projections
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 3. Decoder: 8x8 -> 64x64
        dec_in = latent_dim + dec_class_emb_dim
        self.dec_fc = nn.Linear(dec_in, 256 * 8 * 8)

        # Stage 1: 8x8 -> 16x16
        self.dec_stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            CVAEResBlock(128, num_groups=8)
        )
        # Stage 2: 16x16 -> 32x32
        self.dec_stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            CVAEResBlock(64, num_groups=8)
        )
        # Stage 3: 32x32 -> 64x64
        self.dec_stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            CVAEResBlock(32, num_groups=8)
        )
        self.dec_out = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # output strictly in [0, 1]
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode RGB image x and class label y into posterior parameters (mu, logvar).
        x: (B, 3, 64, 64)
        y: (B,) class indices
        """
        b, _, h, w = x.shape
        c_emb = self.enc_class_emb(y) # (B, 16)
        c_spatial = c_emb.view(b, self.enc_class_emb_dim, 1, 1).expand(-1, -1, h, w)
        x_cond = torch.cat([x, c_spatial], dim=1) # (B, 19, 64, 64)

        h = self.enc_stem(x_cond)
        h = self.enc_stage1(h)
        h = self.enc_stage2(h)
        h = self.enc_stage3(h)
        h = self.enc_pool(h).flatten(1) # (B, 256)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-15.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, exp(logvar)) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Decode latent code z conditioned on class y.
        y can be an integer, a (B,) LongTensor, or a (B, dec_class_emb_dim) dense embedding (for class morphing).
        """
        b = z.shape[0]
        if isinstance(y, int):
            y_t = torch.tensor([y] * b, dtype=torch.long, device=z.device)
            c_emb = self.dec_class_emb(y_t)
        elif isinstance(y, torch.Tensor) and y.dtype == torch.long:
            c_emb = self.dec_class_emb(y)
        elif isinstance(y, torch.Tensor) and (y.ndim == 2 and y.shape[1] == self.dec_class_emb_dim):
            c_emb = y # continuous embedding for class morphing
        else:
            raise ValueError(f"Invalid condition y format: {type(y)}")

        z_cond = torch.cat([z, c_emb], dim=1) # (B, K + dec_dim)
        h = self.dec_fc(z_cond)
        h = h.view(b, 256, 8, 8)
        h = self.dec_stage1(h) # -> (128, 16, 16)
        h = self.dec_stage2(h) # -> (64, 32, 32)
        h = self.dec_stage3(h) # -> (32, 64, 64)
        out = self.dec_out(h)  # -> (3, 64, 64)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns reconstruction, mu, logvar."""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def sample(
        self,
        num_samples: int,
        class_idx: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate unconditional samples from N(0, tau^2 * I) conditioned on class_idx.
        temperature: scaling factor tau in [0.6, 1.0].
        """
        dev = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=dev) * temperature
        with torch.no_grad():
            samples = self.decode(z, class_idx)
        return samples

    def interpolate_slerp(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        class_idx: int,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Interpolate between two latent vectors along a great circle (SLERP)."""
        alphas = np.linspace(0.0, 1.0, num_steps)
        interpolated = []
        with torch.no_grad():
            for a in alphas:
                z_interp = slerp(z0, z1, float(a))
                img = self.decode(z_interp, class_idx)
                interpolated.append(img)
        return torch.cat(interpolated, dim=0)

    def morph_classes(
        self,
        z: torch.Tensor,
        class_from: int,
        class_to: int,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Continuously morph class conditioning with fixed latent code z."""
        dev = z.device
        emb0 = self.dec_class_emb(torch.tensor([class_from], device=dev)) # (1, dec_dim)
        emb1 = self.dec_class_emb(torch.tensor([class_to], device=dev))   # (1, dec_dim)

        alphas = np.linspace(0.0, 1.0, num_steps)
        morphed = []
        with torch.no_grad():
            for a in alphas:
                emb_interp = (1.0 - float(a)) * emb0 + float(a) * emb1
                img = self.decode(z, emb_interp)
                morphed.append(img)
        return torch.cat(morphed, dim=0)


# ==============================================================================
# ELBO Loss Computation & Metrics
# ==============================================================================

def compute_cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    likelihood_type: str = "sigma_vae"
) -> Dict[str, torch.Tensor]:
    """
    Compute rigorous Evidence Lower Bound (ELBO) loss.
    
    Reconstruction loss: summed over all pixels per image, averaged over batch.
    KL divergence: summed over all latent dims per image, averaged over batch.
    
    likelihood_type:
      - 'sigma_vae': Calibrated Gaussian observation variance sigma^2 = MSE(x, recon).
      - 'fixed_mse': Fixed unit variance (pure sum of squared errors).
      - 'naive_baseline': Mean over pixels and mean over latents (classic tutorial pitfall).
    """
    b = x.shape[0]
    d = float(x.shape[1] * x.shape[2] * x.shape[3]) # 3 * 64 * 64 = 12,288

    diff_sq = (recon_x - x) ** 2

    # 1. Pixel MSE for standard reporting
    raw_mse = diff_sq.mean()

    # 2. Per-latent dimension KL: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1.0 + logvar - mu ** 2 - torch.exp(logvar)) # (B, K)
    kl_sum_per_sample = kl_per_dim.sum(dim=1) # (B,)
    kl_mean = kl_sum_per_sample.mean() # scalar

    if likelihood_type == "naive_baseline":
        # Tutorial mistake: mean over pixels and mean over K
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_loss = kl_per_dim.mean()
        total_loss = recon_loss + beta * kl_loss
        elbo_nats = -(recon_loss * d + kl_mean)
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "raw_mse": raw_mse,
            "elbo_nats": elbo_nats,
            "kl_per_dim": kl_per_dim.mean(dim=0).detach()
        }

    # True ELBO: Summed over pixels per sample
    recon_sum_per_sample = diff_sq.view(b, -1).sum(dim=1) # (B,)
    recon_sum_mean = recon_sum_per_sample.mean() # scalar

    if likelihood_type == "sigma_vae":
        # Analytically calibrated decoder variance (Ryzhikov et al., 2020)
        # sigma^2 = 1/D * MSE_sum
        sigma_sq = torch.clamp(recon_sum_mean / d, min=1e-4)
        recon_nll = 0.5 * recon_sum_mean / sigma_sq + 0.5 * d * torch.log(2.0 * math.pi * sigma_sq)
        total_loss = recon_nll + beta * kl_mean
        elbo_nats = -(recon_nll + kl_mean)
        return {
            "loss": total_loss,
            "recon_loss": recon_nll,
            "kl_loss": kl_mean,
            "raw_mse": raw_mse,
            "sigma_sq": sigma_sq.detach(),
            "elbo_nats": elbo_nats,
            "kl_per_dim": kl_per_dim.mean(dim=0).detach()
        }

    else: # 'fixed_mse'
        total_loss = recon_sum_mean + beta * kl_mean
        elbo_nats = -(recon_sum_mean + kl_mean)
        return {
            "loss": total_loss,
            "recon_loss": recon_sum_mean,
            "kl_loss": kl_mean,
            "raw_mse": raw_mse,
            "elbo_nats": elbo_nats,
            "kl_per_dim": kl_per_dim.mean(dim=0).detach()
        }


def compute_active_units(mu_all: torch.Tensor, threshold: float = 0.01) -> Tuple[int, np.ndarray]:
    """
    Compute number of active latent units: Var_{x ~ D}[mu_j(x)] > threshold (Burda et al. 2015).
    mu_all: (N, K) posterior means across dataset.
    Returns:
      (active_count, variances_np)
    """
    variances = torch.var(mu_all, dim=0, unbiased=False).cpu().numpy()
    active_mask = variances > threshold
    return int(np.sum(active_mask)), variances

