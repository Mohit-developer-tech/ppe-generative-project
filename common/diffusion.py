"""
common/diffusion.py
Conditional Denoising Diffusion Probabilistic Model (DDPM / DDIM) for PPE Crop Restoration.

Core Features:
  - SR3-style image-conditioned U-Net: concatenates degraded image (3ch) with noisy state (3ch) -> 6ch input.
  - Sinusoidal time embedding + Residual blocks + GroupNorm + Multi-Head Self-Attention (at 16x16 and 8x8).
  - Cosine noise schedule (Nichol & Dhariwal, ICML 2021) for T=1000 timesteps.
  - Losses: Epsilon-MSE, Epsilon-L1, and low-noise weighted perceptual LPIPS on x0_hat.
  - Fast deterministic DDIM sampler (20, 50, 100 steps) and full stochastic DDPM sampler (1000 steps).
  - Regression U-Net baseline of identical capacity for perception-distortion trade-off benchmarks.
  - EMA shadow weight maintenance for stable sampling.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Cosine Noise Schedule & Diffusion Forward Process
# ==============================================================================

class DiffusionSchedule:
    """
    Cosine Noise Schedule for Denoising Diffusion Probabilistic Models.
    Nichol & Dhariwal, 'Improved Denoising Diffusion Probabilistic Models', ICML 2021.
    """

    def __init__(self, timesteps: int = 1000, s: float = 0.008, device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps
        self.device = device

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Convert to float32 on device
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.alphas_cumprod = alphas_cumprod.float().to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.float().to(device)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device):
        """Move all schedule tensors to specified device."""
        self.device = device
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod", "posterior_variance",
            "posterior_log_variance_clipped", "posterior_mean_coef1", "posterior_mean_coef2"
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diffuse the data x_0 for a given number of diffusion steps t.
        q(x_t | x_0) = N(x_t; sqrt(abar_t) * x_0, (1 - abar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_abar_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_abar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_abar_t * x_start + sqrt_one_minus_abar_t * noise
        return x_t, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Tweedie estimate of clean x_0 from noisy state x_t and predicted noise eps.
        x0_hat = (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t)
        """
        sqrt_recip_abar = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_abar = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_abar * x_t - sqrt_recipm1_abar * eps

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract coefficients at specified timesteps t and reshape to match x_shape."""
        bs = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(bs, *((1,) * (len(x_shape) - 1)))


# ==============================================================================
# 2. Sinusoidal Time Embedding & Building Blocks
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Wide Residual Block with GroupNorm, SiLU, and Time Embedding Injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        groups: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure groups divide channels
        g1 = min(groups, in_channels)
        while in_channels % g1 != 0 and g1 > 1:
            g1 -= 1
        g2 = min(groups, out_channels)
        while out_channels % g2 != 0 and g2 > 1:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            ) if time_emb_dim is not None else None
        )

        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        if self.time_mlp is not None and time_emb is not None:
            time_feat = self.time_mlp(time_emb)[:, :, None, None]
            h = h + time_feat
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for intermediate spatial resolutions."""

    def __init__(self, channels: int, num_heads: int = 4, groups: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        g = min(groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1

        self.norm = nn.GroupNorm(g, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


# ==============================================================================
# 3. Conditional U-Net for Diffusion Restoration (SR3 Backbone)
# ==============================================================================

class ConditionalUNet(nn.Module):
    """
    SR3-Style Conditional U-Net for Image Restoration.
    Takes 6 channels: [x_t (3ch), degraded_y (3ch)] -> predicts eps (3ch).
    STRICTLY NO CLASS CONDITIONING: prevents label leakage into the restoration engine.
    
    Clean 3-Stage Hierarchy for 64x64 images:
      Stage 0 (64x64): channels base_channels (e.g. 64)
      Stage 1 (32x32): channels base_channels * 2 (e.g. 128)
      Stage 2 (16x16): channels base_channels * 4 (e.g. 256) + Attention
      Bottleneck (8x8): channels base_channels * 4 (e.g. 256) + Attention
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        attn_resolutions: Tuple[int, ...] = (16, 8),
        time_emb_dim: int = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim

        ch0 = base_channels
        ch1 = base_channels * 2
        ch2 = base_channels * 4

        # 1. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 2. Input Stem (6 -> ch0)
        self.stem = nn.Conv2d(in_channels, ch0, 3, padding=1)

        # 3. Encoder Stage 0 (64x64)
        self.enc0_b1 = ResidualBlock(ch0, ch0, time_emb_dim)
        self.enc0_b2 = ResidualBlock(ch0, ch0, time_emb_dim)
        self.down0 = nn.Conv2d(ch0, ch1, 3, stride=2, padding=1)  # 64 -> 32

        # 4. Encoder Stage 1 (32x32)
        self.enc1_b1 = ResidualBlock(ch1, ch1, time_emb_dim)
        self.enc1_b2 = ResidualBlock(ch1, ch1, time_emb_dim)
        self.down1 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)  # 32 -> 16

        # 5. Encoder Stage 2 (16x16)
        self.enc2_b1 = ResidualBlock(ch2, ch2, time_emb_dim)
        self.enc2_attn = SelfAttention(ch2)
        self.enc2_b2 = ResidualBlock(ch2, ch2, time_emb_dim)
        self.down2 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)  # 16 -> 8

        # 6. Bottleneck (8x8)
        self.mid_b1 = ResidualBlock(ch2, ch2, time_emb_dim)
        self.mid_attn = SelfAttention(ch2)
        self.mid_b2 = ResidualBlock(ch2, ch2, time_emb_dim)

        # 7. Decoder Stage 2 (8 -> 16)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch2, ch2, 3, padding=1)
        )
        self.dec2_b1 = ResidualBlock(ch2 * 2, ch2, time_emb_dim)
        self.dec2_attn = SelfAttention(ch2)
        self.dec2_b2 = ResidualBlock(ch2, ch2, time_emb_dim)

        # 8. Decoder Stage 1 (16 -> 32)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch2, ch1, 3, padding=1)
        )
        self.dec1_b1 = ResidualBlock(ch1 * 2, ch1, time_emb_dim)
        self.dec1_b2 = ResidualBlock(ch1, ch1, time_emb_dim)

        # 9. Decoder Stage 0 (32 -> 64)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch1, ch0, 3, padding=1)
        )
        self.dec0_b1 = ResidualBlock(ch0 * 2, ch0, time_emb_dim)
        self.dec0_b2 = ResidualBlock(ch0, ch0, time_emb_dim)

        # 10. Output Head
        g_out = min(32, ch0)
        while ch0 % g_out != 0 and g_out > 1:
            g_out -= 1
        self.out_norm = nn.GroupNorm(g_out, ch0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch0, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_noisy: torch.Tensor, degraded: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SR3 conditional U-Net.
        Args:
            x_noisy: Noisy diffusion state x_t, shape [B, 3, 64, 64] in [-1, 1].
            degraded: Degraded conditioning image y, shape [B, 3, 64, 64] in [-1, 1].
            t: Diffusion timestep, shape [B].
        Returns:
            eps_pred: Predicted noise vector, shape [B, 3, 64, 64].
        """
        t_emb = self.time_mlp(t)

        # SR3 Conditioning: Channel concatenation [x_t, degraded] -> [B, 6, 64, 64]
        h = torch.cat([x_noisy, degraded], dim=1)
        h = self.stem(h)

        # Stage 0 (64x64)
        h = self.enc0_b1(h, t_emb)
        skip0 = self.enc0_b2(h, t_emb)
        h = self.down0(skip0)  # 64 -> 32

        # Stage 1 (32x32)
        h = self.enc1_b1(h, t_emb)
        skip1 = self.enc1_b2(h, t_emb)
        h = self.down1(skip1)  # 32 -> 16

        # Stage 2 (16x16)
        h = self.enc2_b1(h, t_emb)
        h = self.enc2_attn(h)
        skip2 = self.enc2_b2(h, t_emb)
        h = self.down2(skip2)  # 16 -> 8

        # Bottleneck (8x8)
        h = self.mid_b1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_b2(h, t_emb)

        # Decoder Stage 2 (8 -> 16)
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2_b1(h, t_emb)
        h = self.dec2_attn(h)
        h = self.dec2_b2(h, t_emb)

        # Decoder Stage 1 (16 -> 32)
        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_b1(h, t_emb)
        h = self.dec1_b2(h, t_emb)

        # Decoder Stage 0 (32 -> 64)
        h = self.up0(h)
        h = torch.cat([h, skip0], dim=1)
        h = self.dec0_b1(h, t_emb)
        h = self.dec0_b2(h, t_emb)

        out = self.out_conv(self.out_act(self.out_norm(h)))
        return out


# ==============================================================================
# 4. Regression U-Net Baseline (Deterministic Perception-Distortion Comparison)
# ==============================================================================

class RegressionUNet(nn.Module):
    """
    Deterministic Regression U-Net of identical capacity.
    Maps degraded image y (3ch) -> clean image x_0 (3ch) directly without diffusion.
    Used to empirically demonstrate the Blau & Michaeli Perception-Distortion Trade-off.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        ch0 = base_channels
        ch1 = base_channels * 2
        ch2 = base_channels * 4

        # Stem
        self.stem = nn.Conv2d(3, ch0, 3, padding=1)

        # Encoder (64 -> 32 -> 16 -> 8)
        self.enc0 = ResidualBlock(ch0, ch0, time_emb_dim=None)
        self.down0 = nn.Conv2d(ch0, ch1, 3, stride=2, padding=1)

        self.enc1 = ResidualBlock(ch1, ch1, time_emb_dim=None)
        self.down1 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)

        self.enc2 = ResidualBlock(ch2, ch2, time_emb_dim=None)
        self.attn2 = SelfAttention(ch2)
        self.down2 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)

        # Bottleneck (8x8)
        self.mid = ResidualBlock(ch2, ch2, time_emb_dim=None)

        # Decoder (8 -> 16 -> 32 -> 64)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch2, ch2, 3, padding=1)
        )
        self.dec2 = ResidualBlock(ch2 * 2, ch2, time_emb_dim=None)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch2, ch1, 3, padding=1)
        )
        self.dec1 = ResidualBlock(ch1 * 2, ch1, time_emb_dim=None)

        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch1, ch0, 3, padding=1)
        )
        self.dec0 = ResidualBlock(ch0 * 2, ch0, time_emb_dim=None)

        g_out = min(32, ch0)
        while ch0 % g_out != 0 and g_out > 1:
            g_out -= 1
        self.out_norm = nn.GroupNorm(g_out, ch0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch0, 3, 3, padding=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Directly map degraded image y in [-1, 1] to predicted clean image in [-1, 1]."""
        x0 = self.enc0(self.stem(y))
        x1 = self.enc1(self.down0(x0))
        x2 = self.attn2(self.enc2(self.down1(x1)))

        m = self.mid(self.down2(x2))

        d2 = self.dec2(torch.cat([self.up2(m), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), x0], dim=1))

        out = self.out_conv(self.out_act(self.out_norm(d0)))
        return torch.tanh(out)


# ==============================================================================
# 5. Exponential Moving Average (EMA)
# ==============================================================================

class EMA:
    """Maintains exponential moving average of model parameters for evaluation and sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ==============================================================================
# 6. Sampling Engines: DDPM (1000 steps) & DDIM (20, 50, 100 steps)
# ==============================================================================

@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    degraded: torch.Tensor,
    schedule: DiffusionSchedule,
    device: torch.device
) -> torch.Tensor:
    """
    Full Stochastic DDPM Ancestral Sampling (1000 steps).
    Ho et al., NeurIPS 2020.
    """
    model.eval()
    bs = degraded.size(0)
    x = torch.randn_like(degraded).to(device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((bs,), i, device=device, dtype=torch.long)
        eps_pred = model(x, degraded, t)

        beta_t = schedule._extract(schedule.betas, t, x.shape)
        sqrt_one_minus_abar_t = schedule._extract(schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alpha_t = schedule._extract(torch.sqrt(schedule.alphas), t, x.shape)

        mean = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_abar_t) * eps_pred)

        if i > 0:
            sigma_t = schedule._extract(torch.sqrt(schedule.posterior_variance), t, x.shape)
            noise = torch.randn_like(x)
            x = mean + sigma_t * noise
        else:
            x = mean

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    degraded: torch.Tensor,
    schedule: DiffusionSchedule,
    steps: int = 50,
    eta: float = 0.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Fast Deterministic DDIM Sampling with Sub-Sequence Striding.
    Song et al., 'Denoising Diffusion Implicit Models', ICLR 2021.
    """
    model.eval()
    bs = degraded.size(0)

    # Sub-sequence of timesteps from T-1 down to 0
    step_indices = np.linspace(0, schedule.timesteps - 1, steps, dtype=int)
    step_indices = np.flip(step_indices)

    x = torch.randn_like(degraded).to(device)

    for i, t_idx in enumerate(step_indices):
        prev_t_idx = step_indices[i + 1] if i + 1 < len(step_indices) else -1

        t = torch.full((bs,), int(t_idx), device=device, dtype=torch.long)
        eps_pred = model(x, degraded, t)

        abar_t = schedule._extract(schedule.alphas_cumprod, t, x.shape)
        if prev_t_idx >= 0:
            t_prev = torch.full((bs,), int(prev_t_idx), device=device, dtype=torch.long)
            abar_prev = schedule._extract(schedule.alphas_cumprod, t_prev, x.shape)
        else:
            abar_prev = torch.ones_like(abar_t)

        # Tweedie estimate of x_0
        pred_x0 = (x - torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(abar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Next state computation (Deterministic if eta == 0.0)
        if eta > 0.0 and prev_t_idx >= 0:
            sigma = eta * torch.sqrt(torch.clamp((1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev), min=0.0))
            dir_xt = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma ** 2, min=0.0)) * eps_pred
            x_prev = torch.sqrt(abar_prev) * pred_x0 + dir_xt + sigma * torch.randn_like(x)
        else:
            dir_xt = torch.sqrt(torch.clamp(1.0 - abar_prev, min=0.0)) * eps_pred
            x_prev = torch.sqrt(abar_prev) * pred_x0 + dir_xt

        x = x_prev

    return torch.clamp(x, -1.0, 1.0)


# ==============================================================================
# 7. Loss Computation with Low-Noise Perceptual Weighting
# ==============================================================================

def compute_diffusion_loss(
    model: nn.Module,
    x_start: torch.Tensor,
    degraded: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    loss_type: str = "eps_l1",
    lpips_fn: Optional[nn.Module] = None,
    lpips_weight: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute diffusion loss under one of the 3 ablation configurations:
      1. 'eps_mse': Standard MSE on noise.
      2. 'eps_l1': L1 loss on noise.
      3. 'eps_l1_lpips': L1 loss on noise + low-noise (t < 250) abar_t-weighted perceptual LPIPS on x0_hat.
    """
    noise = torch.randn_like(x_start)
    x_t, _ = schedule.q_sample(x_start, t, noise=noise)

    eps_pred = model(x_t, degraded, t)

    loss_mse = F.mse_loss(eps_pred, noise)
    loss_l1 = F.l1_loss(eps_pred, noise)

    if loss_type == "eps_mse":
        total_loss = loss_mse
        loss_perc = torch.tensor(0.0, device=x_start.device)
    elif loss_type == "eps_l1":
        total_loss = loss_l1
        loss_perc = torch.tensor(0.0, device=x_start.device)
    elif loss_type == "eps_l1_lpips":
        pred_x0 = schedule.predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        low_noise_mask = (t < (schedule.timesteps // 4)).float()[:, None, None, None]

        if lpips_fn is not None and low_noise_mask.sum() > 0:
            perc_dist = lpips_fn(pred_x0, x_start)
            abar_t = schedule._extract(schedule.alphas_cumprod, t, x_start.shape)
            loss_perc = (perc_dist * abar_t * low_noise_mask).mean()
        else:
            loss_perc = torch.tensor(0.0, device=x_start.device)

        total_loss = loss_l1 + lpips_weight * loss_perc
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    return {
        "loss": total_loss,
        "loss_l1": loss_l1.detach(),
        "loss_mse": loss_mse.detach(),
        "loss_perc": loss_perc.detach()
    }
