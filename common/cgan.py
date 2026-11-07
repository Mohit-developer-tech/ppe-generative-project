"""
common/cgan.py
Conditional Generative Adversarial Network (cGAN) with DiffAugment for 64x64 PPE crops.

Key Architectural Components:
1. DiffAugment (Zhao et al., NeurIPS 2020):
   Differentiable color, translation, and cutout augmentation applied to real & fake
   during discriminator training AND through fakes during generator training.
2. Projection Discriminator (Miyato & Koyama, ICML 2018):
   Spectral Normalization (SN) across all conv & linear layers.
   Combines unconditional realism score psi(phi(x)) with class inner product <phi(x), w_y>.
3. Class-Conditional Batch Normalization (CBN):
   Batch normalization in Generator modulated by learned per-class scale & shift embeddings.
4. Hinge Adversarial Loss:
   Prevents discriminator gradient saturation on limited data.
5. Two-Time-Scale Update Rule (TTUR):
   lr_D = 4e-4, lr_G = 1e-4, Adam(betas=(0.0, 0.9)).
6. Exponential Moving Average (EMA):
   Shadow weights maintained with decay=0.999 for stable evaluation and sampling.
"""

import copy
import math
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from common.data import CLASS_NAMES


# ==============================================================================
# 1. Differentiable Augmentation (DiffAugment)
# ==============================================================================

def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    """
    Apply differentiable augmentations to PyTorch tensor [B, C, H, W].
    Gradients can propagate through both real and fake images.
    """
    if policy:
        for p in policy.split(","):
            p = p.strip()
            if p == "color":
                x = rand_color(x)
            elif p == "translation":
                x = rand_translation(x)
            elif p == "cutout":
                x = rand_cutout(x)
    return x.contiguous()


def rand_color(x: torch.Tensor) -> torch.Tensor:
    """Random brightness, saturation, and contrast adjustment."""
    # Random brightness in [-0.5, 0.5]
    b = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    x = x + b

    # Random saturation in [0.0, 2.0]
    x_mean = x.mean(dim=1, keepdim=True)
    s = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0)
    x = (x - x_mean) * s + x_mean

    # Random contrast in [0.5, 1.5]
    c_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    c = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5)
    x = (x - c_mean) * c + c_mean
    return x


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Random horizontal and vertical integer shift with zero padding."""
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)

    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + translation_x, 0, x.size(2) - 1)
    grid_y = torch.clamp(grid_y + translation_y, 0, x.size(3) - 1)
    # Permute to [B, H, W, C] for contiguous advanced indexing, then back to [B, C, H, W]
    x_perm = x.permute(0, 2, 3, 1)
    x_trans = x_perm[grid_b, grid_x, grid_y]
    return x_trans.permute(0, 3, 1, 2).contiguous()


def rand_cutout(x: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
    """Random square cutout mask filled with zeros."""
    cutout_size = (int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5))
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)

    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(offset_x + grid_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(offset_y + grid_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_b, grid_x, grid_y] = 0.0
    return x * mask.unsqueeze(1)


# ==============================================================================
# 2. Class-Conditional Batch Normalization (CBN)
# ==============================================================================

class ConditionalBatchNorm2d(nn.Module):
    """
    Class-Conditional Batch Normalization.
    Normalizes feature activations, then applies class-specific scale (gamma) and shift (beta).
    """
    def __init__(self, num_features: int, num_classes: int = 3):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        # Initialize gamma around 1 and beta around 0
        nn.init.normal_(self.embed.weight.data[:, :num_features], 1.0, 0.02)
        nn.init.zeros_(self.embed.weight.data[:, num_features:])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma_beta = self.embed(y)  # [B, 2 * num_features]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return out * gamma + beta


# ==============================================================================
# 3. Conditional Generator (DCGAN Backbone + CBN)
# ==============================================================================

class ConditionalGenerator(nn.Module):
    """
    DCGAN-style Class-Conditional Generator with Class-Conditional BatchNorm.
    Maps latent z ~ N(0, I) and class label y in {0, 1, 2} to 64x64 RGB crops.
    """
    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 3,
        base_channels: int = 64,
        out_channels: int = 3
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Project z to 4x4 spatial feature map
        self.fc = nn.Linear(z_dim, base_channels * 8 * 4 * 4)

        # Stage 1: 4x4 -> 8x8 (512 -> 256)
        self.deconv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn1 = ConditionalBatchNorm2d(base_channels * 4, num_classes)

        # Stage 2: 8x8 -> 16x16 (256 -> 128)
        self.deconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn2 = ConditionalBatchNorm2d(base_channels * 2, num_classes)

        # Stage 3: 16x16 -> 32x32 (128 -> 64)
        self.deconv3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn3 = ConditionalBatchNorm2d(base_channels, num_classes)

        # Stage 4: 32x32 -> 64x64 (64 -> 3)
        self.deconv4 = nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.tanh = nn.Tanh()

        self.act = nn.ReLU(inplace=True)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting images in [-1, 1]."""
        # Linear projection to 4x4
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 8, 4, 4)

        # Transposed conv stages with CBN
        h = self.act(self.cbn1(self.deconv1(h), y))
        h = self.act(self.cbn2(self.deconv2(h), y))
        h = self.act(self.cbn3(self.deconv3(h), y))
        out = self.tanh(self.deconv4(h))
        return out

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        to_01: bool = True
    ) -> torch.Tensor:
        """Sample images, optionally mapped to [0, 1] range for evaluation and display."""
        self.eval()
        out = self.forward(z, y)
        if to_01:
            out = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)
        return out


# ==============================================================================
# 4. Projection Discriminator (Spectral Normalization + Miyato & Koyama Projection)
# ==============================================================================

class ProjectionDiscriminator(nn.Module):
    """
    DCGAN-style Discriminator with:
    - Spectral Normalization on all conv and linear layers.
    - Global sum-pooling of intermediate features.
    - Miyato & Koyama (2018) class projection: D(x, y) = psi(phi(x)) + <phi(x), w_y>.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Conv Stage 1: 64x64 -> 32x32 (3 -> 64)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1))
        # Conv Stage 2: 32x32 -> 16x16 (64 -> 128)
        self.conv2 = spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1))
        # Conv Stage 3: 16x16 -> 8x8 (128 -> 256)
        self.conv3 = spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1))
        # Conv Stage 4: 8x8 -> 4x4 (256 -> 512)
        self.conv4 = spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1))

        self.act = nn.LeakyReLU(0.2, inplace=True)

        feature_dim = base_channels * 8  # 512

        # Unconditional realism linear head psi(phi(x))
        self.linear_uncond = spectral_norm(nn.Linear(feature_dim, 1))

        # Class embedding for inner-product projection <phi(x), w_y>
        self.class_embed = spectral_norm(nn.Embedding(num_classes, feature_dim))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract globally pooled feature representation phi(x) in R^512."""
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h))
        # Global sum-pooling over spatial dims (Miyato & Koyama standard)
        phi = torch.sum(h, dim=[2, 3])
        return phi

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator score:
        D(x, y) = psi(phi(x)) + <phi(x), w_y>
        Returns [B, 1] scalar logit score.
        """
        phi = self.extract_features(x)
        # Unconditional score
        out_uncond = self.linear_uncond(phi)  # [B, 1]

        # Conditional projection score
        w_y = self.class_embed(y)  # [B, 512]
        proj = torch.sum(phi * w_y, dim=1, keepdim=True)  # [B, 1]

        return out_uncond + proj


# ==============================================================================
# 5. Hinge Adversarial Loss Formulation
# ==============================================================================

def compute_d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Discriminator Hinge Loss:
    L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]
    """
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    loss_d = loss_real + loss_fake
    return loss_d, float(loss_real.item()), float(loss_fake.item())


def compute_g_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Generator Hinge Loss:
    L_G = -E[D(G(z, y), y)]
    """
    return -d_fake.mean()


# ==============================================================================
# 6. Exponential Moving Average (EMA) for Generator
# ==============================================================================

class EMA:
    """
    Maintains an exponential moving average of model parameters.
    Updated after every generator optimizer step:
    param_ema = decay * param_ema + (1 - decay) * param
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register initial parameter shadows."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        """Update shadow parameters with exponential decay."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Temporarily apply shadow weights to the model (backup original)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


# ==============================================================================
# 7. Spherical Linear Interpolation & Latent Walks
# ==============================================================================

def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two Gaussian latent vectors.
    """
    norm_z0 = z0 / (torch.norm(z0, dim=-1, keepdim=True) + 1e-8)
    norm_z1 = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-8)
    dot = torch.sum(norm_z0 * norm_z1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -0.9995, 0.9995)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    scale0 = torch.sin((1.0 - alpha) * omega) / sin_omega
    scale1 = torch.sin(alpha * omega) / sin_omega
    return scale0 * z0 + scale1 * z1


@torch.no_grad()
def generate_latent_walk(
    generator: ConditionalGenerator,
    class_id: int,
    z_start: Optional[torch.Tensor] = None,
    z_end: Optional[torch.Tensor] = None,
    steps: int = 10,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Generate an interpolation strip of 'steps' images between two latent points
    for a fixed class condition.
    Returns [steps, 3, 64, 64] tensor in [0, 1].
    """
    generator.eval()
    if z_start is None:
        z_start = torch.randn(1, generator.z_dim, device=device)
    if z_end is None:
        z_end = torch.randn(1, generator.z_dim, device=device)

    alphas = np.linspace(0.0, 1.0, steps)
    frames = []
    y_t = torch.tensor([class_id], dtype=torch.long, device=device)

    for alpha in alphas:
        z_interp = slerp(z_start, z_end, float(alpha))
        img = generator.sample(z_interp, y_t, to_01=True)
        frames.append(img.cpu())

    return torch.cat(frames, dim=0)
