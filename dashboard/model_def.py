"""
model_def.py
Exact architecture matching 01_autoencoder_svm.ipynb for model weight loading.
"""
from typing import Tuple
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder with 4 downsampling stages mapping 64x64x3 images
    to a 256-dimensional compact latent representation.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            # Stage 1: (B, 3, 64, 64) -> (B, 32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Stage 2: (B, 32, 32, 32) -> (B, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Stage 3: (B, 64, 16, 16) -> (B, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Stage 4: (B, 128, 8, 8) -> (B, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return self.fc(features)


class ConvDecoder(nn.Module):
    """
    Mirrored Transposed-Convolutional Decoder reconstructing 64x64x3 RGB images
    from a 256-dimensional latent representation, bounded in [0, 1] by Sigmoid.
    """
    def __init__(self, out_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            # Stage 4: (B, 256, 4, 4) -> (B, 128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3: (B, 128, 8, 8) -> (B, 64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 2: (B, 64, 16, 16) -> (B, 32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 1: (B, 32, 32, 32) -> (B, 3, 64, 64)
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)


class _ConvAEV1(nn.Module):
    """
    Baseline Transposed-Convolutional Autoencoder.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class ConvAE(nn.Module):
    """
    Adaptive Convolutional Autoencoder matching 01_autoencoder_svm.ipynb.
    Supports both upgraded high-clarity architecture and legacy baseline weights.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 256, upsample_mode: str = "nearest"):
        super().__init__()
        self.latent_dim = latent_dim
        self.v2_model = ConvAEv2(in_channels=in_channels, latent_dim=latent_dim, upsample_mode=upsample_mode)
        self.v1_model = None
        self.active_mode = "v2"

    def load_state_dict(self, state_dict, strict: bool = False):
        if "encoder.conv.0.weight" in state_dict and state_dict["encoder.conv.0.weight"].shape[0] == 32:
            self.v1_model = _ConvAEV1(in_channels=3, latent_dim=self.latent_dim)
            res = self.v1_model.load_state_dict(state_dict, strict=strict)
            self.active_mode = "v1"
            return res
        else:
            self.active_mode = "v2"
            return self.v2_model.load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.v2_model.to(*args, **kwargs)
        if self.v1_model is not None:
            self.v1_model.to(*args, **kwargs)
        return self

    def eval(self):
        super().eval()
        self.v2_model.eval()
        if self.v1_model is not None:
            self.v1_model.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.v2_model.train(mode)
        if self.v1_model is not None:
            self.v1_model.train(mode)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_mode == "v1" and self.v1_model is not None:
            return self.v1_model.encode(x)
        return self.v2_model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.active_mode == "v1" and self.v1_model is not None:
            return self.v1_model.decode(z)
        return self.v2_model.decode(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.active_mode == "v1" and self.v1_model is not None:
            return self.v1_model(x)
        return self.v2_model(x)


class ResBlock2d(nn.Module):
    """Residual Block with two 3x3 convs, batch norm, and LeakyReLU activation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class ConvEncoderV2(nn.Module):
    """
    AE-v2 Encoder with wider feature hierarchy (48 -> 96 -> 192 -> 256) and residual stages.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            # Stage 1: (B, 3, 64, 64) -> (B, 48, 32, 32)
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(48),

            # Stage 2: (B, 48, 32, 32) -> (B, 96, 16, 16)
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(96),

            # Stage 3: (B, 96, 16, 16) -> (B, 192, 8, 8)
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(192),

            # Stage 4: (B, 192, 8, 8) -> (B, 256, 4, 4)
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(256),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ConvDecoderV2(nn.Module):
    """
    AE-v2 Decoder using Upsample(nearest) + Conv3x3 + ResBlock2d instead of ConvTranspose.
    Completely eliminates checkerboard deconvolution artifacts.
    """
    def __init__(self, out_channels: int = 3, latent_dim: int = 256, upsample_mode: str = "nearest"):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            # Stage 4: (B, 256, 4, 4) -> (B, 192, 8, 8)
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(256, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(192),

            # Stage 3: (B, 192, 8, 8) -> (B, 96, 16, 16)
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(96),

            # Stage 2: (B, 96, 16, 16) -> (B, 48, 32, 32)
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(96, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(48),

            # Stage 1: (B, 48, 32, 32) -> (B, 32, 64, 64)
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(32),

            # Final RGB projection
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class ConvAEv2(nn.Module):
    """
    AE-v2 Architecture: Residual stages, Upsample+Conv3x3 decoder, configurable bottleneck.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 256, upsample_mode: str = "nearest"):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoderV2(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoderV2(out_channels=in_channels, latent_dim=latent_dim, upsample_mode=upsample_mode)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class ConvAEv2Spatial(nn.Module):
    """
    AE-v2 Spatial Bottleneck Variant: 8x8x32 spatial map (flattened to 2048 for SVM).
    """
    def __init__(self, in_channels: int = 3, upsample_mode: str = "nearest"):
        super().__init__()
        self.latent_dim = 2048  # 32 * 8 * 8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(48),

            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(96),

            nn.Conv2d(96, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(192),

            nn.Conv2d(192, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(192),

            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(192, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(96),

            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(96, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(48),

            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock2d(32),

            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h.flatten(start_dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z.view(-1, 32, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

