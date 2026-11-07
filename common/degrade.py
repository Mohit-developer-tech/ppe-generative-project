"""
common/degrade.py
Randomised and deterministic synthetic image degradation pipeline for PPE restoration experiments.
Provides Gaussian blur, Gaussian noise, downscale-upscale, and JPEG artifacts across three severity levels
(mild, medium, severe), plus ONE held-out degradation type (motion blur) used strictly at test time.
"""

import io
import math
import random
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


SEVERITY_LEVELS = ("mild", "medium", "severe")
VALID_TRAIN_DEGRADATIONS = ("gaussian_noise", "gaussian_blur", "downscale", "jpeg")
HELD_OUT_DEGRADATION = "motion_blur"


def _get_rng(seed: Optional[int] = None) -> random.Random:
    """Return a seeded or unseeded Random instance for determinism."""
    return random.Random(seed) if seed is not None else random.Random()


def _ensure_tensor_hwc(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> Tuple[np.ndarray, bool, str]:
    """
    Standardize input into numpy array [H, W, C] in range [0, 1].
    Returns (img_np, is_torch, original_mode).
    """
    is_torch = isinstance(img, torch.Tensor)
    mode = "[0,1]"

    if is_torch:
        t = img.detach().cpu()
        if t.ndim == 3:  # [C, H, W]
            pass
        elif t.ndim == 4:  # [1, C, H, W]
            t = t.squeeze(0)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {img.shape}")

        if t.min() < -0.1:  # Likely [-1, 1]
            mode = "[-1,1]"
            t = torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)
        else:
            t = torch.clamp(t, 0.0, 1.0)

        arr = t.permute(1, 2, 0).numpy()
    elif isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(img, np.ndarray):
        arr = img.copy().astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # [C, H, W]
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    return np.clip(arr, 0.0, 1.0), is_torch, mode


def _restore_output_format(arr: np.ndarray, is_torch: bool, mode: str, device=None) -> Union[torch.Tensor, Image.Image]:
    """Convert numpy [H, W, C] in [0, 1] back to the original input type and range."""
    arr = np.clip(arr, 0.0, 1.0)
    if is_torch:
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        if mode == "[-1,1]":
            t = (t * 2.0) - 1.0
        if device is not None:
            t = t.to(device)
        return t
    else:
        uint8_arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(uint8_arr)


# ==============================================================================
# Individual Degradation Operators
# ==============================================================================

def apply_gaussian_noise(
    img: Union[torch.Tensor, Image.Image, np.ndarray],
    severity: str = "medium",
    seed: Optional[int] = None
) -> Union[torch.Tensor, Image.Image]:
    """Apply zero-mean additive Gaussian noise."""
    arr, is_torch, mode = _ensure_tensor_hwc(img)
    rng = _get_rng(seed)

    sigmas = {
        "mild": (0.01, 0.04),
        "medium": (0.05, 0.10),
        "severe": (0.12, 0.22)
    }
    low, high = sigmas.get(severity, (0.05, 0.10))
    sigma = rng.uniform(low, high)

    # Use seeded numpy Generator for exact determinism
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    noise = np_rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    degraded = np.clip(arr + noise, 0.0, 1.0)

    device = img.device if is_torch else None
    return _restore_output_format(degraded, is_torch, mode, device)


def apply_gaussian_blur(
    img: Union[torch.Tensor, Image.Image, np.ndarray],
    severity: str = "medium",
    seed: Optional[int] = None
) -> Union[torch.Tensor, Image.Image]:
    """Apply Gaussian blur with severity-calibrated kernel sizes and standard deviations."""
    arr, is_torch, mode = _ensure_tensor_hwc(img)
    rng = _get_rng(seed)

    params = {
        "mild": (3, 0.4, 0.8),
        "medium": (5, 0.9, 1.6),
        "severe": (7, 1.8, 3.0)
    }
    ksize, s_low, s_high = params.get(severity, (5, 0.9, 1.6))
    sigma = rng.uniform(s_low, s_high)

    pil_im = Image.fromarray((arr * 255.0).round().astype(np.uint8))
    blurred_pil = TF.gaussian_blur(pil_im, kernel_size=[ksize, ksize], sigma=[sigma, sigma])
    degraded = np.array(blurred_pil, dtype=np.float32) / 255.0

    device = img.device if is_torch else None
    return _restore_output_format(degraded, is_torch, mode, device)


def apply_downscale_upscale(
    img: Union[torch.Tensor, Image.Image, np.ndarray],
    severity: str = "medium",
    seed: Optional[int] = None
) -> Union[torch.Tensor, Image.Image]:
    """Downsample image resolution by 2x or 4x, then bilinearly interpolate back to 64x64."""
    arr, is_torch, mode = _ensure_tensor_hwc(img)
    rng = _get_rng(seed)

    h, w, _ = arr.shape
    scale_factors = {
        "mild": 2,
        "medium": rng.choice([2, 4]),
        "severe": 4
    }
    factor = scale_factors.get(severity, 2)
    down_h, down_w = max(4, h // factor), max(4, w // factor)

    pil_im = Image.fromarray((arr * 255.0).round().astype(np.uint8))
    small = pil_im.resize((down_w, down_h), Image.Resampling.BILINEAR)
    upscaled = small.resize((w, h), Image.Resampling.BILINEAR)
    degraded = np.array(upscaled, dtype=np.float32) / 255.0

    device = img.device if is_torch else None
    return _restore_output_format(degraded, is_torch, mode, device)


def apply_jpeg_compression(
    img: Union[torch.Tensor, Image.Image, np.ndarray],
    severity: str = "medium",
    seed: Optional[int] = None
) -> Union[torch.Tensor, Image.Image]:
    """Simulate blocky Discrete Cosine Transform (DCT) JPEG compression artifacts."""
    arr, is_torch, mode = _ensure_tensor_hwc(img)
    rng = _get_rng(seed)

    qualities = {
        "mild": (50, 75),
        "medium": (25, 45),
        "severe": (10, 22)
    }
    q_low, q_high = qualities.get(severity, (25, 45))
    quality = rng.randint(q_low, q_high)

    pil_im = Image.fromarray((arr * 255.0).round().astype(np.uint8))
    buffer = io.BytesIO()
    pil_im.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    jpeg_im = Image.open(buffer).convert("RGB")
    degraded = np.array(jpeg_im, dtype=np.float32) / 255.0

    device = img.device if is_torch else None
    return _restore_output_format(degraded, is_torch, mode, device)


# ==============================================================================
# HELD-OUT Degradation Operator (Used ONLY at Test Time)
# ==============================================================================

def apply_motion_blur(
    img: Union[torch.Tensor, Image.Image, np.ndarray],
    severity: str = "medium",
    seed: Optional[int] = None
) -> Union[torch.Tensor, Image.Image]:
    """
    HELD-OUT Degradation: Directional linear motion blur convolution.
    This degradation type is held out and MUST NEVER be used during training
    or validation to evaluate zero-shot restoration generalization.
    """
    arr, is_torch, mode = _ensure_tensor_hwc(img)
    rng = _get_rng(seed)

    kernel_sizes = {
        "mild": (3, 5),
        "medium": (7, 11),
        "severe": (13, 19)
    }
    k_min, k_max = kernel_sizes.get(severity, (7, 11))
    size = rng.randint(k_min, k_max)
    if size % 2 == 0:
        size += 1

    angle = rng.uniform(0.0, 180.0)
    theta = math.radians(angle)

    # Build motion blur convolution kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    for i in range(size):
        offset = i - center
        x = int(round(center + offset * cos_t))
        y = int(round(center + offset * sin_t))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0

    # Apply separable 2D convolution via PyTorch F.conv2d
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    k_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # (3, 1, K, K)
    pad = size // 2

    # Circular/reflect padding for seamless edges
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(padded, k_tensor, groups=3)
    degraded = blurred.squeeze(0).permute(1, 2, 0).numpy()

    device = img.device if is_torch else None
    return _restore_output_format(degraded, is_torch, mode, device)


# ==============================================================================
# Pipeline Combiner
# ==============================================================================

def degrade_image(
    img: Union[torch.Tensor, Image.Image, np.ndarray],
    severity: str = "medium",
    degradations: Optional[List[str]] = None,
    seed: Optional[int] = None,
    allow_held_out: bool = False
) -> Union[torch.Tensor, Image.Image]:
    """
    Apply a single or compound degradation pipeline.
    
    Args:
        img: Input image (Tensor [C, H, W] in [0,1] or [-1,1], PIL Image, or ndarray).
        severity: 'mild', 'medium', or 'severe'.
        degradations: Explicit list of operators to apply in sequence, e.g. ['gaussian_blur', 'jpeg'].
                      If None, randomly chooses 1 to 3 operators from VALID_TRAIN_DEGRADATIONS.
        seed: Random seed for deterministic degradation.
        allow_held_out: If True, allows applying 'motion_blur'. If False, raises an error.
    """
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"Invalid severity '{severity}'. Must be one of {SEVERITY_LEVELS}")

    rng = _get_rng(seed)

    # Validate or choose degradation pipeline
    if degradations is None:
        # Choose 1 to 2 random operators for training
        k = rng.choice([1, 2])
        chosen = rng.sample(list(VALID_TRAIN_DEGRADATIONS), k=k)
    else:
        chosen = list(degradations)

    # Check held-out degradation rule
    if HELD_OUT_DEGRADATION in chosen and not allow_held_out:
        raise PermissionError(
            f"'{HELD_OUT_DEGRADATION}' is a strictly held-out test degradation. "
            f"Pass allow_held_out=True only in test evaluation scripts."
        )

    out = img
    for op_name in chosen:
        op_seed = rng.randint(0, 2**31 - 1)
        if op_name == "gaussian_noise":
            out = apply_gaussian_noise(out, severity=severity, seed=op_seed)
        elif op_name == "gaussian_blur":
            out = apply_gaussian_blur(out, severity=severity, seed=op_seed)
        elif op_name == "downscale":
            out = apply_downscale_upscale(out, severity=severity, seed=op_seed)
        elif op_name == "jpeg":
            out = apply_jpeg_compression(out, severity=severity, seed=op_seed)
        elif op_name == "motion_blur":
            out = apply_motion_blur(out, severity=severity, seed=op_seed)
        else:
            raise ValueError(f"Unknown degradation operator: '{op_name}'")

    return out


def degrade_batch(
    batch: torch.Tensor,
    severity: str = "medium",
    degradations: Optional[List[str]] = None,
    seed: Optional[int] = None,
    allow_held_out: bool = False
) -> torch.Tensor:
    """
    Apply degradation across a batch of images [B, C, H, W].
    If seed is provided, ensures reproducible yet distinct degradations across batch items.
    """
    if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
        raise ValueError(f"Expected 4D torch.Tensor [B, C, H, W], got {type(batch)} shape {getattr(batch, 'shape', None)}")

    rng = _get_rng(seed)
    degraded_items = []

    for i in range(batch.size(0)):
        item_seed = rng.randint(0, 2**31 - 1)
        deg_item = degrade_image(
            batch[i],
            severity=severity,
            degradations=degradations,
            seed=item_seed,
            allow_held_out=allow_held_out
        )
        degraded_items.append(deg_item)

    return torch.stack(degraded_items, dim=0)

