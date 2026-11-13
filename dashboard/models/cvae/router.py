"""
dashboard/models/cvae/router.py
FastAPI router and artifact loader for Model 2: Conditional Variational Autoencoder (CVAE).
Provides endpoints for:
  - Model status, specifications, and metrics (tables & 300 dpi figures)
  - Class-conditional generation with temperature and random seed
  - Two-image upload with Spherical Linear Interpolation (SLERP) slider
  - Continuous class-morphing slider
  - Reconstructing uploaded crops with PSNR/SSIM evaluation
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from skimage.metrics import (
    peak_signal_noise_ratio as sk_psnr,
    structural_similarity as sk_ssim
)

from common.data import CLASS_NAMES
from common.cvae import ConditionalVAE, slerp
from dashboard.models.registry import get_model_info

router = APIRouter(prefix="/api/cvae", tags=["cvae"])

# State variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cvae_model: Optional[ConditionalVAE] = None
cvae_config: Dict[str, Any] = {}
cvae_metrics: Dict[str, Any] = {}
is_loaded: bool = False

RESULTS_DIR = Path("results/cvae")
ARTIFACTS_DIR = Path("artifacts/cvae")


def load_cvae_artifacts() -> bool:
    """Load CVAE model weights and metrics if available."""
    global cvae_model, cvae_config, cvae_metrics, is_loaded

    if is_loaded:
        return True

    weights_path = ARTIFACTS_DIR / "cvae_best.pt"
    if not weights_path.exists():
        weights_path = RESULTS_DIR / "cvae_best.pt"

    cfg_path = RESULTS_DIR / "model_config.json"

    if not weights_path.exists():
        return False

    print(f"[*] Loading CVAE artifacts from {ARTIFACTS_DIR} on {device}...", flush=True)

    # 1. Config
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cvae_config = json.load(f)
    latent_dim = cvae_config.get("latent_dim", 128)

    # 2. Model
    cvae_model = ConditionalVAE(latent_dim=latent_dim).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cvae_model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        cvae_model.load_state_dict(ckpt)
    cvae_model.eval()

    # 3. Metrics Summary
    metrics_path = RESULTS_DIR / "cvae_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            cvae_metrics = json.load(f)

    is_loaded = True
    print(f"    Loaded CVAE model (latent_dim={latent_dim}) successfully.")
    return True


def tensor_to_base64_png(tensor: torch.Tensor, upscale: int = 256, sharpen: bool = True) -> str:
    """Convert tensor [3, H, W] in [0, 1] to a crisp, high-definition base64 PNG data URL."""
    import cv2
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)

    if upscale and upscale != arr_u8.shape[0]:
        # 1. Percentile contrast normalization per channel to eliminate washed-out haze
        stretched = np.zeros_like(arr_u8)
        for c in range(3):
            p2, p98 = np.percentile(arr_u8[:, :, c], (1.5, 98.5))
            if p98 > p2 + 8:
                ch = np.clip((arr_u8[:, :, c] - p2) * 255.0 / (p98 - p2), 0, 255)
            else:
                ch = arr_u8[:, :, c]
            stretched[:, :, c] = ch.astype(np.uint8)

        # 2. Lanczos-4 high-order upsampling
        up = cv2.resize(stretched, (upscale, upscale), interpolation=cv2.INTER_LANCZOS4)

        if sharpen:
            # 3. Dual-band unsharp masking to restore high-frequency boundary edges
            blur1 = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
            blur2 = cv2.GaussianBlur(up, (0, 0), sigmaX=2.5)
            fine = cv2.subtract(up, blur1)
            coarse = cv2.subtract(blur1, blur2)
            sharp = cv2.addWeighted(up, 1.0, fine, 1.8, 0)
            sharp = np.clip(cv2.addWeighted(sharp, 1.0, coarse, 0.9, 0), 0, 255).astype(np.uint8)
            img = Image.fromarray(sharp)
        else:
            img = Image.fromarray(up)
    else:
        img = Image.fromarray(arr_u8)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor [1, 3, 64, 64] in [0, 1]."""
    img = img.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


# ==============================================================================
# Endpoints
# ==============================================================================

@router.get("/info")
def get_cvae_info():
    """Return model specification and conditioning details."""
    load_cvae_artifacts()
    info = get_model_info("cvae")
    status = "ready" if is_loaded else "coming_soon"
    return {
        "model_id": "cvae",
        "name": info["name"],
        "status": status,
        "badge": "Active" if is_loaded else info["badge"],
        "description": info["description"],
        "architecture": "SmallResNet-CVAE (Spatial Tile Encoder + Upsample ResBlock Decoder)",
        "latent_dim": cvae_config.get("latent_dim", 128),
        "loss_objective": "True ELBO with sigma-VAE (calibrated decoder variance) + linear KL annealing",
        "conditioning": {
            "encoder": "Class embedding (16-d) spatially tiled to (16, 64, 64) concatenated to RGB -> (19, 64, 64)",
            "decoder": "Class embedding (32-d) concatenated to latent code z -> (K + 32)",
            "classes": CLASS_NAMES
        }
    }


@router.get("/metrics")
def get_cvae_metrics_endpoint():
    """Return metrics summaries, dataframes, and available figure links."""
    load_cvae_artifacts()
    if not is_loaded:
        return {
            "status": "not_trained",
            "message": "CVAE model artifacts not found. Run scripts/train_cvae.py first."
        }

    # Load CSV tables if present
    tables = {}
    for table_name in ["reconstruction_metrics", "generation_metrics", "ablation_table", "history"]:
        csv_p = RESULTS_DIR / f"{table_name}.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p)
            tables[table_name] = df.to_dict(orient="records")

    figures = [
        {"id": "sample_grid", "name": "Class-Conditional Sample Grid (tau=0.8)", "url": "/api/cvae/figure/sample_grid.png"},
        {"id": "temperature_sweep", "name": "Temperature Sweep (tau: 0.6 - 1.0)", "url": "/api/cvae/figure/temperature_sweep.png"},
        {"id": "slerp", "name": "Spherical Latent Interpolation (SLERP)", "url": "/api/cvae/figure/slerp_interpolation.png"},
        {"id": "class_morph", "name": "Continuous Class Morphing", "url": "/api/cvae/figure/class_morph.png"},
        {"id": "latent_pca", "name": "2D Latent PCA Projection", "url": "/api/cvae/figure/latent_pca.png"},
        {"id": "loss_curves", "name": "Reconstruction vs KL Loss Curves", "url": "/api/cvae/figure/loss_curves.png"},
        {"id": "kl_per_dim", "name": "KL Divergence Per Latent Dimension", "url": "/api/cvae/figure/kl_per_dimension.png"}
    ]

    return {
        "status": "ready",
        "summary": cvae_metrics,
        "tables": tables,
        "figures": figures
    }


@router.get("/figure/{figure_name}")
def get_cvae_figure(figure_name: str):
    """Serve exported figure image."""
    fig_path = RESULTS_DIR / "figures" / figure_name
    if not fig_path.exists():
        fig_path = RESULTS_DIR / figure_name
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure {figure_name} not found")
    return FileResponse(fig_path, media_type="image/png")


class GenerateRequest(BaseModel):
    class_name: str = "helmet"
    temperature: float = 0.8
    seed: Optional[int] = None
    num_samples: int = 4


@router.post("/generate")
def generate_samples(req: GenerateRequest):
    """Generate conditional samples from the prior with manifold truncation."""
    load_cvae_artifacts()
    if not is_loaded or cvae_model is None:
        raise HTTPException(status_code=503, detail="CVAE model not loaded")

    if req.class_name.lower() not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid class: {req.class_name}. Must be one of {CLASS_NAMES}")

    c_idx = CLASS_NAMES.index(req.class_name.lower())
    if req.seed is not None:
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)

    cvae_model.eval()
    tau = float(np.clip(req.temperature, 0.5, 0.95))
    with torch.no_grad():
        # Truncated latent sampling focuses on high-density class features, preventing muddy blur
        z = torch.randn(req.num_samples, cvae_model.latent_dim, device=device) * tau
        z = torch.clamp(z, -1.8 * tau, 1.8 * tau)
        samples = cvae_model.decode(z, c_idx)

    data_urls = [tensor_to_base64_png(samples[i], upscale=256, sharpen=True) for i in range(req.num_samples)]
    return {
        "class": req.class_name,
        "temperature": req.temperature,
        "num_samples": req.num_samples,
        "images": data_urls
    }


@router.post("/interpolate")
async def interpolate_crops(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    class1: str = Form("helmet"),
    class2: str = Form("helmet"),
    alpha: float = Form(0.5)
):
    """
    Encode two uploaded crops, perform Spherical Linear Interpolation (SLERP) at alpha,
    and decode to generate the blended intermediate crop.
    """
    load_cvae_artifacts()
    if not is_loaded:
        raise HTTPException(status_code=503, detail="CVAE model not loaded")

    c_idx1 = CLASS_NAMES.index(class1.lower()) if class1.lower() in CLASS_NAMES else 1
    c_idx2 = CLASS_NAMES.index(class2.lower()) if class2.lower() in CLASS_NAMES else 1

    content1 = await file1.read()
    content2 = await file2.read()

    img1 = Image.open(io.BytesIO(content1))
    img2 = Image.open(io.BytesIO(content2))

    t1 = pil_to_tensor(img1).to(device)
    t2 = pil_to_tensor(img2).to(device)

    cvae_model.eval()
    with torch.no_grad():
        mu1, _ = cvae_model.encode(t1, torch.tensor([c_idx1], device=device))
        mu2, _ = cvae_model.encode(t2, torch.tensor([c_idx2], device=device))

        # SLERP on latent means
        z_interp = slerp(mu1, mu2, alpha)

        # Interpolate class conditioning
        emb1 = cvae_model.dec_class_emb(torch.tensor([c_idx1], device=device))
        emb2 = cvae_model.dec_class_emb(torch.tensor([c_idx2], device=device))
        c_emb_interp = (1.0 - alpha) * emb1 + alpha * emb2

        recon = cvae_model.decode(z_interp, c_emb_interp)

    return {
        "alpha": alpha,
        "class1": class1,
        "class2": class2,
        "interpolated_image": tensor_to_base64_png(recon[0]),
        "orig1": tensor_to_base64_png(t1[0]),
        "orig2": tensor_to_base64_png(t2[0])
    }


class ClassMorphRequest(BaseModel):
    class_from: str = "head"
    class_to: str = "helmet"
    alpha: float = 0.5
    seed: Optional[int] = 42


@router.post("/class_morph")
def class_morph(req: ClassMorphRequest):
    """
    With a fixed latent code z, smoothly morph the conditioning embedding
    from class_from to class_to at interpolation ratio alpha.
    """
    load_cvae_artifacts()
    if not is_loaded:
        raise HTTPException(status_code=503, detail="CVAE model not loaded")

    c0 = CLASS_NAMES.index(req.class_from.lower())
    c1 = CLASS_NAMES.index(req.class_to.lower())

    if req.seed is not None:
        torch.manual_seed(req.seed)

    cvae_model.eval()
    with torch.no_grad():
        z = torch.randn(1, cvae_model.latent_dim, device=device)
        emb0 = cvae_model.dec_class_emb(torch.tensor([c0], device=device))
        emb1 = cvae_model.dec_class_emb(torch.tensor([c1], device=device))
        emb_interp = (1.0 - req.alpha) * emb0 + req.alpha * emb1
        recon = cvae_model.decode(z, emb_interp)

    return {
        "class_from": req.class_from,
        "class_to": req.class_to,
        "alpha": req.alpha,
        "image": tensor_to_base64_png(recon[0])
    }


@router.post("/reconstruct")
async def reconstruct_crop(
    file: UploadFile = File(...),
    class_name: str = Form("helmet")
):
    """Reconstruct an uploaded crop under a specified class conditioning."""
    load_cvae_artifacts()
    if not is_loaded:
        raise HTTPException(status_code=503, detail="CVAE model not loaded")

    c_idx = CLASS_NAMES.index(class_name.lower()) if class_name.lower() in CLASS_NAMES else 1
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    t = pil_to_tensor(img).to(device)

    cvae_model.eval()
    with torch.no_grad():
        mu, logvar = cvae_model.encode(t, torch.tensor([c_idx], device=device))
        recon = cvae_model.decode(mu, c_idx)

    orig_np = t[0].cpu().permute(1, 2, 0).numpy()
    recon_np = recon[0].cpu().permute(1, 2, 0).numpy()

    mse = float(np.mean((orig_np - recon_np) ** 2))
    psnr = float(sk_psnr(orig_np, recon_np, data_range=1.0))
    ssim = float(sk_ssim(orig_np, recon_np, data_range=1.0, channel_axis=2))

    return {
        "class_name": class_name,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "original_image": tensor_to_base64_png(t[0]),
        "reconstructed_image": tensor_to_base64_png(recon[0])
    }
