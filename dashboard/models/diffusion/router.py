"""
dashboard/models/diffusion/router.py
FastAPI router and artifact loader for Model 4: Conditional Diffusion Restoration (DDPM / DDIM).

Provides endpoints for:
  - Model status, specifications, and metrics (tables & 300 dpi figures)
  - Interactive image restoration with customizable degradation and DDIM step slider (10 to 100 steps)
  - Side-by-side comparison with the deterministic Regression U-Net baseline
  - Downstream classification before and after restoration (Independent Judge & AE+SVM)
  - Safety-critical hallucination audit: flags when an unprotected head is falsely restored as compliant PPE.
"""

import os
import io
import time
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from common.data import CLASS_NAMES
from common.degrade import (
    degrade_image, SEVERITY_LEVELS, VALID_TRAIN_DEGRADATIONS, HELD_OUT_DEGRADATION
)
from common.diffusion import (
    DiffusionSchedule, ConditionalUNet, RegressionUNet, sample_ddim
)
from common.judge import SmallResNet
from common.metrics import compute_psnr, compute_ssim, compute_mae
from dashboard.models.registry import get_model_info
import joblib

router = APIRouter(prefix="/api/diffusion", tags=["diffusion"])

# State variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion_model: Optional[ConditionalUNet] = None
regression_model: Optional[RegressionUNet] = None
diffusion_schedule: Optional[DiffusionSchedule] = None
diffusion_config: Dict[str, Any] = {}
diffusion_metrics: Dict[str, Any] = {}
diffusion_judge: Optional[SmallResNet] = None
ae_model = None
ae_scaler = None
ae_svm = None
is_loaded: bool = False

RESULTS_DIR = Path("results/diffusion")


def load_diffusion_artifacts() -> bool:
    """Load Diffusion weights, schedule, Regression baseline, config, and metrics."""
    global diffusion_model, regression_model, diffusion_schedule, diffusion_config
    global diffusion_metrics, diffusion_judge, ae_model, ae_scaler, ae_svm, is_loaded

    if is_loaded:
        return True

    weights_path = RESULTS_DIR / "diffusion_best.pt"
    reg_weights_path = RESULTS_DIR / "regression_best.pt"
    cfg_path = RESULTS_DIR / "model_config.json"
    metrics_path = RESULTS_DIR / "diffusion_metrics.json"

    # Always initialize schedule
    diffusion_schedule = DiffusionSchedule(timesteps=1000, device=device)

    # 1. Config
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            diffusion_config = json.load(f)
    base_ch = diffusion_config.get("base_channels", 32)

    # 2. Diffusion Model
    diffusion_model = ConditionalUNet(
        in_channels=6, out_channels=3,
        base_channels=base_ch
    ).to(device)

    # 3. Regression Baseline
    regression_model = RegressionUNet(base_channels=base_ch).to(device)

    if weights_path.exists():
        try:
            ckpt = torch.load(weights_path, map_location=device)
            if "ema_state_dict" in ckpt:
                diffusion_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
            elif "model_state_dict" in ckpt:
                diffusion_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                diffusion_model.load_state_dict(ckpt, strict=False)
            diffusion_model.eval()
            print(f"[*] Loaded Conditional Diffusion U-Net from {weights_path}", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not load diffusion weights: {e}")

    if reg_weights_path.exists():
        try:
            reg_ckpt = torch.load(reg_weights_path, map_location=device)
            regression_model.load_state_dict(reg_ckpt, strict=False)
            regression_model.eval()
            print(f"[*] Loaded Regression U-Net baseline from {reg_weights_path}", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not load regression weights: {e}")

    # 4. Metrics
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            diffusion_metrics = json.load(f)

    # 5. Judge Classifier
    diffusion_judge = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    for judge_p in ["results/judge/judge_seed_42.pt", "results/judge_v1/judge_seed_42.pt"]:
        if os.path.exists(judge_p):
            try:
                j_ckpt = torch.load(judge_p, map_location=device)
                sd = j_ckpt["model_state_dict"] if "model_state_dict" in j_ckpt else j_ckpt
                diffusion_judge.load_state_dict(sd)
                diffusion_judge.eval()
                break
            except Exception:
                pass

    # 6. AE+SVM
    ae_p = "results/autoencoder/ae_best.pt"
    sc_p = "results/autoencoder/scaler.joblib"
    svm_p = "results/autoencoder/svm_model.joblib"
    if os.path.exists(ae_p) and os.path.exists(sc_p) and os.path.exists(svm_p):
        try:
            from dashboard.model_def import ConvAE
            ae_model = ConvAE(latent_dim=256).to(device)
            ae_model.load_state_dict(torch.load(ae_p, map_location=device))
            ae_model.eval()
            ae_scaler = joblib.load(sc_p)
            ae_svm = joblib.load(svm_p)
        except Exception:
            pass

    is_loaded = True
    return True


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def classify_image(img_01: torch.Tensor) -> Dict[str, Any]:
    """Run Judge and AE+SVM on a single [1, 3, 64, 64] image in [0, 1]."""
    load_diffusion_artifacts()
    res = {"judge_class": "unknown", "judge_confidence": 0.0, "svm_class": "unknown"}

    if diffusion_judge is not None:
        diffusion_judge.eval()
        with torch.no_grad():
            logits = diffusion_judge(img_01.to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            res["judge_class"] = CLASS_NAMES[idx]
            res["judge_confidence"] = float(probs[idx])

    if ae_model is not None and ae_scaler is not None and ae_svm is not None:
        ae_model.eval()
        with torch.no_grad():
            latent = ae_model.encode(img_01.to(device)).cpu().numpy()
            scaled = ae_scaler.transform(latent)
            idx = int(ae_svm.predict(scaled)[0])
            res["svm_class"] = CLASS_NAMES[idx]
    else:
        res["svm_class"] = res["judge_class"]

    return res


# ==============================================================================
# Endpoints
# ==============================================================================

@router.get("/info")
def get_diffusion_info():
    """Return Model 4 (Diffusion Restoration) specification and registry metadata."""
    load_diffusion_artifacts()
    info = get_model_info("diffusion")
    has_weights = (RESULTS_DIR / "diffusion_best.pt").exists()
    return {
        "model_id": "diffusion",
        "name": info["name"],
        "status": "ready" if has_weights else "not_trained",
        "badge": "Active (Model 4)" if has_weights else info["badge"],
        "description": info["description"],
        "architecture": "SR3 Conditional U-Net (6ch input) + Sinusoidal Time MLP + Self-Attention + GroupNorm",
        "loss_objective": "Epsilon-L1 + Low-Noise abar_t-weighted LPIPS (t < 250)",
        "features": [
            "SR3 Concatenation Conditioning (Strictly No Label Leakage)",
            "Cosine Noise Schedule (T=1000 timesteps)",
            "Deterministic Fast DDIM Sampling (20, 50, 100 steps)",
            "Regression U-Net Baseline (Perception-Distortion Benchmarking)",
            "Held-Out Degradation Evaluation: Motion Blur",
            "Safety-Critical Class-Flip Audit (Head -> Compliant PPE)"
        ],
        "supported_degradations": list(VALID_TRAIN_DEGRADATIONS) + [HELD_OUT_DEGRADATION],
        "severity_levels": list(SEVERITY_LEVELS)
    }


@router.get("/metrics")
def get_diffusion_metrics_endpoint():
    """Return full quantitative benchmark tables, perception-distortion data, and step sweep."""
    load_diffusion_artifacts()

    bench_csv = RESULTS_DIR / "restoration_benchmark.csv"
    step_csv = RESULTS_DIR / "ddim_step_sweep.csv"
    tradeoff_csv = RESULTS_DIR / "perception_distortion.csv"

    if not bench_csv.exists():
        return {
            "status": "coming_soon",
            "model_id": "diffusion",
            "message": "Diffusion restoration benchmark results not found. Run scripts/train_diffusion.py."
        }

    df_bench = pd.read_csv(bench_csv)
    df_step = pd.read_csv(step_csv) if step_csv.exists() else pd.DataFrame()
    df_tradeoff = pd.read_csv(tradeoff_csv) if tradeoff_csv.exists() else pd.DataFrame()

    figures = [f.name for f in (RESULTS_DIR / "figures").glob("*.png")] if (RESULTS_DIR / "figures").exists() else []

    return {
        "status": "ready",
        "model_id": "diffusion",
        "summary": diffusion_metrics,
        "config": diffusion_config,
        "benchmark_table": df_bench.to_dict(orient="records"),
        "step_sweep_table": df_step.to_dict(orient="records") if not df_step.empty else [],
        "perception_distortion_table": df_tradeoff.to_dict(orient="records") if not df_tradeoff.empty else [],
        "figures": figures
    }


class RestoreRequest(BaseModel):
    sample_class: Optional[str] = "helmet"
    sample_index: Optional[int] = 0
    degradation_type: Optional[str] = "gaussian_noise"
    severity: Optional[str] = "medium"
    ddim_steps: Optional[int] = 20
    seed: Optional[int] = 42


@router.post("/restore")
def restore_crop(
    file: Optional[UploadFile] = File(None),
    degradation_type: str = Form("gaussian_noise"),
    severity: str = Form("medium"),
    ddim_steps: int = Form(20),
    sample_class: Optional[str] = Form(None)
):
    """
    Restore a PPE crop using Conditional DDIM sampling.
    Accepts either an uploaded file or selects a sample crop.
    Returns: clean (if available), degraded, restored, regression baseline, metrics, and safety audit.
    """
    load_diffusion_artifacts()
    clean_im_pil = None
    clean_tensor_01 = None

    # 1. Obtain input image
    if file is not None and getattr(file, "filename", None):
        try:
            contents = file.file.read() if hasattr(file, "file") else file.read()
            clean_im_pil = Image.open(io.BytesIO(contents)).convert("RGB").resize((64, 64))
            arr = np.array(clean_im_pil, dtype=np.float32) / 255.0
            clean_tensor_01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    else:
        # Load sample from test split
        s_class = sample_class or "helmet"
        c_idx = CLASS_NAMES.index(s_class) if s_class in CLASS_NAMES else 1
        cache_path = "data_cache/test_crops.pt"
        if os.path.exists(cache_path):
            test_data = torch.load(cache_path, map_location="cpu")
            mask = (test_data["y"] == c_idx)
            clean_tensor_01 = test_data["X"][mask][:1].float() / 255.0
            arr = clean_tensor_01[0].permute(1, 2, 0).numpy()
            clean_im_pil = Image.fromarray((arr * 255.0).round().astype(np.uint8))
        else:
            clean_tensor_01 = torch.rand(1, 3, 64, 64)
            clean_im_pil = Image.fromarray((clean_tensor_01[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    # 2. Degrade image if degradation_type != 'none'
    if degradation_type != "none":
        is_held_out = (degradation_type == HELD_OUT_DEGRADATION)
        deg_tensor_01 = degrade_image(
            clean_tensor_01[0],
            severity=severity,
            degradations=[degradation_type],
            seed=42,
            allow_held_out=is_held_out
        ).unsqueeze(0)
    else:
        deg_tensor_01 = clean_tensor_01.clone()

    deg_pil = Image.fromarray((deg_tensor_01[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))

    # Map to [-1, 1]
    deg_m11 = (deg_tensor_01.to(device) * 2.0) - 1.0

    # 3. Restore using DDIM
    t_start = time.time()
    steps = min(max(5, ddim_steps), 100)
    with torch.no_grad():
        restored_m11 = sample_ddim(diffusion_model, deg_m11, diffusion_schedule, steps=steps, device=device)
        restored_01 = torch.clamp((restored_m11 + 1.0) / 2.0, 0.0, 1.0).cpu()

        # Regression Baseline
        reg_m11 = regression_model(deg_m11)
        reg_01 = torch.clamp((reg_m11 + 1.0) / 2.0, 0.0, 1.0).cpu()

    latency_ms = (time.time() - t_start) * 1000

    restored_pil = Image.fromarray((restored_01[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))
    reg_pil = Image.fromarray((reg_01[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))

    # 4. Compute Metrics
    psnr_diff = compute_psnr(restored_01[0], clean_tensor_01[0])
    ssim_diff = compute_ssim(restored_01[0], clean_tensor_01[0])
    psnr_reg = compute_psnr(reg_01[0], clean_tensor_01[0])
    ssim_reg = compute_ssim(reg_01[0], clean_tensor_01[0])
    psnr_deg = compute_psnr(deg_tensor_01[0], clean_tensor_01[0])
    ssim_deg = compute_ssim(deg_tensor_01[0], clean_tensor_01[0])

    # 5. Classifications & Safety Audit
    clean_cls = classify_image(clean_tensor_01)
    deg_cls = classify_image(deg_tensor_01)
    restored_cls = classify_image(restored_01)
    reg_cls = classify_image(reg_01)

    # Detect safety-critical flip: head -> helmet/vest
    is_crit_flip = (clean_cls["judge_class"] == "head" and restored_cls["judge_class"] in ("helmet", "vest"))
    class_flipped = (clean_cls["judge_class"] != restored_cls["judge_class"])

    return {
        "status": "success",
        "clean_b64": pil_to_base64(clean_im_pil),
        "degraded_b64": pil_to_base64(deg_pil),
        "restored_b64": pil_to_base64(restored_pil),
        "regression_b64": pil_to_base64(reg_pil),
        "degradation_applied": degradation_type,
        "severity": severity,
        "ddim_steps": steps,
        "latency_ms": round(latency_ms, 1),
        "metrics": {
            "psnr_diffusion": round(psnr_diff, 2),
            "ssim_diffusion": round(ssim_diff, 4),
            "psnr_regression": round(psnr_reg, 2),
            "ssim_regression": round(ssim_reg, 4),
            "psnr_degraded": round(psnr_deg, 2),
            "ssim_degraded": round(ssim_deg, 4)
        },
        "classifications": {
            "clean": clean_cls,
            "degraded": deg_cls,
            "restored_diffusion": restored_cls,
            "restored_regression": reg_cls
        },
        "safety_audit": {
            "class_flipped": class_flipped,
            "safety_critical_flip": is_crit_flip,
            "warning": (
                "CRITICAL SAFETY WARNING: Hallucination detected! An unprotected head was falsely classified as compliant PPE after restoration."
                if is_crit_flip else None
            )
        }
    }


@router.get("/figure/{figure_name}")
def get_diffusion_figure(figure_name: str):
    """Serve publication figures from results/diffusion/figures/."""
    fig_path = RESULTS_DIR / "figures" / figure_name
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure '{figure_name}' not found.")
    return FileResponse(str(fig_path), media_type="image/png")
