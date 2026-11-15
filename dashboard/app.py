"""
dashboard/app.py - FastAPI Backend for PPE Autoencoder & Downstream Classification Dashboard
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import transforms

import io
import glob
import json
import base64
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from PIL import Image
import numpy as np
import pandas as pd
import joblib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_ppe")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.model_def import ConvAE

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "autoencoder")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "autoencoder")
STATIC_DIR = os.path.join(BASE_DIR, "dashboard", "static")
CROPS_TEST_DIR = os.path.join(BASE_DIR, "crops_64", "test")

# Global state loaded once at startup
MODELS: Dict[str, Any] = {}


def load_artifacts():
    """Load all model weights, scalers, and config once at startup."""
    if MODELS.get("model") is not None:
        return MODELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting up: Target device is {device.type.upper()}", flush=True)

    # 1. Load model config
    config_path = os.path.join(RESULTS_DIR, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    else:
        model_config = {"latent_dim": 256, "in_channels": 3, "img_size": 64}

    latent_dim = model_config.get("latent_dim", 256)

    # 2. Load PyTorch ConvAE model
    weights_path = os.path.join(ARTIFACTS_DIR, "ae_best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(RESULTS_DIR, "ae_best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ARTIFACTS_DIR, "latent_256", "ae_best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(RESULTS_DIR, "latent_256", "ae_best.pt")

    model = ConvAE(in_channels=3, latent_dim=latent_dim).to(device)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[*] Successfully loaded ConvAE weights from {weights_path}", flush=True)
    else:
        print(f"[!] Warning: Weights not found at {weights_path}", flush=True)
    model.eval()

    # 3. Load Scikit-Learn Scaler and SVM
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(RESULTS_DIR, "scaler.joblib")

    svm_path = os.path.join(ARTIFACTS_DIR, "svm_model.joblib")
    if not os.path.exists(svm_path):
        svm_path = os.path.join(RESULTS_DIR, "svm_model.joblib")

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    svm = joblib.load(svm_path) if os.path.exists(svm_path) else None

    # 4. Load class names
    class_names_path = os.path.join(RESULTS_DIR, "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = ["head", "helmet", "vest"]

    # 5. Load 2D PCA & PCA Points
    pca_path = os.path.join(ARTIFACTS_DIR, "latent_pca2d.joblib")
    if not os.path.exists(pca_path):
        pca_path = os.path.join(RESULTS_DIR, "latent_pca2d.joblib")
    pca_2d = joblib.load(pca_path) if os.path.exists(pca_path) else None

    pca_points_path = os.path.join(RESULTS_DIR, "latent_pca_points.json")
    pca_points = json.load(open(pca_points_path, "r", encoding="utf-8")) if os.path.exists(pca_points_path) else []

    MODELS["device"] = device
    MODELS["model"] = model
    MODELS["scaler"] = scaler
    MODELS["svm"] = svm
    MODELS["class_names"] = class_names
    MODELS["config"] = model_config
    MODELS["pca_2d"] = pca_2d
    MODELS["pca_points"] = pca_points

    print("[*] All dashboard ML artifacts successfully initialized.", flush=True)
    return MODELS


app = FastAPI(
    title="PPE Generative AI Benchmark & Safety Dashboard",
    description="Multi-Model Generative AI & Safety Inspection Platform for PPE Crops",
    version="2.0.0"
)

# Aggressive cache prevention so browser always receives fresh JS/CSS/API updates
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Load ML artifacts at startup
load_artifacts()

# Model Registry Routers
from dashboard.models import (
    MODEL_REGISTRY,
    get_registered_models,
    get_model_info,
    autoencoder_router,
    cvae_router,
    cgan_router,
    diffusion_router,
    policy_router,
)

app.include_router(autoencoder_router)
app.include_router(cvae_router)
app.include_router(cgan_router)
app.include_router(diffusion_router)
app.include_router(policy_router)


@app.get("/api/models")
def list_benchmark_models():
    """Return catalog of all registered models across the benchmarking suite."""
    return {"status": "success", "models": get_registered_models()}


# Ensure static folder exists
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def pil_to_base64_png(img: Image.Image) -> str:
    """Encode PIL Image to a base64 PNG data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Convert float [0, 1] or uint8 [0, 255] numpy array to base64 PNG."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return pil_to_base64_png(Image.fromarray(arr))


@app.get("/")
def get_root():
    """Serve the single-page application dashboard."""
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {"message": "PPE Dashboard API is running. index.html not found in static/."}


@app.get("/api/health")
def get_health():
    """Health check endpoint indicating model and device status."""
    return {
        "status": "healthy",
        "device": str(MODELS.get("device", "cpu")),
        "model_loaded": MODELS.get("model") is not None,
        "scaler_loaded": MODELS.get("scaler") is not None,
        "svm_loaded": MODELS.get("svm") is not None,
        "latent_dim": MODELS.get("config", {}).get("latent_dim", 256),
        "classes": MODELS.get("class_names", ["head", "helmet", "vest"])
    }


@app.post("/api/predict")
def predict_crop(
    file: UploadFile = File(...),
    crop_x: Optional[float] = Form(None),
    crop_y: Optional[float] = Form(None),
    crop_w: Optional[float] = Form(None),
    crop_h: Optional[float] = Form(None)
):
    """
    Inference pipeline:
    1. Validates upload (PNG, JPG, WEBP, <= 10MB)
    2. Applies user crop box if specified
    3. Resizes to 64x64 LANCZOS, normalizes to [0, 1] via ToTensor
    4. Autoencoder eval with no_grad -> latent vector -> reconstruction
    5. Computes per-image MSE, MAE, PSNR, SSIM, absolute-error heatmap
    6. StandardScaler transform -> SVM predict
    7. Computes softmax over SVM decision_function (labelled "relative score, uncalibrated")
    8. 2D PCA coordinates of latent vector
    9. Violation flag if predicted class is "head"
    """
    valid_content_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    c_type = getattr(file, "content_type", None) or "image/png"
    if c_type not in valid_content_types and ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{c_type}'. Please upload a PNG, JPG, or WEBP image."
        )

    # Read file content safely
    if hasattr(file, "file") and hasattr(file.file, "read"):
        contents = file.file.read()
    else:
        contents = file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum allowed limit of 10 MB."
        )

    try:
        raw_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file: {str(e)}"
        )

    # 2. Crop selection if coordinates provided
    orig_w, orig_h = raw_img.size
    is_pre_cropped = (getattr(file, "filename", "") == "user_crop.png")
    is_sample_crop = (orig_w <= 64 and orig_h <= 64)
    if not is_pre_cropped and not is_sample_crop and all(isinstance(v, (int, float)) for v in [crop_x, crop_y, crop_w, crop_h]):
        if crop_w > 0 and crop_h > 0:
            cx = int(max(0, min(orig_w, crop_x)))
            cy = int(max(0, min(orig_h, crop_y)))
            cw = int(min(orig_w - cx, crop_w))
            ch = int(min(orig_h - cy, crop_h))
            cropped_img = raw_img.crop((cx, cy, cx + cw, cy + ch))
        else:
            cropped_img = raw_img
    else:
        cropped_img = raw_img

    # 3. Model input patch: 64x64 LANCZOS, ToTensor in [0, 1]
    img_64 = cropped_img.resize((64, 64), Image.Resampling.LANCZOS)
    tensor_transform = transforms.ToTensor()
    x = tensor_transform(img_64).unsqueeze(0).to(MODELS["device"])
    orig_64_np = np.array(img_64, dtype=np.float32) / 255.0

    # 4. Autoencoder Forward Pass
    model = MODELS["model"]
    scaler = MODELS["scaler"]
    svm = MODELS["svm"]
    class_names = MODELS["class_names"]
    pca_2d = MODELS["pca_2d"]

    with torch.no_grad():
        recon, z = model(x)

    recon_64_np = np.clip(recon[0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
    z_np = z.cpu().numpy()  # (1, 256)

    # 5. True Authentic Model Metrics (computed in 64x64 domain matching model design)
    diff_raw = orig_64_np - recon_64_np
    mse_val = float(np.mean(diff_raw ** 2))
    mae_val = float(np.mean(np.abs(diff_raw)))
    psnr_val = float(psnr(orig_64_np, recon_64_np, data_range=1.0))
    ssim_val = float(ssim(orig_64_np, recon_64_np, data_range=1.0, channel_axis=2))

    # 6. High-Clarity Edge Definition & Luminance Equalization (Preserves Fine PPE Contours)
    import cv2
    recon_u8 = (recon_64_np * 255.0).astype(np.uint8)
    recon_256 = cv2.resize(recon_u8, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    # Adaptive local contrast equalization (CLAHE on L-channel in LAB)
    lab = cv2.cvtColor(recon_256, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Edge-preserving high-frequency unsharp mask
    blur_edge = cv2.GaussianBlur(rgb_eq, (0, 0), sigmaX=1.2)
    detail = cv2.subtract(rgb_eq, blur_edge)
    sharp_256 = np.clip(cv2.addWeighted(rgb_eq, 1.0, detail, 2.0, 0), 0, 255).astype(np.uint8)
    recon_sharp_disp_pil = Image.fromarray(sharp_256)

    # 7. High-Resolution Display Rendering & Dual-Resolution Comparison
    orig_disp_pil = cropped_img.resize((256, 256), Image.Resampling.LANCZOS)
    orig_64_disp_pil = img_64.resize((256, 256), Image.Resampling.NEAREST)
    recon_raw_disp_pil = Image.fromarray(recon_u8).resize((256, 256), Image.Resampling.NEAREST)
    
    # Color-mapped residual error heatmap (|x - x̂|) using Inferno colormap
    err_map = np.mean(np.abs(diff_raw), axis=2)  # (64, 64)
    err_norm = np.clip(err_map / 0.30, 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps.get_cmap("inferno")
    except AttributeError:
        cmap = matplotlib.cm.get_cmap("inferno")
    rgba = cmap(err_norm)
    heatmap_u8 = (rgba[..., :3] * 255.0).astype(np.uint8)
    heatmap_256 = cv2.resize(heatmap_u8, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    heatmap_disp_pil = Image.fromarray(heatmap_256)

    # 8. Downstream SVM Classification
    if scaler is not None and svm is not None:
        z_scaled = scaler.transform(z_np)
        pred_idx = int(svm.predict(z_scaled)[0])
        pred_class = class_names[pred_idx]
        
        # Softmax over decision function for uncalibrated relative score
        decision = svm.decision_function(z_scaled)[0]
        exp_d = np.exp(decision - np.max(decision))
        softmax_scores = exp_d / np.sum(exp_d)
        class_scores = {c: round(float(s), 4) for c, s in zip(class_names, softmax_scores)}
    else:
        pred_class = "unknown"
        class_scores = {c: 0.0 for c in class_names}

    # 9. 2D PCA Projection
    if pca_2d is not None:
        pca_coords = [round(float(v), 4) for v in pca_2d.transform(z_np)[0]]
    else:
        pca_coords = [0.0, 0.0]

    is_violation = (pred_class == "head")

    return {
        "predicted_class": pred_class,
        "is_violation": is_violation,
        "violation_notice": "Safety violation: no helmet detected!" if is_violation else None,
        "class_scores": class_scores,
        "scores": class_scores,
        "score_type": "relative score, uncalibrated",
        "metrics": {
            "mse": round(mse_val, 6),
            "mae": round(mae_val, 4),
            "psnr": round(psnr_val, 2),
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4)
        },
        "metrics_photoreal": {
            "mse": round(mse_val, 6),
            "mae": round(mae_val, 4),
            "psnr": round(psnr_val, 2),
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4)
        },
        "metrics_raw": {
            "mse": round(mse_val, 6),
            "mae": round(mae_val, 4),
            "psnr": round(psnr_val, 2),
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4)
        },
        "original_crop_b64": pil_to_base64_png(orig_disp_pil),
        "original_b64": pil_to_base64_png(orig_disp_pil),
        "original_64_b64": pil_to_base64_png(orig_64_disp_pil),
        "reconstruction_b64": pil_to_base64_png(recon_raw_disp_pil),
        "reconstruction_sharpened_b64": pil_to_base64_png(recon_sharp_disp_pil),
        "reconstruction_photoreal_b64": pil_to_base64_png(recon_sharp_disp_pil),
        "error_heatmap_b64": pil_to_base64_png(heatmap_disp_pil),
        "latent_vector": [round(float(v), 4) for v in z_np[0]],
        "pca_2d": pca_coords,
        "pca_coords": pca_coords
    }


@app.get("/api/metrics")
def get_metrics():
    """
    Reads and aggregates all real evaluation results from results/autoencoder/*.csv.
    Never hardcoded.
    """
    data: Dict[str, Any] = {}

    # 1. Master summary
    summary_csv = os.path.join(RESULTS_DIR, "final_summary.csv")
    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        data["final_summary"] = df_sum.to_dict(orient="records")

    # 2. Overall reconstruction metrics (Train / Val / Test)
    recon_csv = os.path.join(RESULTS_DIR, "reconstruction_metrics.csv")
    if os.path.exists(recon_csv):
        df_recon = pd.read_csv(recon_csv)
        data["reconstruction_table"] = df_recon.to_dict(orient="records")
        # Direct lookup for convenient summary cards
        test_rows = df_recon[df_recon["Split"].astype(str).str.lower() == "test"]
        if not test_rows.empty:
            r = test_rows.iloc[0]
            data["reconstruction"] = {
                "test_mse": float(r.get("MSE (Mean)", 0.0)),
                "test_mae": float(r.get("MAE (Mean)", 0.0)),
                "test_psnr": float(r.get("PSNR (dB) (Mean)", 0.0)),
                "test_ssim": float(r.get("SSIM (Mean)", 0.0)),
            }

    # 3. Per-class reconstruction metrics
    class_recon_csv = os.path.join(RESULTS_DIR, "reconstruction_per_class_metrics.csv")
    if os.path.exists(class_recon_csv):
        df_cr = pd.read_csv(class_recon_csv)
        # Format for table display
        formatted_rows = []
        for _, row in df_cr.iterrows():
            c_name = str(row.get("Safety Class", ""))
            count = int(row.get("Test Count", 0))
            mse_str = str(row.get("Recon Loss (MSE)", "0.0"))
            mse_val = float(mse_str.split()[0]) if " " in mse_str else float(mse_str)
            psnr_str = str(row.get("PSNR (dB)", "0.0"))
            psnr_val = float(psnr_str.split()[0]) if " " in psnr_str else float(psnr_str)
            ssim_str = str(row.get("SSIM", "0.0"))
            ssim_val = float(ssim_str.split()[0]) if " " in ssim_str else float(ssim_str)
            formatted_rows.append({
                "class": c_name,
                "count": count,
                "mse": mse_val,
                "psnr": psnr_val,
                "ssim": ssim_val
            })
        data["per_class_recon"] = formatted_rows

    # 4. SVM classification metrics (Validation vs Test)
    svm_csv = os.path.join(RESULTS_DIR, "svm_metrics.csv")
    if os.path.exists(svm_csv):
        df_svm = pd.read_csv(svm_csv)
        svm_data: Dict[str, Any] = {}
        for _, row in df_svm.iterrows():
            sp = str(row.get("Split", "")).lower()
            svm_data[sp] = {
                "accuracy": float(row.get("Accuracy", 0.0)),
                "balanced_accuracy": float(row.get("Balanced Accuracy", 0.0)),
                "macro_precision": float(row.get("Macro Precision", 0.0)),
                "macro_recall": float(row.get("Macro Recall", 0.0)),
                "macro_f1": float(row.get("Macro F1", 0.0)),
            }
        data["svm"] = svm_data

    # 5. Baseline PCA comparison
    base_csv = os.path.join(RESULTS_DIR, "ae_vs_pca_baseline.csv")
    if os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
        base_rows = []
        for _, row in df_base.iterrows():
            base_rows.append({
                "metric": str(row.get("Metric", "")),
                "ae_val": str(row.get("AE Latent (256-d)", "")),
                "pca_val": str(row.get("Pixel-PCA (256-d)", "")),
                "delta": str(row.get("Difference (AE - PCA)", ""))
            })
        data["baseline_comparison"] = base_rows

    # 6. SVM Classification Report Text
    report_txt = os.path.join(RESULTS_DIR, "svm_classification_report.txt")
    if os.path.exists(report_txt):
        with open(report_txt, "r", encoding="utf-8") as f:
            data["classification_report_text"] = f.read()

    # 7. Available Figures list
    canonical_figs = [
        "loss_curve.png",
        "svm_confusion_AE-latent.png",
        "latent_pca.png",
        "reconstructions.png"
    ]
    figs_present = [f for f in canonical_figs if os.path.exists(os.path.join(RESULTS_DIR, f))]
    # Also add any other pngs in directory
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".png") and f not in figs_present:
            figs_present.append(f)
    data["figures"] = figs_present

    # 8. PCA Points for Latent scatter
    data["pca_points"] = MODELS.get("pca_points", [])

    return data


CURATED_SAMPLES = {
    "head": [
        "ds1_hard_hat_workers2035_3.png",
        "ds1_hard_hat_workers2488_3.png",
        "ds1_hard_hat_workers3141_4.png",
        "ds1_hard_hat_workers998_0.png",
        "ds1_hard_hat_workers3353_6.png",
        "ds1_hard_hat_workers953_5.png"
    ],
    "helmet": [
        "ds2_helmet_jacket_05379_1.png",
        "ds2_helmet_jacket_01241_1.png",
        "ds2_helmet_jacket_10042_1.png",
        "ds2_helmet_jacket_04586_1.png",
        "ds2_helmet_jacket_04549_1.png",
        "ds1_hard_hat_workers1001_0.png"
    ],
    "vest": [
        "ds2_helmet_jacket_07780_0.png",
        "ds2_helmet_jacket_01970_0.png",
        "ds2_helmet_jacket_05455_0.png",
        "ds2_helmet_jacket_02271_0.png",
        "ds2_helmet_jacket_03613_1.png",
        "ds2_helmet_jacket_00007_0.png"
    ]
}


@app.get("/api/samples")
def get_sample_crops(count_per_class: int = 4, randomize: bool = False):
    """Returns high-quality, prototypical sample crops from crops_64/test with true labels."""
    import random
    samples = []
    class_names = MODELS.get("class_names", ["head", "helmet", "vest"])

    for c_idx, c_name in enumerate(class_names):
        c_dir = os.path.join(CROPS_TEST_DIR, c_name)
        if not os.path.exists(c_dir):
            continue

        if not randomize and c_name in CURATED_SAMPLES:
            chosen = [os.path.join(c_dir, fname) for fname in CURATED_SAMPLES[c_name][:count_per_class]]
            chosen = [p for p in chosen if os.path.exists(p)]
        else:
            all_files = glob.glob(os.path.join(c_dir, "*.png"))
            if not all_files:
                continue
            # Shuffle or sample with quality check
            random.shuffle(all_files)
            chosen = all_files[:count_per_class]

        for p in chosen:
            try:
                im = Image.open(p).convert("RGB")
                samples.append({
                    "id": os.path.basename(p),
                    "filename": os.path.basename(p),
                    "label": c_name,
                    "class_name": c_name,
                    "class_index": c_idx,
                    "image_b64": pil_to_base64_png(im)
                })
            except Exception:
                pass

    return samples


@app.get("/results/{figure_name}")
def get_result_figure(figure_name: str):
    """Serve evaluation figures from results/autoencoder safely."""
    safe_name = os.path.basename(figure_name)
    fig_path = os.path.join(RESULTS_DIR, safe_name)
    if os.path.exists(fig_path) and os.path.isfile(fig_path):
        return FileResponse(fig_path)
    raise HTTPException(status_code=404, detail=f"Figure '{figure_name}' not found.")
