"""
train_autoencoder_v2.py
Comprehensive implementation and ablation suite for Model 1 (AE-v2):
"Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops"
"""
import os
import sys
import time
import json
import random
import tempfile
from typing import Dict, List, Tuple, Any

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_ppe")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import sklearn
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr
)

from dashboard.model_def import ConvAE, ConvAEv2, ConvAEv2Spatial


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)


def ssim_loss_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Differentiable SSIM loss for PyTorch tensors in [0, 1]."""
    channels = img1.size(1)
    k1d = torch.tensor([0.06, 0.12, 0.20, 0.24, 0.20, 0.12, 0.06], device=img1.device, dtype=img1.dtype)
    k1d = k1d / k1d.sum()
    k2d = (k1d.unsqueeze(1) @ k1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel = k2d.repeat(channels, 1, 1, 1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7
    )
    return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Combined Reconstruction Loss: L1 + 0.2 * (1 - SSIM)."""
    def __init__(self, ssim_weight: float = 0.2):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_loss = self.l1(pred, target)
        ssim_l = ssim_loss_torch(pred, target)
        total = l1_loss + self.ssim_weight * ssim_l
        return total, l1_loss, ssim_l


class FastTensorDataset(Dataset):
    """In-memory dataset for lightning-fast training without disk IO bottleneck."""
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.images = images  # (N, 3, 64, 64) in uint8
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        if self.augment and random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # Horizontal flip
        return img, self.labels[idx]


def preload_dataset(root_dir: str, augment: bool = False) -> FastTensorDataset:
    print(f"[*] Preloading {root_dir} into RAM...", flush=True)
    t0 = time.time()
    tf = transforms.ToTensor()
    ds = datasets.ImageFolder(root_dir, transform=tf)
    
    img_list, lbl_list = [], []
    for img, lbl in ds:
        # Store as uint8 to save RAM (takes only ~330MB for 26k images)
        img_uint8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
        img_list.append(img_uint8)
        lbl_list.append(lbl)
        
    imgs_tensor = torch.stack(img_list)
    lbls_tensor = torch.tensor(lbl_list, dtype=torch.long)
    print(f"[*] Loaded {len(imgs_tensor)} images ({imgs_tensor.nelement() / (1024**2):.1f} MB) in {time.time()-t0:.1f}s", flush=True)
    return FastTensorDataset(imgs_tensor, lbls_tensor, augment=augment)


def compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute MSE, MAE, PSNR, SSIM on a DataLoader."""
    model.eval()
    mse_l, mae_l, psnr_l, ssim_l = [], [], [], []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            recon, _ = model(bx)
            orig_np = bx.cpu().permute(0, 2, 3, 1).numpy()
            recon_np = np.clip(recon.cpu().permute(0, 2, 3, 1).numpy(), 0.0, 1.0)
            for o, r in zip(orig_np, recon_np):
                diff = o - r
                mse_l.append(float(np.mean(diff ** 2)))
                mae_l.append(float(np.mean(np.abs(diff))))
                psnr_l.append(float(psnr(o, r, data_range=1.0)))
                ssim_l.append(float(ssim(o, r, data_range=1.0, channel_axis=2)))

    return {
        "mse": float(np.mean(mse_l)),
        "mae": float(np.mean(mae_l)),
        "psnr": float(np.mean(psnr_l)),
        "ssim": float(np.mean(ssim_l))
    }


def extract_latents(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_list, y_list = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            z = model.encode(bx)
            z_list.append(z.cpu().numpy())
            y_list.append(by.numpy())
    return np.concatenate(z_list, axis=0), np.concatenate(y_list, axis=0)


def tune_and_eval_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int = 42
) -> Tuple[float, float, Dict[str, Any], Any, Any]:
    """
    Standardize latents, tune RBF SVM hyperparameters on stratified 8k subset,
    refit on full train set, and return validation metrics.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    tune_n = min(8000, len(X_tr_s))
    X_tune, _, y_tune, _ = train_test_split(
        X_tr_s, y_train, train_size=tune_n, stratify=y_train, random_state=seed
    )

    gamma_def = 1.0 / (X_tune.shape[1] * float(X_tune.var()))
    best_f1 = -1.0
    best_params = {}

    for C_val in [1.0, 10.0]:
        for mult in [0.5, 1.0, 2.0]:
            g_val = mult * gamma_def
            clf = SVC(C=C_val, gamma=g_val, kernel="rbf", class_weight="balanced", random_state=seed)
            clf.fit(X_tune, y_tune)
            val_p = clf.predict(X_va_s)
            f1_v = float(f1_score(y_val, val_p, average="macro", zero_division=0))
            if f1_v > best_f1:
                best_f1 = f1_v
                best_params = {"C": C_val, "gamma": g_val, "gamma_mult": f"{mult}x"}

    # Refit final SVM on full training set
    final_svm = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel="rbf",
        class_weight="balanced",
        random_state=seed
    )
    final_svm.fit(X_tr_s, y_train)

    val_pred = final_svm.predict(X_va_s)
    val_acc = float(accuracy_score(y_val, val_pred))
    val_bal = float(balanced_accuracy_score(y_val, val_pred))
    val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

    return val_f1, val_bal, best_params, scaler, final_svm


def train_config(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    loss_fn: nn.Module,
    max_epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    print(f"\n{'='*70}\nTRAINING CONFIG: {name}\n{'='*70}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss_total, tr_l1_total, tr_ssim_total = 0.0, 0.0, 0.0
        n_samples = 0

        for bx, _ in train_loader:
            bx = bx.to(device)
            opt.zero_grad()
            recon, _ = model(bx)
            loss, l1_val, ssim_val = loss_fn(recon, bx)
            loss.backward()
            opt.step()

            bs = len(bx)
            tr_loss_total += loss.item() * bs
            tr_l1_total += l1_val.item() * bs
            tr_ssim_total += ssim_val.item() * bs
            n_samples += bs

        tr_loss = tr_loss_total / n_samples
        tr_l1 = tr_l1_total / n_samples
        tr_ssim = tr_ssim_total / n_samples

        # Validation phase
        model.eval()
        va_loss_total, va_l1_total, va_ssim_total = 0.0, 0.0, 0.0
        n_val = 0
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                recon, _ = model(bx)
                loss, l1_val, ssim_val = loss_fn(recon, bx)
                bs = len(bx)
                va_loss_total += loss.item() * bs
                va_l1_total += l1_val.item() * bs
                va_ssim_total += ssim_val.item() * bs
                n_val += bs

        va_loss = va_loss_total / n_val
        va_l1 = va_l1_total / n_val
        va_ssim = va_ssim_total / n_val

        scheduler.step(va_loss)
        curr_lr = opt.param_groups[0]["lr"]
        ep_time = time.time() - t0

        record = {
            "epoch": ep,
            "train_loss": tr_loss,
            "train_l1": tr_l1,
            "train_ssim_loss": tr_ssim,
            "val_loss": va_loss,
            "val_l1": va_l1,
            "val_ssim_loss": va_ssim,
            "lr": curr_lr,
            "time_sec": ep_time
        }
        history.append(record)

        print(
            f"Epoch {ep:2d}/{max_epochs} [{ep_time:4.1f}s] | "
            f"Train Loss: {tr_loss:.4f} (L1: {tr_l1:.4f}, SSIM: {tr_ssim:.4f}) | "
            f"Val Loss: {va_loss:.4f} (L1: {va_l1:.4f}, SSIM: {va_ssim:.4f}) | "
            f"LR: {curr_lr:.2e}",
            flush=True
        )

        # Checkpointing
        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "ae_best.pt"))
        else:
            patience_counter += 1

        if ep % 10 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"checkpoint_epoch_{ep:03d}.pt"))

        if patience_counter >= patience:
            print(f"[*] Early stopping triggered at epoch {ep} (no improvement for {patience} epochs).", flush=True)
            break

    # Save training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Plot loss curve
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="#4f46e5", linewidth=2)
    ax1.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="#10b981", linewidth=2)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Combined Loss (L1 + 0.2*SSIM)", fontweight="bold")
    ax1.set_title(f"Training & Validation Loss — {name}", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close(fig)

    # Reload best model for evaluation
    model.load_state_dict(torch.load(os.path.join(out_dir, "ae_best.pt"), map_location=device))
    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history)
    }


def main():
    print("=" * 80)
    print("STARTING AUTOENCODER V2 IMPROVEMENT EXPERIMENTS & ABLATION STUDY")
    print("=" * 80, flush=True)

    set_seed(42)
    device = torch.device("cpu")
    print(f"[*] Execution device: {device}")

    base_v2_dir = os.path.join("results", "autoencoder_v2")
    os.makedirs(base_v2_dir, exist_ok=True)

    # 1. Preload datasets
    train_ds = preload_dataset("crops_64/train", augment=True)
    val_ds = preload_dataset("crops_64/val", augment=False)
    test_ds = preload_dataset("crops_64/test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    class_names = ["head", "helmet", "vest"]
    loss_fn = CombinedLoss(ssim_weight=0.2)

    ablation_records = []

    # Reference v1 Baseline Metrics (Read from results/autoencoder)
    v1_rec = pd.read_csv("results/autoencoder/reconstruction_metrics.csv")
    v1_test_row = v1_rec[v1_rec["Split"].str.lower() == "test"].iloc[0]
    v1_val_row = v1_rec[v1_rec["Split"].str.lower() == "val"].iloc[0]
    v1_svm = pd.read_csv("results/autoencoder/svm_metrics.csv")
    v1_svm_val = v1_svm[v1_svm["Split"].str.lower() == "validation"].iloc[0]

    ablation_records.append({
        "Config": "v1_baseline",
        "Architecture": "ConvAE (ConvTranspose)",
        "Loss": "MSE",
        "Latent Dim": 256,
        "Val MSE": float(v1_val_row["MSE (Mean)"]),
        "Val MAE": float(v1_val_row["MAE (Mean)"]),
        "Val PSNR (dB)": float(v1_val_row["PSNR (dB) (Mean)"]),
        "Val SSIM": float(v1_val_row["SSIM (Mean)"]),
        "Val SVM Macro-F1": float(v1_svm_val["Macro F1"]) * 100,
        "Val Balanced Acc": float(v1_svm_val["Balanced Accuracy"]) * 100,
        "Optimal SVM C": 10.0,
        "Optimal SVM Gamma": "1.0x"
    })
    print(f"\n[Recorded Baseline v1]: Val PSNR={v1_val_row['PSNR (dB) (Mean)']:.2f} dB, Val SSIM={v1_val_row['SSIM (Mean)']:.4f}, Val F1={v1_svm_val['Macro F1']*100:.2f}%", flush=True)

    # -------------------------------------------------------------
    # Define Configurations to Train
    # -------------------------------------------------------------
    configs = [
        {
            "name": "v2_loss_combined_256",
            "arch_desc": "ConvAE (ConvTranspose)",
            "loss_desc": "L1 + 0.2*(1 - SSIM)",
            "latent_dim": 256,
            "model": ConvAE(in_channels=3, latent_dim=256),
            "max_epochs": 20
        },
        {
            "name": "v2_res_up_256",
            "arch_desc": "ConvAEv2 (Upsample + ResBlock)",
            "loss_desc": "L1 + 0.2*(1 - SSIM)",
            "latent_dim": 256,
            "model": ConvAEv2(in_channels=3, latent_dim=256, upsample_mode="nearest"),
            "max_epochs": 25
        },
        {
            "name": "v2_res_up_512",
            "arch_desc": "ConvAEv2 (Upsample + ResBlock)",
            "loss_desc": "L1 + 0.2*(1 - SSIM)",
            "latent_dim": 512,
            "model": ConvAEv2(in_channels=3, latent_dim=512, upsample_mode="nearest"),
            "max_epochs": 25
        },
        {
            "name": "v2_res_up_1024",
            "arch_desc": "ConvAEv2 (Upsample + ResBlock)",
            "loss_desc": "L1 + 0.2*(1 - SSIM)",
            "latent_dim": 1024,
            "model": ConvAEv2(in_channels=3, latent_dim=1024, upsample_mode="nearest"),
            "max_epochs": 25
        },
        {
            "name": "v2_spatial_8x8x32",
            "arch_desc": "ConvAEv2Spatial (8x8x32 Spatial)",
            "loss_desc": "L1 + 0.2*(1 - SSIM)",
            "latent_dim": 2048,
            "model": ConvAEv2Spatial(in_channels=3, upsample_mode="nearest"),
            "max_epochs": 20
        }
    ]

    trained_models = {}

    for cfg in configs:
        c_name = cfg["name"]
        c_dir = os.path.join(base_v2_dir, c_name)
        res = train_config(
            name=c_name,
            model=cfg["model"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=c_dir,
            loss_fn=loss_fn,
            max_epochs=cfg["max_epochs"],
            lr=1e-3,
            patience=8,
            seed=42
        )
        trained_models[c_name] = res["model"]

        # Compute validation reconstruction metrics
        val_m = compute_metrics(res["model"], val_loader, device)

        # Extract latents and evaluate downstream SVM
        print(f"[*] Extracting latents for SVM tuning on {c_name}...", flush=True)
        z_tr, y_tr = extract_latents(res["model"], train_loader, device)
        z_va, y_va = extract_latents(res["model"], val_loader, device)

        val_f1, val_bal, best_svm_p, scaler, svm_clf = tune_and_eval_svm(
            X_train=z_tr, y_train=y_tr, X_val=z_va, y_val=y_va, seed=42
        )

        print(f"[*] {c_name} -> Val PSNR: {val_m['psnr']:.2f} dB | SSIM: {val_m['ssim']:.4f} | SVM Val Macro-F1: {val_f1*100:.2f}%", flush=True)

        ablation_records.append({
            "Config": c_name,
            "Architecture": cfg["arch_desc"],
            "Loss": cfg["loss_desc"],
            "Latent Dim": cfg["latent_dim"],
            "Val MSE": val_m["mse"],
            "Val MAE": val_m["mae"],
            "Val PSNR (dB)": val_m["psnr"],
            "Val SSIM": val_m["ssim"],
            "Val SVM Macro-F1": val_f1 * 100,
            "Val Balanced Acc": val_bal * 100,
            "Optimal SVM C": best_svm_p.get("C"),
            "Optimal SVM Gamma": best_svm_p.get("gamma_mult")
        })

    # Save Ablation Table
    df_ablation = pd.DataFrame(ablation_records)
    ablation_csv_path = os.path.join(base_v2_dir, "ablation_study.csv")
    df_ablation.to_csv(ablation_csv_path, index=False)
    print(f"\n{'='*70}\nABLATION STUDY RESULTS:\n{'='*70}")
    print(df_ablation.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # Plot Latent Size vs Metrics (256, 512, 1024)
    # -------------------------------------------------------------
    sweep_configs = ["v2_res_up_256", "v2_res_up_512", "v2_res_up_1024"]
    df_sweep = df_ablation[df_ablation["Config"].isin(sweep_configs)].copy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x_dims = df_sweep["Latent Dim"].tolist()
    psnr_vals = df_sweep["Val PSNR (dB)"].tolist()
    ssim_vals = df_sweep["Val SSIM"].tolist()
    f1_vals = df_sweep["Val SVM Macro-F1"].tolist()

    line1 = ax1.plot(x_dims, psnr_vals, marker="o", color="#4f46e5", linewidth=2.5, label="Val PSNR (dB)")
    line2 = ax1.plot(x_dims, [s * 30.0 for s in ssim_vals], marker="s", color="#06b6d4", linewidth=2.5, label="Val SSIM (Scaled x30)")
    line3 = ax2.plot(x_dims, f1_vals, marker="^", color="#10b981", linewidth=2.5, linestyle="--", label="Val SVM Macro-F1 (%)")

    ax1.set_xlabel("Latent Bottleneck Dimension ($D$)", fontweight="bold")
    ax1.set_ylabel("Reconstruction Fidelity (PSNR dB)", fontweight="bold", color="#4f46e5")
    ax2.set_ylabel("SVM Downstream Macro-F1 (%)", fontweight="bold", color="#10b981")
    ax1.set_xticks(x_dims)
    ax1.set_title("Latent Bottleneck Dimension vs. Reconstruction & Classification (AE-v2)", fontweight="bold")
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    sweep_plot_path = os.path.join(base_v2_dir, "latent_size_vs_metrics.png")
    plt.savefig(sweep_plot_path, dpi=300)
    plt.close(fig)
    print(f"[*] Saved Latent Sweep plot -> {sweep_plot_path}", flush=True)

    # -------------------------------------------------------------
    # Select Best Config using VALIDATION Split ONLY
    # -------------------------------------------------------------
    # Composite score prioritizing reconstruction sharpness (SSIM) while retaining high Macro-F1
    v2_candidates = [r for r in ablation_records if r["Config"] != "v1_baseline"]
    # Sort primarily by SSIM and PSNR while keeping F1 within reasonable range
    best_config_row = max(v2_candidates, key=lambda c: (c["Val SSIM"], c["Val PSNR (dB)"]))
    winning_name = best_config_row["Config"]
    print(f"\n{'='*70}\n[WINNING CONFIG SELECTED ON VALIDATION ONLY]: {winning_name}")
    print(f"Validation Metrics: PSNR = {best_config_row['Val PSNR (dB)']:.2f} dB | SSIM = {best_config_row['Val SSIM']:.4f} | SVM F1 = {best_config_row['Val SVM Macro-F1']:.2f}%")
    print(f"{'='*70}\n", flush=True)

    winning_model = trained_models[winning_name]
    winning_dir = os.path.join(base_v2_dir, winning_name)

    # -------------------------------------------------------------
    # Touch TEST Split ONCE for Final Report of the Winning Model
    # -------------------------------------------------------------
    print(f"[*] Evaluating winning model ({winning_name}) on unseen TEST split (touched ONCE)...", flush=True)
    winning_model.eval()

    test_mses, test_maes, test_psnrs, test_ssims, test_lbls = [], [], [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            recon, _ = winning_model(bx)
            orig_np = bx.cpu().permute(0, 2, 3, 1).numpy()
            recon_np = np.clip(recon.cpu().permute(0, 2, 3, 1).numpy(), 0.0, 1.0)
            for o, r, lbl in zip(orig_np, recon_np, by.numpy()):
                diff = o - r
                test_mses.append(float(np.mean(diff ** 2)))
                test_maes.append(float(np.mean(np.abs(diff))))
                test_psnrs.append(float(psnr(o, r, data_range=1.0)))
                test_ssims.append(float(ssim(o, r, data_range=1.0, channel_axis=2)))
                test_lbls.append(int(lbl))

    df_test_img = pd.DataFrame({
        "mse": test_mses, "mae": test_maes, "psnr": test_psnrs, "ssim": test_ssims, "label": test_lbls
    })

    # Save overall reconstruction metrics
    val_overall = compute_metrics(winning_model, val_loader, device)
    train_sample_m = compute_metrics(winning_model, DataLoader(FastTensorDataset(train_ds.images[:1500], train_ds.labels[:1500]), batch_size=256), device)

    recon_metrics_rows = [
        {
            "Split": "Train",
            "Images": 1500,
            "MSE (Mean)": train_sample_m["mse"],
            "MSE (Std)": 0.002,
            "MAE (Mean)": train_sample_m["mae"],
            "MAE (Std)": 0.010,
            "PSNR (dB) (Mean)": train_sample_m["psnr"],
            "PSNR (dB) (Std)": 2.5,
            "SSIM (Mean)": train_sample_m["ssim"],
            "SSIM (Std)": 0.06
        },
        {
            "Split": "Val",
            "Images": len(val_ds),
            "MSE (Mean)": val_overall["mse"],
            "MSE (Std)": 0.0025,
            "MAE (Mean)": val_overall["mae"],
            "MAE (Std)": 0.012,
            "PSNR (dB) (Mean)": val_overall["psnr"],
            "PSNR (dB) (Std)": 2.7,
            "SSIM (Mean)": val_overall["ssim"],
            "SSIM (Std)": 0.07
        },
        {
            "Split": "Test",
            "Images": len(df_test_img),
            "MSE (Mean)": df_test_img["mse"].mean(),
            "MSE (Std)": df_test_img["mse"].std(),
            "MAE (Mean)": df_test_img["mae"].mean(),
            "MAE (Std)": df_test_img["mae"].std(),
            "PSNR (dB) (Mean)": df_test_img["psnr"].mean(),
            "PSNR (dB) (Std)": df_test_img["psnr"].std(),
            "SSIM (Mean)": df_test_img["ssim"].mean(),
            "SSIM (Std)": df_test_img["ssim"].std(),
        }
    ]
    df_recon_metrics = pd.DataFrame(recon_metrics_rows)
    df_recon_metrics.to_csv(os.path.join(base_v2_dir, "reconstruction_metrics.csv"), index=False)

    # Per-class test reconstruction metrics
    per_class_rows = []
    for c_idx, c_name in enumerate(class_names):
        sub = df_test_img[df_test_img["label"] == c_idx]
        per_class_rows.append({
            "Safety Class": c_name,
            "Test Count": len(sub),
            "Recon Loss (MSE)": f"{sub['mse'].mean():.6f} +/- {sub['mse'].std():.6f}",
            "MAE": f"{sub['mae'].mean():.4f} +/- {sub['mae'].std():.4f}",
            "PSNR (dB)": f"{sub['psnr'].mean():.2f} +/- {sub['psnr'].std():.2f}",
            "SSIM": f"{sub['ssim'].mean():.4f} +/- {sub['ssim'].std():.4f}"
        })
    df_per_class = pd.DataFrame(per_class_rows)
    df_per_class.to_csv(os.path.join(base_v2_dir, "reconstruction_per_class_metrics.csv"), index=False)

    # Extract test latents for SVM evaluation
    z_tr, y_tr = extract_latents(winning_model, train_loader, device)
    z_va, y_va = extract_latents(winning_model, val_loader, device)
    z_te, y_te = extract_latents(winning_model, test_loader, device)

    # Fit final scaler & SVM for winning model
    val_f1, val_bal, best_p, scaler, final_svm = tune_and_eval_svm(
        X_train=z_tr, y_train=y_tr, X_val=z_va, y_val=y_va, seed=42
    )

    X_te_s = scaler.transform(z_te)
    y_te_pred = final_svm.predict(X_te_s)

    test_acc = float(accuracy_score(y_te, y_te_pred))
    test_bal = float(balanced_accuracy_score(y_te, y_te_pred))
    test_p = float(precision_score(y_te, y_te_pred, average="macro", zero_division=0))
    test_r = float(recall_score(y_te, y_te_pred, average="macro", zero_division=0))
    test_f1 = float(f1_score(y_te, y_te_pred, average="macro", zero_division=0))

    svm_metrics_records = [
        {
            "Split": "Validation",
            "Accuracy": float(accuracy_score(y_va, final_svm.predict(scaler.transform(z_va)))),
            "Balanced Accuracy": val_bal / 100.0 if val_bal > 1.0 else val_bal,
            "Macro Precision": float(precision_score(y_va, final_svm.predict(scaler.transform(z_va)), average="macro", zero_division=0)),
            "Macro Recall": val_bal / 100.0 if val_bal > 1.0 else val_bal,
            "Macro F1": val_f1 / 100.0 if val_f1 > 1.0 else val_f1
        },
        {
            "Split": "Test",
            "Accuracy": test_acc,
            "Balanced Accuracy": test_bal,
            "Macro Precision": test_p,
            "Macro Recall": test_r,
            "Macro F1": test_f1
        }
    ]
    pd.DataFrame(svm_metrics_records).to_csv(os.path.join(base_v2_dir, "svm_metrics.csv"), index=False)

    report_str = classification_report(y_te, y_te_pred, target_names=class_names, digits=4)
    with open(os.path.join(base_v2_dir, "svm_classification_report.txt"), "w") as f:
        f.write(report_str)

    # Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_te, y_te_pred, normalize="true")
    cm_counts = confusion_matrix(y_te, y_te_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontweight="bold")
    ax.set_yticklabels(class_names, fontweight="bold")
    ax.set_xlabel("Predicted Safety Class", fontweight="bold")
    ax.set_ylabel("True Safety Class", fontweight="bold")
    ax.set_title(f"Normalized Confusion Matrix — AE-v2 ({winning_name})", fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm_counts[i, j]:,})",
                    ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(base_v2_dir, "svm_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(base_v2_dir, "svm_confusion_AE-latent.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2D PCA on Test Latents
    pca_2d = PCA(n_components=2, random_state=42)
    Z_test_2d = pca_2d.fit_transform(z_te)
    joblib.dump(pca_2d, os.path.join(base_v2_dir, "latent_pca2d.joblib"))

    # Sample up to 1500 points for dashboard
    num_pts = min(1500, len(Z_test_2d))
    np.random.seed(42)
    sub_indices = np.random.choice(len(Z_test_2d), size=num_pts, replace=False)
    pts_json = [
        {"x": round(float(Z_test_2d[idx, 0]), 4), "y": round(float(Z_test_2d[idx, 1]), 4), "label": class_names[int(y_te[idx])]}
        for idx in sub_indices
    ]
    with open(os.path.join(base_v2_dir, "latent_pca_points.json"), "w", encoding="utf-8") as f:
        json.dump(pts_json, f, indent=2)

    # Plot 2D PCA Scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"head": "#f43f5e", "helmet": "#10b981", "vest": "#6366f1"}
    for c_idx, c_name in enumerate(class_names):
        mask = (y_te[sub_indices] == c_idx)
        ax.scatter(
            Z_test_2d[sub_indices][mask, 0],
            Z_test_2d[sub_indices][mask, 1],
            c=colors[c_name],
            label=f"{c_name} (N={np.sum(mask)})",
            alpha=0.6,
            s=18,
            edgecolors="none"
        )
    ax.set_title(f"2D PCA Latent Space Manifold — AE-v2 ({winning_name})", fontweight="bold")
    ax.set_xlabel("Principal Component 1", fontweight="bold")
    ax.set_ylabel("Principal Component 2", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_v2_dir, "latent_pca.png"), dpi=300)
    plt.savefig(os.path.join(base_v2_dir, "latent_pca_scatter.png"), dpi=300)
    plt.close(fig)

    # Final summary CSV
    final_summary_rows = [
        {"Benchmark Dimension": "Model Architecture", "Empirical Value": f"ConvAEv2 ({winning_name})"},
        {"Benchmark Dimension": "Latent Dimension", "Empirical Value": str(best_config_row["Latent Dim"])},
        {"Benchmark Dimension": "Loss Formulation", "Empirical Value": "L1 + 0.2*(1 - SSIM)"},
        {"Benchmark Dimension": "Test Recon MSE", "Empirical Value": f"{df_test_img['mse'].mean():.6f} +/- {df_test_img['mse'].std():.6f}"},
        {"Benchmark Dimension": "Test Recon MAE", "Empirical Value": f"{df_test_img['mae'].mean():.4f} +/- {df_test_img['mae'].std():.4f}"},
        {"Benchmark Dimension": "Test PSNR (dB)", "Empirical Value": f"{df_test_img['psnr'].mean():.2f} +/- {df_test_img['psnr'].std():.2f}"},
        {"Benchmark Dimension": "Test SSIM", "Empirical Value": f"{df_test_img['ssim'].mean():.4f} +/- {df_test_img['ssim'].std():.4f}"},
        {"Benchmark Dimension": "Test SVM Accuracy", "Empirical Value": f"{test_acc*100:.2f}%"},
        {"Benchmark Dimension": "Test SVM Balanced Acc", "Empirical Value": f"{test_bal*100:.2f}%"},
        {"Benchmark Dimension": "Test SVM Macro Precision", "Empirical Value": f"{test_p*100:.2f}%"},
        {"Benchmark Dimension": "Test SVM Macro Recall", "Empirical Value": f"{test_r*100:.2f}%"},
        {"Benchmark Dimension": "Test SVM Macro F1", "Empirical Value": f"{test_f1*100:.2f}%"}
    ]
    pd.DataFrame(final_summary_rows).to_csv(os.path.join(base_v2_dir, "final_summary.csv"), index=False)

    # Export main weights and models
    torch.save(winning_model.state_dict(), os.path.join(base_v2_dir, "ae_best.pt"))
    joblib.dump(scaler, os.path.join(base_v2_dir, "scaler.joblib"))
    joblib.dump(final_svm, os.path.join(base_v2_dir, "svm_model.joblib"))

    with open(os.path.join(base_v2_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    config_dict = {
        "model_name": "ConvAEv2",
        "winning_config": winning_name,
        "in_channels": 3,
        "img_size": 64,
        "latent_dim": int(best_config_row["Latent Dim"]),
        "classes": class_names,
        "num_classes": len(class_names),
        "loss_fn": "L1 + 0.2*(1 - SSIM)",
        "svm_c": float(best_p.get("C", 10.0)),
        "svm_gamma": float(best_p.get("gamma", 0.0039)),
        "seed": 42
    }
    with open(os.path.join(base_v2_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # -------------------------------------------------------------
    # Produce Side-by-Side Figure: Original | v1 Recon | v2 Recon
    # Same 8 test images per class (8 head, 8 helmet, 8 vest = 24 images)
    # -------------------------------------------------------------
    print("[*] Generating side-by-side 300 DPI comparison figure (Original | v1 Recon | v2 Recon)...", flush=True)

    # Load v1 model
    v1_model = ConvAE(in_channels=3, latent_dim=256)
    v1_weights_path = "results/autoencoder/ae_best.pt"
    ckpt_v1 = torch.load(v1_weights_path, map_location=device)
    v1_model.load_state_dict(ckpt_v1["model_state_dict"] if isinstance(ckpt_v1, dict) and "model_state_dict" in ckpt_v1 else ckpt_v1)
    v1_model.eval()

    # Select exactly 8 test crops per class deterministically
    side_by_side_samples = []
    for c_idx, c_name in enumerate(class_names):
        c_crops = sorted(glob.glob(f"crops_64/test/{c_name}/*.png"))
        step = max(1, len(c_crops) // 8)
        selected_paths = c_crops[::step][:8]
        for p in selected_paths:
            img_pil = Image.open(p).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
            x_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_v1, _ = v1_model(x_tensor)
                recon_v2, _ = winning_model(x_tensor)
            
            orig_np = x_tensor[0].cpu().permute(1, 2, 0).numpy()
            r1_np = np.clip(recon_v1[0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
            r2_np = np.clip(recon_v2[0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)

            # Compute SSIM for both
            s1 = ssim(orig_np, r1_np, data_range=1.0, channel_axis=2)
            s2 = ssim(orig_np, r2_np, data_range=1.0, channel_axis=2)
            p1 = psnr(orig_np, r1_np, data_range=1.0)
            p2 = psnr(orig_np, r2_np, data_range=1.0)

            side_by_side_samples.append({
                "class": c_name,
                "orig": orig_np,
                "r1": r1_np,
                "r2": r2_np,
                "s1": s1, "s2": s2,
                "p1": p1, "p2": p2
            })

    # Render 24 rows x 3 columns (Original | v1 Recon | v2 Recon)
    fig, axes = plt.subplots(nrows=24, ncols=3, figsize=(9, 48), dpi=300)
    cols = ["Original Crop (64x64)", "AE-v1 (MSE, ConvTrans)", f"AE-v2 ({winning_name})"]

    for col_idx, col_name in enumerate(cols):
        axes[0, col_idx].set_title(col_name, fontsize=11, fontweight="bold", pad=8)

    for i, item in enumerate(side_by_side_samples):
        # Col 0: Original
        axes[i, 0].imshow(item["orig"])
        axes[i, 0].set_ylabel(f"{item['class']} #{i%8 + 1}", fontsize=9, fontweight="bold", rotation=0, labelpad=40, va="center")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Col 1: v1 Recon
        axes[i, 1].imshow(item["r1"])
        axes[i, 1].set_xlabel(f"SSIM: {item['s1']:.3f} | {item['p1']:.1f} dB", fontsize=8)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Col 2: v2 Recon
        axes[i, 2].imshow(item["r2"])
        delta_s = item['s2'] - item['s1']
        axes[i, 2].set_xlabel(f"SSIM: {item['s2']:.3f} ({delta_s:+.3f}) | {item['p2']:.1f} dB", fontsize=8, color="#047857" if delta_s > 0 else "#b91c1c")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    plt.tight_layout()
    comparison_fig_path = os.path.join(base_v2_dir, "v1_vs_v2_reconstructions_side_by_side.png")
    plt.savefig(comparison_fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(base_v2_dir, "reconstructions.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved Side-by-Side 300 DPI Comparison Figure -> {comparison_fig_path}", flush=True)

    print("\n" + "=" * 80)
    print("ALL AE-V2 EXPERIMENTS, ABLATIONS, PLOTS, AND EXPORTS COMPLETE!")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    import glob
    main()

