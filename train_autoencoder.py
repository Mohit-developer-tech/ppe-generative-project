"""
train_autoencoder.py
Reference Implementation for Model 1 of 5:
"Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops"

Full pipeline:
- Step 0: Verification, hardware detection, dataset check against crops_64/stats.csv
- Step 1: Full training of baseline ConvAE (latent_dim=256) with checkpointing every 10 epochs
- Step 2: Latent dimension sweep across [64, 128, 256, 512]
- Step 3: AE-v2 architectural variant with combined MSE + SSIM loss
- Step 4: Final evaluation on test split (touched once), raw-pixel PCA baseline, exports
"""
import os
import sys
import glob
import time
import math
import random
import shutil
import tempfile
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import sklearn
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


@dataclass
class Config:
    data_dir: str = "crops_64"
    output_dir: str = "results/autoencoder"
    in_channels: int = 3
    img_size: int = 64
    latent_dim: int = 256
    sweep_latent_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 10
    lr_patience: int = 4
    lr_factor: float = 0.5
    checkpoint_interval: int = 10
    seed: int = 42
    num_workers: int = 0
    use_amp: bool = False
    svm_tune_n: int = 8000
    run_baseline: bool = True
    run_sweep: bool = True
    run_ae_v2: bool = True
    smoke_test: bool = False
    force_train: bool = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)


class ConvAE(nn.Module):
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
        return self.decode(z), z


class ConvEncoderV2(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ConvDecoderV2(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(x)


class ConvAEv2(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoderV2(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoderV2(out_channels=in_channels, latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def ssim_loss_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
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
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
    return 1.0 - ssim_map.mean()


def get_data_loaders(config: Config, device: torch.device):
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])
    
    data_path = Path(config.data_dir)
    raw_train_ds = datasets.ImageFolder(root=str(data_path / "train"), transform=train_transform)
    raw_val_ds = datasets.ImageFolder(root=str(data_path / "val"), transform=eval_transform)
    raw_test_ds = datasets.ImageFolder(root=str(data_path / "test"), transform=eval_transform)
    class_names = raw_train_ds.classes
    
    if config.smoke_test:
        def make_subset(ds, n, seed):
            idx, _ = train_test_split(np.arange(len(ds)), train_size=min(n, len(ds)), stratify=np.array(ds.targets), random_state=seed)
            return Subset(ds, idx)
        train_ds = make_subset(raw_train_ds, 500, config.seed)
        val_ds = make_subset(raw_val_ds, 150, config.seed)
        test_ds = make_subset(raw_test_ds, 150, config.seed)
    else:
        train_ds, val_ds, test_ds = raw_train_ds, raw_val_ds, raw_test_ds
        
    pin = (device.type == "cuda")
    loaders = {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=pin),
        "val": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin),
        "test": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin)
    }
    return loaders, {"train": train_ds, "val": val_ds, "test": test_ds}, class_names


def find_latest_checkpoint(target_dir: str) -> Optional[str]:
    ckpts = glob.glob(os.path.join(target_dir, "checkpoint_epoch_*.pt"))
    if not ckpts:
        return None
    def get_epoch(p):
        try:
            return int(Path(p).stem.split("_")[-1])
        except Exception:
            return 0
    return sorted(ckpts, key=get_epoch)[-1]


def train_model_harness(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    save_dir: str,
    loss_mode: str = "mse",
    epochs_override: Optional[int] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = epochs_override if epochs_override is not None else (2 if config.smoke_test else config.epochs)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience, min_lr=1e-6
    )
    
    best_weights_path = os.path.join(save_dir, "ae_best.pt")
    history_csv_path = os.path.join(save_dir, "history.csv")
    
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    
    latest_ckpt = find_latest_checkpoint(save_dir)
    if latest_ckpt is not None and not config.force_train:
        print(f"[*] Resuming from checkpoint: {latest_ckpt}")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
        start_epoch = ckpt_data["epoch"] + 1
        best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
        patience_counter = ckpt_data.get("patience_counter", 0)
        history = ckpt_data.get("history", history)
    elif os.path.exists(best_weights_path) and not config.force_train:
        print(f"[*] Found trained model at {best_weights_path}. Loading weights.")
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        return model, history

    scaler_amp = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    for epoch in range(start_epoch, num_epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss_acc, train_samples = 0.0, 0
        for bx, _ in train_loader:
            bx = bx.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                recon, _ = model(bx)
                if loss_mode == "combined":
                    loss = F.mse_loss(recon, bx) + 0.1 * ssim_loss_torch(recon, bx)
                else:
                    loss = F.mse_loss(recon, bx)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss_acc += loss.item() * bx.size(0)
            train_samples += bx.size(0)
        ep_train = train_loss_acc / train_samples
        
        model.eval()
        val_loss_acc, val_samples = 0.0, 0
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    recon, _ = model(bx)
                    if loss_mode == "combined":
                        loss = F.mse_loss(recon, bx) + 0.1 * ssim_loss_torch(recon, bx)
                    else:
                        loss = F.mse_loss(recon, bx)
                val_loss_acc += loss.item() * bx.size(0)
                val_samples += bx.size(0)
        ep_val = val_loss_acc / val_samples
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(ep_val)
        
        history["epoch"].append(epoch)
        history["train_loss"].append(ep_train)
        history["val_loss"].append(ep_val)
        history["lr"].append(cur_lr)
        
        saved_tag = ""
        if ep_val < best_val_loss:
            best_val_loss = ep_val
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            saved_tag = " -> [Best Saved]"
        else:
            patience_counter += 1
            
        dur = time.time() - ep_start
        print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Train: {ep_train:.6f} | Val: {ep_val:.6f} | LR: {cur_lr:.2e} | Pat: {patience_counter:02d}/{config.patience:02d} | Time: {dur:.1f}s{saved_tag}", flush=True)
        
        if epoch % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "history": history
            }, ckpt_path)
            
        pd.DataFrame(history).to_csv(history_csv_path, index=False)
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    return model, history


def compute_dataloader_reconstruction_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Compute average per-image MSE, MAE, PSNR, and SSIM across a DataLoader."""
    model.eval()
    mse_vals, mae_vals, psnr_vals, ssim_vals = [], [], [], []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device, non_blocking=True)
            recon, _ = model(bx)
            orig_np = bx.cpu().permute(0, 2, 3, 1).numpy()
            recon_np = np.clip(recon.cpu().permute(0, 2, 3, 1).numpy(), 0.0, 1.0)
            
            for orig, rec in zip(orig_np, recon_np):
                diff = orig - rec
                img_mse = float(np.mean(diff ** 2))
                img_mae = float(np.mean(np.abs(diff)))
                img_psnr = float(psnr(orig, rec, data_range=1.0))
                img_ssim = float(ssim(orig, rec, data_range=1.0, channel_axis=2))
                
                mse_vals.append(img_mse)
                mae_vals.append(img_mae)
                psnr_vals.append(img_psnr)
                ssim_vals.append(img_ssim)
                
    return {
        "mse_mean": float(np.mean(mse_vals)),
        "mse_std": float(np.std(mse_vals)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals)),
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
    }


def extract_features_and_eval_val_svm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Config
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Extract latent features, tune SVM on a stratified train subset,
    and report the validation Macro-F1 score and optimal hyperparameters.
    """
    model.eval()
    def get_latents(dl):
        X_l, y_l = [], []
        with torch.no_grad():
            for bx, by in dl:
                bx = bx.to(device, non_blocking=True)
                z = model.encode(bx)
                X_l.append(z.cpu().numpy())
                y_l.append(by.numpy())
        return np.concatenate(X_l, axis=0), np.concatenate(y_l, axis=0)

    X_train, y_train = get_latents(train_loader)
    X_val, y_val = get_latents(val_loader)
    
    # Standardize features using training split statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Stratified tuning subset
    tune_n = min(config.svm_tune_n, len(X_train_scaled))
    if tune_n < len(X_train_scaled):
        X_tune, _, y_tune, _ = train_test_split(
            X_train_scaled, y_train, train_size=tune_n, stratify=y_train, random_state=config.seed
        )
    else:
        X_tune, y_tune = X_train_scaled, y_train

    gamma_default = 1.0 / (X_tune.shape[1] * float(X_tune.var()))
    best_f1 = -1.0
    best_bal_acc = -1.0
    best_params = {}
    
    for C_c in [1.0, 10.0]:
        for mult in [0.5, 1.0, 2.0]:
            g_c = mult * gamma_default
            clf = SVC(C=C_c, gamma=g_c, kernel="rbf", class_weight="balanced", random_state=config.seed)
            clf.fit(X_tune, y_tune)
            val_p = clf.predict(X_val_scaled)
            f1_val = f1_score(y_val, val_p, average="macro", zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_bal_acc = balanced_accuracy_score(y_val, val_p)
                best_params = {"C": C_c, "gamma": g_c, "gamma_mult": f"{mult}x"}
                
    return best_f1, best_bal_acc, best_params


def evaluate_full_test_split(model: nn.Module, loader: DataLoader, device: torch.device):
    """Compute per-image test reconstruction metrics across test split."""
    model.eval()
    mse_l, mae_l, psnr_l, ssim_l, lbl_l = [], [], [], [], []
    pairs = []
    
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            recon, _ = model(bx)
            orig_np = bx.cpu().permute(0, 2, 3, 1).numpy()
            recon_np = np.clip(recon.cpu().permute(0, 2, 3, 1).numpy(), 0.0, 1.0)
            
            for orig, rec, lbl in zip(orig_np, recon_np, by.numpy()):
                diff = orig - rec
                mse_l.append(float(np.mean(diff ** 2)))
                mae_l.append(float(np.mean(np.abs(diff))))
                psnr_l.append(float(psnr(orig, rec, data_range=1.0)))
                ssim_l.append(float(ssim(orig, rec, data_range=1.0, channel_axis=2)))
                lbl_l.append(int(lbl))
                if len(pairs) < 60:
                    pairs.append((orig, rec, int(lbl)))
                    
    df_imgs = pd.DataFrame({"mse": mse_l, "mae": mae_l, "psnr": psnr_l, "ssim": ssim_l, "label": lbl_l})
    return df_imgs, pairs


def extract_all_latents(model: nn.Module, loader: DataLoader, device: torch.device):
    """Extract latent vectors and labels from a DataLoader."""
    model.eval()
    X_l, y_l = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            z = model.encode(bx)
            X_l.append(z.cpu().numpy())
            y_l.append(by.numpy())
    return np.concatenate(X_l, axis=0), np.concatenate(y_l, axis=0)


def main():
    parser = argparse.ArgumentParser(description="ConvAE Training and Evaluation Benchmark")
    parser.add_argument("--step", type=str, default="all", choices=["all", "0", "1", "2", "3", "4"], help="Step to execute")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--no_force_train", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()

    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.patience = args.patience
    config.smoke_test = args.smoke_test
    if args.no_force_train:
        config.force_train = False

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Step 0: Integrity Check
    print("=" * 70)
    print("STEP 0 — RESET, INTEGRITY CHECKS & HARDWARE DETECTION")
    print("=" * 70)
    stats_csv_path = os.path.join(config.data_dir, "stats.csv")
    if os.path.exists(stats_csv_path):
        ref_stats = pd.read_csv(stats_csv_path)
        print("Reference stats (crops_64/stats.csv):")
        print(ref_stats.to_string(index=False))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nExecution Target Device: {device.type.upper()}")
    if device.type == "cpu":
        print("[!] WARNING: Running on CPU. Training with 100 epochs takes ~15.7 minutes per epoch.")
        print("[!] Checkpoints will be saved every 10 epochs for resilience.")
    else:
        print("[*] CUDA GPU detected. Enabling mixed precision autocast.")
        config.use_amp = True
        config.batch_size = 256
        
    loaders, datasets_dict, class_names = get_data_loaders(config, device)
    for split_name in ["train", "val", "test"]:
        ds = datasets_dict[split_name]
        targets = [lbl for _, lbl in ds] if isinstance(ds, Subset) else ds.targets
        counts = pd.Series(targets).value_counts().sort_index().to_dict()
        print(f"  Split: {split_name:<5} | Total: {len(ds):<5} | Class Breakdown: {counts}")

    if args.step == "0":
        print("\nStep 0 sanity checks complete.")
        return

    # Step 1: Baseline ConvAE (latent_dim=256)
    print("\n" + "=" * 70)
    print("STEP 1 — FULL TRAINING (BASELINE CONVAE, LATENT_DIM=256)")
    print("=" * 70)
    base_output_dir = os.path.join(config.output_dir, "latent_256")
    model_baseline = ConvAE(config.in_channels, 256).to(device)
    model_baseline, base_history = train_model_harness(
        model=model_baseline,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        device=device,
        save_dir=base_output_dir,
        loss_mode="mse"
    )
    val_metrics_256 = compute_dataloader_reconstruction_metrics(model_baseline, loaders["val"], device)
    val_f1_256, val_bal_acc_256, best_svm_params_256 = extract_features_and_eval_val_svm(
        model=model_baseline,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        config=config
    )
    print(f"Baseline Validation PSNR: {val_metrics_256['psnr_mean']:.2f} dB | Val SSIM: {val_metrics_256['ssim_mean']:.4f}")
    print(f"Baseline Validation SVM Macro-F1: {val_f1_256:.4f} (Optimal C: {best_svm_params_256.get('C')}, gamma: {best_svm_params_256.get('gamma_mult')})")
    
    # Save loss curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base_history["epoch"], base_history["train_loss"], label="Train Loss (MSE)", color="steelblue", lw=2)
    ax.plot(base_history["epoch"], base_history["val_loss"], label="Val Loss (MSE)", color="coral", lw=2)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Reconstruction Loss", fontweight="bold")
    ax.set_title("ConvAE (256-d) Convergence Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    curve_path = os.path.join(base_output_dir, "ae_loss_curve.png")
    plt.savefig(curve_path, dpi=300)
    plt.close(fig)
    
    # Mirror root weights & curve
    shutil.copy2(os.path.join(base_output_dir, "ae_best.pt"), os.path.join(config.output_dir, "ae_best.pt"))
    shutil.copy2(curve_path, os.path.join(config.output_dir, "ae_loss_curve.png"))
    pd.DataFrame(base_history).to_csv(os.path.join(config.output_dir, "history.csv"), index=False)

    if args.step == "1":
        print("\nStep 1 baseline training and evaluation complete.")
        return

    # Step 2: Latent Size Sweep
    sweep_results = []
    if config.run_sweep and not config.smoke_test:
        print("\n" + "=" * 70)
        print("STEP 2 — LATENT SIZE SWEEP [64, 128, 256, 512]")
        print("=" * 70)
        for l_dim in config.sweep_latent_dims:
            sub_dir = os.path.join(config.output_dir, f"latent_{l_dim}")
            if l_dim == 256:
                sw_model = model_baseline
                sw_metrics = val_metrics_256
                sw_f1, sw_bal_acc, sw_params = val_f1_256, val_bal_acc_256, best_svm_params_256
            else:
                print(f"\n--- Training Latent Dimension: {l_dim} ---")
                sw_model = ConvAE(in_channels=config.in_channels, latent_dim=l_dim).to(device)
                sw_model, sw_hist = train_model_harness(
                    model=sw_model,
                    train_loader=loaders["train"],
                    val_loader=loaders["val"],
                    config=config,
                    device=device,
                    save_dir=sub_dir,
                    loss_mode="mse"
                )
                sw_metrics = compute_dataloader_reconstruction_metrics(sw_model, loaders["val"], device)
                sw_f1, sw_bal_acc, sw_params = extract_features_and_eval_val_svm(
                    model=sw_model,
                    train_loader=loaders["train"],
                    val_loader=loaders["val"],
                    device=device,
                    config=config
                )
            
            sweep_results.append({
                "Latent Dim": l_dim,
                "Val MSE": sw_metrics["mse_mean"],
                "Val PSNR (dB)": sw_metrics["psnr_mean"],
                "Val SSIM": sw_metrics["ssim_mean"],
                "Val SVM Macro-F1": sw_f1,
                "Val SVM Balanced Acc": sw_bal_acc,
                "Optimal C": sw_params.get("C"),
                "Optimal Gamma": sw_params.get("gamma_mult")
            })
            
        df_sweep = pd.DataFrame(sweep_results)
        sweep_csv_path = os.path.join(config.output_dir, "latent_sweep_comparison.csv")
        df_sweep.to_csv(sweep_csv_path, index=False)
        print(f"\nLatent Sweep Comparison Table saved -> {sweep_csv_path}")
        print(df_sweep.to_string(index=False))

    # Step 3: AE-v2 Variant
    model_v2 = None
    val_metrics_v2 = None
    val_f1_v2 = None
    if config.run_ae_v2:
        print("\n" + "=" * 70)
        print("STEP 3 — ARCHITECTURAL VARIANT (AE-V2 DUAL CONV + COMBINED LOSS)")
        print("=" * 70)
        v2_output_dir = os.path.join(config.output_dir, "ae_v2_variant")
        model_v2 = ConvAEv2(in_channels=config.in_channels, latent_dim=256).to(device)
        model_v2, history_v2 = train_model_harness(
            model=model_v2,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=config,
            device=device,
            save_dir=v2_output_dir,
            loss_mode="combined"
        )
        val_metrics_v2 = compute_dataloader_reconstruction_metrics(model_v2, loaders["val"], device)
        val_f1_v2, val_bal_acc_v2, best_svm_params_v2 = extract_features_and_eval_val_svm(
            model=model_v2,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            config=config
        )
        v2_comp_df = pd.DataFrame([
            {
                "Model Variant": "ConvAE (Reference, MSE)",
                "Val MSE": val_metrics_256["mse_mean"],
                "Val PSNR (dB)": val_metrics_256["psnr_mean"],
                "Val SSIM": val_metrics_256["ssim_mean"],
                "Val SVM Macro-F1": val_f1_256,
            },
            {
                "Model Variant": "ConvAEv2 (Dual-Conv + SSIM Loss)",
                "Val MSE": val_metrics_v2["mse_mean"],
                "Val PSNR (dB)": val_metrics_v2["psnr_mean"],
                "Val SSIM": val_metrics_v2["ssim_mean"],
                "Val SVM Macro-F1": val_f1_v2,
            }
        ])
        v2_csv_path = os.path.join(config.output_dir, "ae_v1_vs_v2_comparison.csv")
        v2_comp_df.to_csv(v2_csv_path, index=False)
        print(f"AE-v1 vs AE-v2 Comparison saved -> {v2_csv_path}")
        print(v2_comp_df.to_string(index=False))

    # Model Selection strictly on Val
    candidates = [
        {"name": "ConvAE (latent_dim=256)", "model": model_baseline, "val_f1": val_f1_256, "val_psnr": val_metrics_256["psnr_mean"], "dir": base_output_dir}
    ]
    if config.run_ae_v2 and model_v2 is not None and val_f1_v2 is not None:
        candidates.append({
            "name": "ConvAEv2 (latent_dim=256, dual-loss)", "model": model_v2, "val_f1": val_f1_v2, "val_psnr": val_metrics_v2["psnr_mean"], "dir": v2_output_dir
        })
    best_cand = max(candidates, key=lambda c: (c["val_f1"], c["val_psnr"]))
    winning_model = best_cand["model"]
    winning_name = best_cand["name"]
    print(f"\n[WINNER SELECTED ON VALIDATION]: {winning_name} (Val Macro-F1: {best_cand['val_f1']:.4f})")

    # Step 4: Final Evaluation on Test Split (touched ONCE)
    print("\n" + "=" * 70)
    print("STEP 4 — FINAL BENCHMARK EVALUATION (TEST SPLIT TOUCHED ONCE)")
    print("=" * 70)
    test_df_imgs, sample_pairs = evaluate_full_test_split(winning_model, loaders["test"], device)
    
    df_test_overall = pd.DataFrame([{
        "Model": winning_name,
        "Test Samples": len(test_df_imgs),
        "Recon Loss (MSE)": f"{test_df_imgs['mse'].mean():.6f} +/- {test_df_imgs['mse'].std():.6f}",
        "MAE": f"{test_df_imgs['mae'].mean():.4f} +/- {test_df_imgs['mae'].std():.4f}",
        "PSNR (dB)": f"{test_df_imgs['psnr'].mean():.2f} +/- {test_df_imgs['psnr'].std():.2f}",
        "SSIM": f"{test_df_imgs['ssim'].mean():.4f} +/- {test_df_imgs['ssim'].std():.4f}",
    }])
    df_test_overall.to_csv(os.path.join(config.output_dir, "reconstruction_metrics.csv"), index=False)
    
    class_rows = []
    for c_idx, c_name in enumerate(class_names):
        c_sub = test_df_imgs[test_df_imgs["label"] == c_idx]
        class_rows.append({
            "Safety Class": c_name,
            "Test Count": len(c_sub),
            "Recon Loss (MSE)": f"{c_sub['mse'].mean():.6f} +/- {c_sub['mse'].std():.6f}",
            "MAE": f"{c_sub['mae'].mean():.4f} +/- {c_sub['mae'].std():.4f}",
            "PSNR (dB)": f"{c_sub['psnr'].mean():.2f} +/- {c_sub['psnr'].std():.2f}",
            "SSIM": f"{c_sub['ssim'].mean():.4f} +/- {c_sub['ssim'].std():.4f}",
        })
    df_per_class = pd.DataFrame(class_rows)
    df_per_class.to_csv(os.path.join(config.output_dir, "reconstruction_per_class_metrics.csv"), index=False)

    # Extract all latents
    X_tr, y_tr = extract_all_latents(winning_model, loaders["train"], device)
    X_va, y_va = extract_all_latents(winning_model, loaders["val"], device)
    X_te, y_te = extract_all_latents(winning_model, loaders["test"], device)
    np.savez_compressed(
        os.path.join(config.output_dir, "latent_features.npz"),
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        classes=np.array(class_names)
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    tune_n = min(config.svm_tune_n, len(X_tr_s))
    X_tune, _, y_tune, _ = train_test_split(X_tr_s, y_tr, train_size=tune_n, stratify=y_tr, random_state=config.seed)
    gamma_def = 1.0 / (X_tune.shape[1] * float(X_tune.var()))
    best_val_f1, best_svm_params = -1.0, {}
    tuning_records = []
    for C_val in [1.0, 10.0]:
        for mult in [0.5, 1.0, 2.0]:
            g_val = mult * gamma_def
            clf = SVC(C=C_val, gamma=g_val, kernel="rbf", class_weight="balanced", random_state=config.seed)
            clf.fit(X_tune, y_tune)
            pred_val = clf.predict(X_va_s)
            f1_v = f1_score(y_va, pred_val, average="macro", zero_division=0)
            tuning_records.append({
                "C": C_val, "gamma_mult": f"{mult}x", "gamma": g_val,
                "Val Macro-F1": f1_v, "Val Balanced Acc": balanced_accuracy_score(y_va, pred_val)
            })
            if f1_v > best_val_f1:
                best_val_f1 = f1_v
                best_svm_params = {"C": C_val, "gamma": g_val}
    pd.DataFrame(tuning_records).to_csv(os.path.join(config.output_dir, "svm_hyperparameter_tuning.csv"), index=False)

    best_svm = SVC(C=best_svm_params["C"], gamma=best_svm_params["gamma"], kernel="rbf", class_weight="balanced", random_state=config.seed)
    best_svm.fit(X_tr_s, y_tr)
    y_te_pred = best_svm.predict(X_te_s)

    test_svm_metrics = {
        "Model": winning_name,
        "Accuracy": accuracy_score(y_te, y_te_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_te, y_te_pred),
        "Macro Precision": precision_score(y_te, y_te_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_te, y_te_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_te, y_te_pred, average="macro", zero_division=0)
    }
    df_svm_test = pd.DataFrame([test_svm_metrics])
    df_svm_test.to_csv(os.path.join(config.output_dir, "svm_metrics.csv"), index=False)

    report_txt = classification_report(y_te, y_te_pred, target_names=class_names, digits=4)
    with open(os.path.join(config.output_dir, "svm_classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Confusion matrix
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
    ax.set_title(f"Normalized Confusion Matrix — {winning_name}", fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm_counts[i, j]:,})",
                    ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "svm_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Raw pixel baseline
    if config.run_baseline:
        def get_raw_flattened(dl):
            fl_x, fl_y = [], []
            for bx, by in dl:
                fl_x.append(bx.flatten(start_dim=1).numpy())
                fl_y.append(by.numpy())
            return np.concatenate(fl_x, axis=0), np.concatenate(fl_y, axis=0)

        X_tr_raw, y_tr_raw = get_raw_flattened(loaders["train"])
        X_va_raw, y_va_raw = get_raw_flattened(loaders["val"])
        X_te_raw, y_te_raw = get_raw_flattened(loaders["test"])
        
        pca_baseline = PCA(n_components=256, random_state=config.seed)
        X_tr_pca = pca_baseline.fit_transform(X_tr_raw)
        X_va_pca = pca_baseline.transform(X_va_raw)
        X_te_pca = pca_baseline.transform(X_te_raw)
        
        tune_n_p = min(config.svm_tune_n, len(X_tr_pca))
        X_tune_p, _, y_tune_p, _ = train_test_split(X_tr_pca, y_tr_raw, train_size=tune_n_p, stratify=y_tr_raw, random_state=config.seed)
        g_def_pca = 1.0 / (X_tune_p.shape[1] * float(X_tune_p.var()))
        best_pca_f1, best_pca_params = -1.0, {}
        for C_p in [1.0, 10.0]:
            for mult in [0.5, 1.0, 2.0]:
                g_p = mult * g_def_pca
                clf_p = SVC(C=C_p, gamma=g_p, kernel="rbf", class_weight="balanced", random_state=config.seed)
                clf_p.fit(X_tune_p, y_tune_p)
                pv = clf_p.predict(X_va_pca)
                f1_pv = f1_score(y_va_raw, pv, average="macro", zero_division=0)
                if f1_pv > best_pca_f1:
                    best_pca_f1 = f1_pv
                    best_pca_params = {"C": C_p, "gamma": g_p}
        pca_svm = SVC(C=best_pca_params["C"], gamma=best_pca_params["gamma"], kernel="rbf", class_weight="balanced", random_state=config.seed)
        pca_svm.fit(X_tr_pca, y_tr_raw)
        y_te_pca_pred = pca_svm.predict(X_te_pca)
        pca_metrics = {
            "Accuracy": accuracy_score(y_te_raw, y_te_pca_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_te_raw, y_te_pca_pred),
            "Macro Precision": precision_score(y_te_raw, y_te_pca_pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_te_raw, y_te_pca_pred, average="macro", zero_division=0),
            "Macro F1": f1_score(y_te_raw, y_te_pca_pred, average="macro", zero_division=0)
        }
        comp_rows = []
        for m_key in ["Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
            ae_val = test_svm_metrics[m_key]
            pca_val = pca_metrics[m_key]
            comp_rows.append({
                "Evaluation Metric": m_key,
                "Winning AE Latents": ae_val,
                "Raw Pixel PCA(256)": pca_val,
                "Delta (AE - PCA)": ae_val - pca_val
            })
        df_baseline_comp = pd.DataFrame(comp_rows)
        df_baseline_comp.to_csv(os.path.join(config.output_dir, "ae_vs_pca_baseline.csv"), index=False)

    # Master summary table
    master_rows = [
        {"Dimension": "Model Architecture", "Value": winning_name},
        {"Dimension": "Latent Dimensionality", "Value": str(winning_model.latent_dim)},
        {"Dimension": "Input Resolution", "Value": f"{config.img_size}x{config.img_size}x3"},
        {"Dimension": "Test Recon MSE", "Value": f"{test_df_imgs['mse'].mean():.6f} +/- {test_df_imgs['mse'].std():.6f}"},
        {"Dimension": "Test Recon MAE", "Value": f"{test_df_imgs['mae'].mean():.4f} +/- {test_df_imgs['mae'].std():.4f}"},
        {"Dimension": "Test PSNR (dB)", "Value": f"{test_df_imgs['psnr'].mean():.2f} +/- {test_df_imgs['psnr'].std():.2f}"},
        {"Dimension": "Test SSIM", "Value": f"{test_df_imgs['ssim'].mean():.4f} +/- {test_df_imgs['ssim'].std():.4f}"},
        {"Dimension": "Test SVM Accuracy", "Value": f"{test_svm_metrics['Accuracy']:.4f}"},
        {"Dimension": "Test SVM Balanced Acc", "Value": f"{test_svm_metrics['Balanced Accuracy']:.4f}"},
        {"Dimension": "Test SVM Macro Precision", "Value": f"{test_svm_metrics['Macro Precision']:.4f}"},
        {"Dimension": "Test SVM Macro Recall", "Value": f"{test_svm_metrics['Macro Recall']:.4f}"},
        {"Dimension": "Test SVM Macro F1", "Value": f"{test_svm_metrics['Macro F1']:.4f}"},
    ]
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(os.path.join(config.output_dir, "final_summary.csv"), index=False)
    print("\nMaster summary saved -> results/autoencoder/final_summary.csv")


if __name__ == "__main__":
    main()
