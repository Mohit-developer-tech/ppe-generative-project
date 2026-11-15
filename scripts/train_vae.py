"""
train_vae.py
Reference Implementation for Model 2 of 5:
"Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops"
Model 2: Convolutional Variational Autoencoder (ConvVAE) Representation Learning & Generative Sampling

Pipeline:
- Step 0: Verification, hardware detection, dataset check against crops_64/stats.csv
- Step 1: Full training of baseline ConvVAE (latent_dim=256, beta=1e-4) with checkpointing every 10 epochs
- Step 2: Evaluation on Validation split & Beta/Latent tuning
- Step 3: Novel crop generation via prior sampling z ~ N(0, I)
- Step 4: Final evaluation on unseen Test split (reconstruction, downstream SVM on mu latents, comparative benchmark vs AE and PCA)
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
class VAEConfig:
    data_dir: str = "crops_64"
    output_dir: str = "results/vae"
    ae_reference_dir: str = "results/autoencoder"
    in_channels: int = 3
    img_size: int = 64
    latent_dim: int = 256
    beta: float = 1e-4             # Weight for KL divergence term relative to mean-MSE
    kl_warmup_epochs: int = 5      # Epochs to linearly anneal beta from 0 to target beta
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


class ConvVAEEncoder(nn.Module):
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
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.flatten(self.conv(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvVAEDecoder(nn.Module):
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
        x = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(x)


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvVAEEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvVAEDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent extraction for downstream classification (mu)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Prior sampling z ~ N(0, I) to generate novel crops."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Negative Evidence Lower Bound (ELBO):
    Loss = MSE(x, recon_x) + beta * KL(q(z|x) || p(z))
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    # KL divergence per batch item: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def get_data_loaders(config: VAEConfig, device: torch.device):
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


def train_vae_harness(
    model: ConvVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: VAEConfig,
    device: torch.device,
    save_dir: str,
    epochs_override: Optional[int] = None
) -> Tuple[ConvVAE, Dict[str, List[float]]]:
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = epochs_override if epochs_override is not None else (2 if config.smoke_test else config.epochs)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience, min_lr=1e-6
    )
    
    best_weights_path = os.path.join(save_dir, "vae_best.pt")
    history_csv_path = os.path.join(save_dir, "history.csv")
    
    # If already fully trained and force_train is False, load best weights immediately
    if os.path.exists(best_weights_path) and not config.force_train:
        print(f"[*] Found trained model at {best_weights_path}. Loading best weights.")
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        history = pd.read_csv(history_csv_path).to_dict(orient="list") if os.path.exists(history_csv_path) else {}
        return model, history

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "train_mse": [], "train_kl": [], "val_loss": [], "val_mse": [], "val_kl": [], "lr": []}
    
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

    print(f"Starting VAE Training: {num_epochs} Epochs | Target Beta: {config.beta:.1e} | Output: {save_dir}")
    print("-" * 85)

    scaler_amp = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    for epoch in range(start_epoch, num_epochs + 1):
        ep_start = time.time()
        model.train()
        
        # Linear KL warmup: beta anneals from 0 to target beta
        current_beta = config.beta * min(1.0, epoch / max(1, config.kl_warmup_epochs))
        
        train_tot, train_rec, train_kld, train_samples = 0.0, 0.0, 0.0, 0
        for bx, _ in train_loader:
            bx = bx.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=config.use_amp):
                recon, mu, logvar = model(bx)
                tot_loss, rec_loss, kl_loss = vae_loss_function(recon, bx, mu, logvar, beta=current_beta)
            scaler_amp.scale(tot_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            b_sz = bx.size(0)
            train_tot += tot_loss.item() * b_sz
            train_rec += rec_loss.item() * b_sz
            train_kld += kl_loss.item() * b_sz
            train_samples += b_sz
            
        ep_train = train_tot / train_samples
        ep_train_rec = train_rec / train_samples
        ep_train_kl = train_kld / train_samples
        
        # Validation pass
        model.eval()
        val_tot, val_rec, val_kld, val_samples = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=config.use_amp):
                    recon, mu, logvar = model(bx)
                    tot_loss, rec_loss, kl_loss = vae_loss_function(recon, bx, mu, logvar, beta=current_beta)
                b_sz = bx.size(0)
                val_tot += tot_loss.item() * b_sz
                val_rec += rec_loss.item() * b_sz
                val_kld += kl_loss.item() * b_sz
                val_samples += b_sz
                
        ep_val = val_tot / val_samples
        ep_val_rec = val_rec / val_samples
        ep_val_kl = val_kld / val_samples
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(ep_val)
        
        history["epoch"].append(epoch)
        history["train_loss"].append(ep_train)
        history["train_mse"].append(ep_train_rec)
        history["train_kl"].append(ep_train_kl)
        history["val_loss"].append(ep_val)
        history["val_mse"].append(ep_val_rec)
        history["val_kl"].append(ep_val_kl)
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
        print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Total: {ep_train:.5f} (Rec:{ep_train_rec:.5f}, KL:{ep_train_kl:.2f}) | Val: {ep_val:.5f} (Rec:{ep_val_rec:.5f}, KL:{ep_val_kl:.2f}) | Pat: {patience_counter:02d}/{config.patience:02d} | Time: {dur:.1f}s{saved_tag}", flush=True)
        
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


def compute_dataloader_reconstruction_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_vals, mae_vals, psnr_vals, ssim_vals = [], [], [], []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device, non_blocking=True)
            recon, _, _ = model(bx)
            orig_np = bx.cpu().permute(0, 2, 3, 1).numpy()
            recon_np = np.clip(recon.cpu().permute(0, 2, 3, 1).numpy(), 0.0, 1.0)
            
            for orig, rec in zip(orig_np, recon_np):
                diff = orig - rec
                mse_vals.append(float(np.mean(diff ** 2)))
                mae_vals.append(float(np.mean(np.abs(diff))))
                psnr_vals.append(float(psnr(orig, rec, data_range=1.0)))
                ssim_vals.append(float(ssim(orig, rec, data_range=1.0, channel_axis=2)))
                
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


def extract_features_and_eval_val_svm(model: ConvVAE, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, config: VAEConfig):
    model.eval()
    def get_latents(dl):
        X_l, y_l = [], []
        with torch.no_grad():
            for bx, by in dl:
                bx = bx.to(device, non_blocking=True)
                mu = model.encode(bx)
                X_l.append(mu.cpu().numpy())
                y_l.append(by.numpy())
        return np.concatenate(X_l, axis=0), np.concatenate(y_l, axis=0)

    X_train, y_train = get_latents(train_loader)
    X_val, y_val = get_latents(val_loader)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    tune_n = min(config.svm_tune_n, len(X_train_scaled))
    if tune_n < len(X_train_scaled):
        X_tune, _, y_tune, _ = train_test_split(
            X_train_scaled, y_train, train_size=tune_n, stratify=y_train, random_state=config.seed
        )
    else:
        X_tune, y_tune = X_train_scaled, y_train

    gamma_default = 1.0 / (X_tune.shape[1] * float(X_tune.var()))
    best_f1, best_bal_acc, best_params = -1.0, -1.0, {}
    
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


def evaluate_full_test_split(model: ConvVAE, loader: DataLoader, device: torch.device):
    model.eval()
    mse_l, mae_l, psnr_l, ssim_l, lbl_l = [], [], [], [], []
    pairs = []
    
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            recon, _, _ = model(bx)
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


def extract_all_latents(model: ConvVAE, loader: DataLoader, device: torch.device):
    model.eval()
    X_l, y_l = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            mu = model.encode(bx)
            X_l.append(mu.cpu().numpy())
            y_l.append(by.numpy())
    return np.concatenate(X_l, axis=0), np.concatenate(y_l, axis=0)


def generate_prior_samples(model: ConvVAE, num_samples: int, device: torch.device, save_path: str):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, device=device)
        samples_np = samples.cpu().permute(0, 2, 3, 1).numpy()

    cols = 8
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()
    for idx in range(len(axes)):
        if idx < num_samples:
            axes[idx].imshow(np.clip(samples_np[idx], 0.0, 1.0))
        axes[idx].axis("off")
    plt.suptitle("ConvVAE Novel Generated PPE Crops (z ~ N(0, I))", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prior generative samples -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="ConvVAE Representation Learning and Benchmarking")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--beta", type=float, default=1e-4, help="KL Divergence loss weight")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--no_force_train", action="store_true", help="Load cached weights if present")
    args = parser.parse_args()

    config = VAEConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.beta = args.beta
    config.smoke_test = args.smoke_test
    if args.no_force_train:
        config.force_train = False

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("STEP 0 — SETUP, INTEGRITY CHECKS & HARDWARE DETECTION (MODEL 2: VAE)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device.type.upper()}")
    
    loaders, datasets_dict, class_names = get_data_loaders(config, device)
    for s_name in ["train", "val", "test"]:
        ds = datasets_dict[s_name]
        targets = [lbl for _, lbl in ds] if isinstance(ds, Subset) else ds.targets
        counts = pd.Series(targets).value_counts().sort_index().to_dict()
        print(f"  Split: {s_name:<5} | Total: {len(ds):<5} | Class Breakdown: {counts}")

    print("\n" + "=" * 70)
    print("STEP 1 — CONVVAE TRAINING (LATENT_DIM=256, BETA=1e-4)")
    print("=" * 70)
    model = ConvVAE(config.in_channels, config.latent_dim).to(device)
    model, history = train_vae_harness(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        device=device,
        save_dir=config.output_dir
    )

    # Save convergence curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(history["epoch"], history["train_mse"], label="Train Recon (MSE)", color="steelblue", lw=2)
    ax1.plot(history["epoch"], history["val_mse"], label="Val Recon (MSE)", color="coral", lw=2)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Reconstruction MSE", fontweight="bold")
    ax1.set_title("Reconstruction Loss Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(history["epoch"], history["train_kl"], label="Train KL Divergence", color="forestgreen", lw=2)
    ax2.plot(history["epoch"], history["val_kl"], label="Val KL Divergence", color="purple", lw=2)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("KL Divergence", fontweight="bold")
    ax2.set_title("Latent KL Divergence Curve", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_curve_path = os.path.join(config.output_dir, "vae_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close(fig)

    # Generative prior sampling
    sample_grid_path = os.path.join(config.output_dir, "vae_generated_samples.png")
    generate_prior_samples(model, num_samples=32, device=device, save_path=sample_grid_path)

    # Step 4: Final Test Split Evaluation
    print("\n" + "=" * 70)
    print("STEP 4 — FINAL BENCHMARK EVALUATION (TEST SPLIT TOUCHED ONCE)")
    print("=" * 70)
    test_df_imgs, sample_pairs = evaluate_full_test_split(model, loaders["test"], device)
    
    df_test_overall = pd.DataFrame([{
        "Model": "ConvVAE (latent_dim=256, beta=1e-4)",
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
    pd.DataFrame(class_rows).to_csv(os.path.join(config.output_dir, "reconstruction_per_class_metrics.csv"), index=False)

    # Reconstruction Grid Figure (300 DPI)
    fig, axes = plt.subplots(len(class_names) * 2, 5, figsize=(11.0, len(class_names) * 4.0))
    for c_idx, c_name in enumerate(class_names):
        c_pairs = [p for p in sample_pairs if p[2] == c_idx][:5]
        row_o = c_idx * 2
        row_r = c_idx * 2 + 1
        for col_idx, (orig, rec, _) in enumerate(c_pairs):
            axes[row_o, col_idx].imshow(orig)
            axes[row_o, col_idx].axis("off")
            if col_idx == 0:
                axes[row_o, col_idx].set_title(f"[{c_name.upper()}] Original", fontsize=10, fontweight="bold", loc="left")
            axes[row_r, col_idx].imshow(rec)
            axes[row_r, col_idx].axis("off")
            p_val = psnr(orig, rec, data_range=1.0)
            s_val = ssim(orig, rec, data_range=1.0, channel_axis=2)
            axes[row_r, col_idx].set_title(f"PSNR:{p_val:.1f} SSIM:{s_val:.2f}", fontsize=8)
            if col_idx == 0:
                axes[row_r, col_idx].text(-0.1, 0.5, "Reconstruction", va="center", ha="right", fontsize=9, fontweight="bold", transform=axes[row_r, col_idx].transAxes)
    plt.suptitle("Qualitative Reconstruction Grid — ConvVAE", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "vae_reconstructions.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Latent Feature Extraction
    X_tr, y_tr = extract_all_latents(model, loaders["train"], device)
    X_va, y_va = extract_all_latents(model, loaders["val"], device)
    X_te, y_te = extract_all_latents(model, loaders["test"], device)
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
        "Model": "ConvVAE (latent_dim=256, beta=1e-4)",
        "Accuracy": accuracy_score(y_te, y_te_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_te, y_te_pred),
        "Macro Precision": precision_score(y_te, y_te_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_te, y_te_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_te, y_te_pred, average="macro", zero_division=0)
    }
    pd.DataFrame([test_svm_metrics]).to_csv(os.path.join(config.output_dir, "svm_metrics.csv"), index=False)

    report_txt = classification_report(y_te, y_te_pred, target_names=class_names, digits=4)
    with open(os.path.join(config.output_dir, "svm_classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Confusion matrix heatmap (300 DPI)
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
    ax.set_title("Normalized Confusion Matrix — ConvVAE", fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm_counts[i, j]:,})",
                    ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "svm_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Comparative benchmark vs Model 1 (ConvAE)
    comp_rows = []
    ae_summary_path = os.path.join(config.ae_reference_dir, "final_summary.csv")
    ae_dict = {}
    if os.path.exists(ae_summary_path):
        ae_df = pd.read_csv(ae_summary_path)
        ae_dict = dict(zip(ae_df["Dimension"], ae_df["Value"]))

    master_rows = [
        {"Dimension": "Model Architecture", "ConvAE (Model 1)": ae_dict.get("Model Architecture", "ConvAE"), "ConvVAE (Model 2)": "ConvVAE (Probabilistic)"},
        {"Dimension": "Latent Bottleneck", "ConvAE (Model 1)": "256 (Deterministic z)", "ConvVAE (Model 2)": "256 (mu, logvar)"},
        {"Dimension": "Generative Capability", "ConvAE (Model 1)": "Reconstruction Only", "ConvVAE (Model 2)": "Prior Sampling z ~ N(0, I)"},
        {"Dimension": "Test Recon MSE", "ConvAE (Model 1)": ae_dict.get("Test Recon MSE", "N/A"), "ConvVAE (Model 2)": f"{test_df_imgs['mse'].mean():.6f} +/- {test_df_imgs['mse'].std():.6f}"},
        {"Dimension": "Test PSNR (dB)", "ConvAE (Model 1)": ae_dict.get("Test PSNR (dB)", "N/A"), "ConvVAE (Model 2)": f"{test_df_imgs['psnr'].mean():.2f} +/- {test_df_imgs['psnr'].std():.2f}"},
        {"Dimension": "Test SSIM", "ConvAE (Model 1)": ae_dict.get("Test SSIM", "N/A"), "ConvVAE (Model 2)": f"{test_df_imgs['ssim'].mean():.4f} +/- {test_df_imgs['ssim'].std():.4f}"},
        {"Dimension": "Test SVM Accuracy", "ConvAE (Model 1)": ae_dict.get("Test SVM Accuracy", "N/A"), "ConvVAE (Model 2)": f"{test_svm_metrics['Accuracy']:.4f}"},
        {"Dimension": "Test SVM Balanced Acc", "ConvAE (Model 1)": ae_dict.get("Test SVM Balanced Acc", "N/A"), "ConvVAE (Model 2)": f"{test_svm_metrics['Balanced Accuracy']:.4f}"},
        {"Dimension": "Test SVM Macro F1", "ConvAE (Model 1)": ae_dict.get("Test SVM Macro F1", "N/A"), "ConvVAE (Model 2)": f"{test_svm_metrics['Macro F1']:.4f}"},
    ]
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(os.path.join(config.output_dir, "final_summary.csv"), index=False)
    print("\nMaster Comparative Benchmark (ConvAE vs ConvVAE):")
    print(df_master.to_string(index=False))


if __name__ == "__main__":
    main()

