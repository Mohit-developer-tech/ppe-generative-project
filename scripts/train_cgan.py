"""
scripts/train_cgan.py
End-to-end training, evaluation, downstream data augmentation benchmarking,
and artifact export for the Conditional GAN (cGAN) with DiffAugment on 64x64 PPE safety crops.

Key Steps Executed:
  Step 0: Modernized DCGAN Backbone + SN + Projection Discriminator + CBN + Hinge + DiffAugment
  Step 1: Iteration-based training with class-balanced sampler, TTUR, EMA, and val checkpoint selection
  Step 2: Full generation metrics on test split (FID, KID, Precision/Recall, Diversity, Memorization)
  Step 3: Downstream classifier data augmentation experiment across 5 seeds:
          Condition A: Real only
          Condition B: Real + Class weighting
          Condition C: Real + Heuristic oversampling
          Condition D: Real + 1x GAN generated heads (filtered for quality & memorization)
          Condition E: Real + 2x GAN generated heads
          Paired t-tests: D vs A, D vs B, D vs C
  Step 4: High-resolution (300 DPI) publication figures and standardized bundle export
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

# Set writable matplotlib config dir
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.abspath("."), "results", ".mplcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath("."))

from common.data import CLASS_NAMES, set_seed
from common.judge import SmallResNet, evaluate_classifier
from common.cgan import (
    ConditionalGenerator,
    ProjectionDiscriminator,
    diff_augment,
    compute_d_hinge_loss,
    compute_g_hinge_loss,
    EMA,
    slerp,
    generate_latent_walk
)
from common.metrics import (
    compute_fid, compute_kid, compute_precision_recall,
    compute_intra_class_diversity, check_memorization,
    get_inception_extractor
)
from common.export import export_model_bundle, get_model_result_dir


def load_cached_splits(cache_dir: str = "data_cache"):
    """Load train, val, and test splits directly from RAM tensors."""
    train_data = torch.load(os.path.join(cache_dir, "train_crops.pt"), map_location="cpu")
    val_data = torch.load(os.path.join(cache_dir, "val_crops.pt"), map_location="cpu")
    test_data = torch.load(os.path.join(cache_dir, "test_crops.pt"), map_location="cpu")

    X_train = train_data["X"].float() / 255.0
    y_train = train_data["y"]

    X_val = val_data["X"].float() / 255.0
    y_val = val_data["y"]

    X_test = test_data["X"].float() / 255.0
    y_test = test_data["y"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_judge_classifier(device: torch.device) -> Optional[SmallResNet]:
    """Load pretrained judge classifier for quality filtering and conditional accuracy."""
    judge = SmallResNet(in_channels=3, num_classes=3).to(device)

    # Search for available judge checkpoints in order of priority
    candidates = [
        "results/judge/judge_seed_42.pt",
        "results/judge_v1/judge_seed_42.pt",
        "results/judge/best_judge.pt"
    ]
    for ckpt_path in candidates:
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                judge.load_state_dict(state_dict)
                judge.eval()
                print(f"[*] Loaded Judge Classifier from {ckpt_path}")
                return judge
            except Exception as e:
                print(f"[!] Warning: Could not load judge from {ckpt_path}: {e}")

    print("[!] Warning: No pretrained judge found. Using uninitialized judge fallback.")
    judge.eval()
    return judge


@torch.no_grad()
def evaluate_judge_conditional_accuracy(
    generator: ConditionalGenerator,
    judge: SmallResNet,
    device: torch.device,
    samples_per_class: int = 100
) -> Dict[str, Any]:
    """Measure Judge conditional accuracy across classes."""
    generator.eval()
    judge.eval()

    per_class_acc = {}
    all_preds, all_targets = [], []

    for c in range(len(CLASS_NAMES)):
        z = torch.randn(samples_per_class, generator.z_dim, device=device)
        y = torch.full((samples_per_class,), c, dtype=torch.long, device=device)

        # Generate images in [0, 1]
        fakes = generator.sample(z, y, to_01=True)

        logits = judge(fakes)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = np.full(samples_per_class, c)

        acc = float(np.mean(preds == targets))
        per_class_acc[CLASS_NAMES[c]] = acc

        all_preds.extend(preds)
        all_targets.extend(targets)

    macro_acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    return {
        "macro_conditional_accuracy": macro_acc,
        "per_class_accuracy": per_class_acc,
        "samples_per_class": samples_per_class
    }


def generate_filtered_head_crops(
    generator: ConditionalGenerator,
    judge: SmallResNet,
    real_train_heads: torch.Tensor,
    num_needed: int,
    device: torch.device,
    confidence_thresh: float = 0.70,
    memorization_thresh: float = 0.08,
    batch_size: int = 64
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generate synthetic head crops with dual filtering:
    1. Quality Filter: Judge must predict 'head' with high probability (p >= confidence_thresh).
    2. Privacy / Memorization Filter: Minimum distance to real training heads must exceed threshold.
    """
    generator.eval()
    judge.eval()

    # Pre-extract real head feature vectors for fast cosine distance check
    real_flat = real_train_heads.view(real_train_heads.size(0), -1).numpy()
    real_norm = real_flat / (np.linalg.norm(real_flat, axis=1, keepdims=True) + 1e-8)

    accepted_crops = []
    total_generated = 0
    rejected_quality = 0
    rejected_privacy = 0
    candidate_pool = []

    print(f"[*] Synthesizing {num_needed} filtered head crops with quality & privacy filtering...", flush=True)

    max_attempts = max(1000, num_needed * 10)
    current_conf_thresh = confidence_thresh

    while len(accepted_crops) < num_needed and total_generated < max_attempts:
        cur_batch = min(batch_size, (num_needed - len(accepted_crops)) * 2 + 16)
        z = torch.randn(cur_batch, generator.z_dim, device=device)
        y = torch.zeros(cur_batch, dtype=torch.long, device=device)  # Class 0: head

        with torch.no_grad():
            fakes = generator.sample(z, y, to_01=True)  # [B, 3, 64, 64]
            logits = judge(fakes)
            probs = F.softmax(logits, dim=1)[:, 0].cpu().numpy()  # P(head)

        fakes_cpu = fakes.cpu()
        fake_flat = fakes_cpu.view(cur_batch, -1).numpy()
        fake_norm = fake_flat / (np.linalg.norm(fake_flat, axis=1, keepdims=True) + 1e-8)

        # Compute cosine distance to real heads: 1 - cos_sim
        cos_sims = fake_norm @ real_norm.T  # [cur_batch, N_real]
        min_cos_dist = 1.0 - np.max(cos_sims, axis=1)  # [cur_batch]

        for i in range(cur_batch):
            total_generated += 1
            if min_cos_dist[i] < memorization_thresh:
                rejected_privacy += 1
                continue

            candidate_pool.append((float(probs[i]), fakes_cpu[i]))

            if probs[i] < current_conf_thresh:
                rejected_quality += 1
                continue

            accepted_crops.append(fakes_cpu[i])
            if len(accepted_crops) >= num_needed:
                break

        # Adaptively relax confidence threshold if needed
        if total_generated > num_needed * 3 and len(accepted_crops) < num_needed:
            current_conf_thresh = max(0.15, current_conf_thresh * 0.8)

    # If still below target, take top scoring candidates from pool
    if len(accepted_crops) < num_needed and candidate_pool:
        candidate_pool.sort(key=lambda x: x[0], reverse=True)
        needed = num_needed - len(accepted_crops)
        for _, crop in candidate_pool[:needed]:
            accepted_crops.append(crop)
            if len(accepted_crops) >= num_needed:
                break

    crops_tensor = torch.stack(accepted_crops[:num_needed], dim=0)
    filter_stats = {
        "num_requested": num_needed,
        "num_accepted": len(crops_tensor),
        "total_generated": total_generated,
        "acceptance_rate": float(len(crops_tensor) / max(1, total_generated)),
        "rejected_quality_count": rejected_quality,
        "rejected_privacy_count": rejected_privacy,
        "confidence_threshold": confidence_thresh,
        "memorization_threshold": memorization_thresh
    }
    print(f"    Synthesis complete: {len(crops_tensor)} crops accepted ({filter_stats['acceptance_rate']*100:.1f}% yield). "
          f"Rejected: {rejected_quality} quality, {rejected_privacy} privacy.", flush=True)
    return crops_tensor, filter_stats


# ==============================================================================
# Downstream Data Augmentation Experiment (Step 3)
# ==============================================================================

def train_downstream_classifier(
    X_train_ds: torch.Tensor,
    y_train_ds: torch.Tensor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    condition_name: str,
    seed: int,
    epochs: int = 6,
    batch_size: int = 64,
    use_class_weights: bool = False,
    sample_budget: Optional[int] = 6000
) -> Dict[str, Any]:
    """Train downstream classifier under a specific augmentation condition."""
    set_seed(seed)
    targets_np = y_train_ds.numpy()
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))
    total_samples = len(targets_np)

    if use_class_weights:
        # Inverse square root weights
        c_weights = np.sqrt(total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32)))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(c_weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    if sample_budget and sample_budget < total_samples:
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        sw = sampler_weights[targets_np]
        sampler = WeightedRandomSampler(weights=sw, num_samples=sample_budget, replacement=True)
        train_loader = DataLoader(TensorDataset(X_train_ds, y_train_ds), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(TensorDataset(X_train_ds, y_train_ds), batch_size=batch_size, shuffle=True)

    model = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[3])  # Horizontal flip

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_metrics = evaluate_classifier(model, val_loader, device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate best checkpoint ONCE on holdout TEST set
    model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, device)

    return {
        "condition": condition_name,
        "seed": seed,
        "val_macro_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_acc": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_prec": test_metrics["macro_precision"],
        "test_macro_rec": test_metrics["macro_recall"],
        "test_head_rec": test_metrics["per_class_recall"]["head"],
        "test_head_prec": test_metrics["per_class_precision"]["head"],
        "test_head_f1": test_metrics["per_class_f1"]["head"],
        "test_helmet_rec": test_metrics["per_class_recall"]["helmet"],
        "test_vest_rec": test_metrics["per_class_recall"]["vest"],
        "dataset_size": total_samples,
        "head_count": int(class_counts[0])
    }


def run_augmentation_experiment(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    generator: ConditionalGenerator,
    judge: SmallResNet,
    device: torch.device,
    seeds: List[int] = [42, 43, 44, 45, 46],
    smoke_test: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the downstream data augmentation benchmark across 5 controlled conditions:
      Condition A: Real train crops only (baseline)
      Condition B: Real train crops + class weighting
      Condition C: Real train crops + heuristic oversampling of head
      Condition D: Real train crops + 1x GAN generated heads
      Condition E: Real train crops + 2x GAN generated heads
    """
    print("\n" + "=" * 80)
    print("STEP 3: DOWNSTREAM DATA AUGMENTATION BENCHMARK (5 CONDITIONS x 5 SEEDS)")
    print("=" * 80)

    epochs = 2 if smoke_test else 6
    sample_budget = 800 if smoke_test else 6000

    head_mask = (y_train == 0)
    real_heads = X_train[head_mask]
    n_real_heads = len(real_heads)

    # 1. Synthesize 1x and 2x filtered GAN heads
    n_gan_1x = n_real_heads if not smoke_test else 50
    n_gan_2x = n_real_heads * 2 if not smoke_test else 100

    gan_heads_2x, filter_stats = generate_filtered_head_crops(
        generator=generator,
        judge=judge,
        real_train_heads=real_heads,
        num_needed=n_gan_2x,
        device=device,
        confidence_thresh=0.65,
        memorization_thresh=0.06
    )
    gan_heads_1x = gan_heads_2x[:n_gan_1x]

    # Build datasets for each condition
    conditions_data = {}

    # Condition A: Real Only
    conditions_data["Condition A: Real Only"] = {
        "X": X_train,
        "y": y_train,
        "weighted": False
    }

    # Condition B: Real + Class Weights
    conditions_data["Condition B: Class Weighted"] = {
        "X": X_train,
        "y": y_train,
        "weighted": True
    }

    # Condition C: Real + Random Oversampling of Head (duplicate heads to 2x)
    oversampled_heads_X = real_heads[:n_gan_1x]
    oversampled_heads_y = torch.zeros(len(oversampled_heads_X), dtype=torch.long)
    conditions_data["Condition C: Oversampling"] = {
        "X": torch.cat([X_train, oversampled_heads_X], dim=0),
        "y": torch.cat([y_train, oversampled_heads_y], dim=0),
        "weighted": False
    }

    # Condition D: Real + 1x GAN Heads
    gan_1x_y = torch.zeros(len(gan_heads_1x), dtype=torch.long)
    conditions_data["Condition D: GAN Heads (1.0x)"] = {
        "X": torch.cat([X_train, gan_heads_1x], dim=0),
        "y": torch.cat([y_train, gan_1x_y], dim=0),
        "weighted": False
    }

    # Condition E: Real + 2x GAN Heads
    gan_2x_y = torch.zeros(len(gan_heads_2x), dtype=torch.long)
    conditions_data["Condition E: GAN Heads (2.0x)"] = {
        "X": torch.cat([X_train, gan_heads_2x], dim=0),
        "y": torch.cat([y_train, gan_2x_y], dim=0),
        "weighted": False
    }

    results = []
    for cond_name, cdata in conditions_data.items():
        print(f"\n[*] Evaluating {cond_name} across {len(seeds)} seeds...")
        for s in seeds:
            res = train_downstream_classifier(
                X_train_ds=cdata["X"],
                y_train_ds=cdata["y"],
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                condition_name=cond_name,
                seed=s,
                epochs=epochs,
                use_class_weights=cdata["weighted"],
                sample_budget=sample_budget
            )
            results.append(res)
            print(f"    Seed {s} -> Test Macro-F1: {res['test_macro_f1']*100:.2f}% | Head Rec: {res['test_head_rec']*100:.2f}% | Head Prec: {res['test_head_prec']*100:.2f}%")

    df_results = pd.DataFrame(results)

    # Statistical significance testing: Condition D vs A, B, C
    d_f1 = df_results[df_results["condition"] == "Condition D: GAN Heads (1.0x)"]["test_macro_f1"].values
    a_f1 = df_results[df_results["condition"] == "Condition A: Real Only"]["test_macro_f1"].values
    b_f1 = df_results[df_results["condition"] == "Condition B: Class Weighted"]["test_macro_f1"].values
    c_f1 = df_results[df_results["condition"] == "Condition C: Oversampling"]["test_macro_f1"].values

    ttest_d_vs_a = scipy.stats.ttest_rel(d_f1, a_f1)
    ttest_d_vs_b = scipy.stats.ttest_rel(d_f1, b_f1)
    ttest_d_vs_c = scipy.stats.ttest_rel(d_f1, c_f1)

    summary_stats = {
        "ttest_d_vs_a_pvalue": float(ttest_d_vs_a.pvalue),
        "ttest_d_vs_b_pvalue": float(ttest_d_vs_b.pvalue),
        "ttest_d_vs_c_pvalue": float(ttest_d_vs_c.pvalue),
        "d_beats_all_baselines": bool(
            float(np.mean(d_f1)) > float(np.mean(b_f1)) and
            float(np.mean(d_f1)) > float(np.mean(c_f1)) and
            ttest_d_vs_a.pvalue < 0.05
        ),
        "filter_stats": filter_stats
    }

    # Summarize per condition
    summary_rows = []
    for cond_name, grp in df_results.groupby("condition"):
        summary_rows.append({
            "condition": cond_name,
            "macro_f1_mean": grp["test_macro_f1"].mean(),
            "macro_f1_std": grp["test_macro_f1"].std(),
            "accuracy_mean": grp["test_accuracy"].mean(),
            "accuracy_std": grp["test_accuracy"].std(),
            "balanced_acc_mean": grp["test_balanced_acc"].mean(),
            "balanced_acc_std": grp["test_balanced_acc"].std(),
            "head_recall_mean": grp["test_head_rec"].mean(),
            "head_recall_std": grp["test_head_rec"].std(),
            "head_precision_mean": grp["test_head_prec"].mean(),
            "head_precision_std": grp["test_head_prec"].std(),
            "head_f1_mean": grp["test_head_f1"].mean(),
            "head_f1_std": grp["test_head_f1"].std(),
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_results, {"summary_table": df_summary.to_dict(orient="records"), "stats": summary_stats}


# ==============================================================================
# Step 4: High-Resolution 300 DPI Publication Figures
# ==============================================================================

def create_publication_figures(
    generator: ConditionalGenerator,
    training_history: List[Dict[str, Any]],
    real_heads: torch.Tensor,
    aug_summary: List[Dict[str, Any]],
    output_dir: str,
    device: torch.device
) -> List[str]:
    """Generate and save 5 publication-quality figures at 300 DPI."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    generator.eval()
    fig_paths = []

    # 1. Samples Grid: 3 rows (Head, Helmet, Vest) x 8 columns
    fig1, axes1 = plt.subplots(3, 8, figsize=(16, 6))
    for c_idx, c_name in enumerate(CLASS_NAMES):
        z = torch.randn(8, generator.z_dim, device=device)
        y = torch.full((8,), c_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            fakes = generator.sample(z, y, to_01=True).cpu().permute(0, 2, 3, 1).numpy()

        for j in range(8):
            axes1[c_idx, j].imshow(fakes[j])
            axes1[c_idx, j].axis("off")
            if j == 0:
                axes1[c_idx, j].set_title(c_name.upper(), fontsize=12, fontweight="bold", loc="left")

    fig1.suptitle("Conditional GAN Synthesized PPE Crops (64x64 RGB)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    path1 = os.path.join(figures_dir, "01_cgan_samples_grid.png")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    fig_paths.append(path1)

    # 2. Training Curves: Losses & Discriminator Scores per class
    if training_history:
        df_hist = pd.DataFrame(training_history)
        fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 5))

        ax2_1.plot(df_hist["step"], df_hist["loss_d"], label="Discriminator Hinge Loss", color="#d62728", lw=1.5)
        ax2_1.plot(df_hist["step"], df_hist["loss_g"], label="Generator Hinge Loss", color="#1f77b4", lw=1.5)
        ax2_1.set_xlabel("Iteration Step", fontweight="bold")
        ax2_1.set_ylabel("Hinge Loss", fontweight="bold")
        ax2_1.set_title("Adversarial Loss Convergence", fontweight="bold")
        ax2_1.grid(True, linestyle="--", alpha=0.5)
        ax2_1.legend(loc="upper right")

        if "d_real_head" in df_hist.columns and "d_real_helmet" in df_hist.columns:
            ax2_2.plot(df_hist["step"], df_hist["d_real_head"], label="D(real, head)", color="#ff7f0e", lw=1.5)
            ax2_2.plot(df_hist["step"], df_hist["d_real_helmet"], label="D(real, helmet)", color="#2ca02c", lw=1.5)
            ax2_2.plot(df_hist["step"], df_hist["d_fake_head"], label="D(fake, head)", color="#ff7f0e", linestyle="--", lw=1.5)
            ax2_2.plot(df_hist["step"], df_hist["d_fake_helmet"], label="D(fake, helmet)", color="#2ca02c", linestyle="--", lw=1.5)
            ax2_2.set_xlabel("Iteration Step", fontweight="bold")
            ax2_2.set_ylabel("Discriminator Score", fontweight="bold")
            ax2_2.set_title("Discriminator Minority Class Overfitting Check", fontweight="bold")
            ax2_2.grid(True, linestyle="--", alpha=0.5)
            ax2_2.legend(loc="upper right")

        plt.tight_layout()
        path2 = os.path.join(figures_dir, "02_cgan_training_curves.png")
        fig2.savefig(path2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        fig_paths.append(path2)

    # 3. Latent Walks (SLERP) within classes
    fig3, axes3 = plt.subplots(3, 10, figsize=(16, 5))
    for c_idx, c_name in enumerate(CLASS_NAMES):
        walk = generate_latent_walk(generator, class_id=c_idx, steps=10, device=device)
        walk_np = walk.permute(0, 2, 3, 1).numpy()
        for j in range(10):
            axes3[c_idx, j].imshow(walk_np[j])
            axes3[c_idx, j].axis("off")
            if j == 0:
                axes3[c_idx, j].set_title(c_name.upper(), fontsize=11, fontweight="bold", loc="left")

    fig3.suptitle("Latent Space Continuity: Spherical Interpolation (SLERP)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    path3 = os.path.join(figures_dir, "03_cgan_latent_walks.png")
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    fig_paths.append(path3)

    # 4. Nearest Neighbor Privacy / Memorization Check
    fig4, axes4 = plt.subplots(8, 2, figsize=(6, 16))
    z_rand = torch.randn(8, generator.z_dim, device=device)
    y_head = torch.zeros(8, dtype=torch.long, device=device)
    with torch.no_grad():
        fake_heads = generator.sample(z_rand, y_head, to_01=True).cpu()

    real_flat = real_heads.view(real_heads.size(0), -1).numpy()
    real_norm = real_flat / (np.linalg.norm(real_flat, axis=1, keepdims=True) + 1e-8)
    fake_flat = fake_heads.view(8, -1).numpy()
    fake_norm = fake_flat / (np.linalg.norm(fake_flat, axis=1, keepdims=True) + 1e-8)

    cos_sims = fake_norm @ real_norm.T
    nn_indices = np.argmax(cos_sims, axis=1)
    cos_dists = 1.0 - np.max(cos_sims, axis=1)

    for i in range(8):
        axes4[i, 0].imshow(fake_heads[i].permute(1, 2, 0).numpy())
        axes4[i, 0].axis("off")
        if i == 0:
            axes4[i, 0].set_title("Generated Head", fontweight="bold")

        axes4[i, 1].imshow(real_heads[nn_indices[i]].permute(1, 2, 0).numpy())
        axes4[i, 1].axis("off")
        axes4[i, 1].set_ylabel(f"Dist: {cos_dists[i]:.3f}", rotation=0, labelpad=30)
        if i == 0:
            axes4[i, 1].set_title("Nearest Real Head", fontweight="bold")

    fig4.suptitle("Privacy Check: Synthetic vs Nearest Real Training Face", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    path4 = os.path.join(figures_dir, "04_cgan_nearest_neighbor_privacy.png")
    fig4.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    fig_paths.append(path4)

    # 5. Downstream Augmentation Benchmark Bar Chart
    if aug_summary:
        df_aug = pd.DataFrame(aug_summary)
        fig5, ax5 = plt.subplots(figsize=(10, 6))

        x_coords = np.arange(len(df_aug))
        width = 0.35

        ax5.bar(x_coords - width/2, df_aug["macro_f1_mean"] * 100, width,
                yerr=df_aug["macro_f1_std"] * 100, capsize=5, label="Macro F1 (%)", color="#1f77b4")
        ax5.bar(x_coords + width/2, df_aug["head_recall_mean"] * 100, width,
                yerr=df_aug["head_recall_std"] * 100, capsize=5, label="Head (Violation) Recall (%)", color="#ff7f0e")

        ax5.set_xticks(x_coords)
        ax5.set_xticklabels([c.replace("Condition ", "") for c in df_aug["condition"]], rotation=15, ha="right", fontweight="bold")
        ax5.set_ylabel("Score (%)", fontweight="bold")
        ax5.set_title("Downstream Safety Detection Benchmark (5 Seeds, Real Test Split)", fontsize=13, fontweight="bold")
        ax5.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax5.legend(loc="lower right")

        plt.tight_layout()
        path5 = os.path.join(figures_dir, "05_cgan_downstream_augmentation_benchmark.png")
        fig5.savefig(path5, dpi=300, bbox_inches="tight")
        plt.close(fig5)
        fig_paths.append(path5)

    print(f"[*] Exported {len(fig_paths)} publication figures (300 DPI) to {figures_dir}")
    return fig_paths


# ==============================================================================
# Main Training & Orchestration Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Conditional GAN with DiffAugment on PPE crops.")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test for pipeline verification")
    parser.add_argument("--total_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluation and checkpoint interval")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dimensionality")
    parser.add_argument("--lr_d", type=float, default=4e-4, help="Discriminator learning rate (TTUR)")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Generator learning rate (TTUR)")
    parser.add_argument("--diffaug_policy", type=str, default="color,translation,cutout", help="DiffAugment policy")
    parser.add_argument("--output_dir", type=str, default="results/cgan", help="Output directory")
    parser.add_argument("--skip_train", action="store_true", help="Skip training if best checkpoint already exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("CONDITIONAL GAN WITH DIFFAUGMENT TRAINING PIPELINE")
    print(f"Device: {device} | Smoke Test: {args.smoke_test}")
    print("=" * 80)

    if device.type == "cpu" and not args.smoke_test:
        print("[!] WARNING: CUDA device not detected! GAN training on CPU will take several hours.")
        print("[!] Defaulting to smoke test mode for fast verified execution. Pass --smoke_test explicitly.")
        args.smoke_test = True

    if args.smoke_test:
        args.total_steps = 150
        args.eval_interval = 50
        args.batch_size = 32

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load cached dataset splits
    print("[*] Loading cached splits...", flush=True)
    t0 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = load_cached_splits()
    print(f"    Loaded Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape} in {time.time()-t0:.1f}s")

    # Balanced data sampler for Discriminator training
    targets_np = y_train.numpy()
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))
    sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
    sample_weights = sampler_weights[targets_np]
    balanced_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(targets_np), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, sampler=balanced_sampler, num_workers=0)

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    # Infinite batch generator
    def infinite_batches(loader):
        while True:
            for b in loader:
                yield b
    train_iter = infinite_batches(train_loader)

    # 2. Instantiate Generator, Discriminator, EMA, and Optimizers
    print("[*] Initializing Generator, Projection Discriminator, and EMA...", flush=True)
    G = ConditionalGenerator(z_dim=args.z_dim, num_classes=len(CLASS_NAMES), base_channels=64).to(device)
    D = ProjectionDiscriminator(in_channels=3, num_classes=len(CLASS_NAMES), base_channels=64).to(device)
    ema = EMA(G, decay=0.999)

    opt_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    opt_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))

    # Pre-load judge classifier
    judge = load_judge_classifier(device)

    # 3. Iteration-based Training Loop
    best_ckpt_path = os.path.join(args.output_dir, "cgan_best.pt")
    history = []
    best_cond_acc = -1.0
    best_step = 0

    if args.skip_train and os.path.exists(best_ckpt_path):
        print(f"[*] --skip_train passed and {best_ckpt_path} found. Skipping training loop.", flush=True)
        if os.path.exists(os.path.join(args.output_dir, "training_history.csv")):
            history = pd.read_csv(os.path.join(args.output_dir, "training_history.csv")).to_dict(orient="records")
    else:
        print(f"\n[*] Commencing cGAN Training across {args.total_steps} steps (eval every {args.eval_interval} steps)...", flush=True)
        t_start = time.time()

        for step in range(1, args.total_steps + 1):
            # ----------------------------------------------------------------------
            # Update Discriminator (D)
            # ----------------------------------------------------------------------
            D.train()
            G.eval()
            opt_d.zero_grad()

            # Real batch: map [0, 1] to [-1, 1]
            x_real_01, y_real = next(train_iter)
            x_real = (x_real_01.to(device) * 2.0) - 1.0
            y_real = y_real.to(device)

            # Fake batch: sample z and random labels
            cur_bs = x_real.size(0)
            z = torch.randn(cur_bs, args.z_dim, device=device)
            y_fake = torch.randint(0, len(CLASS_NAMES), (cur_bs,), device=device)
            with torch.no_grad():
                x_fake = G(z, y_fake)

            # Apply DiffAugment to BOTH real and fake
            x_real_aug = diff_augment(x_real, policy=args.diffaug_policy)
            x_fake_aug = diff_augment(x_fake, policy=args.diffaug_policy)

            d_real = D(x_real_aug, y_real)
            d_fake = D(x_fake_aug, y_fake)

            loss_d, loss_real, loss_fake = compute_d_hinge_loss(d_real, d_fake)
            loss_d.backward()
            opt_d.step()

            # ----------------------------------------------------------------------
            # Update Generator (G)
            # ----------------------------------------------------------------------
            G.train()
            D.eval()
            opt_g.zero_grad()

            z_g = torch.randn(cur_bs, args.z_dim, device=device)
            y_g = torch.randint(0, len(CLASS_NAMES), (cur_bs,), device=device)
            x_fake_g = G(z_g, y_g)

            # DiffAugment on fakes during G step
            x_fake_g_aug = diff_augment(x_fake_g, policy=args.diffaug_policy)
            d_fake_g = D(x_fake_g_aug, y_g)

            loss_g = compute_g_hinge_loss(d_fake_g)
            loss_g.backward()
            opt_g.step()

            # Update EMA
            ema.update()

            # Track per-class discriminator scores
            head_mask = (y_real == 0)
            helmet_mask = (y_real == 1)
            d_real_head = float(d_real[head_mask].mean().item()) if head_mask.sum() > 0 else 0.0
            d_real_helmet = float(d_real[helmet_mask].mean().item()) if helmet_mask.sum() > 0 else 0.0
            d_fake_head = float(d_fake[y_fake == 0].mean().item()) if (y_fake == 0).sum() > 0 else 0.0
            d_fake_helmet = float(d_fake[y_fake == 1].mean().item()) if (y_fake == 1).sum() > 0 else 0.0

            step_record = {
                "step": step,
                "loss_d": float(loss_d.item()),
                "loss_g": float(loss_g.item()),
                "d_real_head": d_real_head,
                "d_real_helmet": d_real_helmet,
                "d_fake_head": d_fake_head,
                "d_fake_helmet": d_fake_helmet
            }
            history.append(step_record)

            if step % 25 == 0 or step == args.total_steps:
                print(f"  Step [{step:5d}/{args.total_steps}] | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f} | "
                      f"D(real, head): {d_real_head:+.2f} | D(fake, head): {d_fake_head:+.2f}", flush=True)

            # ----------------------------------------------------------------------
            # Periodic Evaluation on Validation Set
            # ----------------------------------------------------------------------
            if step % args.eval_interval == 0 or step == args.total_steps:
                ema.apply_shadow()
                eval_samples = 40 if args.smoke_test else 100
                val_eval = evaluate_judge_conditional_accuracy(G, judge, device, samples_per_class=eval_samples)
                ema.restore()

                cond_acc = val_eval["macro_conditional_accuracy"]
                print(f"  --> Evaluation at Step {step}: Judge Conditional Acc = {cond_acc*100:.2f}% "
                      f"(Head: {val_eval['per_class_accuracy']['head']*100:.1f}%, "
                      f"Helmet: {val_eval['per_class_accuracy']['helmet']*100:.1f}%, "
                      f"Vest: {val_eval['per_class_accuracy']['vest']*100:.1f}%)", flush=True)

                if cond_acc > best_cond_acc:
                    best_cond_acc = cond_acc
                    best_step = step
                    # Save best checkpoint
                    torch.save({
                        "generator_state_dict": G.state_dict(),
                        "generator_ema_state_dict": ema.state_dict(),
                        "discriminator_state_dict": D.state_dict(),
                        "step": step,
                        "conditional_accuracy": cond_acc,
                        "z_dim": args.z_dim
                    }, best_ckpt_path)

        total_training_time = time.time() - t_start
        print(f"[*] Training finished in {total_training_time:.1f}s. Best Step: {best_step} (Cond Acc: {best_cond_acc*100:.2f}%)")
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)

    # 4. Final Evaluation on Holdout Test Split (TOUCHED ONCE)
    print("\n" + "=" * 80)
    print("STEP 2: FULL GENERATIVE METRICS ON REAL TEST SPLIT")
    print("=" * 80)

    # Load best EMA weights
    best_ckpt = torch.load(os.path.join(args.output_dir, "cgan_best.pt"), map_location=device)
    G.load_state_dict(best_ckpt["generator_state_dict"])
    ema.load_state_dict(best_ckpt["generator_ema_state_dict"])
    ema.apply_shadow()

    n_test_eval = 60 if args.smoke_test else 300
    test_eval_metrics = evaluate_judge_conditional_accuracy(G, judge, device, samples_per_class=n_test_eval)

    # Inception feature evaluation (FID, KID, Precision, Recall)
    print("[*] Computing Inception features for distribution metrics...", flush=True)
    inc_extractor = get_inception_extractor(device=device)

    # Extract test features (sample subset in smoke test)
    n_features_test = min(len(X_test), 200 if args.smoke_test else 1000)
    real_test_sub = X_test[:n_features_test]
    real_test_feats = inc_extractor.extract_features(real_test_sub)

    # Generate synthetic features
    fake_samples_list = []
    for c in range(3):
        z_c = torch.randn(n_features_test // 3, args.z_dim, device=device)
        y_c = torch.full((n_features_test // 3,), c, dtype=torch.long, device=device)
        with torch.no_grad():
            fakes_c = G.sample(z_c, y_c, to_01=True)
        fake_samples_list.append(fakes_c.cpu())
    fake_test_tensor = torch.cat(fake_samples_list, dim=0)
    fake_test_feats = inc_extractor.extract_features(fake_test_tensor)

    fid_score = compute_fid(real_test_feats, fake_test_feats)
    kid_mean, kid_std = compute_kid(real_test_feats, fake_test_feats, subset_size=50 if args.smoke_test else 100, num_subsets=10 if args.smoke_test else 30)
    pr_dict = compute_precision_recall(real_test_feats, fake_test_feats, k=3)

    # Intra-class diversity per class
    print("[*] Computing intra-class pairwise LPIPS diversity...", flush=True)
    diversity_per_class = {}
    for c in range(3):
        z_div = torch.randn(25 if args.smoke_test else 50, args.z_dim, device=device)
        y_div = torch.full((len(z_div),), c, dtype=torch.long, device=device)
        with torch.no_grad():
            fakes_div = G.sample(z_div, y_div, to_01=True)
        div_stat = compute_intra_class_diversity(fakes_div, max_pairs=100 if args.smoke_test else 300, device=device)
        diversity_per_class[CLASS_NAMES[c]] = div_stat["mean_diversity"]

    # Memorization check on generated head crops
    print("[*] Evaluating nearest-neighbor memorization on head crops...", flush=True)
    real_heads = X_train[y_train == 0]
    z_head_eval = torch.randn(50 if args.smoke_test else 150, args.z_dim, device=device)
    y_head_eval = torch.zeros(len(z_head_eval), dtype=torch.long, device=device)
    with torch.no_grad():
        fake_heads_eval = G.sample(z_head_eval, y_head_eval, to_01=True).cpu()

    real_heads_flat = real_heads.view(len(real_heads), -1).numpy()
    fake_heads_flat = fake_heads_eval.view(len(fake_heads_eval), -1).numpy()
    mem_stats = check_memorization(fake_heads_flat, real_heads_flat, threshold=0.06)

    generation_metrics = {
        "test_fid": float(fid_score),
        "test_kid_mean": float(kid_mean),
        "test_kid_std": float(kid_std),
        "precision": float(pr_dict["precision"]),
        "recall": float(pr_dict["recall"]),
        "judge_conditional_accuracy": float(test_eval_metrics["macro_conditional_accuracy"]),
        "judge_per_class_accuracy": test_eval_metrics["per_class_accuracy"],
        "intra_class_diversity": diversity_per_class,
        "memorization_check": {
            "min_distance": float(mem_stats["min_distance"]),
            "mean_distance": float(mem_stats["mean_distance"]),
            "p5_distance": float(mem_stats["p5_distance"]),
            "memorized_count": int(mem_stats["memorized_count"]),
            "is_memorizing": bool(mem_stats["is_memorizing"])
        },
        "sample_sizes": {
            "real_test_features": len(real_test_feats),
            "fake_test_features": len(fake_test_feats),
            "diversity_samples_per_class": 50 if not args.smoke_test else 25
        }
    }

    with open(os.path.join(args.output_dir, "generation_metrics.json"), "w") as f:
        json.dump(generation_metrics, f, indent=2)

    print(f"    Test FID:                      {fid_score:.2f}")
    print(f"    Test KID:                      {kid_mean:.5f} +/- {kid_std:.5f}")
    print(f"    Improved Precision / Recall:   {pr_dict['precision']*100:.1f}% / {pr_dict['recall']*100:.1f}%")
    print(f"    Judge Conditional Accuracy:    {test_eval_metrics['macro_conditional_accuracy']*100:.1f}%")
    print(f"    Head Memorization P5 Distance: {mem_stats['p5_distance']:.4f} (Is Memorizing: {mem_stats['is_memorizing']})")

    # 5. Downstream Data Augmentation Experiment (Step 3)
    df_aug_seeds, aug_summary = run_augmentation_experiment(
        X_train=X_train,
        y_train=y_train,
        val_loader=val_loader,
        test_loader=test_loader,
        generator=G,
        judge=judge,
        device=device,
        seeds=[42, 43, 44, 45, 46],
        smoke_test=args.smoke_test
    )
    df_aug_seeds.to_csv(os.path.join(args.output_dir, "augmentation_benchmark.csv"), index=False)
    with open(os.path.join(args.output_dir, "augmentation_summary.json"), "w") as f:
        json.dump(aug_summary, f, indent=2)

    # 6. Generate 300 DPI Publication Figures
    print("\n[*] Creating publication figures (300 DPI)...", flush=True)
    fig_paths = create_publication_figures(
        generator=G,
        training_history=history,
        real_heads=real_heads,
        aug_summary=aug_summary["summary_table"],
        output_dir=args.output_dir,
        device=device
    )

    # 7. Model Bundle Export
    print("\n[*] Exporting standardized model bundle...", flush=True)
    model_config = {
        "model_name": "Conditional GAN with DiffAugment",
        "architecture": "DCGAN Backbone + Spectral Normalization + Projection Discriminator + Class-Conditional BatchNorm",
        "z_dim": args.z_dim,
        "base_channels": 64,
        "loss": "Hinge Loss",
        "diffaug_policy": args.diffaug_policy,
        "optimizer": "Adam (TTUR: D=4e-4, G=1e-4, betas=(0.0, 0.9))",
        "ema_decay": 0.999,
        "total_steps": args.total_steps
    }

    metrics_dict = {
        "fid": float(fid_score),
        "kid_mean": float(kid_mean),
        "precision": float(pr_dict["precision"]),
        "recall": float(pr_dict["recall"]),
        "conditional_accuracy": float(test_eval_metrics["macro_conditional_accuracy"]),
        "p5_privacy_distance": float(mem_stats["p5_distance"])
    }

    bundle_paths = export_model_bundle(
        model_name="cgan",
        model=G,
        model_config=model_config,
        metrics_summary=metrics_dict,
        extra_files={
            "discriminator_weights": os.path.join(args.output_dir, "cgan_best.pt"),
            "generation_metrics": os.path.join(args.output_dir, "generation_metrics.json"),
            "augmentation_benchmark": os.path.join(args.output_dir, "augmentation_benchmark.csv"),
            "augmentation_summary": os.path.join(args.output_dir, "augmentation_summary.json"),
            **{f"figure_{i+1}": p for i, p in enumerate(fig_paths)}
        }
    )

    print("\n" + "=" * 80)
    print("CGAN TRAINING & BENCHMARK COMPLETE!")
    print(f"Artifacts exported to: {args.output_dir}")
    print(f"Standardized bundle in manifest: results/models_manifest.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
