"""
scripts/ablate_judge_val.py
Systematic 5-ablation experimental runner for improving the independent judge classifier.
Evaluated strictly across 3 random seeds (42, 43, 44) on the VALIDATION split (crops_64/val/).
ZERO TEST LEAKAGE: The test split is never loaded or touched during ablations.

Ablations tested:
  a. Softer class weighting (inverse square root of class frequency)
  b. Logit adjustment or a bias term tuned on VAL to maximize macro-F1
  c. Label smoothing (0.08)
  d. Pretrained ResNet-18 fine-tuned on crops
  e. Stronger data augmentation (RandomResizedCrop + ColorJitter + Flip)
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import torchvision.transforms.v2 as v2

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath("."))

from common.data import CLASS_NAMES, set_seed
from common.judge import SmallResNet, evaluate_classifier


def load_cached_splits(cache_dir: str = "data_cache"):
    """Load cached train and val splits directly from RAM tensors."""
    train_path = os.path.join(cache_dir, "train_crops.pt")
    val_path = os.path.join(cache_dir, "val_crops.pt")

    train_data = torch.load(train_path, map_location="cpu")
    val_data = torch.load(val_path, map_location="cpu")

    X_train = train_data["X"].float() / 255.0
    y_train = train_data["y"]

    X_val = val_data["X"].float() / 255.0
    y_val = val_data["y"]

    return X_train, y_train, X_val, y_val


def get_pretrained_resnet18(num_classes: int = 3) -> nn.Module:
    """Pretrained ResNet18 adapted for 3-class classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class PretrainedResNet18Wrapper(nn.Module):
    """
    Wraps ResNet18 with ImageNet normalization and fine-tunes on crops.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet = get_pretrained_resnet18(num_classes=num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.resnet(x_norm)


def run_single_experiment(
    config_name: str,
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 7,
    batch_size: int = 64,
    sample_budget: int = 8000
) -> Dict[str, Any]:
    """Train and evaluate one seed under a specific ablation config strictly on VAL."""
    set_seed(seed)
    targets_np = y_train.numpy()
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))
    total_samples = len(targets_np)

    # 1. Configuration parameters
    batch_transform = None
    use_resnet18 = False
    label_smoothing = 0.0
    lr = 2e-3

    if config_name == "a_soft_weights":
        # Inverse square root weights: sqrt(N / (C * N_c))
        c_weights = np.sqrt(total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32)))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
    elif config_name == "c_label_smoothing":
        # Full inverse weights + label smoothing 0.08
        c_weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        label_smoothing = 0.08
    elif config_name == "d_pretrained_resnet18":
        # Pretrained ResNet-18 fine-tuning
        c_weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        use_resnet18 = True
        lr = 5e-4
    elif config_name == "e_stronger_aug":
        # Stronger augmentation: RandomResizedCrop + ColorJitter + Flip
        c_weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        batch_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(64, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02)
        ])
    else:
        # Default baseline (judge_v1 training settings)
        c_weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)

    sample_weights = sampler_weights[targets_np]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=sample_budget, replacement=True
    )

    # Use pure TensorDataset for near-instant memory loading
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # Model and Loss
    if use_resnet18:
        model = PretrainedResNet18Wrapper(num_classes=len(CLASS_NAMES)).to(device)
    else:
        model = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)

    class_weights_t = torch.tensor(c_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Apply batch augmentation
            if batch_transform is not None:
                x = batch_transform(x)
            else:
                # Baseline horizontal flip
                if torch.rand(1).item() > 0.5:
                    x = torch.flip(x, dims=[3])

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation evaluation
        val_eval = evaluate_classifier(model, val_loader, device)
        val_f1 = val_eval["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint and evaluate final VAL
    model.load_state_dict(best_state)
    val_final = evaluate_classifier(model, val_loader, device)
    dur = time.time() - t0

    # Also compute logits for post-hoc analysis
    model.eval()
    all_logits = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device).float()
            all_logits.append(model(x).cpu())
    val_logits = torch.cat(all_logits, dim=0).numpy()

    return {
        "config": config_name,
        "seed": seed,
        "val_macro_f1": float(val_final["macro_f1"]),
        "val_macro_prec": float(val_final["macro_precision"]),
        "val_macro_rec": float(val_final["macro_recall"]),
        "val_acc": float(val_final["accuracy"]),
        "val_bal_acc": float(val_final["balanced_accuracy"]),
        "head_prec": float(val_final["per_class_precision"]["head"]),
        "head_rec": float(val_final["per_class_recall"]["head"]),
        "head_f1": float(val_final["per_class_f1"]["head"]),
        "helmet_prec": float(val_final["per_class_precision"]["helmet"]),
        "helmet_rec": float(val_final["per_class_recall"]["helmet"]),
        "helmet_f1": float(val_final["per_class_f1"]["helmet"]),
        "vest_prec": float(val_final["per_class_precision"]["vest"]),
        "vest_rec": float(val_final["per_class_recall"]["vest"]),
        "vest_f1": float(val_final["per_class_f1"]["vest"]),
        "duration_s": float(dur),
        "val_logits": val_logits,
        "best_state": best_state
    }


def evaluate_logit_adjustment(
    records: List[Dict[str, Any]],
    y_val: torch.Tensor,
    prefix: str = "b_logit_adj_baseline"
) -> List[Dict[str, Any]]:
    """
    Logit adjustment or bias term tuned on VAL to maximize macro-F1.
    """
    pi = np.array([867, 16041, 9991], dtype=np.float32) / 26899.0
    log_pi = np.log(pi)
    targets = y_val.numpy()

    tau_candidates = np.linspace(0.0, 1.5, 31)
    results = []

    for rec in records:
        seed = rec["seed"]
        logits = rec["val_logits"]

        best_tau = 0.0
        best_f1 = -1.0
        best_eval = None

        for tau in tau_candidates:
            adj_logits = logits + tau * log_pi
            preds = np.argmax(adj_logits, axis=1)

            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
            acc = float(accuracy_score(targets, preds))
            bal_acc = float(balanced_accuracy_score(targets, preds))
            f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
            prec = float(precision_score(targets, preds, average="macro", zero_division=0))
            rec_all = float(recall_score(targets, preds, average="macro", zero_division=0))

            c_rec = recall_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)
            c_prec = precision_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)
            c_f1 = f1_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
                best_eval = {
                    "val_macro_f1": f1,
                    "val_macro_prec": prec,
                    "val_macro_rec": rec_all,
                    "val_acc": acc,
                    "val_bal_acc": bal_acc,
                    "head_prec": float(c_prec[0]),
                    "head_rec": float(c_rec[0]),
                    "head_f1": float(c_f1[0]),
                    "helmet_prec": float(c_prec[1]),
                    "helmet_rec": float(c_rec[1]),
                    "helmet_f1": float(c_f1[1]),
                    "vest_prec": float(c_prec[2]),
                    "vest_rec": float(c_rec[2]),
                    "vest_f1": float(c_f1[2]),
                    "best_tau": best_tau
                }

        results.append({"config": prefix, "seed": seed, "duration_s": 0.1, **best_eval})

    return results


def main():
    parser = argparse.ArgumentParser(description="Run judge ablations on VAL split.")
    parser.add_argument("--configs", nargs="+", default=["baseline", "a_soft_weights", "c_label_smoothing", "d_pretrained_resnet18", "e_stronger_aug"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output_dir", type=str, default="results/judge_ablations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Judge Ablations on device: {device}")
    print(f"[*] Configurations: {args.configs}")
    print(f"[*] Seeds: {args.seeds}")

    X_train, y_train, X_val, y_val = load_cached_splits()
    print(f"[*] Loaded cached splits: Train={X_train.shape}, Val={X_val.shape}")

    all_seed_results = []
    baseline_records = []

    # Check if a_soft_weights or other runs exist in ablation_seeds.csv
    seeds_csv = os.path.join(args.output_dir, "ablation_seeds.csv")
    if os.path.exists(seeds_csv):
        existing_df = pd.read_csv(seeds_csv)
        all_seed_results = existing_df.to_dict(orient="records")
        print(f"[*] Loaded {len(all_seed_results)} existing seed records from {seeds_csv}")

    for cfg in args.configs:
        # Check if already completed
        done_seeds = [r["seed"] for r in all_seed_results if r.get("config") == cfg]
        remaining_seeds = [s for s in args.seeds if s not in done_seeds]

        if not remaining_seeds:
            print(f"\n[*] Config {cfg} already completed for seeds {args.seeds}. Skipping training.")
            continue

        print(f"\n=================================================================")
        print(f"RUNNING CONFIGURATION: {cfg.upper()} (Seeds: {remaining_seeds})")
        print(f"=================================================================")
        for s in remaining_seeds:
            res = run_single_experiment(
                config_name=cfg,
                seed=s,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                device=device,
                epochs=7,
                batch_size=64,
                sample_budget=8000
            )
            print(f"  [{cfg}] Seed {s:2d} -> Val F1: {res['val_macro_f1']:.4f} | "
                  f"Val Acc: {res['val_acc']*100:.2f}% | "
                  f"Head Prec: {res['head_prec']*100:.2f}% | "
                  f"Helmet Rec: {res['helmet_rec']*100:.2f}% | "
                  f"({res['duration_s']:.1f}s)", flush=True)

            if cfg == "baseline":
                baseline_records.append(res)

            row = {k: v for k, v in res.items() if k not in ("val_logits", "best_state")}
            all_seed_results.append(row)

            # Keep CSV updated
            pd.DataFrame(all_seed_results).to_csv(seeds_csv, index=False)

    # Evaluate Logit adjustment on baseline if baseline ran or from saved v1
    b_done = any(r.get("config") == "b_logit_adjustment" for r in all_seed_results)
    if not b_done:
        print(f"\n=================================================================")
        print(f"RUNNING CONFIGURATION: B_LOGIT_ADJUSTMENT (Post-hoc tuned on VAL)")
        print(f"=================================================================")
        # Load baseline logits from judge_v1 checkpoints
        v1_records = []
        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        for s in args.seeds:
            ckpt_path = f"results/judge_v1/judge_seed_{s}.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                model = SmallResNet().to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                all_l = []
                with torch.no_grad():
                    for x, _ in val_loader:
                        all_l.append(model(x.to(device)))
                val_logits = torch.cat(all_l, dim=0).cpu().numpy()
                v1_records.append({"seed": s, "val_logits": val_logits})

        if v1_records:
            b_results = evaluate_logit_adjustment(v1_records, y_val, prefix="b_logit_adjustment")
            for r in b_results:
                print(f"  [b_logit_adjustment] Seed {r['seed']:2d} (Tau={r['best_tau']:.2f}) -> "
                      f"Val F1: {r['val_macro_f1']:.4f} | "
                      f"Val Acc: {r['val_acc']*100:.2f}% | "
                      f"Head Prec: {r['head_prec']*100:.2f}% | "
                      f"Helmet Rec: {r['helmet_rec']*100:.2f}%", flush=True)
                all_seed_results.append(r)
            pd.DataFrame(all_seed_results).to_csv(seeds_csv, index=False)

    # Aggregate by configuration
    df = pd.DataFrame(all_seed_results)
    summary_rows = []
    for cfg, grp in df.groupby("config"):
        summary_rows.append({
            "config": cfg,
            "num_seeds": len(grp),
            "val_macro_f1_mean": grp["val_macro_f1"].mean(),
            "val_macro_f1_std": grp["val_macro_f1"].std(),
            "val_acc_mean": grp["val_acc"].mean(),
            "val_acc_std": grp["val_acc"].std(),
            "val_bal_acc_mean": grp["val_bal_acc"].mean(),
            "val_bal_acc_std": grp["val_bal_acc"].std(),
            "head_prec_mean": grp["head_prec"].mean(),
            "head_prec_std": grp["head_prec"].std(),
            "head_rec_mean": grp["head_rec"].mean(),
            "head_rec_std": grp["head_rec"].std(),
            "helmet_rec_mean": grp["helmet_rec"].mean(),
            "helmet_rec_std": grp["helmet_rec"].std(),
            "helmet_prec_mean": grp["helmet_prec"].mean(),
            "helmet_prec_std": grp["helmet_prec"].std(),
            "vest_f1_mean": grp["vest_f1"].mean(),
            "vest_f1_std": grp["vest_f1"].std(),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("val_macro_f1_mean", ascending=False)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("ABLATION SUMMARY TABLE ON VALIDATION SET (3 SEEDS):")
    print("=" * 90)
    print(df_summary[["config", "val_macro_f1_mean", "head_prec_mean", "helmet_rec_mean", "val_acc_mean"]].to_string(index=False))
    print("=" * 90)
    print(f"[*] Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
