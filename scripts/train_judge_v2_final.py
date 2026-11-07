"""
scripts/train_judge_v2_final.py
Executes Step 4 & 5:
Trains the winning Judge configuration across 5 random seeds (42, 43, 44, 45, 46).
Performs post-hoc Temperature Scaling strictly on the VALIDATION split.
Evaluates on the holdout TEST split EXACTLY ONCE for final benchmark reporting.
Saves model weights, calibrated temperatures, benchmark CSV, and summary JSON to results/judge/.
Keeps results/judge_v1/ completely untouched.
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

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath("."))

from common.data import CLASS_NAMES, set_seed
from common.judge import SmallResNet, evaluate_classifier
from scripts.calibrate_temperature import compute_ece, fit_temperature


def load_all_cached_splits(cache_dir: str = "data_cache"):
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


def evaluate_with_temperature(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """Evaluate model with temperature-scaled probabilities."""
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)

    logits_t = torch.cat(all_logits, dim=0)
    targets_np = torch.cat(all_targets, dim=0).numpy()

    # Temperature-scaled probabilities
    scaled_logits = logits_t / temperature
    probs_np = F.softmax(scaled_logits, dim=1).numpy()
    preds_np = np.argmax(probs_np, axis=1)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
    acc = float(accuracy_score(targets_np, preds_np))
    bal_acc = float(balanced_accuracy_score(targets_np, preds_np))
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    macro_prec = float(precision_score(targets_np, preds_np, average="macro", zero_division=0))
    macro_rec = float(recall_score(targets_np, preds_np, average="macro", zero_division=0))

    labels_all = list(range(len(CLASS_NAMES)))
    recalls = recall_score(targets_np, preds_np, labels=labels_all, average=None, zero_division=0)
    precs = precision_score(targets_np, preds_np, labels=labels_all, average=None, zero_division=0)
    f1s = f1_score(targets_np, preds_np, labels=labels_all, average=None, zero_division=0)

    ece, ece_info = compute_ece(probs_np, targets_np, n_bins=10)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_recall": {CLASS_NAMES[i]: float(recalls[i]) for i in range(len(CLASS_NAMES))},
        "per_class_precision": {CLASS_NAMES[i]: float(precs[i]) for i in range(len(CLASS_NAMES))},
        "per_class_f1": {CLASS_NAMES[i]: float(f1s[i]) for i in range(len(CLASS_NAMES))},
        "ece": ece,
        "ece_info": ece_info,
        "temperature": temperature,
        "logits": logits_t,
        "targets": targets_np
    }


def train_and_calibrate_seed(
    seed: int,
    config_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    epochs: int = 7,
    batch_size: int = 64,
    sample_budget: int = 8000
) -> Dict[str, Any]:
    """Train single seed with chosen config, calibrate on VAL, and test ONCE on TEST."""
    set_seed(seed)
    targets_np = y_train.numpy()
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))
    total_samples = len(targets_np)

    # Configure based on chosen ablation
    if config_name == "a_soft_weights":
        # Softer weights: inverse square root
        c_weights = np.sqrt(total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32)))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        lr = 2e-3
        label_smoothing = 0.0
    elif config_name == "c_label_smoothing":
        c_weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        lr = 2e-3
        label_smoothing = 0.08
    else:
        # Default fallback
        c_weights = np.sqrt(total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32)))
        sampler_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
        lr = 2e-3
        label_smoothing = 0.0

    sample_weights = sampler_weights[targets_np]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=sample_budget, replacement=True
    )

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    model = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    class_weights_t = torch.tensor(c_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None

    print(f"\n[*] Training Seed {seed} ({config_name}) across {epochs} epochs...", flush=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
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

        val_eval = evaluate_classifier(model, val_loader, device)
        val_f1 = val_eval["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    dur = time.time() - t0
    model.load_state_dict(best_state)

    # 1. Evaluate on VAL (uncalibrated)
    val_uncal = evaluate_with_temperature(model, val_loader, device, temperature=1.0)
    val_ece_pre = val_uncal["ece"]

    # 2. Temperature scaling strictly on VAL split
    val_logits = val_uncal["logits"]
    val_labels = torch.from_numpy(val_uncal["targets"]).long()
    T_star = fit_temperature(val_logits, val_labels)

    # 3. Evaluate on VAL (calibrated)
    val_cal = evaluate_with_temperature(model, val_loader, device, temperature=T_star)
    val_ece_post = val_cal["ece"]

    print(f"    Seed {seed} Val Calibration: T*={T_star:.4f} | ECE: {val_ece_pre*100:.2f}% -> {val_ece_post*100:.2f}%", flush=True)

    # 4. Final Evaluation on holdout TEST split (TOUCHED ONCE)
    test_uncal = evaluate_with_temperature(model, test_loader, device, temperature=1.0)
    test_cal = evaluate_with_temperature(model, test_loader, device, temperature=T_star)

    # Save checkpoint with calibrated temperature
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"judge_seed_{seed}.pt")
    torch.save({
        "model_state_dict": best_state,
        "seed": seed,
        "config": config_name,
        "temperature": T_star,
        "val_ece_uncalibrated": val_ece_pre,
        "val_ece_calibrated": val_ece_post,
        "val_metrics": {k: v for k, v in val_cal.items() if k not in ("logits", "targets", "ece_info")},
        "test_metrics": {k: v for k, v in test_cal.items() if k not in ("logits", "targets", "ece_info")},
        "training_time_s": dur
    }, ckpt_path)

    print(f"    Seed {seed} TEST Result: Acc={test_cal['accuracy']*100:.2f}% | BalAcc={test_cal['balanced_accuracy']*100:.2f}% | MacroF1={test_cal['macro_f1']*100:.2f}% | ECE={test_cal['ece']*100:.2f}%", flush=True)

    return {
        "seed": seed,
        "config": config_name,
        "temperature": T_star,
        "val_ece_pre": val_ece_pre,
        "val_ece_post": val_ece_post,
        "val_macro_f1": val_cal["macro_f1"],
        "val_accuracy": val_cal["accuracy"],
        "val_balanced_acc": val_cal["balanced_accuracy"],
        "test_accuracy": test_cal["accuracy"],
        "test_balanced_acc": test_cal["balanced_accuracy"],
        "test_macro_f1": test_cal["macro_f1"],
        "test_macro_precision": test_cal["macro_precision"],
        "test_macro_recall": test_cal["macro_recall"],
        "test_ece_uncal": test_uncal["ece"],
        "test_ece_cal": test_cal["ece"],
        "recall_head": test_cal["per_class_recall"]["head"],
        "recall_helmet": test_cal["per_class_recall"]["helmet"],
        "recall_vest": test_cal["per_class_recall"]["vest"],
        "prec_head": test_cal["per_class_precision"]["head"],
        "prec_helmet": test_cal["per_class_precision"]["helmet"],
        "prec_vest": test_cal["per_class_precision"]["vest"],
        "f1_head": test_cal["per_class_f1"]["head"],
        "f1_helmet": test_cal["per_class_f1"]["helmet"],
        "f1_vest": test_cal["per_class_f1"]["vest"],
        "training_time_s": dur,
        "checkpoint": ckpt_path
    }


def main():
    parser = argparse.ArgumentParser(description="Train winning judge configuration on 5 seeds.")
    parser.add_argument("--config", type=str, default="a_soft_weights", help="Winning ablation config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output_dir", type=str, default="results/judge")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--budget", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"FINAL 5-SEED JUDGE RETRAINING & CALIBRATION: {args.config.upper()}")
    print(f"Seeds: {args.seeds} | Device: {device} | Epochs: {args.epochs}")
    print("=" * 80)

    t0 = time.time()
    print("[*] Loading all dataset splits (Train, Val, Test)...", flush=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_cached_splits()
    print(f"    Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape} in {time.time()-t0:.1f}s")

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    seed_results = []
    total_start = time.time()

    for s in args.seeds:
        res = train_and_calibrate_seed(
            seed=s,
            config_name=args.config,
            X_train=X_train,
            y_train=y_train,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=64,
            sample_budget=args.budget
        )
        seed_results.append(res)

    total_time = time.time() - total_start
    df_results = pd.DataFrame(seed_results)
    benchmark_csv = os.path.join(args.output_dir, "judge_benchmark.csv")
    df_results.to_csv(benchmark_csv, index=False)

    summary_stats = {
        "num_seeds": len(args.seeds),
        "chosen_config": args.config,
        "total_training_time_s": total_time,
        "temperature_mean": float(df_results["temperature"].mean()),
        "temperature_std": float(df_results["temperature"].std()),
        "val_ece_pre_mean": float(df_results["val_ece_pre"].mean()),
        "val_ece_post_mean": float(df_results["val_ece_post"].mean()),
        "test_ece_uncal_mean": float(df_results["test_ece_uncal"].mean()),
        "test_ece_cal_mean": float(df_results["test_ece_cal"].mean()),
        "test_accuracy_mean": float(df_results["test_accuracy"].mean()),
        "test_accuracy_std": float(df_results["test_accuracy"].std()),
        "test_balanced_acc_mean": float(df_results["test_balanced_acc"].mean()),
        "test_balanced_acc_std": float(df_results["test_balanced_acc"].std()),
        "test_macro_f1_mean": float(df_results["test_macro_f1"].mean()),
        "test_macro_f1_std": float(df_results["test_macro_f1"].std()),
        "test_macro_precision_mean": float(df_results["test_macro_precision"].mean()),
        "test_macro_precision_std": float(df_results["test_macro_precision"].std()),
        "test_macro_recall_mean": float(df_results["test_macro_recall"].mean()),
        "test_macro_recall_std": float(df_results["test_macro_recall"].std()),
        "recall_head_mean": float(df_results["recall_head"].mean()),
        "recall_head_std": float(df_results["recall_head"].std()),
        "recall_helmet_mean": float(df_results["recall_helmet"].mean()),
        "recall_helmet_std": float(df_results["recall_helmet"].std()),
        "recall_vest_mean": float(df_results["recall_vest"].mean()),
        "recall_vest_std": float(df_results["recall_vest"].std()),
        "prec_head_mean": float(df_results["prec_head"].mean()),
        "prec_head_std": float(df_results["prec_head"].std()),
        "prec_helmet_mean": float(df_results["prec_helmet"].mean()),
        "prec_helmet_std": float(df_results["prec_helmet"].std()),
        "prec_vest_mean": float(df_results["prec_vest"].mean()),
        "prec_vest_std": float(df_results["prec_vest"].std()),
        "f1_head_mean": float(df_results["f1_head"].mean()),
        "f1_head_std": float(df_results["f1_head"].std()),
        "f1_helmet_mean": float(df_results["f1_helmet"].mean()),
        "f1_helmet_std": float(df_results["f1_helmet"].std()),
        "f1_vest_mean": float(df_results["f1_vest"].mean()),
        "f1_vest_std": float(df_results["f1_vest"].std()),
    }

    summary_json = os.path.join(args.output_dir, "judge_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL 5-SEED BENCHMARK REPORT (TOUCHED TEST SPLIT ONCE):")
    print("=" * 80)
    print(f"Config:                {args.config.upper()}")
    print(f"Calibrated Temp T*:    {summary_stats['temperature_mean']:.4f} +/- {summary_stats['temperature_std']:.4f}")
    print(f"Val ECE (Pre -> Post): {summary_stats['val_ece_pre_mean']*100:.2f}% -> {summary_stats['val_ece_post_mean']*100:.2f}%")
    print(f"Test ECE:              {summary_stats['test_ece_uncal_mean']*100:.2f}% (uncal) -> {summary_stats['test_ece_cal_mean']*100:.2f}% (cal)")
    print(f"Test Accuracy:         {summary_stats['test_accuracy_mean']*100:.2f}% +/- {summary_stats['test_accuracy_std']*100:.2f}%")
    print(f"Balanced Accuracy:     {summary_stats['test_balanced_acc_mean']*100:.2f}% +/- {summary_stats['test_balanced_acc_std']*100:.2f}%")
    print(f"Macro F1 Score:        {summary_stats['test_macro_f1_mean']*100:.2f}% +/- {summary_stats['test_macro_f1_std']*100:.2f}%")
    print(f"Macro Precision:       {summary_stats['test_macro_precision_mean']*100:.2f}% +/- {summary_stats['test_macro_precision_std']*100:.2f}%")
    print(f"Macro Recall:          {summary_stats['test_macro_recall_mean']*100:.2f}% +/- {summary_stats['test_macro_recall_std']*100:.2f}%")
    print(f"  Head:   Recall={summary_stats['recall_head_mean']*100:.2f}% | Prec={summary_stats['prec_head_mean']*100:.2f}% | F1={summary_stats['f1_head_mean']*100:.2f}%")
    print(f"  Helmet: Recall={summary_stats['recall_helmet_mean']*100:.2f}% | Prec={summary_stats['prec_helmet_mean']*100:.2f}% | F1={summary_stats['f1_helmet_mean']*100:.2f}%")
    print(f"  Vest:   Recall={summary_stats['recall_vest_mean']*100:.2f}% | Prec={summary_stats['prec_vest_mean']*100:.2f}% | F1={summary_stats['f1_vest_mean']*100:.2f}%")
    print("=" * 80)
    print(f"[*] Checkpoint & summary saved to {args.output_dir}")


if __name__ == "__main__":
    main()

