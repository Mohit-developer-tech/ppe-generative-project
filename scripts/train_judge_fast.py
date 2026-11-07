"""
scripts/train_judge_fast.py
High-performance 5-seed training runner for the independent SmallResNet judge.
Uses pre-cached uint8 tensors from data_cache/ for near-instant memory loading.
Trains across 5 random seeds (42, 43, 44, 45, 46) on real train crops with
class-balanced cross-entropy and random horizontal flip augmentation.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath("."))

from common.data import CLASS_NAMES, set_seed
from common.judge import SmallResNet, evaluate_classifier


def load_cached_split(split_name: str, cache_dir: str = "data_cache"):
    """Load pre-cached tensor split instantly into RAM."""
    path = os.path.join(cache_dir, f"{split_name}_crops.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cached split not found at {path}. Run scripts/cache_dataset.py first.")
    data = torch.load(path, map_location="cpu")
    # Convert uint8 to float32 [0, 1]
    X = data["X"].float() / 255.0
    y = data["y"]
    return X, y


def train_single_seed(
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: str,
    device: torch.device,
    epochs: int = 7,
    batch_size: int = 64,
    lr: float = 2e-3,
    sample_budget: int = 8000
) -> Dict[str, Any]:
    """Train one seed of the SmallResNet judge."""
    set_seed(seed)
    
    # Class-balanced sampling
    targets_np = y_train.numpy()
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))
    weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
    sample_weights = weights[targets_np]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=sample_budget,
        replacement=True
    )

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    # Class-balanced loss weights
    total_samples = len(targets_np)
    c_weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
    class_weights_t = torch.tensor(c_weights, dtype=torch.float32, device=device)

    model = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None

    print(f"\n=======================================================", flush=True)
    print(f"TRAINING JUDGE SEED {seed} ({epochs} Epochs, Budget: {sample_budget})", flush=True)
    print(f"=======================================================", flush=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Random horizontal flip
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
        epoch_loss = train_loss / max(1, n_batches)

        # Validation evaluation
        val_eval = evaluate_classifier(model, val_loader, device)
        val_f1 = val_eval["macro_f1"]
        val_acc = val_eval["accuracy"]
        val_bal = val_eval["balanced_accuracy"]
        ep_dur = time.time() - ep_t0

        flag = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = " -> [BEST VAL F1]"

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({ep_dur:.1f}s) | "
              f"Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | Bal Acc: {val_bal*100:.2f}%{flag}", flush=True)

    # Restore best checkpoint
    model.load_state_dict(best_state)
    dur = time.time() - t0
    print(f"[*] Seed {seed} training finished in {dur:.1f}s. Best Val F1: {best_val_f1:.4f}", flush=True)

    # Evaluate on full holdout test set (7470 samples)
    test_eval = evaluate_classifier(model, test_loader, device)
    val_final = evaluate_classifier(model, val_loader, device)

    ckpt_path = os.path.join(output_dir, f"judge_seed_{seed}.pt")
    torch.save({
        "model_state_dict": best_state,
        "seed": seed,
        "best_val_f1": best_val_f1,
        "val_metrics": {k: v for k, v in val_final.items() if k not in ("preds", "targets", "probs")},
        "test_metrics": {k: v for k, v in test_eval.items() if k not in ("preds", "targets", "probs")},
        "training_time_s": dur
    }, ckpt_path)
    print(f"    Saved Checkpoint -> {ckpt_path}", flush=True)

    return {
        "seed": seed,
        "test_accuracy": test_eval["accuracy"],
        "test_balanced_acc": test_eval["balanced_accuracy"],
        "test_macro_f1": test_eval["macro_f1"],
        "test_macro_precision": test_eval["macro_precision"],
        "test_macro_recall": test_eval["macro_recall"],
        "recall_head": test_eval["per_class_recall"]["head"],
        "recall_helmet": test_eval["per_class_recall"]["helmet"],
        "recall_vest": test_eval["per_class_recall"]["vest"],
        "checkpoint": ckpt_path,
        "training_time_s": dur
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("FAST INDEPENDENT SMALL-RESNET JUDGE TRAINING (5 SEEDS)", flush=True)
    print(f"Hardware Device: {device}", flush=True)
    print("=" * 80, flush=True)

    output_dir = "results/judge"
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    print("[*] Loading cached datasets...", flush=True)
    X_train, y_train = load_cached_split("train")
    X_val, y_val = load_cached_split("val")
    X_test, y_test = load_cached_split("test")
    print(f"    Loaded in {time.time()-t0:.1f}s: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}", flush=True)

    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    seeds = [42, 43, 44, 45, 46]
    results = []

    total_start = time.time()
    for s in seeds:
        res = train_single_seed(
            seed=s,
            X_train=X_train,
            y_train=y_train,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=output_dir,
            device=device,
            epochs=7,
            batch_size=64,
            lr=2e-3,
            sample_budget=8000
        )
        results.append(res)

    total_time = time.time() - total_start

    # Export Benchmark Tables
    df_results = pd.DataFrame(results)
    benchmark_csv = os.path.join(output_dir, "judge_benchmark.csv")
    df_results.to_csv(benchmark_csv, index=False)

    summary_stats = {
        "num_seeds": len(seeds),
        "total_training_time_s": total_time,
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
    }

    summary_json = os.path.join(output_dir, "judge_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("FINAL INDEPENDENT JUDGE BENCHMARK SUMMARY (5 SEEDS):", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Ensemble Training Time: {total_time/60:.2f} minutes", flush=True)
    print(f"Test Accuracy:         {summary_stats['test_accuracy_mean']*100:.2f}% +/- {summary_stats['test_accuracy_std']*100:.2f}%", flush=True)
    print(f"Balanced Accuracy:     {summary_stats['test_balanced_acc_mean']*100:.2f}% +/- {summary_stats['test_balanced_acc_std']*100:.2f}%", flush=True)
    print(f"Macro F1 Score:        {summary_stats['test_macro_f1_mean']*100:.2f}% +/- {summary_stats['test_macro_f1_std']*100:.2f}%", flush=True)
    print(f"Macro Precision:       {summary_stats['test_macro_precision_mean']*100:.2f}% +/- {summary_stats['test_macro_precision_std']*100:.2f}%", flush=True)
    print(f"Macro Recall:          {summary_stats['test_macro_recall_mean']*100:.2f}% +/- {summary_stats['test_macro_recall_std']*100:.2f}%", flush=True)
    print(f"  - Head Recall (Viol):{summary_stats['recall_head_mean']*100:.2f}% +/- {summary_stats['recall_head_std']*100:.2f}%", flush=True)
    print(f"  - Helmet Recall:     {summary_stats['recall_helmet_mean']*100:.2f}% +/- {summary_stats['recall_helmet_std']*100:.2f}%", flush=True)
    print(f"  - Vest Recall:       {summary_stats['recall_vest_mean']*100:.2f}% +/- {summary_stats['recall_vest_std']*100:.2f}%", flush=True)
    print("=" * 80, flush=True)

    # Comparison against AE+SVM 92.33%
    ae_svm_accuracy = 0.9233
    diff = (summary_stats["test_accuracy_mean"] - ae_svm_accuracy) * 100
    if summary_stats["test_accuracy_mean"] < ae_svm_accuracy:
        print(f"\n[ALERT] Judge test accuracy ({summary_stats['test_accuracy_mean']*100:.2f}%) is BELOW AE+SVM's 92.33%!", flush=True)
    else:
        print(f"\n[SUCCESS] Judge test accuracy ({summary_stats['test_accuracy_mean']*100:.2f}%) exceeds AE+SVM baseline ({ae_svm_accuracy*100:.2f}%) by {diff:+.2f}%.", flush=True)

    print(f"[*] Exported benchmark table -> {benchmark_csv}", flush=True)
    print(f"[*] Exported summary stats -> {summary_json}", flush=True)


if __name__ == "__main__":
    main()

