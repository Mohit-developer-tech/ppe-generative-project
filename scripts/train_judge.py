"""
scripts/train_judge.py
Executes the multi-seed training pipeline for the independent SmallResNet judge.
Trains across 5 random seeds (42, 43, 44, 45, 46) on real train crops with
class-balanced cross entropy, random horizontal flip augmentation, and early stopping
on validation macro-F1.
Exports all checkpoints and benchmarks to results/judge/.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from common.data import CLASS_NAMES, set_seed
from common.judge import SmallResNet, evaluate_classifier


class FastInMemoryCropDataset(Dataset):
    """
    High-performance in-memory dataset storing crops as uint8 arrays.
    Eliminates all disk I/O during training, accelerating CPU training ~5x.
    Memory footprint: ~330 MB for all 26,899 crops.
    """
    def __init__(self, root_dir: str, augment: bool = False):
        self.augment = augment
        self.samples = []
        self.targets = []
        
        t0 = time.time()
        print(f"[*] Pre-loading {root_dir} into RAM...", end="", flush=True)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            for fname in sorted(os.listdir(cls_folder)):
                if fname.lower().endswith(".png"):
                    fpath = os.path.join(cls_folder, fname)
                    with Image.open(fpath) as img:
                        arr = np.array(img.convert("RGB"), dtype=np.uint8)  # (64, 64, 3)
                        self.samples.append(arr)
                        self.targets.append(cls_idx)

        self.samples = np.stack(self.samples)  # (N, 64, 64, 3) uint8
        self.targets = np.array(self.targets, dtype=np.int64)  # (N,)
        elapsed = time.time() - t0
        print(f" Loaded {len(self.samples)} images in {elapsed:.1f}s ({self.samples.nbytes / 1e6:.1f} MB)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = self.samples[idx]  # (64, 64, 3)
        # Random horizontal flip for data augmentation
        if self.augment and np.random.rand() > 0.5:
            img_np = np.fliplr(img_np).copy()

        # Convert to tensor [3, 64, 64] in [0, 1]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return tensor, target


def train_single_judge_seed(
    seed: int,
    train_ds: FastInMemoryCropDataset,
    val_ds: FastInMemoryCropDataset,
    test_ds: FastInMemoryCropDataset,
    output_dir: str,
    device: torch.device,
    max_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5
) -> Dict[str, Any]:
    """Train a single seed of the SmallResNet judge."""
    set_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Class-balanced loss weights: N / (C * N_c)
    targets_np = train_ds.targets
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))
    total_samples = len(targets_np)
    weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    print(f"\n=======================================================")
    print(f"TRAINING JUDGE: SEED {seed} (Max {max_epochs} Epochs)")
    print(f"=======================================================")
    print(f"Class Weights: Head={weights[0]:.2f}, Helmet={weights[1]:.2f}, Vest={weights[2]:.2f}")

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = train_loss / max(1, n_batches)

        # Validation
        val_eval = evaluate_classifier(model, val_loader, device)
        val_f1 = val_eval["macro_f1"]
        val_acc = val_eval["accuracy"]
        val_bal = val_eval["balanced_accuracy"]
        ep_dur = time.time() - ep_t0

        flag = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            flag = " -> [BEST VAL F1]"
        else:
            patience_counter += 1

        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] ({ep_dur:.1f}s) | "
              f"Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | Bal Acc: {val_bal*100:.2f}%{flag}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    print(f"Seed {seed} finished in {elapsed:.1f}s. Best Val F1: {best_val_f1:.4f}")

    # Evaluate on holdout test set
    test_eval = evaluate_classifier(model, test_loader, device)
    val_final = evaluate_classifier(model, val_loader, device)

    # Save model checkpoint
    ckpt_path = os.path.join(output_dir, f"judge_seed_{seed}.pt")
    torch.save({
        "model_state_dict": best_state,
        "seed": seed,
        "best_val_f1": best_val_f1,
        "val_metrics": {k: v for k, v in val_final.items() if k not in ("preds", "targets", "probs")},
        "test_metrics": {k: v for k, v in test_eval.items() if k not in ("preds", "targets", "probs")},
        "training_time_s": elapsed
    }, ckpt_path)
    print(f"Saved Checkpoint -> {ckpt_path}")

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
        "training_time_s": elapsed
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Judge Training on device: {device}")
    
    output_dir = "results/judge"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load in-memory datasets once
    train_ds = FastInMemoryCropDataset("crops_64/train", augment=True)
    val_ds = FastInMemoryCropDataset("crops_64/val", augment=False)
    test_ds = FastInMemoryCropDataset("crops_64/test", augment=False)

    # 2. Train 5 seeds
    seeds = [42, 43, 44, 45, 46]
    all_results = []
    
    total_start = time.time()
    for s in seeds:
        res = train_single_judge_seed(
            seed=s,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            output_dir=output_dir,
            device=device,
            max_epochs=12,
            batch_size=128,
            lr=1.2e-3,
            patience=4
        )
        all_results.append(res)

    total_time = time.time() - total_start

    # 3. Export Summary Benchmark
    df_results = pd.DataFrame(all_results)
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

    print("\n" + "=" * 80)
    print("FINAL 5-SEED INDEPENDENT JUDGE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Ensemble Training Time: {total_time/60:.1f} minutes")
    print(f"Test Accuracy:         {summary_stats['test_accuracy_mean']*100:.2f}% +/- {summary_stats['test_accuracy_std']*100:.2f}%")
    print(f"Balanced Accuracy:     {summary_stats['test_balanced_acc_mean']*100:.2f}% +/- {summary_stats['test_balanced_acc_std']*100:.2f}%")
    print(f"Macro F1 Score:        {summary_stats['test_macro_f1_mean']*100:.2f}% +/- {summary_stats['test_macro_f1_std']*100:.2f}%")
    print(f"Macro Precision:       {summary_stats['test_macro_precision_mean']*100:.2f}% +/- {summary_stats['test_macro_precision_std']*100:.2f}%")
    print(f"Macro Recall:          {summary_stats['test_macro_recall_mean']*100:.2f}% +/- {summary_stats['test_macro_recall_std']*100:.2f}%")
    print(f"  - Head Recall (Violation): {summary_stats['recall_head_mean']*100:.2f}% +/- {summary_stats['recall_head_std']*100:.2f}%")
    print(f"  - Helmet Recall:           {summary_stats['recall_helmet_mean']*100:.2f}% +/- {summary_stats['recall_helmet_std']*100:.2f}%")
    print(f"  - Vest Recall:             {summary_stats['recall_vest_mean']*100:.2f}% +/- {summary_stats['recall_vest_std']*100:.2f}%")
    print("=" * 80)

    # Check against AE+SVM 92.33% threshold
    ae_svm_acc = 0.9233
    if summary_stats["test_accuracy_mean"] < ae_svm_acc:
        print(f"\n[ALERT] Judge test accuracy ({summary_stats['test_accuracy_mean']*100:.2f}%) is BELOW AE+SVM's 92.33%!")
    else:
        print(f"\n[SUCCESS] Judge test accuracy ({summary_stats['test_accuracy_mean']*100:.2f}%) comfortably exceeds AE+SVM baseline ({ae_svm_acc*100:.2f}%).")


if __name__ == "__main__":
    main()

