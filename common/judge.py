"""
common/judge.py
Supervised Small-ResNet classifier serving as the INDEPENDENT evaluation judge
for all generative and restoration PPE experiments.
Trained strictly on real train crops with class-balanced loss, flip augmentation,
and early stopping on validation macro-F1.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from common.data import CLASS_NAMES, get_dataloader, set_seed


# ==============================================================================
# Model Architecture: Small ResNet for 64x64 Crops
# ==============================================================================

class ResBlock(nn.Module):
    """Residual Block with two 3x3 convs and skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class SmallResNetV1(nn.Module):
    """
    Judge-v1 Architecture: Fast 3-stage ResNet for 64x64 inputs.
    Stem: 3 -> 24 (stride 2, 64x64 -> 32x32)
    Layer 1: 24 -> 48 (stride 1)
    Layer 2: 48 -> 96 (stride 2, 32x32 -> 16x16)
    Layer 3: 96 -> 160 (stride 2, 16x16 -> 8x8)
    Pool + Dropout + FC: 160 -> num_classes
    Total parameters: ~550K.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock(24, 48, stride=1)
        self.layer2 = ResBlock(48, 96, stride=2)
        self.layer3 = ResBlock(96, 160, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(160, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 160-dimensional representation before final classification."""
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        feats = self.dropout(feats)
        return self.fc(feats)


# Standard alias so existing checkpoints and imports load seamlessly
SmallResNet = SmallResNetV1


# ==============================================================================
# Evaluation Function
# ==============================================================================

def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate classifier performance over an entire DataLoader."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None

            x = x.to(device, non_blocking=True).float()
            # If in [-1, 1], convert to [0, 1]
            if x.min() < -0.1:
                x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            if y is not None:
                all_targets.append(y.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)

    if not all_targets:
        return {"preds": preds_arr, "probs": probs_arr}

    targets_arr = np.concatenate(all_targets, axis=0)
    acc = float(accuracy_score(targets_arr, preds_arr))
    bal_acc = float(balanced_accuracy_score(targets_arr, preds_arr))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_prec = float(precision_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(targets_arr, preds_arr, average="macro", zero_division=0))

    # Per-class recall, precision, and F1
    labels_all = list(range(len(CLASS_NAMES)))
    class_recalls = recall_score(targets_arr, preds_arr, labels=labels_all, average=None, zero_division=0)
    class_precisions = precision_score(targets_arr, preds_arr, labels=labels_all, average=None, zero_division=0)
    class_f1s = f1_score(targets_arr, preds_arr, labels=labels_all, average=None, zero_division=0)

    per_class_recall = {
        CLASS_NAMES[i]: float(class_recalls[i]) for i in range(len(CLASS_NAMES))
    }
    per_class_precision = {
        CLASS_NAMES[i]: float(class_precisions[i]) for i in range(len(CLASS_NAMES))
    }
    per_class_f1 = {
        CLASS_NAMES[i]: float(class_f1s[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "preds": preds_arr,
        "targets": targets_arr,
        "probs": probs_arr
    }


# ==============================================================================
# Generic Training Function (Reused by Data Augmentation Experiment)
# ==============================================================================

def train_classifier(
    train_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    val_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    test_data: Optional[Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]]] = None,
    seed: int = 42,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a SmallResNet with class-balanced cross entropy and early stopping on validation macro-F1.
    Exposed so that downstream data augmentation experiments can reuse the IDENTICAL architecture and budget.
    """
    set_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Format DataLoaders if raw tensors passed
    if isinstance(train_data, tuple):
        train_loader = DataLoader(TensorDataset(train_data[0], train_data[1]), batch_size=128, shuffle=True)
    else:
        train_loader = train_data

    if isinstance(val_data, tuple):
        val_loader = DataLoader(TensorDataset(val_data[0], val_data[1]), batch_size=128, shuffle=False)
    else:
        val_loader = val_data

    test_loader = None
    if test_data is not None:
        if isinstance(test_data, tuple):
            test_loader = DataLoader(TensorDataset(test_data[0], test_data[1]), batch_size=128, shuffle=False)
        else:
            test_loader = test_data

    # Compute class frequencies from train loader
    all_y = []
    for batch in train_loader:
        y = batch[1] if isinstance(batch, (list, tuple)) else batch
        all_y.append(y)
    targets_np = torch.cat(all_y).cpu().numpy()
    class_counts = np.bincount(targets_np, minlength=len(CLASS_NAMES))

    # Inverse class weights: N / (C * N_c)
    total_samples = len(targets_np)
    weights = total_samples / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(np.float32))
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=dev)

    if verbose:
        print(f"[*] Training SmallResNet Judge [Seed {seed}] on {total_samples} samples across {len(CLASS_NAMES)} classes.")
        print(f"    Class weights: head={weights[0]:.2f}, helmet={weights[1]:.2f}, vest={weights[2]:.2f}")

    model = SmallResNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_f1": [], "val_acc": [], "val_bal_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y = batch[0].to(dev).float(), batch[1].to(dev).long()
            # Random horizontal flip augmentation during training
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[3])

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, n_batches)

        # Validation phase
        val_eval = evaluate_classifier(model, val_loader, dev)
        val_f1 = val_eval["macro_f1"]
        val_acc = val_eval["accuracy"]
        val_bal_acc = val_eval["balanced_accuracy"]

        history["train_loss"].append(epoch_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            flag = " -> [Best Val F1 Checkpoint]"
        else:
            patience_counter += 1
            flag = ""

        if verbose:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc*100:.2f}% | Bal Acc: {val_bal_acc*100:.2f}%{flag}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate best model on validation and test
    final_val = evaluate_classifier(model, val_loader, dev)
    final_test = evaluate_classifier(model, test_loader, dev) if test_loader else None

    return {
        "model": model,
        "seed": seed,
        "best_val_f1": best_val_f1,
        "val_metrics": final_val,
        "test_metrics": final_test,
        "history": history
    }


# ==============================================================================
# Multi-Seed Judge Training Suite
# ==============================================================================

def train_judge_ensemble(
    seeds: List[int] = [42, 43, 44, 45, 46],
    data_dir: str = "crops_64",
    output_dir: str = "results/judge",
    epochs: int = 15,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Train 5 distinct seeds of the SmallResNet judge classifier on real train crops.
    Saves all weights and benchmark tables into results/judge/.
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("TRAINING SUPERVISED SMALL-RESNET JUDGE CLASSIFIER (5 SEEDS)")
    print("=" * 75)
    print(f"Hardware Device: {dev}")
    print(f"Seeds: {seeds}")
    print(f"Results Directory: {output_dir}")

    # Build persistent fast loaders (preloaded in memory for speed on CPU)
    print("[*] Loading full datasets for judge training...")
    t0 = time.time()
    train_loader = get_dataloader("train", data_dir=data_dir, batch_size=128, balanced=False, augment=False)
    val_loader = get_dataloader("val", data_dir=data_dir, batch_size=256, balanced=False, augment=False)
    test_loader = get_dataloader("test", data_dir=data_dir, batch_size=256, balanced=False, augment=False)

    seed_results = []
    for s in seeds:
        print(f"\n--- Training Judge Seed {s} ---")
        res = train_classifier(
            train_data=train_loader,
            val_data=val_loader,
            test_data=test_loader,
            seed=s,
            epochs=epochs,
            patience=5,
            device=dev,
            verbose=True
        )
        test_m = res["test_metrics"]
        val_m = res["val_metrics"]

        # Save model checkpoint
        ckpt_path = os.path.join(output_dir, f"judge_seed_{s}.pt")
        torch.save({
            "model_state_dict": res["model"].state_dict(),
            "seed": s,
            "val_metrics": {k: v for k, v in val_m.items() if k not in ("preds", "targets", "probs")},
            "test_metrics": {k: v for k, v in test_m.items() if k not in ("preds", "targets", "probs")}
        }, ckpt_path)

        seed_results.append({
            "seed": s,
            "test_accuracy": test_m["accuracy"],
            "test_balanced_acc": test_m["balanced_accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_macro_precision": test_m["macro_precision"],
            "test_macro_recall": test_m["macro_recall"],
            "recall_head": test_m["per_class_recall"]["head"],
            "recall_helmet": test_m["per_class_recall"]["helmet"],
            "recall_vest": test_m["per_class_recall"]["vest"],
            "checkpoint": ckpt_path
        })

    df_seeds = pd.DataFrame(seed_results)
    benchmark_csv = os.path.join(output_dir, "judge_benchmark.csv")
    df_seeds.to_csv(benchmark_csv, index=False)

    # Compute Aggregate Benchmark (mean +/- std)
    summary_stats = {
        "num_seeds": len(seeds),
        "test_accuracy_mean": float(df_seeds["test_accuracy"].mean()),
        "test_accuracy_std": float(df_seeds["test_accuracy"].std()),
        "test_balanced_acc_mean": float(df_seeds["test_balanced_acc"].mean()),
        "test_balanced_acc_std": float(df_seeds["test_balanced_acc"].std()),
        "test_macro_f1_mean": float(df_seeds["test_macro_f1"].mean()),
        "test_macro_f1_std": float(df_seeds["test_macro_f1"].std()),
        "test_macro_precision_mean": float(df_seeds["test_macro_precision"].mean()),
        "test_macro_precision_std": float(df_seeds["test_macro_precision"].std()),
        "test_macro_recall_mean": float(df_seeds["test_macro_recall"].mean()),
        "test_macro_recall_std": float(df_seeds["test_macro_recall"].std()),
        "recall_head_mean": float(df_seeds["recall_head"].mean()),
        "recall_head_std": float(df_seeds["recall_head"].std()),
        "recall_helmet_mean": float(df_seeds["recall_helmet"].mean()),
        "recall_helmet_std": float(df_seeds["recall_helmet"].std()),
        "recall_vest_mean": float(df_seeds["recall_vest"].mean()),
        "recall_vest_std": float(df_seeds["recall_vest"].std()),
    }

    summary_json = os.path.join(output_dir, "judge_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 75)
    print("INDEPENDENT SMALL-RESNET JUDGE BENCHMARK SUMMARY (5 SEEDS):")
    print("=" * 75)
    print(f"Test Accuracy:         {summary_stats['test_accuracy_mean']*100:.2f}% +/- {summary_stats['test_accuracy_std']*100:.2f}%")
    print(f"Balanced Accuracy:     {summary_stats['test_balanced_acc_mean']*100:.2f}% +/- {summary_stats['test_balanced_acc_std']*100:.2f}%")
    print(f"Macro F1 Score:        {summary_stats['test_macro_f1_mean']*100:.2f}% +/- {summary_stats['test_macro_f1_std']*100:.2f}%")
    print(f"Macro Precision:       {summary_stats['test_macro_precision_mean']*100:.2f}% +/- {summary_stats['test_macro_precision_std']*100:.2f}%")
    print(f"Macro Recall:          {summary_stats['test_macro_recall_mean']*100:.2f}% +/- {summary_stats['test_macro_recall_std']*100:.2f}%")
    print(f"  - Head Recall (Viol):{summary_stats['recall_head_mean']*100:.2f}% +/- {summary_stats['recall_head_std']*100:.2f}%")
    print(f"  - Helmet Recall:     {summary_stats['recall_helmet_mean']*100:.2f}% +/- {summary_stats['recall_helmet_std']*100:.2f}%")
    print(f"  - Vest Recall:       {summary_stats['recall_vest_mean']*100:.2f}% +/- {summary_stats['recall_vest_std']*100:.2f}%")
    print("=" * 75)

    # Check against AE+SVM 92.33% threshold
    ae_svm_accuracy = 0.9233
    if summary_stats["test_accuracy_mean"] < ae_svm_accuracy:
        print(f"\n[ALERT] Judge test accuracy ({summary_stats['test_accuracy_mean']*100:.2f}%) is BELOW AE+SVM's 92.33%!")
    else:
        print(f"\n[SUCCESS] Judge test accuracy ({summary_stats['test_accuracy_mean']*100:.2f}%) comfortably exceeds AE+SVM baseline (92.33%).")

    return summary_stats


# ==============================================================================
# Runtime Ensemble Judge Scorer
# ==============================================================================

class EnsembleJudge:
    """
    Ensemble of trained SmallResNet models used to score generated / restored PPE crops.
    """
    def __init__(self, model_dir: str = "results/judge", device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List[SmallResNet] = []

        model_paths = sorted(Path(model_dir).glob("judge_seed_*.pt"))
        if not model_paths:
            # Fallback check
            raise FileNotFoundError(f"No trained judge checkpoints found in {model_dir}")

        for p in model_paths:
            m = SmallResNet(in_channels=3, num_classes=3)
            ckpt = torch.load(p, map_location=self.device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval().to(self.device)
            self.models.append(m)

    def score_crops(self, crops_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Score a batch of image crops [B, 3, 64, 64] with the 5-seed ensemble.
        Returns predicted classes, probabilities, violation flags, and prediction entropy.
        """
        t = crops_tensor.to(self.device).float()
        if t.min() < -0.1:
            t = torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)

        all_probs = []
        with torch.no_grad():
            for m in self.models:
                logits = m(t)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Average probabilities across ensemble seeds
        mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (B, 3)
        preds = torch.argmax(mean_probs, dim=1).cpu().numpy()
        probs_np = mean_probs.cpu().numpy()

        # Violation is True if predicted class is 0 (head = safety violation)
        violations = (preds == 0).tolist()
        pred_classes = [CLASS_NAMES[idx] for idx in preds]

        # Normalized entropy: H = - sum(p * log(p)) / log(3)
        eps = 1e-8
        entropy = - np.sum(probs_np * np.log(probs_np + eps), axis=1) / np.log(3.0)

        return {
            "predicted_indices": preds.tolist(),
            "predicted_classes": pred_classes,
            "probabilities": probs_np.tolist(),
            "is_violation": violations,
            "confidence": np.max(probs_np, axis=1).tolist(),
            "entropy": entropy.tolist()
        }

    def __call__(self, crops_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass returning averaged logits for PyTorch module compatibility."""
        t = crops_tensor.to(self.device).float()
        if t.min() < -0.1:
            t = torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)
        with torch.no_grad():
            all_logits = [m(t) for m in self.models]
        return torch.stack(all_logits, dim=0).mean(dim=0)


def load_judge(model_dir: str = "results/judge", device: Optional[torch.device] = None) -> EnsembleJudge:
    """Helper function to instantiate and return the trained Judge ensemble."""
    return EnsembleJudge(model_dir=model_dir, device=device)


