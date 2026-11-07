"""
scripts/diagnose_judge_val.py
Step 1: In-depth validation error diagnosis for judge_v1.
Computes raw and normalized confusion matrices on the VAL split (not test),
per-class precision, recall, and F1 across all 5 seeds,
and exports 24 misclassified VAL crops per error type into high-res montages.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score
)

sys.path.insert(0, os.path.abspath("."))
from common.data import CLASS_NAMES
from common.judge import SmallResNetV1


def load_val_data():
    """Load cached validation tensor and match each index to its file path."""
    val_cache = torch.load("data_cache/val_crops.pt", map_location="cpu")
    X_val = val_cache["X"].float() / 255.0  # (N, 3, 64, 64)
    y_val = val_cache["y"].numpy()          # (N,)

    # Map indices to exact file paths (crops_64/val/<class>/*.png sorted)
    file_paths = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join("crops_64", "val", cls_name)
        fnames = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(".png")])
        for fn in fnames:
            file_paths.append(os.path.join(cls_dir, fn))

    assert len(file_paths) == len(y_val), f"Mismatch: {len(file_paths)} files vs {len(y_val)} labels"
    return X_val, y_val, file_paths


def evaluate_models_on_val(X_val, y_val, model_dir="results/judge_v1"):
    """Evaluate all 5 seeds and ensemble on VAL."""
    seeds = [42, 43, 44, 45, 46]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_ds = TensorDataset(X_val, torch.from_numpy(y_val))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    seed_probs = []
    seed_metrics = []

    for s in seeds:
        ckpt_path = os.path.join(model_dir, f"judge_seed_{s}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = SmallResNetV1().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs_list = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                logits = model(x)
                p = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(p)
        
        probs_arr = np.concatenate(probs_list, axis=0)  # (N, 3)
        preds_arr = np.argmax(probs_arr, axis=1)

        acc = accuracy_score(y_val, preds_arr)
        bal_acc = balanced_accuracy_score(y_val, preds_arr)
        p_c, r_c, f_c, _ = precision_recall_fscore_support(y_val, preds_arr, labels=[0, 1, 2], zero_division=0)

        seed_probs.append(probs_arr)
        seed_metrics.append({
            "seed": s,
            "acc": acc,
            "bal_acc": bal_acc,
            "prec_head": p_c[0], "rec_head": r_c[0], "f1_head": f_c[0],
            "prec_helmet": p_c[1], "rec_helmet": r_c[1], "f1_helmet": f_c[1],
            "prec_vest": p_c[2], "rec_vest": r_c[2], "f1_vest": f_c[2],
            "macro_f1": float(np.mean(f_c)),
            "macro_prec": float(np.mean(p_c)),
            "macro_rec": float(np.mean(r_c))
        })

    # Ensemble average (soft voting)
    ensemble_probs = np.mean(seed_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    return seed_metrics, ensemble_probs, ensemble_preds, seed_probs


def create_error_montage(error_samples, out_path, error_title, max_crops=24):
    """
    Stitches up to 24 misclassified crops into a neat 4x6 grid
    with true label, predicted label, and confidence score.
    """
    n_show = min(len(error_samples), max_crops)
    if n_show == 0:
        return

    cols = 6
    rows = int(np.ceil(n_show / cols))
    cell_w, cell_h = 100, 120
    montage = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 243, 246))
    draw = ImageDraw.Draw(montage)

    for i in range(n_show):
        sample = error_samples[i]
        r = i // cols
        c = i % cols
        x_off = c * cell_w
        y_off = r * cell_h

        # Load and paste crop (scaled up to 72x72 for visibility)
        try:
            with Image.open(sample["path"]) as img:
                img_resized = img.convert("RGB").resize((72, 72), Image.Resampling.NEAREST)
                montage.paste(img_resized, (x_off + 14, y_off + 6))
        except Exception:
            pass

        # Text captions
        conf_str = f"P({sample['pred_name']})={sample['pred_prob']*100:.0f}%"
        draw.text((x_off + 6, y_off + 82), sample['fname'][:14], fill=(40, 40, 40))
        draw.text((x_off + 6, y_off + 96), conf_str, fill=(200, 30, 30))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    montage.save(out_path)
    print(f"[*] Saved {n_show} error montage -> {out_path}")


def main():
    print("=" * 80)
    print("STEP 1: JUDGE-V1 VALIDATION DIAGNOSTICS & ERROR TAXONOMY")
    print("=" * 80)

    X_val, y_val, file_paths = load_val_data()
    print(f"Loaded {len(X_val)} validation crops across {len(CLASS_NAMES)} classes:")
    val_counts = np.bincount(y_val, minlength=3)
    for idx, c in enumerate(CLASS_NAMES):
        print(f"  - Class {idx} ({c:<7}): {val_counts[idx]} crops ({val_counts[idx]/len(y_val)*100:.2f}%)")

    seed_metrics, ensemble_probs, ensemble_preds, seed_probs = evaluate_models_on_val(X_val, y_val)
    df_metrics = pd.DataFrame(seed_metrics)

    print("\n" + "-" * 80)
    print("VAL PERFORMANCE ACROSS 5 SEEDS (Mean +/- Std):")
    print("-" * 80)
    print(f"Raw Accuracy:      {df_metrics['acc'].mean()*100:.2f}% +/- {df_metrics['acc'].std()*100:.2f}%")
    print(f"Balanced Accuracy: {df_metrics['bal_acc'].mean()*100:.2f}% +/- {df_metrics['bal_acc'].std()*100:.2f}%")
    print(f"Macro F1 Score:    {df_metrics['macro_f1'].mean()*100:.2f}% +/- {df_metrics['macro_f1'].std()*100:.2f}%")
    print(f"Macro Precision:   {df_metrics['macro_prec'].mean()*100:.2f}% +/- {df_metrics['macro_prec'].std()*100:.2f}%")
    print(f"Macro Recall:      {df_metrics['macro_rec'].mean()*100:.2f}% +/- {df_metrics['macro_rec'].std()*100:.2f}%")
    print("\nPer-Class Breakdown on VAL:")
    print(f"  Head   -> Recall: {df_metrics['rec_head'].mean()*100:.2f}% +/- {df_metrics['rec_head'].std()*100:.2f}% | "
          f"Precision: {df_metrics['prec_head'].mean()*100:.2f}% +/- {df_metrics['prec_head'].std()*100:.2f}% | "
          f"F1: {df_metrics['f1_head'].mean()*100:.2f}%")
    print(f"  Helmet -> Recall: {df_metrics['rec_helmet'].mean()*100:.2f}% +/- {df_metrics['rec_helmet'].std()*100:.2f}% | "
          f"Precision: {df_metrics['prec_helmet'].mean()*100:.2f}% +/- {df_metrics['prec_helmet'].std()*100:.2f}% | "
          f"F1: {df_metrics['f1_helmet'].mean()*100:.2f}%")
    print(f"  Vest   -> Recall: {df_metrics['rec_vest'].mean()*100:.2f}% +/- {df_metrics['rec_vest'].std()*100:.2f}% | "
          f"Precision: {df_metrics['prec_vest'].mean()*100:.2f}% +/- {df_metrics['prec_vest'].std()*100:.2f}% | "
          f"F1: {df_metrics['f1_vest'].mean()*100:.2f}%")

    # Use Seed 42 for concrete sample extraction (or Ensemble)
    # Both are reported
    preds_s42 = np.argmax(seed_probs[0], axis=1)
    probs_s42 = seed_probs[0]

    # Confusion Matrices on VAL
    raw_cm = confusion_matrix(y_val, preds_s42, labels=[0, 1, 2])
    norm_cm = raw_cm.astype(float) / raw_cm.sum(axis=1, keepdims=True)

    print("\n" + "=" * 80)
    print("RAW CONFUSION MATRIX ON VAL (Seed 42): [Rows: True, Cols: Pred]")
    print("=" * 80)
    print(f"{'':<10} {'Pred head':<12} {'Pred helmet':<12} {'Pred vest':<12} {'Total'}")
    for idx, c in enumerate(CLASS_NAMES):
        row = raw_cm[idx]
        print(f"True {c:<7} {row[0]:<12} {row[1]:<12} {row[2]:<12} {val_counts[idx]}")

    print("\n" + "=" * 80)
    print("NORMALISED CONFUSION MATRIX ON VAL (Seed 42):")
    print("=" * 80)
    print(f"{'':<10} {'Pred head':<12} {'Pred helmet':<12} {'Pred vest':<12}")
    for idx, c in enumerate(CLASS_NAMES):
        row = norm_cm[idx]
        print(f"True {c:<7} {row[0]*100:>8.2f}%    {row[1]*100:>8.2f}%    {row[2]*100:>8.2f}%")

    # Catalog Misclassification Error Types
    out_dir = "results/judge_v1/diagnosis"
    os.makedirs(out_dir, exist_ok=True)

    error_buckets = {}
    for true_idx in range(3):
        for pred_idx in range(3):
            if true_idx != pred_idx:
                key = f"true_{CLASS_NAMES[true_idx]}_pred_{CLASS_NAMES[pred_idx]}"
                error_buckets[key] = []

    for i in range(len(y_val)):
        true_lbl = y_val[i]
        pred_lbl = preds_s42[i]
        if true_lbl != pred_lbl:
            key = f"true_{CLASS_NAMES[true_lbl]}_pred_{CLASS_NAMES[pred_lbl]}"
            error_buckets[key].append({
                "idx": i,
                "path": file_paths[i],
                "fname": os.path.basename(file_paths[i]),
                "true_name": CLASS_NAMES[true_lbl],
                "pred_name": CLASS_NAMES[pred_lbl],
                "pred_prob": float(probs_s42[i, pred_lbl]),
                "head_prob": float(probs_s42[i, 0]),
                "helmet_prob": float(probs_s42[i, 1]),
                "vest_prob": float(probs_s42[i, 2])
            })

    print("\n" + "=" * 80)
    print("MISCLASSIFICATION ERROR COUNTS ON VAL:")
    print("=" * 80)
    for k, samples in error_buckets.items():
        print(f"  - {k:<28}: {len(samples)} crops")

    # Export Top 24 Montages for each error type
    diagnosis_meta = {}
    for k, samples in error_buckets.items():
        # Sort by prediction confidence descending
        samples_sorted = sorted(samples, key=lambda x: x["pred_prob"], reverse=True)
        montage_path = os.path.join(out_dir, f"{k}_montage.png")
        create_error_montage(samples_sorted, montage_path, k, max_crops=24)
        
        # Save sample metadata
        json_path = os.path.join(out_dir, f"{k}_samples.json")
        with open(json_path, "w") as f:
            json.dump(samples_sorted[:24], f, indent=2)
        diagnosis_meta[k] = {
            "total_count": len(samples),
            "sample_24": samples_sorted[:24]
        }

    # Save summary report JSON
    with open(os.path.join(out_dir, "diagnosis_summary.json"), "w") as f:
        json.dump({
            "val_counts": {CLASS_NAMES[i]: int(val_counts[i]) for i in range(3)},
            "raw_confusion_matrix": raw_cm.tolist(),
            "norm_confusion_matrix": norm_cm.tolist(),
            "error_counts": {k: len(v) for k, v in error_buckets.items()},
            "seed_metrics": seed_metrics
        }, f, indent=2)

    print("\n[*] Validation diagnosis export completed!")


if __name__ == "__main__":
    main()

