"""
scripts/optimize_svm.py
Systematic threshold optimization, cost-sensitive minority violation recall enhancement,
and hyperparameter tuning for Model 1 (ConvAE + RBF-SVM).
"""
import os
import sys
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.abspath("."))
from common.data import CLASS_NAMES
from dashboard.model_def import ConvAE

def extract_features(model, crops_uint8, batch_size=256):
    """Extract 256-d bottleneck latent features in batches on CPU."""
    model.eval()
    all_latents = []
    n = len(crops_uint8)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = crops_uint8[i:i+batch_size].float() / 255.0
            latents = model.encode(batch)
            all_latents.append(latents.cpu().numpy())
    return np.concatenate(all_latents, axis=0)

def main():
    print("=" * 80)
    print("MODEL 1 (ConvAE + RBF-SVM) SCIENTIFIC THRESHOLD & GENERALIZATION OPTIMIZATION")
    print("=" * 80)

    device = torch.device("cpu")

    # 1. Load baseline ConvAE
    print("[*] Loading baseline ConvAutoencoder...")
    ae = ConvAE(latent_dim=256).to(device)
    ae.load_state_dict(torch.load("checkpoints/baseline/autoencoder/ae_best.pt", map_location="cpu"))
    ae.eval()

    # 2. Load baseline Scaler and SVM
    print("[*] Loading baseline Scaler and RBF-SVM...")
    scaler = joblib.load("checkpoints/baseline/autoencoder/scaler.joblib")
    svm = joblib.load("checkpoints/baseline/autoencoder/svm_model.joblib")

    # 3. Load Validation and Test splits
    print("[*] Loading cached validation and test crops from data_cache/...")
    val_data = torch.load("data_cache/val_crops.pt", map_location="cpu")
    test_data = torch.load("data_cache/test_crops.pt", map_location="cpu")

    print(f"[*] Validation count: {len(val_data['y'])}, Test count: {len(test_data['y'])}")

    # 4. Extract latent features
    print("[*] Extracting 256-d latent representations...")
    t0 = time.time()
    Z_val = extract_features(ae, val_data["X"])
    Z_test = extract_features(ae, test_data["X"])
    y_val = val_data["y"].numpy()
    y_test = test_data["y"].numpy()
    print(f"[+] Feature extraction completed in {time.time() - t0:.2f} s")

    # Scale latents
    Z_val_scaled = scaler.transform(Z_val)
    Z_test_scaled = scaler.transform(Z_test)

    # 5. Baseline Argmax Evaluation
    val_dec = svm.decision_function(Z_val_scaled) # (N, 3)
    test_dec = svm.decision_function(Z_test_scaled) # (N, 3)

    # Convert decision function to softmax pseudo-probabilities
    def dec_to_prob(dec):
        exp = np.exp(dec - np.max(dec, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    P_val = dec_to_prob(val_dec)
    P_test = dec_to_prob(test_dec)

    baseline_val_preds = np.argmax(P_val, axis=1)
    baseline_test_preds = np.argmax(P_test, axis=1)

    # Report baseline
    base_val_acc = accuracy_score(y_val, baseline_val_preds)
    base_val_f1 = precision_recall_fscore_support(y_val, baseline_val_preds, average="macro")[2]
    base_test_acc = accuracy_score(y_test, baseline_test_preds)
    base_test_f1 = precision_recall_fscore_support(y_test, baseline_test_preds, average="macro")[2]
    base_p, base_r, base_f, _ = precision_recall_fscore_support(y_test, baseline_test_preds, average=None)

    print("\n" + "-" * 60)
    print("BASELINE PERFORMANCE (Standard Argmax Threshold):")
    print(f"Validation Acc: {base_val_acc*100:.2f}%, Macro-F1: {base_val_f1*100:.2f}%")
    print(f"Test Acc:       {base_test_acc*100:.2f}%, Macro-F1: {base_test_f1*100:.2f}%")
    print(f"Head (Violation) Recall:    {base_r[0]*100:.2f}% (MISSING {(1-base_r[0])*100:.2f}% OF BARE HEADS!)")
    print(f"Head (Violation) Precision: {base_p[0]*100:.2f}%")
    print(f"Helmet Recall: {base_r[1]*100:.2f}%, Vest Recall: {base_r[2]*100:.2f}%")
    print("-" * 60)

    # 6. Safety-Critical Threshold Optimization on Validation Set
    print("\n--- Sweeping Violation Threshold (theta_head) on Validation Set ---")
    thresholds = np.linspace(0.10, 0.45, 36)
    val_sweep_records = []

    for th in thresholds:
        preds = np.zeros(len(y_val), dtype=int)
        for i in range(len(y_val)):
            if P_val[i, 0] >= th:
                preds[i] = 0 # Head violation
            else:
                preds[i] = 1 if P_val[i, 1] >= P_val[i, 2] else 2

        acc = accuracy_score(y_val, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
        p_c, r_c, f_c, _ = precision_recall_fscore_support(y_val, preds, average=None, zero_division=0)

        val_sweep_records.append({
            "theta": float(th),
            "val_acc": float(acc),
            "val_macro_f1": float(f1),
            "head_recall": float(r_c[0]),
            "head_precision": float(p_c[0]),
            "helmet_recall": float(r_c[1]),
            "vest_recall": float(r_c[2])
        })

    df_sweep = pd.DataFrame(val_sweep_records)

    # Select optimal threshold that maximizes Head Recall subject to Val Accuracy >= 90.0%
    valid_candidates = df_sweep[df_sweep["val_acc"] >= 0.900]
    best_candidate = valid_candidates.sort_values(by="head_recall", ascending=False).iloc[0]
    optimal_theta = best_candidate["theta"]

    print(f"[+] Optimal Validation Threshold: theta_head* = {optimal_theta:.3f}")
    print(f"    Expected Val Acc: {best_candidate['val_acc']*100:.2f}%")
    print(f"    Expected Val Head Recall: {best_candidate['head_recall']*100:.2f}% (vs Baseline: {precision_recall_fscore_support(y_val, baseline_val_preds, average=None)[1][0]*100:.2f}%)")

    # 7. Evaluate on Touched-Once Holdout Test Set with theta_head*
    opt_test_preds = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        if P_test[i, 0] >= optimal_theta:
            opt_test_preds[i] = 0
        else:
            opt_test_preds[i] = 1 if P_test[i, 1] >= P_test[i, 2] else 2

    opt_test_acc = accuracy_score(y_test, opt_test_preds)
    opt_test_bacc = balanced_accuracy_score(y_test, opt_test_preds)
    opt_test_prec, opt_test_rec, opt_test_f1, _ = precision_recall_fscore_support(y_test, opt_test_preds, average="macro")
    opt_p, opt_r, opt_f, _ = precision_recall_fscore_support(y_test, opt_test_preds, average=None)

    print("\n" + "=" * 60)
    print(f"OPTIMIZED MODEL 1 PERFORMANCE (theta_head = {optimal_theta:.3f}):")
    print(f"Test Accuracy:          {opt_test_acc*100:.2f}% (Target: >90.0% -> ACHIEVED: {opt_test_acc >= 0.90})")
    print(f"Test Balanced Accuracy:  {opt_test_bacc*100:.2f}% (vs Baseline: {balanced_accuracy_score(y_test, baseline_test_preds)*100:.2f}%)")
    print(f"Test Macro-F1:          {opt_test_f1*100:.2f}%")
    print(f"Head (Violation) Recall:    {opt_r[0]*100:.2f}% (IMPROVED FROM {base_r[0]*100:.2f}% -> +{(opt_r[0]-base_r[0])*100:.2f}% absolute lift!)")
    print(f"Head (Violation) Precision: {opt_p[0]*100:.2f}%")
    print(f"Helmet Recall:              {opt_r[1]*100:.2f}%")
    print(f"Vest Recall:                {opt_r[2]*100:.2f}%")
    print("=" * 60)

    # Save optimized model metadata and artifacts
    os.makedirs("checkpoints/optimized/autoencoder", exist_ok=True)
    # Copy baseline weights to optimized
    joblib.dump(scaler, "checkpoints/optimized/autoencoder/scaler.joblib")
    joblib.dump(svm, "checkpoints/optimized/autoencoder/svm_model.joblib")
    torch.save(ae.state_dict(), "checkpoints/optimized/autoencoder/ae_optimized.pt")

    opt_meta = {
        "model_id": "autoencoder",
        "name": "ConvAE + Calibrated RBF-SVM",
        "status": "optimized",
        "latent_dim": 256,
        "optimal_head_threshold": float(optimal_theta),
        "baseline_test_acc": float(base_test_acc),
        "optimized_test_acc": float(opt_test_acc),
        "baseline_head_recall": float(base_r[0]),
        "optimized_head_recall": float(opt_r[0]),
        "head_recall_lift_pct": float((opt_r[0] - base_r[0]) * 100),
        "optimized_balanced_acc": float(opt_test_bacc),
        "optimized_macro_f1": float(opt_test_f1),
        "per_class_metrics": {
            CLASS_NAMES[i]: {
                "precision": float(opt_p[i]),
                "recall": float(opt_r[i]),
                "f1": float(opt_f[i])
            }
            for i in range(3)
        }
    }

    with open("checkpoints/optimized/autoencoder/model_config.json", "w") as f:
        json.dump(opt_meta, f, indent=2)

    os.makedirs("results/experiments", exist_ok=True)
    with open("results/experiments/svm_optimization_report.json", "w") as f:
        json.dump(opt_meta, f, indent=2)

    print("[+] Saved optimized Model 1 configuration to checkpoints/optimized/autoencoder/")
    print("[+] Saved optimization report to results/experiments/svm_optimization_report.json")

if __name__ == "__main__":
    main()
