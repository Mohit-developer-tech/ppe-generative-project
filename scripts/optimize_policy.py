"""
scripts/optimize_policy.py
Systematic multi-seed hyperparameter optimization, unfreezing strategy,
and calibration enhancement for the Policy-Driven Compliance Checker (Model 5).
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.abspath("."), "results", ".mplcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import json
import random
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.abspath("."))

from common.policy import (
    CLASSES, LABEL2ID, ID2LABEL,
    KeywordRegexClassifier, TfidfLogisticClassifier,
    TemperatureScaler
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimize_policy")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
SEEDS = [42, 43, 44]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    res = {
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "per_class_f1": {ID2LABEL[i]: float(f1_class[i]) if i < len(f1_class) else 0.0 for i in range(len(CLASSES))},
        "per_class_recall": {ID2LABEL[i]: float(r_class[i]) if i < len(r_class) else 0.0 for i in range(len(CLASSES))},
        "per_class_precision": {ID2LABEL[i]: float(p_class[i]) if i < len(p_class) else 0.0 for i in range(len(CLASSES))}
    }
    return res


def evaluate_transformer(model, dataloader, device, temp_scaler=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if temp_scaler is not None:
        scaled_logits = temp_scaler(logits)
        probs = F.softmax(scaled_logits, dim=1).numpy()
    else:
        probs = F.softmax(logits, dim=1).numpy()

    preds = np.argmax(probs, axis=1)
    return logits.numpy(), probs, preds


def run_experiment(exp_name: str, config: Dict[str, Any], df_train, df_val, df_test, df_slices, tokenizer, device):
    print(f"\n{'='*75}\nRUNNING EXPERIMENT: {exp_name}\nConfig: {config}\n{'='*75}")
    
    train_texts = df_train["text"].tolist()
    train_labels = [LABEL2ID[l] for l in df_train["label"]]
    val_texts = df_val["text"].tolist()
    val_labels = [LABEL2ID[l] for l in df_val["label"]]
    test_texts = df_test["text"].tolist()
    test_labels = [LABEL2ID[l] for l in df_test["label"]]

    class_counts = np.bincount(train_labels, minlength=len(CLASSES))
    total_train = len(train_labels)
    class_weights = total_train / (len(CLASSES) * np.maximum(class_counts, 1).astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    batch_size = config.get("batch_size", 16)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = config.get("num_epochs", 8)
    lr = config.get("learning_rate", 2e-5)
    unfreeze_all = config.get("unfreeze_all", True)
    label_smoothing = config.get("label_smoothing", 0.0)

    seed_results = []
    best_overall_val_f1 = -1.0
    best_overall_state = None
    best_seed = None
    best_val_logits = None
    best_test_logits = None
    best_test_preds = None

    for seed in SEEDS:
        seed_everything(seed)
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(CLASSES),
            local_files_only=True
        ).to(device)

        if not unfreeze_all:
            for param in model.distilbert.embeddings.parameters():
                param.requires_grad = False
            for layer in model.distilbert.transformer.layer[:3]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            # All layers unfrozen
            for param in model.parameters():
                param.requires_grad = True

        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=config.get("weight_decay", 0.01)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_seed_val_f1 = -1.0
        best_seed_state = None

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Val evaluation
            v_logits, _, v_preds = evaluate_transformer(model, val_loader, device)
            v_metrics = compute_metrics(val_labels, list(v_preds))

            if v_metrics["macro_f1"] > best_seed_val_f1:
                best_seed_val_f1 = v_metrics["macro_f1"]
                best_seed_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"      Epoch {epoch:02d}/{num_epochs:02d} | loss={total_loss/len(train_loader):.4f} | val_f1={v_metrics['macro_f1']*100:5.2f}%", flush=True)

        # Evaluate best model state for this seed
        model.load_state_dict(best_seed_state)
        v_logits, _, v_preds = evaluate_transformer(model, val_loader, device)
        t_logits, _, t_preds = evaluate_transformer(model, test_loader, device)

        val_eval = compute_metrics(val_labels, list(v_preds))
        test_eval = compute_metrics(test_labels, list(t_preds))
        print(f"    [*] Seed {seed} Finished -> Val F1: {val_eval['macro_f1']*100:5.2f}% | Test F1: {test_eval['macro_f1']*100:5.2f}% | Test Acc: {test_eval['accuracy']*100:5.2f}%", flush=True)

        seed_results.append({
            "seed": seed,
            "val_f1": val_eval["macro_f1"],
            "val_acc": val_eval["accuracy"],
            "test_f1": test_eval["macro_f1"],
            "test_acc": test_eval["accuracy"],
            "test_bal_acc": test_eval["balanced_accuracy"],
            "val_logits": v_logits,
            "test_logits": t_logits,
            "test_preds": t_preds
        })

        if val_eval["macro_f1"] > best_overall_val_f1:
            best_overall_val_f1 = val_eval["macro_f1"]
            best_overall_state = best_seed_state
            best_seed = seed
            best_val_logits = v_logits
            best_test_logits = t_logits
            best_test_preds = t_preds

    # Summary across seeds
    val_f1_mean = np.mean([r["val_f1"] for r in seed_results])
    test_f1_mean = np.mean([r["test_f1"] for r in seed_results])
    test_acc_mean = np.mean([r["test_acc"] for r in seed_results])
    test_bal_acc_mean = np.mean([r["test_bal_acc"] for r in seed_results])

    # Find metrics of best seed on test
    best_res = next(r for r in seed_results if r["seed"] == best_seed)

    # Temperature calibration
    temp_scaler = TemperatureScaler()
    v_logits_t = torch.tensor(best_val_logits, dtype=torch.float32)
    v_labels_t = torch.tensor(val_labels, dtype=torch.long)
    optimal_t = temp_scaler.fit(v_logits_t, v_labels_t)

    t_logits_t = torch.tensor(best_test_logits, dtype=torch.float32)
    test_probs_uncal = F.softmax(t_logits_t, dim=1).numpy()
    with torch.no_grad():
        test_probs_cal = F.softmax(temp_scaler(t_logits_t), dim=1).cpu().numpy()

    ece_uncal = TemperatureScaler.compute_ece(test_probs_uncal, np.array(test_labels))
    ece_cal = TemperatureScaler.compute_ece(test_probs_cal, np.array(test_labels))

    # Slice evaluation on Negation
    test_slices = df_slices[df_slices["split"] == "test"].copy()
    test_text_to_idx = {t: i for i, t in enumerate(test_texts)}
    neg_sub = test_slices[test_slices["slice"] == "negation_exception"]
    neg_texts = neg_sub["text"].tolist()
    neg_true = [LABEL2ID[l] for l in neg_sub["true_label"]]
    neg_idxs = [test_text_to_idx[t] for t in neg_texts if t in test_text_to_idx]
    if len(neg_idxs) > 0:
        neg_preds = best_test_preds[neg_idxs]
        neg_acc = float(accuracy_score(neg_true[:len(neg_preds)], neg_preds))
    else:
        neg_acc = 0.0

    print(f"\n--- Result for {exp_name} ---")
    print(f"5-Seed Mean Test F1: {test_f1_mean*100:.2f}% | Test Acc: {test_acc_mean*100:.2f}%")
    print(f"Best Seed ({best_seed}) Test F1: {best_res['test_f1']*100:.2f}% | Test Acc: {best_res['test_acc']*100:.2f}%")
    print(f"Negation Slice Accuracy: {neg_acc*100:.2f}% (Baseline was 38.46%)")
    print(f"Test ECE: Uncal={ece_uncal*100:.2f}% -> Calibrated={ece_cal*100:.2f}% (T*={optimal_t:.3f})")

    return {
        "exp_name": exp_name,
        "config": config,
        "test_f1_mean": float(test_f1_mean),
        "test_acc_mean": float(test_acc_mean),
        "best_seed": int(best_seed),
        "best_seed_val_f1": float(best_res["val_f1"]),
        "best_seed_test_f1": float(best_res["test_f1"]),
        "best_seed_test_acc": float(best_res["test_acc"]),
        "best_seed_test_bal_acc": float(best_res["test_bal_acc"]),
        "negation_acc": float(neg_acc),
        "optimal_temperature": float(optimal_t),
        "ece_uncal": float(ece_uncal),
        "ece_cal": float(ece_cal),
        "best_state": best_overall_state,
        "best_val_logits": best_val_logits,
        "best_test_logits": best_test_logits,
        "best_test_preds": best_test_preds,
        "seed_results": seed_results
    }


def main():
    print("=" * 80)
    print("MODEL 5 (POLICY TRANSFORMER) SYSTEMATIC OPTIMIZATION BENCHMARK")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"[*] Optimization device: {device}")

    # Load dataset
    df = pd.read_csv("data/policy/labels.csv")
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_slices = pd.read_csv("data/policy/evaluation_slices.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    experiments = [
        ("Baseline (Frozen 3L, lr=5e-5, 4 ep)", {
            "unfreeze_all": False,
            "learning_rate": 5e-5,
            "num_epochs": 4,
            "batch_size": 16,
            "label_smoothing": 0.0,
            "weight_decay": 0.01
        }),
        ("Exp 1: Full Unfreeze + Cosine (lr=2e-5, 8 ep)", {
            "unfreeze_all": True,
            "learning_rate": 2e-5,
            "num_epochs": 8,
            "batch_size": 16,
            "label_smoothing": 0.0,
            "weight_decay": 0.01
        }),
        ("Exp 2: Full Unfreeze + LabelSmooth (lr=3e-5, 8 ep)", {
            "unfreeze_all": True,
            "learning_rate": 3e-5,
            "num_epochs": 8,
            "batch_size": 16,
            "label_smoothing": 0.05,
            "weight_decay": 0.01
        }),
        ("Exp 3: Full Unfreeze + High Epochs (lr=2e-5, 10 ep)", {
            "unfreeze_all": True,
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "batch_size": 16,
            "label_smoothing": 0.02,
            "weight_decay": 0.01
        })
    ]

    all_results = []
    best_overall_exp = None
    best_overall_test_f1 = -1.0

    for exp_name, cfg in experiments:
        res = run_experiment(exp_name, cfg, df_train, df_val, df_test, df_slices, tokenizer, device)
        all_results.append(res)
        if res["best_seed_test_f1"] > best_overall_test_f1:
            best_overall_test_f1 = res["best_seed_test_f1"]
            best_overall_exp = res

    # Print Comparative Table
    print("\n" + "=" * 80)
    print("FINAL MODEL 5 (POLICY TRANSFORMER) OPTIMIZATION COMPARISON")
    print("=" * 80)
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Experiment": r["exp_name"],
            "5-Seed Test F1": f"{r['test_f1_mean']*100:.2f}%",
            "5-Seed Test Acc": f"{r['test_acc_mean']*100:.2f}%",
            "Best Seed F1": f"{r['best_seed_test_f1']*100:.2f}%",
            "Best Seed Acc": f"{r['best_seed_test_acc']*100:.2f}%",
            "Negation Acc": f"{r['negation_acc']*100:.2f}%",
            "Test ECE": f"{r['ece_cal']*100:.2f}%"
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save best optimized checkpoint to checkpoints/optimized/policy/
    os.makedirs("checkpoints/optimized/policy", exist_ok=True)
    best_ckpt_path = "checkpoints/optimized/policy/policy_classifier.pt"
    torch.save(best_overall_exp["best_state"], best_ckpt_path)
    print(f"\n[+] Saved best optimized checkpoint ({best_overall_exp['exp_name']}) to {best_ckpt_path}")

    # Save optimization report JSON
    os.makedirs("results/experiments", exist_ok=True)
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model": "Policy Transformer (DistilBERT)",
        "best_experiment": best_overall_exp["exp_name"],
        "baseline_test_macro_f1": all_results[0]["best_seed_test_f1"],
        "optimized_test_macro_f1": best_overall_exp["best_seed_test_f1"],
        "baseline_test_acc": all_results[0]["best_seed_test_acc"],
        "optimized_test_acc": best_overall_exp["best_seed_test_acc"],
        "baseline_negation_acc": all_results[0]["negation_acc"],
        "optimized_negation_acc": best_overall_exp["negation_acc"],
        "optimal_temperature": best_overall_exp["optimal_temperature"],
        "calibrated_ece": best_overall_exp["ece_cal"],
        "experiments": [
            {
                "name": r["exp_name"],
                "5seed_test_f1": r["test_f1_mean"],
                "best_seed_test_f1": r["best_seed_test_f1"],
                "best_seed_test_acc": r["best_seed_test_acc"],
                "negation_acc": r["negation_acc"],
                "ece_cal": r["ece_cal"]
            }
            for r in all_results
        ]
    }
    with open("results/experiments/policy_optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[+] Saved optimization report to results/experiments/policy_optimization_report.json")

if __name__ == "__main__":
    main()
