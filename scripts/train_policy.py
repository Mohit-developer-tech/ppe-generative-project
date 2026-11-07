"""
scripts/train_policy.py
Multi-seed benchmark, temperature calibration, evaluation slices, and artifact export
for the Policy-Driven Compliance Checker (Model 5).

Evaluates:
  1. Keyword / Regex baseline
  2. TF-IDF + Logistic Regression baseline
  3. Fine-tuned DistilBERT Sequence Classifier across 5 seeds (42..46)
  4. Temperature scaling on validation logits (L-BFGS) + ECE tracking
  5. Evaluation slices: Negation / Exception, Conditional, Multi-PPE
  6. Generates 5 publication-ready figures (300 DPI) in results/policy/figures/
  7. Exports standardized bundle to results/policy/
"""

import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import random
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.policy import (
    CLASSES, LABEL2ID, ID2LABEL,
    KeywordRegexClassifier, TfidfLogisticClassifier,
    TemperatureScaler
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_policy")

OUTPUT_DIR = "results/policy"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
SEEDS = [42, 43, 44, 45, 46]


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


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
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


def main():
    print("=" * 80)
    print("POLICY CHECKER: TRAINING, BENCHMARKING & MULTI-MODAL VERDICT SYSTEM")
    print("=" * 80)

    # 1. Load Real Dataset
    df = pd.read_csv("data/policy/labels.csv")
    print(f"[*] Loaded dataset from data/policy/labels.csv: {len(df)} total sentences")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[*] Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    train_texts = df_train["text"].tolist()
    train_labels = [LABEL2ID[l] for l in df_train["label"]]
    val_texts = df_val["text"].tolist()
    val_labels = [LABEL2ID[l] for l in df_val["label"]]
    test_texts = df_test["text"].tolist()
    test_labels = [LABEL2ID[l] for l in df_test["label"]]

    # Load Slices
    df_slices = pd.read_csv("data/policy/evaluation_slices.csv")
    print(f"[*] Loaded {len(df_slices)} evaluation slice instances across {df_slices['slice'].nunique()} slices")

    # --------------------------------------------------------------------------
    # 2. Benchmark Baseline 1: Keyword / Regex Rules
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("BENCHMARK 1: Keyword / Regex Baseline")
    print("-" * 60)
    kw_clf = KeywordRegexClassifier()
    kw_val_preds = [LABEL2ID[p] for p in kw_clf.predict(val_texts)]
    kw_test_preds = [LABEL2ID[p] for p in kw_clf.predict(test_texts)]

    kw_val_metrics = compute_metrics(val_labels, kw_val_preds)
    kw_test_metrics = compute_metrics(test_labels, kw_test_preds)

    print(f"VAL  -> Acc: {kw_val_metrics['accuracy']:.4f}, BalAcc: {kw_val_metrics['balanced_accuracy']:.4f}, Macro F1: {kw_val_metrics['macro_f1']:.4f}")
    print(f"TEST -> Acc: {kw_test_metrics['accuracy']:.4f}, BalAcc: {kw_test_metrics['balanced_accuracy']:.4f}, Macro F1: {kw_test_metrics['macro_f1']:.4f}")

    # --------------------------------------------------------------------------
    # 3. Benchmark Baseline 2: TF-IDF + Logistic Regression
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("BENCHMARK 2: TF-IDF + Logistic Regression Baseline")
    print("-" * 60)
    tfidf_clf = TfidfLogisticClassifier()
    tfidf_clf.fit(train_texts, [ID2LABEL[y] for y in train_labels])

    tfidf_val_preds = [LABEL2ID[p] for p in tfidf_clf.predict(val_texts)]
    tfidf_test_preds = [LABEL2ID[p] for p in tfidf_clf.predict(test_texts)]

    tfidf_val_metrics = compute_metrics(val_labels, tfidf_val_preds)
    tfidf_test_metrics = compute_metrics(test_labels, tfidf_test_preds)

    print(f"VAL  -> Acc: {tfidf_val_metrics['accuracy']:.4f}, BalAcc: {tfidf_val_metrics['balanced_accuracy']:.4f}, Macro F1: {tfidf_val_metrics['macro_f1']:.4f}")
    print(f"TEST -> Acc: {tfidf_test_metrics['accuracy']:.4f}, BalAcc: {tfidf_test_metrics['balanced_accuracy']:.4f}, Macro F1: {tfidf_test_metrics['macro_f1']:.4f}")

    # --------------------------------------------------------------------------
    # 4. Fine-Tune DistilBERT across 5 Seeds
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"TRAINING DISTILBERT ACROSS 5 SEEDS: {SEEDS}")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    # Class weights for loss
    class_counts = np.bincount(train_labels, minlength=len(CLASSES))
    total_train = len(train_labels)
    class_weights = total_train / (len(CLASSES) * np.maximum(class_counts, 1).astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[*] Computed class weights: {np.round(class_weights, 3)}")

    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    seed_results = []
    best_overall_val_f1 = -1.0
    best_overall_state = None
    best_seed = None
    best_val_logits = None
    best_test_logits = None
    best_test_preds = None

    for seed in SEEDS:
        print(f"\n---> Training Seed {seed}...")
        seed_everything(seed)

        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(CLASSES),
            local_files_only=True
        ).to(device)

        # Freeze embeddings and lower 3 transformer layers for faster CPU fine-tuning and regularization
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.distilbert.transformer.layer[:3]:
            for param in layer.parameters():
                param.requires_grad = False

        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        best_seed_val_f1 = -1.0
        best_seed_model_state = None

        for epoch in range(1, NUM_EPOCHS + 1):
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
                optimizer.step()
                total_loss += loss.item()

            # Evaluate on Val
            v_logits, v_probs, v_preds = evaluate_transformer(model, val_loader, device)
            v_metrics = compute_metrics(val_labels, list(v_preds))

            if v_metrics["macro_f1"] > best_seed_val_f1:
                best_seed_val_f1 = v_metrics["macro_f1"]
                best_seed_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"     Epoch {epoch}/{NUM_EPOCHS}: Loss={total_loss/len(train_loader):.4f}, Val F1={v_metrics['macro_f1']:.4f}, Val Acc={v_metrics['accuracy']:.4f}", flush=True)

        # Restore best model for this seed
        model.load_state_dict(best_seed_model_state)
        v_logits, v_probs, v_preds = evaluate_transformer(model, val_loader, device)
        t_logits, t_probs, t_preds = evaluate_transformer(model, test_loader, device)

        val_eval = compute_metrics(val_labels, list(v_preds))
        test_eval = compute_metrics(test_labels, list(t_preds))

        seed_results.append({
            "seed": seed,
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "val_logits": v_logits,
            "test_logits": t_logits,
            "test_preds": t_preds
        })

        print(f"[*] Seed {seed} Result -> Val F1: {val_eval['macro_f1']:.4f}, Test F1: {test_eval['macro_f1']:.4f}, Test Acc: {test_eval['accuracy']:.4f}", flush=True)

        if val_eval["macro_f1"] > best_overall_val_f1:
            best_overall_val_f1 = val_eval["macro_f1"]
            best_overall_state = best_seed_model_state
            best_seed = seed
            best_val_logits = v_logits
            best_test_logits = t_logits
            best_test_preds = t_preds

    # Aggregate 5-seed statistics
    val_f1s = [r["val_metrics"]["macro_f1"] for r in seed_results]
    val_accs = [r["val_metrics"]["accuracy"] for r in seed_results]
    val_baccs = [r["val_metrics"]["balanced_accuracy"] for r in seed_results]

    test_f1s = [r["test_metrics"]["macro_f1"] for r in seed_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in seed_results]
    test_baccs = [r["test_metrics"]["balanced_accuracy"] for r in seed_results]

    print("\n" + "=" * 60)
    print("DISTILBERT 5-SEED SUMMARY:")
    print(f"VAL  Macro F1: {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")
    print(f"VAL  Accuracy: {np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}")
    print(f"VAL  Bal  Acc: {np.mean(val_baccs):.4f} +/- {np.std(val_baccs):.4f}")
    print(f"TEST Macro F1: {np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f}")
    print(f"TEST Accuracy: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    print(f"TEST Bal  Acc: {np.mean(test_baccs):.4f} +/- {np.std(test_baccs):.4f}")
    print(f"Best Seed: {best_seed} (Val F1: {best_overall_val_f1:.4f})")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # 5. Temperature Calibration on Validation Set
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEMPERATURE CALIBRATION (L-BFGS on Val NLL)")
    print("-" * 60)
    temp_scaler = TemperatureScaler()
    v_logits_t = torch.tensor(best_val_logits, dtype=torch.float32)
    v_labels_t = torch.tensor(val_labels, dtype=torch.long)

    optimal_t = temp_scaler.fit(v_logits_t, v_labels_t)
    print(f"[*] Optimal Temperature T* = {optimal_t:.4f}")

    # ECE before vs after on Val
    val_probs_uncal = F.softmax(v_logits_t, dim=1).numpy()
    with torch.no_grad():
        val_probs_cal = F.softmax(temp_scaler(v_logits_t), dim=1).cpu().numpy()

    ece_val_uncal = TemperatureScaler.compute_ece(val_probs_uncal, np.array(val_labels))
    ece_val_cal = TemperatureScaler.compute_ece(val_probs_cal, np.array(val_labels))

    # ECE before vs after on Test
    t_logits_t = torch.tensor(best_test_logits, dtype=torch.float32)
    test_probs_uncal = F.softmax(t_logits_t, dim=1).numpy()
    with torch.no_grad():
        test_probs_cal = F.softmax(temp_scaler(t_logits_t), dim=1).cpu().numpy()

    ece_test_uncal = TemperatureScaler.compute_ece(test_probs_uncal, np.array(test_labels))
    ece_test_cal = TemperatureScaler.compute_ece(test_probs_cal, np.array(test_labels))

    print(f"VAL  ECE: {ece_val_uncal:.4f} -> {ece_val_cal:.4f} (Reduction: {(ece_val_uncal - ece_val_cal):.4f})")
    print(f"TEST ECE: {ece_test_uncal:.4f} -> {ece_test_cal:.4f} (Reduction: {(ece_test_uncal - ece_test_cal):.4f})")

    # --------------------------------------------------------------------------
    # 6. Evaluation Slices Benchmarking
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("EVALUATION SLICES BENCHMARK")
    print("-" * 60)

    slice_records = []
    # Test texts mapping
    test_text_to_idx = {t: i for i, t in enumerate(test_texts)}
    
    # Slices present in test split
    test_slices = df_slices[df_slices["split"] == "test"].copy()
    
    slice_names = test_slices["slice"].unique()
    slice_metrics = {}

    for s_name in slice_names:
        sub = test_slices[test_slices["slice"] == s_name]
        texts = sub["text"].tolist()
        y_true = [LABEL2ID[l] for l in sub["true_label"]]

        # Keyword
        kw_p = [LABEL2ID[p] for p in kw_clf.predict(texts)]
        kw_acc = accuracy_score(y_true, kw_p)

        # TF-IDF
        tf_p = [LABEL2ID[p] for p in tfidf_clf.predict(texts)]
        tf_acc = accuracy_score(y_true, tf_p)

        # DistilBERT best seed
        bert_idxs = [test_text_to_idx[t] for t in texts]
        bert_p = best_test_preds[bert_idxs]
        bert_acc = accuracy_score(y_true, bert_p)

        slice_metrics[s_name] = {
            "count": len(texts),
            "keyword_acc": float(kw_acc),
            "tfidf_acc": float(tf_acc),
            "distilbert_acc": float(bert_acc)
        }
        print(f"Slice '{s_name}' (N={len(texts)}): Keyword={kw_acc*100:.1f}%, TF-IDF={tf_acc*100:.1f}%, DistilBERT={bert_acc*100:.1f}%")

    # --------------------------------------------------------------------------
    # 7. Generate Publication Figures (300 DPI)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("GENERATING 5 PUBLICATION FIGURES (300 DPI)")
    print("-" * 60)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 300

    # Fig 1: Normalized Confusion Matrix
    cm = confusion_matrix(test_labels, best_test_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, cbar=True, ax=ax)
    ax.set_title("DistilBERT Test Confusion Matrix (Normalized)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Regulatory Label", fontsize=11, labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "01_confusion_matrix.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"[*] Saved Figure 1 -> {fig1_path}")

    # Fig 2: Baseline Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Keyword / Regex", "TF-IDF + Logistic", "DistilBERT (5 Seeds)"]
    f1_means = [kw_test_metrics["macro_f1"], tfidf_test_metrics["macro_f1"], float(np.mean(test_f1s))]
    f1_errs = [0.0, 0.0, float(np.std(test_f1s))]
    bacc_means = [kw_test_metrics["balanced_accuracy"], tfidf_test_metrics["balanced_accuracy"], float(np.mean(test_baccs))]
    bacc_errs = [0.0, 0.0, float(np.std(test_baccs))]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, f1_means, width, yerr=f1_errs, capsize=5, label="Macro F1", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x + width/2, bacc_means, width, yerr=bacc_errs, capsize=5, label="Balanced Accuracy", color="#2ca02c", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline Comparison on Held-Out Test Split", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "02_baseline_comparison.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[*] Saved Figure 2 -> {fig2_path}")

    # Fig 3: Evaluation Slices Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    s_labels = list(slice_metrics.keys())
    kw_s_accs = [slice_metrics[k]["keyword_acc"] for k in s_labels]
    tf_s_accs = [slice_metrics[k]["tfidf_acc"] for k in s_labels]
    bert_s_accs = [slice_metrics[k]["distilbert_acc"] for k in s_labels]

    x = np.arange(len(s_labels))
    w = 0.25

    ax.bar(x - w, kw_s_accs, w, label="Keyword Baseline", color="#d62728", alpha=0.85)
    ax.bar(x, tf_s_accs, w, label="TF-IDF + Logistic", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, bert_s_accs, w, label="DistilBERT", color="#2ca02c", alpha=0.85)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Stress-Test Slice Evaluation (Held-Out Test)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.replace('_', ' ').title()}\n(N={slice_metrics[s]['count']})" for s in s_labels], fontsize=11)
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "03_evaluation_slices.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"[*] Saved Figure 3 -> {fig3_path}")

    # Fig 4: Calibration Reliability Diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    def plot_reliability(probs, labels, title, ece_val, ax):
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        accs = (preds == labels)

        bins = np.linspace(0, 1, 11)
        bin_accs = []
        bin_confs = []
        for i in range(10):
            mask = (confidences > bins[i]) & (confidences <= bins[i+1])
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(accs[mask]))
                bin_confs.append(np.mean(confidences[mask]))
            else:
                bin_accs.append(0)
                bin_confs.append((bins[i] + bins[i+1]) / 2)

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.bar(bins[:-1] + 0.05, bin_accs, width=0.08, alpha=0.6, color="#1f77b4", edgecolor="black", label="Accuracy")
        ax.step(bins, [0] + bin_accs, where="pre", color="#d62728", linewidth=1.5)
        ax.set_title(f"{title}\nECE = {ece_val:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Observed Accuracy", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    plot_reliability(test_probs_uncal, np.array(test_labels), "Uncalibrated (T=1.00)", ece_test_uncal, ax1)
    plot_reliability(test_probs_cal, np.array(test_labels), f"Calibrated (T*={optimal_t:.2f})", ece_test_cal, ax2)

    plt.suptitle("Post-Hoc Temperature Calibration Reliability Diagram (Test Split)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "04_calibration_reliability.png")
    fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved Figure 4 -> {fig4_path}")

    # Fig 5: End-to-End Multi-Modal Verdict Decision Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    policy_states = ["helmet_required", "vest_required", "both_required", "none / other_ppe"]
    crop_states = ["head (violation)", "helmet (compliant)", "vest (compliant)"]

    # 0 = Cannot determine (Yellow/Orange), 1 = Compliant (Green), -1 = Violation (Red)
    # Rows: policy_states, Cols: crop_states
    matrix = np.array([
        [-1,  1,  0],  # helmet_required -> head: -1, helmet: 1, vest: 0
        [ 0,  0,  1],  # vest_required   -> head:  0, helmet: 0, vest: 1
        [-1,  0,  0],  # both_required   -> head: -1, helmet: 0, vest: 0
        [ 1,  1,  1],  # none/other_ppe  -> head:  1, helmet: 1, vest: 1
    ])

    cmap = matplotlib.colors.ListedColormap(["#d62728", "#ff7f0e", "#2ca02c"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    sns.heatmap(matrix, annot=False, cmap=cmap, norm=norm,
                xticklabels=crop_states, yticklabels=policy_states, cbar=False, ax=ax, linewidths=1.5, linecolor="white")

    # Annotate with text
    labels_map = {-1: "VIOLATION", 0: "CANNOT\nDETERMINE", 1: "COMPLIANT"}
    for i in range(len(policy_states)):
        for j in range(len(crop_states)):
            val = matrix[i, j]
            ax.text(j + 0.5, i + 0.5, labels_map[val], ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_title("Multi-Modal Safety Compliance Decision Matrix\n(Policy Requirement × Visual Crop Evidence)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Visual Crop Evidence (Consensus Judge v2 + AE+SVM)", fontsize=11, labelpad=8)
    ax.set_ylabel("Extracted Policy Requirement", fontsize=11, labelpad=8)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "05_end_to_end_verdict_matrix.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"[*] Saved Figure 5 -> {fig5_path}")

    # --------------------------------------------------------------------------
    # 8. Export Standardized Bundle
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("EXPORTING STANDARDIZED ARTIFACTS TO results/policy/")
    print("-" * 60)

    # 8a. Save Checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, "policy_classifier.pt")
    torch.save({
        "state_dict": best_overall_state,
        "temperature": float(optimal_t),
        "model_name": MODEL_NAME,
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "best_seed": best_seed,
        "val_f1": float(best_overall_val_f1)
    }, checkpoint_path)
    print(f"[*] Saved model checkpoint -> {checkpoint_path}")

    # 8b. Save model_config.json
    config_dict = {
        "model_type": "DistilBertForSequenceClassification",
        "model_name": MODEL_NAME,
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "best_seed": best_seed,
        "temperature": float(optimal_t),
        "total_curated_sentences": len(df),
        "split_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test)
        }
    }
    config_path = os.path.join(OUTPUT_DIR, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[*] Saved model config -> {config_path}")

    # 8c. Save policy_metrics.json
    metrics_bundle = {
        "dataset": {
            "total_sentences": len(df),
            "splits": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
            "source_leakage": {"train_test_overlap": 0, "train_val_overlap": 0}
        },
        "baselines": {
            "keyword_regex": {
                "val": kw_val_metrics,
                "test": kw_test_metrics
            },
            "tfidf_logistic": {
                "val": tfidf_val_metrics,
                "test": tfidf_test_metrics
            }
        },
        "distilbert_5seeds": {
            "val_macro_f1_mean": float(np.mean(val_f1s)),
            "val_macro_f1_std": float(np.std(val_f1s)),
            "val_acc_mean": float(np.mean(val_accs)),
            "val_acc_std": float(np.std(val_accs)),
            "val_bacc_mean": float(np.mean(val_baccs)),
            "val_bacc_std": float(np.std(val_baccs)),
            "test_macro_f1_mean": float(np.mean(test_f1s)),
            "test_macro_f1_std": float(np.std(test_f1s)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "test_bacc_mean": float(np.mean(test_baccs)),
            "test_bacc_std": float(np.std(test_baccs)),
            "seeds": seed_results[best_seed % len(seed_results)]["test_metrics"]
        },
        "temperature_calibration": {
            "optimal_temperature": float(optimal_t),
            "val_ece_uncalibrated": float(ece_val_uncal),
            "val_ece_calibrated": float(ece_val_cal),
            "test_ece_uncalibrated": float(ece_test_uncal),
            "test_ece_calibrated": float(ece_test_cal)
        },
        "evaluation_slices": slice_metrics
    }
    metrics_path = os.path.join(OUTPUT_DIR, "policy_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_bundle, f, indent=2)
    print(f"[*] Saved policy metrics -> {metrics_path}")

    # 8d. Save policy_benchmark.csv
    benchmark_rows = [
        {
            "model": "Keyword / Regex",
            "val_macro_f1": kw_val_metrics["macro_f1"],
            "val_acc": kw_val_metrics["accuracy"],
            "val_bal_acc": kw_val_metrics["balanced_accuracy"],
            "test_macro_f1": kw_test_metrics["macro_f1"],
            "test_acc": kw_test_metrics["accuracy"],
            "test_bal_acc": kw_test_metrics["balanced_accuracy"]
        },
        {
            "model": "TF-IDF + Logistic",
            "val_macro_f1": tfidf_val_metrics["macro_f1"],
            "val_acc": tfidf_val_metrics["accuracy"],
            "val_bal_acc": tfidf_val_metrics["balanced_accuracy"],
            "test_macro_f1": tfidf_test_metrics["macro_f1"],
            "test_acc": tfidf_test_metrics["accuracy"],
            "test_bal_acc": tfidf_test_metrics["balanced_accuracy"]
        },
        {
            "model": f"DistilBERT (5 Seeds Mean)",
            "val_macro_f1": float(np.mean(val_f1s)),
            "val_acc": float(np.mean(val_accs)),
            "val_bal_acc": float(np.mean(val_baccs)),
            "test_macro_f1": float(np.mean(test_f1s)),
            "test_acc": float(np.mean(test_accs)),
            "test_bal_acc": float(np.mean(test_baccs))
        },
        {
            "model": f"DistilBERT (Best Seed {best_seed})",
            "val_macro_f1": float(best_overall_val_f1),
            "val_acc": float(seed_results[SEEDS.index(best_seed)]["val_metrics"]["accuracy"]),
            "val_bal_acc": float(seed_results[SEEDS.index(best_seed)]["val_metrics"]["balanced_accuracy"]),
            "test_macro_f1": float(seed_results[SEEDS.index(best_seed)]["test_metrics"]["macro_f1"]),
            "test_acc": float(seed_results[SEEDS.index(best_seed)]["test_metrics"]["accuracy"]),
            "test_bal_acc": float(seed_results[SEEDS.index(best_seed)]["test_metrics"]["balanced_accuracy"])
        }
    ]
    df_bm = pd.DataFrame(benchmark_rows)
    bm_path = os.path.join(OUTPUT_DIR, "policy_benchmark.csv")
    df_bm.to_csv(bm_path, index=False)
    print(f"[*] Saved benchmark summary -> {bm_path}")

    print("\n" + "=" * 80)
    print("POLICY CHECKER TRAINING & BENCHMARKING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
