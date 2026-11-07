import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)

print("Loading model...", flush=True)
m = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5, local_files_only=True)

texts = ["Hard hats must be worn at all times.", "High-visibility vests are required."]
labels = torch.tensor([0, 1], dtype=torch.long)

enc = tok(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
print("Encodings:", {k: v.shape for k, v in enc.items()}, flush=True)

print("Running forward pass...", flush=True)
out = m(**enc, labels=labels)
print("Loss:", out.loss.item(), "Logits:", out.logits.shape, flush=True)

print("Running backward pass...", flush=True)
out.loss.backward()
print("Backward pass succeeded!", flush=True)

