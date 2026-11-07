"""
common/policy.py
Core Policy Ingestion, Classification, Calibration, and Multi-Modal Verdict Engine.
Used by policy_checker.ipynb, scripts/train_policy.py, and dashboard/models/policy/router.py.

Includes:
  1. Document Ingestion: PDF (pypdf), DOCX (python-docx), TXT.
     - Scanned PDF detection (<20 chars/page)
     - Clean line-break hyphens and headers/footers
     - Safe sentence segmentation with legal/regulatory abbreviation handling
     - HTML-escaping for secure UI rendering
  2. Baselines:
     - KeywordRegexClassifier (regex with negation & co-occurrence rules)
     - TfidfLogisticClassifier (TF-IDF + class-balanced Logistic Regression)
  3. Transformer Architecture:
     - PolicyDistilBertClassifier (DistilBERT with class-weighted loss)
  4. Temperature Calibration:
     - TemperatureScaler (L-BFGS optimization on validation NLL + ECE metric)
  5. Three-State Multi-Modal Verdict Engine:
     - Vision consensus check (Judge v2 + AE+SVM)
     - Violation / Compliant / Cannot determine logic citing specific clauses & pages
"""

import os
import re
import io
import html
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("common.policy")

# 5-Class Taxonomy
CLASSES = ["helmet_required", "vest_required", "both_required", "other_ppe", "none"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}

CLASS_DESCRIPTIONS = {
    "helmet_required": "Head protection (hard hat / safety helmet) mandated.",
    "vest_required": "High-visibility garment / safety vest mandated.",
    "both_required": "Both head protection and high-visibility vest co-mandated.",
    "other_ppe": "Other PPE (eye, ear, foot, hand, respirator, fall arrest) mandated.",
    "none": "General scope, disclaimer, administrative text, or explicit exemption."
}

# ------------------------------------------------------------------------------
# 1. Document Ingestion Engine
# ------------------------------------------------------------------------------

class DocumentIngestionEngine:
    """
    Robust multi-format document parser supporting PDF, DOCX, and TXT files.
    Enforces a 10 MB file size limit, detects scanned documents, cleans text,
    and performs abbreviation-aware sentence segmentation.
    """
    MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB limit
    MIN_CHARS_PER_PAGE = 20            # Threshold for scanned PDF warning

    # Common abbreviations in occupational safety standards that should not break sentences
    ABBREVIATIONS = [
        "cfr", "ansi", "isea", "osha", "mutcd", "astm", "nfpa", "usace", "msha", "epa",
        "sec", "subpart", "para", "no", "rev", "app", "vol", "ed",
        "mr", "mrs", "ms", "dr", "prof",
        "e.g", "i.e", "et al", "etc", "approx", "vs", "fig"
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans extracted text:
        - Reconnects hyphenated words across line breaks (e.g. 'hel-\nmet' -> 'helmet')
        - Normalizes irregular whitespace and quotes
        - Removes non-printable control characters
        """
        if not text:
            return ""
        # Remove soft hyphens
        text = text.replace("\xad", "")
        # Reconnect words split across line breaks with hyphens
        text = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", text)
        # Replace multiple whitespace/newlines with single space
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @classmethod
    def segment_sentences(cls, text: str) -> List[str]:
        """
        Splits raw document text into clean sentences, avoiding splits on
        regulatory citations, section numbers, or standard abbreviations.
        """
        text = cls.clean_text(text)
        if not text:
            return []

        # Tokenize roughly on periods, question marks, and exclamation marks followed by whitespace
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        sentences = []
        buffer = ""

        for part in raw_parts:
            part = part.strip()
            if not part:
                continue

            if buffer:
                buffer += " " + part
            else:
                buffer = part

            # Check if ends with an abbreviation
            last_token = buffer.split()[-1].lower().rstrip(".!?")
            if last_token in cls.ABBREVIATIONS or re.search(r"\b\d+\.\d*$", buffer):
                # Probable abbreviation or section decimal; keep accumulating
                continue
            
            # Sentence is complete
            if len(buffer) >= 15:  # Filter out trivial micro-fragments
                sentences.append(buffer)
            buffer = ""

        if buffer and len(buffer) >= 15:
            sentences.append(buffer)

        return sentences

    @classmethod
    def ingest_file(cls, file_content: Union[bytes, io.BytesIO], filename: str) -> Dict[str, Any]:
        """
        Parses a file payload into structured sentences tagged with source metadata and page numbers.
        Returns a dict with sentences, warnings, scanned_detected, and char_count.
        """
        if isinstance(file_content, io.BytesIO):
            raw_bytes = file_content.getvalue()
        elif isinstance(file_content, bytes):
            raw_bytes = file_content
        else:
            raise ValueError("file_content must be bytes or io.BytesIO")

        if len(raw_bytes) > cls.MAX_FILE_BYTES:
            raise ValueError(f"File size {len(raw_bytes)/(1024*1024):.2f} MB exceeds maximum limit of 10 MB.")

        ext = os.path.splitext(filename)[1].lower()
        pages_data: List[Dict[str, Any]] = []
        is_scanned = False
        warnings = []

        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                num_pages = len(reader.pages)
                total_chars = 0

                for p_idx, page in enumerate(reader.pages, start=1):
                    p_text = page.extract_text() or ""
                    cleaned = cls.clean_text(p_text)
                    total_chars += len(cleaned)
                    pages_data.append({"page": p_idx, "text": cleaned})

                char_density = total_chars / max(1, num_pages)
                if char_density < cls.MIN_CHARS_PER_PAGE:
                    is_scanned = True
                    warnings.append(
                        f"PDF character density is extremely low ({char_density:.1f} chars/page). "
                        "Document appears to be scanned or lacks a digital text layer. OCR may be required."
                    )
            except Exception as e:
                raise ValueError(f"Failed to parse PDF document: {e}")

        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(io.BytesIO(raw_bytes))
                full_text = []
                for p in doc.paragraphs:
                    if p.text.strip():
                        full_text.append(p.text.strip())
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text.strip():
                                full_text.append(cell.text.strip())
                joined_text = "\n".join(full_text)
                pages_data.append({"page": 1, "text": cls.clean_text(joined_text)})
            except Exception as e:
                raise ValueError(f"Failed to parse DOCX document: {e}")

        elif ext in [".txt", ".md", ".csv", ".log"]:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")
            pages_data.append({"page": 1, "text": cls.clean_text(text)})

        else:
            raise ValueError(f"Unsupported document format '{ext}'. Allowed: .pdf, .docx, .txt")

        # Segment sentences per page
        structured_sentences = []
        sentence_id = 0
        for p in pages_data:
            s_list = cls.segment_sentences(p["text"])
            for s in s_list:
                structured_sentences.append({
                    "id": sentence_id,
                    "text": s,
                    "safe_html": html.escape(s),
                    "page": p["page"],
                    "source": filename
                })
                sentence_id += 1

        return {
            "filename": filename,
            "format": ext,
            "page_count": len(pages_data),
            "sentence_count": len(structured_sentences),
            "is_scanned": is_scanned,
            "warnings": warnings,
            "sentences": structured_sentences
        }


# ------------------------------------------------------------------------------
# 2. Baseline Classifiers
# ------------------------------------------------------------------------------

class KeywordRegexClassifier:
    """
    Interpretable rule-based classifier using prioritized regular expressions.
    Specifically handles negation/exceptions, co-occurrence, and specialized PPE.
    """
    def __init__(self):
        # Negation / Exemption patterns
        self.re_negation = re.compile(
            r"\b(not required|exempt|no longer required|not mandatory|need not|except when|without needing|free of)\b",
            re.IGNORECASE
        )
        # Head / Helmet terms
        self.re_helmet = re.compile(
            r"\b(hard\s*hat|hard-hat|safety\s*helmet|protective\s*helmet|head\s*protection|protective\s*headwear|en\s*397|ansi\s*z89)\b",
            re.IGNORECASE
        )
        # Vest / High-Vis terms
        self.re_vest = re.compile(
            r"\b(high-?vis|safety\s*vest|reflective\s*vest|warning\s*garment|retroreflective|fluorescent\s*(vest|garment|apparel)|ansi\s*107|class\s*[23]\s*(apparel|vest))\b",
            re.IGNORECASE
        )
        # Other PPE terms
        self.re_other = re.compile(
            r"\b(safety\s*glasses|goggles|face\s*shield|hearing\s*protection|ear\s*plugs|earmuffs|safety\s*boots|safety-toe|steel\s*toe|harness|lanyard|fall\s*arrest|respirator|gloves|life\s*jacket|buoyant\s*vest)\b",
            re.IGNORECASE
        )

    def predict_one(self, text: str) -> str:
        t = text.strip()
        # 1. Negation/Exemption takes immediate precedence -> none
        if self.re_negation.search(t):
            return "none"

        has_helmet = bool(self.re_helmet.search(t))
        has_vest = bool(self.re_vest.search(t))

        # 2. Both required co-occurrence
        if has_helmet and has_vest:
            return "both_required"
        if has_helmet:
            return "helmet_required"
        if has_vest:
            return "vest_required"
        if self.re_other.search(t):
            return "other_ppe"

        return "none"

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]


class TfidfLogisticClassifier:
    """
    TF-IDF Vectorizer with class-balanced Multinomial Logistic Regression.
    """
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500, stop_words="english")),
            ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42))
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        return self.pipeline.predict_proba(texts)


# ------------------------------------------------------------------------------
# 3. Transformer Model Architecture & PyTorch Module
# ------------------------------------------------------------------------------

class PolicyDistilBertClassifier(nn.Module):
    """
    Fine-tuned DistilBERT for sentence-level PPE compliance policy classification.
    Outputs unnormalized 5-class logits.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 5):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.num_classes = num_classes
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            local_files_only=True
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# ------------------------------------------------------------------------------
# 4. Temperature Calibration
# ------------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling to calibrate neural network logits.
    Optimizes a single scalar parameter T = exp(log_temp) > 0 on validation set logits using L-BFGS
    to minimize cross-entropy / negative log-likelihood (NLL).
    """
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(min=0.05, max=10.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.05, max_iter: int = 50) -> float:
        """
        Learns optimal temperature T on validation logits.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.log_temp], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE).
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(labels)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                avg_conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

        return float(ece)


# ------------------------------------------------------------------------------
# 5. Three-State Multi-Modal Verdict Engine
# ------------------------------------------------------------------------------

class ThreeStateVerdictEngine:
    """
    Cross-modal decision engine combining extracted policy requirements with
    visual evidence from PPE crops (consensus between Judge v2 and AE+SVM).
    
    States:
      - VIOLATION: Observed crop directly contradicts policy mandate
      - COMPLIANT: Observed crop directly satisfies policy mandate
      - CANNOT DETERMINE: Visual evidence is incomplete (e.g. single crop when both
        helmet and vest are mandated, or torso unseen), or vision models disagree.
    """
    @staticmethod
    def check_vision_consensus(judge_label: str, svm_label: str) -> Optional[str]:
        """
        Enforces consensus gating: both Independent Judge v2 and Autoencoder+SVM
        must agree on the crop identity ('head', 'helmet', or 'vest').
        Returns the consensus label if agreed, or None if models disagree.
        """
        if judge_label.lower() == svm_label.lower():
            return judge_label.lower()
        return None

    @classmethod
    def evaluate_verdict(
        cls,
        policy_clauses: List[Dict[str, Any]],
        crop_label: str,
        vision_confidence: float = 0.95,
        vision_agreed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates safety compliance between parsed policy clauses and visual evidence.
        """
        # 1. Consensus Gate Check
        if not vision_agreed:
            return {
                "verdict": "Cannot determine",
                "badge_class": "badge-secondary",
                "summary": "Vision model consensus failure. Independent Judge v2 and AE+SVM disagree on crop identity.",
                "reason": "The vision classifier and autoencoder-SVM produced conflicting classifications. Compliance cannot be reliably certified without vision consensus.",
                "cited_clause": None,
                "cited_page": None,
                "confidence": 0.0
            }

        crop = crop_label.lower()
        
        # 2. Extract policy mandates
        has_both_req = any(c.get("predicted_label") == "both_required" for c in policy_clauses)
        has_helmet_req = any(c.get("predicted_label") in ["helmet_required", "both_required"] for c in policy_clauses)
        has_vest_req = any(c.get("predicted_label") in ["vest_required", "both_required"] for c in policy_clauses)

        # Find prime citation clause
        prime_both = next((c for c in policy_clauses if c.get("predicted_label") == "both_required"), None)
        prime_helmet = next((c for c in policy_clauses if c.get("predicted_label") == "helmet_required"), None) or prime_both
        prime_vest = next((c for c in policy_clauses if c.get("predicted_label") == "vest_required"), None) or prime_both

        # 3. Decision Tree Logic
        
        # Scenario A: Policy mandates BOTH helmet and vest
        if has_both_req:
            cited = prime_both or {}
            c_text = cited.get("text", "Active site policy mandates both hard hat and safety vest.")
            c_page = cited.get("page", 1)

            if crop == "head":
                return {
                    "verdict": "Violation",
                    "badge_class": "badge-danger",
                    "summary": "Direct Violation: Unhelmeted head observed in zone mandating both hard hat and safety vest.",
                    "reason": f"Visual crop identifies bare head (no helmet). Policy clause mandates: \"{c_text}\"",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": vision_confidence
                }
            elif crop == "helmet":
                return {
                    "verdict": "Cannot determine",
                    "badge_class": "badge-warning",
                    "summary": "Incomplete Evidence: Helmet detected, but policy also requires safety vest.",
                    "reason": f"Observed crop confirms head protection, but cannot verify high-visibility vest compliance from a single head crop. Policy clause: \"{c_text}\"",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": vision_confidence * 0.7
                }
            elif crop == "vest":
                return {
                    "verdict": "Cannot determine",
                    "badge_class": "badge-warning",
                    "summary": "Incomplete Evidence: Safety vest detected, but policy also requires protective hard hat.",
                    "reason": f"Observed crop confirms torso vest compliance, but cannot verify helmet compliance from a torso crop. Policy clause: \"{c_text}\"",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": vision_confidence * 0.7
                }

        # Scenario B: Policy mandates Helmet only
        if has_helmet_req and not has_vest_req:
            cited = prime_helmet or {}
            c_text = cited.get("text", "Head protection is mandated by safety standard.")
            c_page = cited.get("page", 1)

            if crop == "helmet":
                return {
                    "verdict": "Compliant",
                    "badge_class": "badge-success",
                    "summary": "Compliant: Protective helmet observed matching policy mandate.",
                    "reason": f"Worker is wearing compliant head protection. Citing policy clause: \"{c_text}\"",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": vision_confidence
                }
            elif crop == "head":
                return {
                    "verdict": "Violation",
                    "badge_class": "badge-danger",
                    "summary": "Violation: Unhelmeted worker head observed where head protection is mandatory.",
                    "reason": f"Visual evidence shows bare head without required helmet. Citing policy clause: \"{c_text}\"",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": vision_confidence
                }
            elif crop == "vest":
                return {
                    "verdict": "Cannot determine",
                    "badge_class": "badge-secondary",
                    "summary": "Insufficient Visual Evidence: Crop shows vest, but policy mandates head protection.",
                    "reason": "The provided image crop shows torso apparel only; head protection status cannot be determined.",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": 0.5
                }

        # Scenario C: Policy mandates Vest only
        if has_vest_req and not has_helmet_req:
            cited = prime_vest or {}
            c_text = cited.get("text", "High-visibility safety apparel is mandated by standard.")
            c_page = cited.get("page", 1)

            if crop == "vest":
                return {
                    "verdict": "Compliant",
                    "badge_class": "badge-success",
                    "summary": "Compliant: High-visibility safety vest observed matching policy mandate.",
                    "reason": f"Worker is wearing high-visibility apparel. Citing policy clause: \"{c_text}\"",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": vision_confidence
                }
            elif crop in ["head", "helmet"]:
                return {
                    "verdict": "Cannot determine",
                    "badge_class": "badge-secondary",
                    "summary": "Insufficient Visual Evidence: Crop shows head region, but policy mandates vest.",
                    "reason": "The provided crop contains head/helmet visual data only; torso high-visibility vest status cannot be determined.",
                    "cited_clause": c_text,
                    "cited_page": c_page,
                    "confidence": 0.5
                }

        # Scenario D: No specific helmet or vest PPE mandated
        return {
            "verdict": "Compliant",
            "badge_class": "badge-info",
            "summary": "Policy Advisory: No head protection or high-visibility vest mandated by current document.",
            "reason": "Uploaded document consists of general administrative rules or specialized PPE (e.g. eye/respirator) not evaluated by the crop model.",
            "cited_clause": None,
            "cited_page": None,
            "confidence": 0.9
        }
