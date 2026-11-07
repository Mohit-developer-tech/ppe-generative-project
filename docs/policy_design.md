# Design Review: Model 5 — Policy-Driven Compliance Checker
**Research Project:** Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops  
**Component:** Natural Language Policy Ingestion, Sentence-Level Classification, and Multi-Modal Vision Verdict Integration  
**Date:** November 2026

---

## 1. Problem Formulation & System Scope

Industrial sites, construction environments, and chemical processing plants do not operate under monolithic safety rules. Regulatory mandates (OSHA 29 CFR 1926, ANSI/ISEA 107, MUTCD) and site-specific standard operating procedures (SOPs) establish granular, conditional, and zone-dependent Personal Protective Equipment (PPE) requirements.

Automating safety compliance requires bridging the gap between:
1. **Unstructured Regulatory/Corporate Documents**: Safety policy handbooks, contractor orientation manuals, and OSHA regulatory codes written in English.
2. **Visual Inspection Models**: Computer vision classifiers and generative restoration engines operating on $64\times 64$ worker safety crops (`head`, `helmet`, `vest`).

This module implements **Model 5: Policy-Driven Compliance Checker**, an NLP transformer pipeline integrated with an explicit three-state multi-modal decision engine.

```
                    ┌──────────────────────────────┐
                    │  Uploaded Policy Document    │
                    │      (PDF / DOCX / TXT)      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Document Ingestion &       │
                    │   Sentence Segmentation      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Sentence PPE Classifier     │
                    │ (DistilBERT / TF-IDF / Rules)│
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Calibrated Confidence Filter │
                    │   (Temperature Scaling T*)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Aggregated Policy Reqs:      │
                    │ {helmet, vest, both, other}  │
                    └──────────────┬───────────────┘
                                   │
      ┌────────────────────────────┴───────────────────────────┐
      │                                                        │
      ▼                                                        ▼
┌──────────────┐                                       ┌──────────────┐
│  Judge v2    │                                       │   AE + SVM   │
│ (SmallResNet)│                                       │   Baseline   │
└──────┬───────┘                                       └──────┬───────┘
       │                                                      │
       └───────────────────────────┬──────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │    Vision Consensus Fusion   │
                    │   (Agreement Required Gate)  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Three-State Policy Verdict │
                    │ [Violation|Compliant|Cannot] │
                    └──────────────────────────────┘
```

---

## 2. Failure Modes & Mitigations

### Failure Mode A: Label Collapse and Forcing Wrong Categories
- **Risk**: In a 3-class system (`helmet_required`, `vest_required`, `both_required`), rules mandating eye protection, ear muffs, gloves, or safety boots are forced into a hardhat or vest label, creating phantom safety requirements. Similarly, administrative disclaimers and scope definitions are misclassified as PPE mandates.
- **Mitigation**: Formulate a mandatory **5-class taxonomy**:
  1. `helmet_required`: Mandatory hard hat or protective helmet (OSHA 1926.100).
  2. `vest_required`: Mandatory high-visibility garments (MUTCD 6D.03, ANSI/ISEA 107).
  3. `both_required`: Explicit co-requirement of hard hat and high-vis vest in the same clause/zone.
  4. `other_ppe`: Safety glasses, gloves, steel-toe boots, respirators, hearing protection, fall arrest harnesses.
  5. `none`: Administrative text, definitions, penalties, general safety philosophy without direct PPE mandates.

### Failure Mode B: Data Leakage via Random Sentence Splitting
- **Risk**: Many regulatory sentences share nearly identical boilerplate phrasing across clauses (e.g. *"Employees shall be provided with..."*). Random sentence-level train/test splitting leaks phrase structures, yielding artificially inflated test metrics ($>98\%$) that fail in real deployment.
- **Mitigation**: **Strict document/section-level splitting**. The training set is derived from OSHA 29 CFR 1926 Subpart E (General PPE, Head Protection, Eye Protection, Foot Protection); validation is held out from MUTCD Section 6D.03; and the test set is held out from OSHA 1926 Subpart G (Signaling/Flaggers), UK HSE standards, and external site rules.

### Failure Mode C: Synthetic Data Illusions
- **Risk**: Generating hundreds of synthetic template sentences (*"Wear your helmet in zone X"*) produces an unrealistically clean distribution that masks real-world syntactic complexity, leading to brittle real-world behavior.
- **Mitigation**: Build the benchmark dataset ($N \ge 300$ sentences) exclusively from **real public safety texts**. Synthetic template sentences are strictly quarantined into an isolated stress-test slice used exclusively for robustness auditing.

### Failure Mode D: Negation and Exception Blindness
- **Risk**: Bag-of-words and small-data transformers frequently latch onto keywords like *"helmet"* and ignore negations or exceptions (*"Hard hats are not required inside enclosed excavator cabs"*), mistakenly declaring a helmet requirement.
- **Mitigation**:
  1. Explicitly audit an isolated **Negation / Exception evaluation slice**.
  2. Incorporate negative polarity phrases (*"not required"*, *"exempt"*, *"except when"*, *"excluding"*) into feature representations.
  3. Transparently document negation handling as a known architectural limitation in the dashboard and research notes.

### Failure Mode E: Premature Whole-Worker Compliance Claims
- **Risk**: Claiming a worker is "100% Compliant" based on a single crop showing a hard hat, even when the policy requires both hard hat and high-visibility vest.
- **Mitigation**: Implement a strict **Three-State Verdict Architecture**:
  - `Violation`: Observed crop directly violates the policy (e.g., bare head when helmet is required).
  - `Compliant`: Observed crop directly satisfies the policy (e.g., helmet crop when helmet is required).
  - `Cannot determine`: The crop provides incomplete visual evidence (e.g., policy requires `both_required`, but crop only shows a helmet; or crop shows `vest` while helmet is required).

### Failure Mode F: Vision Misclassification Propagation
- **Risk**: If the vision model misclassifies a degraded crop, the policy checker will issue an invalid compliance citation.
- **Mitigation**: **Multi-Model Consensus Gating**. Both the Independent Judge Classifier v2 and the Autoencoder + SVM baseline must agree on the crop identity. If they disagree, the verdict defaults to `Cannot determine` with an explicit vision disagreement audit flag.

---

## 3. Data Specification & Label Distribution

The dataset `data/policy/labels.csv` comprises real regulatory and corporate clauses categorized into 5 distinct classes:

| Class Label | Description | Canonical Example |
| :--- | :--- | :--- |
| `helmet_required` | Mandatory head protection | *"Employees working in areas where there is danger of head injury shall wear protective helmets."* (29 CFR 1926.100a) |
| `vest_required` | Mandatory high-visibility apparel | *"All workers within the right-of-way exposed to traffic shall wear high-visibility safety apparel."* (MUTCD 6D.03) |
| `both_required` | Simultaneous helmet & vest requirement | *"All personnel entering the active laydown area must wear an ANSI approved hard hat and Class 2 high-vis vest."* |
| `other_ppe` | Other PPE (hearing, eye, gloves, boots) | *"Protective eye and face equipment shall be required where there is danger of flying particles."* (29 CFR 1926.102) |
| `none` | General scope, administrative, disclaimers | *"The employer shall verify that the required workplace hazard assessment has been performed."* (29 CFR 1910.132d) |

### Split Strategy (Zero Cross-Section Leakage)
- **Train Split** (~60%, 200 sentences): OSHA 29 CFR 1926 Subpart E (1926.95, 1926.96, 1926.100, 1926.102, 1926.104), OSHA 1910.132 General Requirements.
- **Validation Split** (~20%, 65 sentences): MUTCD Part 6 Section 6D.03 High-Visibility apparel, OSHA 1926.106 Water safety.
- **Test Split** (~20%, 75 sentences): OSHA 29 CFR 1926 Subpart G (1926.200, 1926.201 Flaggers), UK HSE / ANSI-107 site manuals.

---

## 4. Modeling & Baseline Suite

To verify whether deep contextual representations are genuinely necessary for this domain, we implement and benchmark four competing models across 5 random seeds:

1. **Keyword / Regex Heuristic Rule Classifier**:
   - Matches priority patterns: `both` > `helmet` > `vest` > `other` > `none`.
   - Incorporates negative lookaheads and exception patterns (`"not required"`, `"except"`).
2. **TF-IDF + Multinomial Logistic Regression**:
   - Sublinear term frequency, n-gram range $(1, 2)$, $L_2$ regularization with class weighting.
3. **Zero-Shot NLI Baseline**:
   - Entailment hypothesis testing against candidate label templates.
4. **Fine-Tuned DistilBERT Classifier**:
   - Pretrained `distilbert-base-uncased`, sequence length 128, linear classification head, AdamW ($lr = 3\times 10^{-5}$, weight decay 0.01), 10% warmup, early stopping on validation macro-F1.

---

## 5. Calibration & Uncertainty Engine

Raw transformer soft-max probabilities are systematically overconfident. To ensure safety-critical reliability:
1. **Temperature Scaling ($T^*$)**:
   $$\hat{p}_i = \frac{\exp(z_i / T^*)}{\sum_j \exp(z_j / T^*)}$$
   Optimized via negative log-likelihood on the validation split.
2. **Expected Calibration Error (ECE)**:
   Reported before and after scaling with 10 confidence bins.
3. **Review Flag**:
   Any sentence with calibrated confidence $\max_k \hat{p}_k < \tau$ (default $\tau = 0.65$) is marked as `needs_human_review = True`, preventing automated verdict decisions on ambiguous clauses.

---

## 6. Document Ingestion Pipeline

The ingestion service handles real-world formats:
- **PDF**: Stream-level text extraction via `pypdf` with page number preservation.
- **Scanned PDF Trap**: Detects files yielding $<20$ characters per page across all pages. Emits a clean user alert: *"Scanned document detected with no text layer. OCR is out of scope; please upload a digital PDF, DOCX, or TXT file."*
- **DOCX**: Paragraph parsing via `python-docx` with heading level extraction.
- **TXT**: UTF-8/Latin-1 plain text parsing with sentence boundary disambiguation.
- **Sanitization**: All text is HTML-escaped (`html.escape`) before rendering in the dashboard DOM to eliminate Cross-Site Scripting (XSS) vectors.
- **Upload Limit**: Enforced $10\text{ MB}$ maximum payload.

---

## 7. Multi-Modal Verdict Decision Rules

Let $\mathcal{R}_{\text{policy}} \subseteq \{\text{helmet}, \text{vest}\}$ be the union of active, confident PPE requirements extracted from the policy document.
Let $\hat{y}_{\text{judge}} \in \{\text{head}, \text{helmet}, \text{vest}\}$ and $\hat{y}_{\text{svm}} \in \{\text{head}, \text{helmet}, \text{vest}\}$ be the vision model predictions.

1. **Consensus Gating**:
   $$\hat{y}_{\text{vision}} = \begin{cases} \hat{y}_{\text{judge}}, & \text{if } \hat{y}_{\text{judge}} = \hat{y}_{\text{svm}} \\ \text{UNCERTAIN}, & \text{otherwise} \end{cases}$$
2. **Verdict State Logic**:
   - If $\hat{y}_{\text{vision}} == \text{UNCERTAIN}$: **`Cannot determine`** *(Vision ensemble disagreement)*.
   - If $\text{helmet} \in \mathcal{R}_{\text{policy}}$ and $\hat{y}_{\text{vision}} == \text{head}$: **`Violation`** *(Unprotected head detected in mandatory helmet zone)*.
   - If $\text{helmet} \in \mathcal{R}_{\text{policy}}$ and $\hat{y}_{\text{vision}} == \text{helmet}$ and $\text{vest} \notin \mathcal{R}_{\text{policy}}$: **`Compliant`** *(Helmet verified)*.
   - If $\text{vest} \in \mathcal{R}_{\text{policy}}$ and $\hat{y}_{\text{vision}} == \text{vest}$ and $\text{helmet} \notin \mathcal{R}_{\text{policy}}$: **`Compliant`** *(High-vis vest verified)*.
   - If $\text{both} \in \mathcal{R}_{\text{policy}}$:
     - If $\hat{y}_{\text{vision}} == \text{head}$: **`Violation`** *(Head unprotected)*.
     - If $\hat{y}_{\text{vision}} \in \{\text{helmet}, \text{vest}\}$: **`Cannot determine`** *(Crop confirms one item, but full worker compliance requires full-body inspection)*.
   - If $\mathcal{R}_{\text{policy}} == \emptyset$: **`Cannot determine`** *(No PPE requirements identified in uploaded text)*.

