# Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops

A rigorous deep learning research and evaluation suite benchmarking generative representation models (Convolutional Autoencoder, Conditional VAE, Conditional GAN, Score-based Diffusion Restoration, and Safety Policy Auditor) on safety-critical Personal Protective Equipment (PPE) crops.

---

## 🔒 Ethics & Privacy Guarantee

> **IMPORTANT PRIVACY & RESEARCH ETHICS NOTICE**  
> 1. **Transient Processing Only:** Head crops depict real people's faces. The backend server and inference pipelines process uploaded photos strictly in-memory. Uploaded images are **never persisted to disk** or retained in server logs.  
> 2. **Synthetic Identities:** Generative synthetic faces produced by any model in this suite (cVAE, cGAN, Diffusion) are artificial samples and **must not be published, claimed, or indexed as real individuals**.  
> 3. **Memorization Safeguards:** All generative models are audited using `common.metrics.check_nearest_real_neighbor` to verify that synthetic crops are novel samples along the continuous manifold and do not replicate or memorize training identity crops.

---

## 🏗️ Project Architecture

```
ppe-generative-project/
├── crops_64/                       # 64x64 RGB PPE crops (ImageFolder structure)
│   ├── train/                      # head=867, helmet=15991, vest=10041 (Total: 26,899)
│   ├── val/                        # head=231, helmet=4265, vest=2669 (Total: 7,165)
│   ├── test/                       # head=241, helmet=4446, vest=2783 (Total: 7,470)
│   └── stats.csv                   # Split and class breakdown verification ground truth
├── data_cache/                     # Pre-cached contiguous uint8 tensors for fast CPU training
├── common/                         # Shared foundation package across all notebooks and dashboard
│   ├── __init__.py                 # Clean package symbol exports
│   ├── data.py                     # Loaders ([0,1] & [-1,1]), WeightedRandomSampler, audit check
│   ├── degrade.py                  # Multi-severity degradations & held-out motion blur
│   ├── metrics.py                  # PSNR, SSIM, LPIPS, FID, KID, Precision/Recall, Diversity, Floor
│   ├── judge.py                    # Independent SmallResNet classifier (5-seed ensemble)
│   └── export.py                   # Standardized bundle layout manager (results/<model_name>/)
├── tests/
│   └── test_common.py              # 12 unit tests verifying metrics, degradations, judge, and audit
├── scripts/
│   ├── cache_dataset.py            # Converts ImageFolder into zero-copy binary .pt cache
│   ├── train_judge_fast.py         # Trains 5-seed independent SmallResNet judge on real crops
│   └── compute_noise_floor.py      # Computes empirical FID noise floor: FID(test vs train)
├── dashboard/                      # Production FastAPI + Modern Responsive Frontend
│   ├── app.py                      # FastAPI application with model registry routers
│   ├── model_def.py                # Exact ConvAE architecture matching training state_dict
│   ├── test_api.py                 # Backend automated test suite
│   ├── models/                     # Decoupled model registry and routers
│   │   ├── registry.py             # Catalog of 5 generative benchmark models
│   │   ├── autoencoder/            # Model 1: Autoencoder router and pipeline
│   │   ├── cvae/                   # Model 2: Conditional VAE router
│   │   ├── cgan/                   # Model 3: Conditional GAN router
│   │   ├── diffusion/              # Model 4: Diffusion restoration router
│   │   └── policy/                 # Model 5: Safety policy checker router
│   └── static/
│       ├── index.html              # Modern responsive UI with model tabs and dark mode
│       ├── app.js                  # Crop selector, Chart.js, tabs, dynamic performance hub
│       └── styles.css              # Custom styling
└── results/
    ├── autoencoder/                # Model 1 artifacts: ae_best.pt, svm_model.joblib, metrics, figures
    ├── judge/                      # Independent SmallResNet judge checkpoints (seeds 42..46) & benchmark
    └── noise_floor.json            # Inception empirical noise floor metrics
```

---

## 📊 Shared Foundation (`common/`)

### 1. `common/data.py`
- **Data Loaders:** Provides reproducible PyTorch `ImageFolder` loaders supporting both `[0, 1]` and `[-1, 1]` pixel ranges.
- **Extreme Class Imbalance:** Automatically computes inverted class frequencies for `WeightedRandomSampler` to counter the heavy imbalance (867 heads vs 15,991 helmets).
- **Integrity Audit:** `verify_dataset_counts()` programmatically validates split counts against `crops_64/stats.csv`.

### 2. `common/metrics.py`
- **Reconstruction Metrics:** Per-image PSNR (with zero-MSE protection), SSIM (guaranteed 1.0 for identical inputs), MSE, and MAE.
- **Deep Perceptual Distance:** AlexNet-based LPIPS using PyTorch Hub cached weights.
- **Distributional Quality (FID & KID):** Standardized PyTorch InceptionV3 pool3 (2048-d) feature extractor with bilinear upsampling to $299\times 299$. Fixed-sample, fixed-seed polynomial kernel MMD for KID.
- **Distributional Coverage:** Improved Precision & Recall ($k$-NN manifold estimation; Sajjadi et al., Kynkäänniemi et al. NeurIPS 2019).
- **Intra-Class Diversity:** Mean pairwise LPIPS distance across generated instances.
- **Memorization Guard:** Nearest-real-neighbor distance distribution audit (`check_nearest_real_neighbor`).
- **FID Noise Floor:** Empirical lower-bound baseline $\text{FID}(\text{real test} \parallel \text{real train})$.

### 3. `common/judge.py`
- **Independent Supervised SmallResNet:** A fast, 3-stage residual architecture with a stride-2 stem designed for discriminative representation without compute explosion.
- **No AE+SVM Contamination:** Generative models are evaluated by this independent supervised classifier, preventing autoencoder representation biases from contaminating benchmark results.
- **5-Seed Ensemble:** Evaluated across seeds `[42, 43, 44, 45, 46]` reporting mean $\pm$ standard deviation for accuracy, balanced accuracy, macro-F1, and per-class recall.

### 4. `common/degrade.py`
- **Standardized Corruptions:** Gaussian blur, Gaussian noise, downscale-upscale resampling ($2\times, 4\times$), JPEG compression, and composite corruptions.
- **Severity Tiers:** Three calibrated severity levels (`mild`, `medium`, `severe`) with deterministic seed support.
- **Held-Out Test Degradation:** Motion blur is strictly quarantined for zero-shot testing (`allow_held_out=True`).

### 5. `common/export.py`
- Standardizes saving and loading model bundles under `results/<model_name>/` (weights, `model_config.json`, `class_names.json`, sample grids, and metric records).

---

## 🚀 Running the Web Dashboard

A single command starts the entire system (FastAPI backend + responsive frontend):

```bash
uvicorn dashboard.app:app --reload
```

Open your browser at:
```
http://127.0.0.1:8000
```

### Dashboard Features
- **Model Registry Navigation:** Switch seamlessly between:
  1. **Autoencoder (Model 1):** Active full inference pipeline, 64x64 interactive crop box, real-time reconstruction, perceptual error heatmap, 256-d latent bar chart, 2D PCA scatter plot, and dynamic performance tables.
  2. **Conditional VAE (Model 2):** Class condition selector, latent $\beta$-ELBO slider, and dynamic performance hub.
  3. **Conditional GAN (Model 3):** Truncation $\psi$ parameter explorer and adversarial synthesis preview.
  4. **Diffusion Restoration (Model 4):** Corruption simulator and reverse diffusion restoration visualizer.
  5. **Safety Policy Checker (Model 5):** Counterfactual PPE compliance rule auditor.
- **Zero Fabrication:** The dashboard reads real numbers dynamically from result files in `results/<model_name>/` and never hardcodes metric values.

---

## 👥 Code Ownership & Contribution Workflow

We maintain a strict code ownership model (`.github/CODEOWNERS`) to guarantee stability and prevent regressions across shared assets:

| Path | Owner | Description |
|---|---|---|
| `/common/` | `@Mohit-developer-tech` | Core foundation (data, metrics, degradations, judge) |
| `/dashboard/` | `@Mohit-developer-tech` | Production FastAPI server & frontend |
| `/models/autoencoder/` | `@Mohit-developer-tech` | Baseline ConvAE & SVM classifier |
| `/models/vae/` | `@KhOmkar` | Conditional VAE generation |
| `/models/policy/` | `@KhOmkar` | Policy Compliance Checker & DistilBERT |
| `/models/gan/` | `@SachinB18` | Conditional GAN & DiffAugment augmentation |
| `/models/diffusion/` | `@SachinB18` | DDPM / DDIM score restoration |

### Pull Request & Review Rules
- **Component PRs:** Pull requests that touch only a person's own folders can be merged after **one review**.
- **Shared Infrastructure PRs:** Pull requests that touch `/common/` or `/dashboard/` **need Mohit's approval** (`@Mohit-developer-tech`).
- **Feature Branches:**
  - `feature/vae` (Owner: `@KhOmkar`)
  - `feature/policy` (Owner: `@KhOmkar`)
  - `feature/gan` (Owner: `@SachinB18`)
  - `feature/diffusion` (Owner: `@SachinB18`)

---

## 🧪 Verification & Testing

Run the automated test suite verifying metrics, degradations, dataset audit, and judge training:

```bash
python -m unittest tests/test_common.py -v
```

All 12 unit tests pass deterministically.

