# Master AI Model Performance Audit & Optimization Blueprint
## Project: Generative AI Framework for Automated Construction Site Safety Monitoring and Hazard Prediction

> **Author:** Antigravity AI Engineering & Research Team  
> **Date:** November 2026  
> **Target Hardware:** NVIDIA RTX 3050 (4 GB VRAM) / High-Throughput Memory-Efficient CPU Pipeline  
> **Core Strategy:** Checkpoint-First Scientific Optimization (No Unnecessary Retraining)

---

## Executive Summary of Project Audit

This audit evaluates the five foundational machine learning and deep learning models comprising the Generative PPE Benchmarking Suite:
1. **Model 1: ConvAE + RBF-SVM** (Deterministic Self-Supervised Manifold & Classification)
2. **Model 2: Conditional $\beta$-VAE / $\sigma$-VAE** (Probabilistic Generative Latent Manifold)
3. **Model 3: Conditional GAN with DiffAugment** (Minority Class Adversarial Synthesis)
4. **Model 4: Conditional Diffusion (DDPM/DDIM)** (Score-Based Image Restoration)
5. **Model 5: Policy Transformer (DistilBERT)** (Regulatory Clause Classification & Calibration)
*Plus Auxiliary:* **SmallResNet 5-Seed Ensemble Judge** (Independent Evaluation Arbiter)

All original checkpoints have been cataloged, verified, and safely backed up to `checkpoints/baseline/`. No baseline checkpoint will be deleted or overwritten.

---

## 1. Checkpoint Inventory & Baseline Verification

| Model | Checkpoint Location | File Size | Epochs / Steps | Stored Components | Compatibility Status | Baseline Strategy |
|:---|:---|:---:|:---:|:---|:---:|:---:|
| **Model 1: ConvAE** | `artifacts/autoencoder/ae_best.pt`<br>`artifacts/autoencoder/checkpoints/checkpoint_epoch_100.pt` | 13.31 MB<br>39.91 MB | 100 epochs (Best Val: Ep 94) | `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `history` | 100% Verified Compatible | **Fine-Tuning & SVM Sweep (Strategy C)** |
| **Model 1: SVM** | `artifacts/autoencoder/svm_model.joblib`<br>`scaler.joblib`, `latent_pca2d.joblib` | 25.99 MB<br>0.01 MB | 5-fold CV Grid Search | RBF-SVM ($C=10.0, \gamma=0.0039$), StandardScaler, 256-d PCA | 100% Verified Compatible | **Threshold & Latent Optimization** |
| **Model 2: cVAE** | `artifacts/cvae/cvae_best.pt` | 20.70 MB | 100 epochs ($\beta=1.0$) | Full model `state_dict` (Encoder, Decoder, Condition Embeddings) | 100% Verified Compatible | **Targeted Fine-Tuning (Strategy C)** |
| **Model 3: cGAN** | `artifacts/cgan/cgan_best.pt` | 39.68 MB | 100 epochs (26,000 steps) | `generator_state_dict`, `generator_ema_state_dict`, `discriminator_state_dict` | 100% Verified Compatible | **Inference & Filter Audit (Strategy A/C)** |
| **Model 4: Diffusion** | `artifacts/diffusion/diffusion_best.pt`<br>`regression_best.pt` | 13.16 MB<br>7.11 MB | 100 epochs (Cosine $T=1000$) | SR3 U-Net `state_dict`, Time MLP, Self-Attention | 100% Verified Compatible | **Guided Residual Sampling (Strategy C/D)** |
| **Model 5: Policy** | `artifacts/policy/policy_classifier.pt` | 255.46 MB | 4 epochs fine-tuning | DistilBERT `state_dict`, Scaler $T^*=0.5982$, Class Maps | 100% Verified Compatible | **Class-Weighted Re-tuning (Strategy C)** |
| **Judge Arbiter** | `artifacts/judge/judge_seed_{42..46}.pt` | 2.12 MB each (5 seeds) | 10 epochs each | SmallResNet 3-stage `state_dict` (~550K params) | 100% Verified Compatible | **Frozen Neutral Arbiter (Strategy A)** |

---

## 2. Model-by-Model Deep Audit & Problem Identification

### 2.1 Model 1: ConvAE + RBF-SVM
* **Architecture:** 4-layer strided convolutional encoder ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$) with LeakyReLU(0.2) + BatchNorm, projecting to a 256-dimensional bottleneck vector $z \in \mathbb{R}^{256}$. Symmetrical transposed convolution decoder with Sigmoid output.
* **Current Empirical Metrics:**
  - Test Reconstruction MSE: $0.006199 \pm 0.0042$
  - Test PSNR: $23.12 \pm 3.20\text{ dB}$
  - Test SSIM: $0.7132 \pm 0.0929$
  - SVM Test Accuracy: **$91.93\%$**
  - SVM Macro-F1: **$86.11\%$** (Head F1: $74.07\%$, Helmet F1: $93.38\%$, Vest F1: $90.88\%$)
* **Identified Vulnerabilities & Issues:**
  1. *Minority Violation Recall Bottleneck:* While helmet recall is $93.52\%$ and vest recall is $91.47\%$, minority `head` recall is only **$65.42\%$**. More than one-third ($34.58\%$) of critical unhelmeted head violations are missed by the argmax decision boundary.
  2. *Single Latent Dimension:* Checkpoints exist exclusively for $d=256$. Intermediate evaluations for $d=384$ and $d=512$ were not saved in standardized checkpoint formats.
  3. *Pure Pixel Loss Limitations:* Pure MSE loss causes high-frequency texture attenuation (e.g., retroreflective vest micro-patterns blur into solid yellow-green blocks).

---

### 2.2 Model 2: Conditional $\beta$-VAE / $\sigma$-VAE
* **Architecture:** Conditional VAE with spatial class embedding concatenation on encoder ($64\times 64\times 19$) and latent vector concatenation on decoder ($z \in \mathbb{R}^{128} \oplus c \in \mathbb{R}^{32} \rightarrow \mathbb{R}^{160}$). GroupNorm and SiLU residual blocks.
* **Current Empirical Metrics:**
  - Test Reconstruction MSE: $0.009514$
  - Test PSNR: $21.19\text{ dB}$
  - Test SSIM: $0.6068$
  - Test ELBO: $11,178.61\text{ nats}$ ($-1.312\text{ bits/dim}$)
  - Active Latent Units: **$128 / 128\ (100\%)$**
  - Judge Conditional Fidelity: **$62.00\%$** (Head: $79.0\%$, Vest: $75.5\%$, **Helmet: $31.5\%$**)
* **Identified Vulnerabilities & Root Causes:**
  1. *Severe Helmet-to-Head Collapse:* The independent judge scores generated helmets at only **$31.5\%$ conditional accuracy**. Smooth Gaussian decoding removes the sharp brim curvature and specular highlights that distinguish a hard hat from hair/scalp, causing the judge to classify synthetic helmets as bare heads.
  2. *Reconstruction vs Regularization Tradeoff:* Fixed $\beta=1.0$ under the analytical $\sigma$-VAE variance ($D/2 = 6,144$) imposes excessive regularization pressure on high-frequency details.
  3. *Lack of Structural SSIM Constraint:* The reconstruction objective only penalizes squared pixel differences, ignoring local luminance and contrast covariance.

---

### 2.3 Model 3: Conditional GAN (DiffAugment)
* **Architecture:** DCGAN-style transposed convolution generator with Conditional BatchNorm; Projection Discriminator with Spectral Normalization on all convolution kernels; DiffAugment pipeline (color jitter, translation $\pm 4\text{px}$, cutout $8\times 8$).
* **Current Empirical Metrics:**
  - Discriminator Loss: $0.4974$ (Hinge Loss)
  - Generator Loss: $1.6244$
  - Judge Conditional Accuracy: **$89.56\%$** (Helmet: $96.7\%$, Head: $87.3\%$, Vest: $84.7\%$)
  - FID: $134.62$
  - KID: $0.0725 \pm 0.0039$
  - 5th-Percentile Privacy Distance ($p_5$): **$0.2637$** (vs $0.06$ noise floor)
  - Downstream Augmentation Gain: Increases minority Head F1 from $70.76\%$ to **$76.81\%$** ($+6.05\%$).
* **Identified Vulnerabilities & Observations:**
  1. *Elevated FID Score ($134.62$):* Due to calculating Inception features on $64\times 64$ crops upsampled to $299\times 299$, bilinear upsampling artifacts inflate the distribution distance relative to the $24.73$ real-vs-real noise floor.
  2. *Precision / Recall Imbalance:* Inception Precision is $0.0961$ while Inception Recall is $0.0760$, indicating that generated samples occupy a relatively narrow sub-manifold of the real feature distribution.

---

### 2.4 Model 4: Conditional Diffusion (DDPM/DDIM)
* **Architecture:** SR3-style Conditional U-Net accepting 6-channel inputs ($3\text{ch degraded} \oplus 3\text{ch noisy state } x_t$). Sinusoidal time embedding, self-attention at $16\times 16$, GroupNorm.
* **Current Empirical Metrics:**
  - Regression Baseline U-Net: PSNR = $21.45\text{ dB}$, SSIM = $0.7606$, Judge Accuracy = **$87.44\%$**
  - Unconstrained DDIM-50 Diffusion: PSNR = $5.41\text{ dB}$, SSIM = $0.0034$, LPIPS = **$0.1600$**
  - Flip Rate: **$100\%$ on safety-critical head crops**
* **Identified Vulnerabilities & Root Causes:**
  1. *Catastrophic Safety Class Flips:* When bare heads are corrupted by noise or blur, unconstrained score-based reverse diffusion samples toward the dominant data prior (helmets represent $59.5\%$ of training data), hallucinating hard hat textures onto unhelmeted heads.
  2. *Low PSNR (5.41 dB):* Without an explicit structural reconstruction penalty anchoring reverse diffusion to the clean regression target, high-frequency stochastic generation drifts in pixel space, producing the classic Blau-Michaeli failure mode.

---

### 2.5 Model 5: Policy Transformer (DistilBERT)
* **Architecture:** `distilbert-base-uncased` (6 layers, 12 attention heads, 768 hidden dimension, 66M parameters) + linear 5-class classification head.
* **Current Empirical Metrics:**
  - Test Accuracy: **$77.53\%$**
  - Balanced Accuracy: **$79.72\%$**
  - Test Macro-F1: **$77.72\%$** (5 seeds mean: $76.36\% \pm 2.20\%$)
  - Baseline Comparison: Beats Keyword Regex ($45.16\%$) by **$+32.56\%$**, but trails TF-IDF + Logistic Regression ($79.32\%$) on the small test split ($N=89$).
  - Calibration: Temperature scaling ($T^*=0.5982$) reduces test ECE from $21.20\%$ to **$10.80\%$**.
* **Identified Vulnerabilities & Root Causes:**
  1. *Small Training Corpus Overfitting:* The training set has only 155 sentences. DistilBERT with 66M parameters risks overfitting compared to TF-IDF.
  2. *Negation & Exception Blindspot:* Performance on negation/exception clauses ($N=13$) is only **$38.46\%$**, because the self-attention mechanism fails to properly weight negative scopes ("not required", "exempt", "except") without targeted training instances.
  3. *Frozen Layer Under-adaptation:* Freezing the lower 3 transformer layers prevented the encoder from adapting domain-specific legal vocabulary (e.g., ANSI classifications, MUTCD section codes).

---

## 3. Dataset Integrity & Split Verification Audit

### 3.1 Visual Dataset (`crops_64/`)
* **Split Counts Verification:** Programmatically verified against `crops_64/stats.csv`:
  - **Train ($N=26,899$):** `head` = 867 (3.22%), `helmet` = 15,991 (59.45%), `vest` = 10,041 (37.33%)
  - **Val ($N=7,165$):** `head` = 249 (3.48%), `helmet` = 4,278 (59.71%), `vest` = 2,638 (36.82%)
  - **Test ($N=7,470$):** `head` = 214 (2.86%), `helmet` = 4,395 (58.84%), `vest` = 2,861 (38.30%)
* **Leakage Audit:** Images are strictly partitioned by scene origin hashes; zero crop overlap exists across train, validation, and test directories.

### 3.2 Regulatory Policy Corpus (`data/policy/`)
* **Distribution:** 321 sentences (155 Train, 77 Val, 89 Test).
* **Leakage Audit:** Exact-phrase hashing confirms zero sentence overlap between splits.

---

## 4. Hardware Resource & GPU Memory Optimization Profile

* **Available Hardware Profile:**
  - Machine Environment: Intel CPU / NVIDIA RTX 3050 (4 GB VRAM limit).
  - VRAM Budget: 4,096 MB maximum allocatable.
* **Safe Operating Parameters:**
  - Image Batch Size: 64 for ConvAE, cVAE, and cGAN ($64 \times 3 \times 64 \times 64 \times 4\text{ bytes} \approx 3.15\text{ MB}$ activations per batch).
  - Diffusion Batch Size: 16–32 (to prevent U-Net self-attention peak memory spikes).
  - Transformer Batch Size: 16 (max sequence length 128 tokens, peak memory $\approx 850\text{ MB}$).
  - Mixed Precision (`torch.cuda.amp.autocast`): Reduces forward-pass memory footprint by $\approx 45\%$ with float16 linear projections.

---

## 5. Model-Wise Optimization Roadmap (Phase 2)

Following the mandatory **Checkpoint-First** strategy, the optimization sequence is prioritized as follows:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           OPTIMIZATION ROADMAP PIPELINE                          │
├────────────────────────────┬─────────────────────────────────────────────────────┤
│ 1. Conditional β-VAE       │ • Load `cvae_best.pt` baseline                      │
│    (High Priority)         │ • Sweep β ∈ {0.1, 0.5, 1.0} with MSE + SSIM loss   │
│                            │ • Fine-tune decoder to eliminate helmet blur         │
│                            │ • Evaluate conditional fidelity vs active units     │
├────────────────────────────┼─────────────────────────────────────────────────────┤
│ 2. Policy Transformer      │ • Load `policy_classifier.pt` baseline               │
│    (High Priority)         │ • Unfreeze upper layers with differential LR        │
│                            │ • Add negation-clause focal loss                    │
│                            │ • Target > 80% Macro-F1 (surpassing TF-IDF)          │
├────────────────────────────┼─────────────────────────────────────────────────────┤
│ 3. ConvAE + RBF-SVM        │ • Load `ae_best.pt` baseline                         │
│                            │ • Optimize SVM decision threshold for Head recall   │
│                            │ • Fine-tune with composite SSIM loss                │
│                            │ • Evaluate d ∈ {256, 384, 512} representation       │
├────────────────────────────┼─────────────────────────────────────────────────────┤
│ 4. Conditional Diffusion   │ • Load `diffusion_best.pt` baseline                 │
│                            │ • Implement residual anchoring to clean regression  │
│                            │ • Suppress class-flipping on degraded bare heads     │
│                            │ • Sweep DDIM steps (10, 20, 50) for latency/quality │
├────────────────────────────┼─────────────────────────────────────────────────────┤
│ 5. Conditional GAN         │ • Load `cgan_best.pt` baseline                      │
│                            │ • Evaluate EMA generator weights                    │
│                            │ • Verify downstream augmentation stability          │
└────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 6. Audit Sign-Off & Verification

All original checkpoints are secured in `checkpoints/baseline/`. The baseline evaluation metrics are frozen and established as ground truth. We now proceed to executing Phase 2 optimizations systematically.

