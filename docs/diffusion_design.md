# Model 4: Conditional Diffusion Restoration (DDPM / DDIM) Design Document

**Project:** Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops  
**Model Focus:** SR3-Style Conditional Denoising Diffusion Probabilistic Model for PPE Crop Restoration  
**Task Domain:** Restoring low-quality, noisy, downscaled, JPEG-compressed, and motion-blurred surveillance crops ($64\times 64$ RGB) for downstream safety verification.

---

## 1. Overview and Problem Formulation

Surveillance video feeds in real-world industrial and construction facilities frequently suffer from severe optical and transmission degradations:
- Low-resolution crops from distant cameras ($2\times$ to $4\times$ downscale).
- Sensor noise in low-light environments (additive Gaussian noise).
- Optical defocus and atmospheric blur (Gaussian blur).
- Bandwidth-constrained streaming compression artifacts (blocky JPEG DCT quantization).
- Rapid worker movement or camera vibration (directional motion blur).

In this benchmark, **Model 4 (Conditional Diffusion Restoration)** aims to invert these degradations without hallucinating false compliance. While Model 1 (Autoencoder) performed deterministic encoding, Model 2 (cVAE) enabled smooth probabilistic interpolation, and Model 3 (cGAN) performed minority-class data augmentation, Model 4 acts as a **super-resolution and restoration engine** conditioned directly on the degraded image.

---

## 2. Architectural Design & Design Decisions

### a. Conditioning Protocol: SR3 Concatenation (Strictly No Label Conditioning)
Following the SR3 paradigm (Saharia et al., IEEE TPAMI 2022):
- The degraded image $y \in [-1, 1]^{3 \times 64 \times 64}$ is concatenated directly with the noisy latent state $x_t \in [-1, 1]^{3 \times 64 \times 64}$ along the channel axis.
- The input to the U-Net is therefore a $6$-channel tensor:
  $$\mathbf{z}_t = [x_t \,\|\, y] \in \mathbb{R}^{B \times 6 \times 64 \times 64}$$
- **CRITICAL SAFETY DESIGN CHOICE — NO CLASS LABELS**:
  Under no circumstances is the true class label $c \in \{\text{head}, \text{helmet}, \text{vest}\}$ provided to the restoration network. If the model were conditioned on $c$, it would suffer from catastrophic **label leakage**:
  1. The network would use the label prior to draw a hardhat on ambiguous crops rather than recovering true physical pixels.
  2. Downstream compliance evaluators (the Independent Judge Classifier and AE+SVM) would receive artificially inflated accuracy scores.
  3. In a real-world automated safety monitoring system, the ground-truth compliance status is unknown at inference time; the model must restore the crop strictly from visual evidence.

### b. Network Backbone (`ConditionalUNet`)
The denoising network is a customized conditional U-Net designed specifically for $64\times 64$ images:
- **Input / Output**: Input is $6$ channels ($3$ noisy $+ 3$ degraded condition); output is $3$ channels ($\hat{\epsilon}_\theta$).
- **Time Embeddings**: Sinusoidal positional embeddings (dimension $d_{\text{emb}} = 256$) mapped through a 2-layer MLP with SiLU activations.
- **Encoder Stages**:
  - Stage 1 ($64\times 64$): Conv2d stem ($6 \to 64$), 2 Residual Blocks with GroupNorm (32 groups) + SiLU.
  - Downsample 1: Strided Conv2d ($64\times 64 \to 32\times 32$, channels $64 \to 128$).
  - Stage 2 ($32\times 32$): 2 Residual Blocks ($128$ channels).
  - Downsample 2: Strided Conv2d ($32\times 32 \to 16\times 16$, channels $128 \to 256$).
  - Stage 3 ($16\times 16$): 2 Residual Blocks ($256$ channels) + Multi-Head Self-Attention ($16\times 16$ spatial resolution, 4 attention heads).
  - Downsample 3: Strided Conv2d ($16\times 16 \to 8\times 8$, channels $256 \to 256$).
  - Stage 4 ($8\times 8$): 2 Residual Blocks ($256$ channels) + Multi-Head Self-Attention ($8\times 8$ spatial resolution, 4 attention heads).
- **Middle Bottleneck ($8\times 8$)**:
  - Residual Block ($256$ channels) $\to$ Self-Attention ($8\times 8$, 4 heads) $\to$ Residual Block ($256$ channels).
- **Decoder Stages (Symmetric with Skip Connections)**:
  - Upsample 1 ($8\times 8 \to 16\times 16$ via Nearest Upsample + Conv2d) $\to$ Concatenate skip connections $\to$ 2 Residual Blocks + Self-Attention.
  - Upsample 2 ($16\times 16 \to 32\times 32$) $\to$ Concatenate skip connections $\to$ 2 Residual Blocks.
  - Upsample 3 ($32\times 32 \to 64\times 64$) $\to$ Concatenate skip connections $\to$ 2 Residual Blocks.
- **Output Head**: GroupNorm $\to$ SiLU $\to$ Conv2d ($64 \to 3$) initialized near zero.
- **Exponential Moving Average (EMA)**:
  Shadow weights maintained with decay $\beta_{\text{EMA}} = 0.999$, updated every training step and used for all validation, test sampling, and dashboard serving.

### c. Noise Scheduling: Cosine vs Linear Schedule
Standard linear beta schedules ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$) drop signal-to-noise ratio too quickly for small $64\times 64$ images, causing the center of the diffusion trajectory to consist mostly of pure noise.
We implement the **Cosine Noise Schedule** (Nichol & Dhariwal, ICML 2021):
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2, \quad s = 0.008$$
with $\beta_t = \text{clip}\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.0001, 0.999\right)$ for $T = 1000$ timesteps. This maintains structural image content across a broader range of timesteps.

---

## 3. Loss Formulations & Perceptual Stability

Standard diffusion models optimize the variational lower bound using $\epsilon$-prediction:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### Loss Formulations
1. **Baseline $\epsilon$-MSE Loss**:
   $$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, y, t)\|_2^2 \right]$$
2. **Main Loss ($\epsilon$-L1 Loss)**:
   $$\mathcal{L}_{\text{L1}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, y, t)\|_1 \right]$$
   L1 loss produces sharper high-frequency gradients than MSE and is less sensitive to outlier noise predictions.
3. **Low-Noise Perceptual Term ($\epsilon$-L1 + Weighted LPIPS)**:
   From any predicted noise vector $\hat{\epsilon} = \epsilon_\theta(x_t, y, t)$, the Tweedie estimate of the clean image is:
   $$\hat{x}_0(x_t, \hat{\epsilon}) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$
   **Instability Hazard**: At large timesteps ($t \approx T$), $\bar{\alpha}_t \to 0$, causing $\frac{1}{\sqrt{\bar{\alpha}_t}} \to \infty$. A small error in $\hat{\epsilon}$ causes severe amplification in $\hat{x}_0$, leading to exploded gradients if LPIPS is backpropagated at high noise.
   **Solution**: We apply perceptual LPIPS loss **strictly at low-noise timesteps** ($t < T/4 = 250$) and scale it by $\bar{\alpha}_t$:
   $$\mathcal{L}_{\text{total}} = \|\epsilon - \hat{\epsilon}\|_1 + \lambda_{\text{perc}} \cdot \mathbb{I}(t < 250) \cdot \bar{\alpha}_t \cdot \text{LPIPS}(\hat{x}_0, x_0)$$
   where $\lambda_{\text{perc}} = 0.1$.

### Validation Ablation Matrix
| Config Name | Loss Formulation | Perceptual Weighting | Target Hypothesis |
| :--- | :--- | :--- | :--- |
| `eps_mse` | Epsilon MSE | None | Standard baseline, smooth |
| `eps_l1` | Epsilon L1 | None | Sharper edges, robust to outliers |
| `eps_l1_lpips` | Epsilon L1 + LPIPS | $\bar{\alpha}_t \cdot \mathbb{I}(t < 250)$ | Superior structural realism & LPIPS |

Model selection is conducted strictly on the **Validation Split** using LPIPS, FID, and Judge Classification Recovery.

---

## 4. Sampling Algorithms: DDPM vs DDIM

### a. Full Stochastic DDPM ($1000$ steps)
Following Ho et al. (NeurIPS 2020):
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, y, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$
While theoretically exact, $1000$ sequential network forward passes require $\sim 15 - 20$ seconds per image on CPU, making it impractical for interactive dashboard serving.

### b. Deterministic DDIM ($S \in \{20, 50, 100\}$ steps)
Following Song et al. (ICLR 2021), we define a sub-sequence of timesteps $\tau = [\tau_1, \tau_2, \dots, \tau_S]$ with $\eta = 0.0$ (deterministic non-Markovian trajectory):
$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \left( \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \hat{\epsilon}}{\sqrt{\bar{\alpha}_{\tau_i}}} \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}} \hat{\epsilon}$$
- **50 steps**: Achieves $98\%$ of the visual quality of full DDPM at $20\times$ speedup.
- **20 steps**: Instant interactive preview ($< 0.5$s per image) suitable for dashboard sliders.

---

## 5. The Perception-Distortion Trade-off & Regression Baseline

A fundamental law of image restoration (Blau & Michaeli, CVPR 2018) dictates that **distortion** (e.g. MSE, PSNR, SSIM) and **perceptual quality** (e.g. LPIPS, FID) are mutually conflicting:
- A deterministic regression network trained with MSE/L1 minimizes pixel variance by predicting the conditional expectation $\mathbb{E}[x_0 \mid y]$. Under high uncertainty, this expectation is the blur average of all plausible completions.
- Diffusion models sample from the true conditional distribution $p(x_0 \mid y)$. They reconstruct crisp high-frequency details (textures, weaves, edges), achieving superior LPIPS and FID, even if the exact pixel locations deviate slightly from the ground truth (yielding lower PSNR).

### Benchmark Comparison Baselines
Every test evaluation compares 4 distinct approaches:
1. **No Restoration**: Degraded crop evaluated directly.
2. **Bicubic Upscaling**: Standard non-learning interpolation baseline.
3. **Regression U-Net Baseline**: A deterministic network with identical capacity (same depth and channels) trained strictly with L1 + LPIPS loss to directly map $y \to x_0$.
4. **Conditional Diffusion (DDIM 50 steps)**: Our proposed generative restoration engine.

---

## 6. Degradation Protocol & Held-Out Generalization

Following `common/degrade.py`, the training set is dynamically perturbed with 4 standard degradation types across 3 severity levels:
- `gaussian_noise`: $\sigma \in [0.01, 0.22]$
- `gaussian_blur`: kernel $3 \times 3$ to $7 \times 7$, $\sigma \in [0.4, 3.0]$
- `downscale`: $2\times$ to $4\times$ downsampling followed by bilinear upsampling
- `jpeg`: quality $75$ down to $10$

### Held-Out Test Degradation: `motion_blur`
To prevent the model from overfitting to synthetic artifacts, **linear directional motion blur** (kernels $3\times 3$ to $19\times 19$, random angles $0^\circ - 180^\circ$) is **STRICTLY HELD OUT** during all training and validation phases. It is evaluated exclusively on the test split to assess zero-shot restoration generalization under unseen camera motions.

---

## 7. Safety-Critical Hallucination Audit

In industrial safety monitoring, an image restoration tool must never hallucinate safety compliance where a hazard exists. We formalize and evaluate:

1. **Overall Class-Flip Rate**:
   $$\text{Flip}_{\text{all}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{C}(\text{restored}_i) \ne \hat{C}(\text{clean}_i))$$
2. **Safety-Critical Flip Rate ($\text{Hazard} \to \text{Compliant}$)**:
   $$\text{Flip}_{\text{critical}} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{C}(\text{clean}_i) = \text{head} \;\land\; \hat{C}(\text{restored}_i) \in \{\text{helmet}, \text{vest}\})}{\sum_{i=1}^N \mathbb{I}(\hat{C}(\text{clean}_i) = \text{head})}$$
   A non-zero critical flip rate indicates that the model is inventing synthetic PPE features onto unprotected workers, posing a dangerous safety vulnerability.
3. **Downstream Classification Accuracy**:
   Evaluated with both the Independent Judge Classifier v2 (`SmallResNet`) and the Model 1 Autoencoder + SVM baseline.

