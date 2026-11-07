# Conditional Variational Autoencoder (CVAE) Design Document
**Research Project:** Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops  
**Model 2:** Class-Conditional Variational Autoencoder (CVAE) on $64\times 64$ RGB Crops  
**Classes:** `head` (Safety Violation = 0), `helmet` (Compliant = 1), `vest` (Compliant = 2)

---

## 1. Executive Summary & Design Goals
The goal of Model 2 is to build a class-conditional generative model capable of:
1. Synthesizing high-fidelity $64\times 64$ PPE crops conditioned strictly on class $y \in \{\text{head}, \text{helmet}, \text{vest}\}$.
2. Establishing a continuous, structured, and disentangled latent manifold $z \sim \mathcal{N}(0, I_K)$ for smooth interpolation (slerp) and class morphing.
3. Enabling exact log-likelihood bounds (ELBO) and information-theoretic metrics (KL per dimension, bits/dim, active units) that plain Autoencoders cannot provide.

To prevent the well-documented failure modes of standard VAEs on small image crops, this document details each failure mode, its mathematical cause, and the exact architectural/objective solutions implemented.

---

## 2. Failure Modes and Systematic Mitigations

### 2.1 Loss Scaling & True ELBO Formulation
* **The Failure Mode**: A ubiquitous bug in VAE implementations is writing:
  ```python
  recon_loss = F.mse_loss(recon, x, reduction='mean') # averages over 3*64*64 = 12,288 pixels
  kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - exp(logvar), dim=1)) # summed over K=128
  ```
  By averaging over $D = C \times H \times W = 12,288$ pixels, the reconstruction term is implicitly divided by $12,288$ relative to the KL divergence. Consequently, the KL regularizer overpowers reconstruction by $\sim 10^4\times$, forcing $q(z|x) \to \mathcal{N}(0, I)$ immediately and causing complete posterior collapse.
* **The Mathematical Fix**:
  The Evidence Lower Bound (ELBO) is defined as:
  $$\log p_\theta(x|y) \ge \mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - \beta D_{KL}(q_\phi(z|x, y) \parallel p(z))$$
  Both the reconstruction log-likelihood and the KL divergence must be **summed across their respective feature dimensions per sample**, and only **averaged across the batch**:
  $$\text{Recon}_{\text{sum}} = \sum_{c=1}^3 \sum_{h=1}^{64} \sum_{w=1}^{64} (x_{c,h,w} - \hat{x}_{c,h,w})^2$$
  $$D_{KL,\text{sum}} = -\frac{1}{2} \sum_{j=1}^K \left(1 + \ln \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
  Under Gaussian observation likelihood $p(x|z, y) = \mathcal{N}(\hat{x}, \sigma^2 I)$ with calibrated decoder variance ($\sigma$-VAE):
  $$\mathcal{L}_{\text{ELBO}} = \frac{1}{2\sigma^2} \text{Recon}_{\text{sum}} + \frac{D}{2} \ln(2\pi \sigma^2) + \beta D_{KL,\text{sum}}$$
  where $\sigma^2$ is dynamically calibrated via running batch MSE: $\sigma^{*2} = \frac{1}{D}\text{Recon}_{\text{sum}}$.
  Fixed-variance MSE and L1 likelihoods are evaluated alongside as controlled ablation baselines.

---

### 2.2 Posterior Collapse & Inactive Units
* **The Failure Mode**: In early training iterations, the decoder generates uninformative outputs. The encoder can instantly minimize loss by setting $\mu \to 0$ and $\sigma \to 1$, completely dropping information from $x$. Once collapsed, gradients from the decoder vanish, trapping latent dimensions in an inactive state.
* **The Mathematical Fix**:
  1. **KL Annealing Warm-up**: Linearly ramp $\beta$ from $0.0$ to $\beta_{\text{target}}$ over the first $T_{\text{warmup}} = 15$ epochs:
     $$\beta(t) = \beta_{\text{target}} \cdot \min\left(1.0, \frac{t}{T_{\text{warmup}}}\right)$$
     This allows the encoder and decoder to establish strong reconstruction pathways before the Gaussian prior constraint is enforced.
  2. **Active Units Diagnostic ($A_u$)**:
     Following Burda et al. (2015), we track the variance of the posterior mean across the entire validation dataset for each latent unit $j \in \{1, \dots, K\}$:
     $$A_j = \text{Var}_{x \sim \mathcal{D}_{\text{val}}}[\mu_j(x)]$$
     A unit is defined as **active** if $A_j > 0.01$. The total active count $N_{\text{active}} = \sum_{j=1}^K \mathbf{1}(A_j > 0.01)$ is reported for every epoch and checkpoint.
  3. **Bits Per Dimension (BPD)**:
     $$\text{BPD} = \frac{\mathcal{L}_{\text{recon}} + D_{KL}}{D \cdot \ln 2}$$
     providing an exact, standardized measure of generative compression efficiency.

---

### 2.3 Condition Leakage and Verification
* **The Failure Mode**: In poorly designed conditional VAEs, the decoder learns to generate images solely from the latent code $z$, ignoring the class label $y$. Conversely, if $z$ captures class information redundantly, the condition is not disentangled from the continuous style variables.
* **The Architectural Fix**:
  1. **Encoder Conditioning**: Class index $y \in \{0, 1, 2\}$ is embedded into a learnable dense embedding vector $e_y \in \mathbb{R}^{16}$, spatially broadcasted to $16 \times 64 \times 64$, and concatenated directly to the RGB input tensor:
     $$x_{\text{in}} = [x_{\text{RGB}}, \text{Broadcast}(e_y, 64 \times 64)] \in \mathbb{R}^{19 \times 64 \times 64}$$
     The first convolution layer immediately binds spatial features with class identity.
  2. **Decoder Conditioning**: The class embedding $e_y \in \mathbb{R}^{32}$ is concatenated directly to the latent code $z \in \mathbb{R}^K$:
     $$z_{\text{in}} = [z, e_y] \in \mathbb{R}^{K + 32}$$
     ensuring that every stage of decompression is steered by the label.
  3. **Verification Protocol**:
     - **Class-Swap Test**: Pass a fixed sample $z \sim \mathcal{N}(0, I)$ through the decoder 3 times with $y \in \{0, 1, 2\}$. The underlying head pose, lighting, and camera angle must be preserved while helmet/head/vest visual cues switch cleanly.
     - **Judge Conditional Accuracy**: Generate $N=2,000$ crops per class from the prior, pass them to the independent Supervised Judge Classifier, and measure per-class recall and overall accuracy:
       $$\text{CondAcc} = \frac{1}{3 \times 2000} \sum_{c=0}^2 \sum_{i=1}^{2000} \mathbf{1}\left(\text{Judge}(G(z_i, c)) = c\right)$$
       A score near 33.3% indicates conditioning failure; a score $\ge 90\%$ confirms robust conditioning.

---

### 2.4 Blur Mitigation Without Breaking the ELBO
* **The Failure Mode**: Standard deconvolutional decoders using `ConvTranspose2d` introduce high-frequency checkerboard artifacts, while L2 pixel loss naturally penalizes high-frequency details, yielding blurry reconstructions. Adding an arbitrary LPIPS loss breaks the probabilistic ELBO interpretation because LPIPS is not a normalized log-likelihood density.
* **The Architectural Fix**:
  - **Upsample + Conv Blocks**: All decoder stages use bilinear/nearest upsampling followed by $3\times 3$ convolutions, completely eliminating checkerboard artifacts.
  - **Residual Refinement & GroupNorm**: Each resolution stage ($8\times 8 \to 16\times 16 \to 32\times 32 \to 64\times 64$) incorporates residual blocks stabilized with Group Normalization (`num_groups=8`), which does not suffer from batch-size instability.
  - **Ablation Transparency**: Pure ELBO models (Gaussian $\sigma$-VAE and MSE) are benchmarked rigorously. An optional LPIPS perceptual auxiliary loss is trained and reported strictly as a separate, clearly labelled ablation variant (`cvae_lpips_ablation`).

---

### 2.5 Prior-Hole Mismatch & Sampling Temperature
* **The Failure Mode**: The aggregate posterior $q(z) = \frac{1}{N}\sum_{i=1}^N q(z|x_i)$ does not uniformly fill the standard Gaussian prior $\mathcal{N}(0, I)$. In high dimensions ($K \ge 64$), sampling from the outer tails of $\mathcal{N}(0, I)$ hits low-density regions ("prior holes"), generating noisy or unconvincing artifacts.
* **The Mathematical Fix**:
  - **Empirical Moment Matching**: Measure empirical mean $\mathbb{E}[z]$ and covariance $\text{Cov}[z]$ of aggregate posterior to quantify deviation from $(0, I)$.
  - **Temperature-Controlled Sampling**: Introduce a sampling scale parameter $\tau \in [0.6, 1.0]$:
    $$z \sim \mathcal{N}(0, \tau^2 I)$$
    Lower temperatures ($\tau = 0.7-0.8$) concentrate probability mass inside high-density empirical regions, yielding crisp, artifact-free samples while preserving class characteristics.

---

### 2.6 Latent Manifold Interpolation: SLERP vs LERP
* **The Failure Mode**: Due to the Gaussian Annulus Theorem (geometry of high-dimensional Gaussians), random samples $z \sim \mathcal{N}(0, I_K)$ concentrate in a thin spherical shell of radius $r \approx \sqrt{K}$. Linear interpolation $\text{lerp}(z_1, z_2, \alpha) = (1 - \alpha)z_1 + \alpha z_2$ passes through the interior of the sphere at $\alpha = 0.5$, where Gaussian probability density is virtually zero, resulting in blurry, unnatural intermediate images.
* **The Mathematical Fix**:
  - **Spherical Linear Interpolation (SLERP)**:
    $$\text{slerp}(z_1, z_2, \alpha) = \frac{\sin((1 - \alpha)\Omega)}{\sin \Omega} z_1 + \frac{\sin(\alpha \Omega)}{\sin \Omega} z_2$$
    where $\Omega = \arccos\left(\frac{z_1 \cdot z_2}{\|z_1\|_2 \|z_2\|_2}\right)$. SLERP keeps all intermediate points on the high-probability manifold shell of radius $\sqrt{K}$.
  - **Class Morphing**: With fixed $z^*$, continuously interpolate the class conditioning embedding:
    $$e(\alpha) = (1 - \alpha) e_{y_1} + \alpha e_{y_2}$$
    and decode $G(z^*, e(\alpha))$ to smoothly transition compliance features (e.g., bare head gradually forming a hardhat).

---

## 3. Architecture Specification

```
[Input: Crop (3, 64, 64) + Class Label y in {0, 1, 2}]
                 │
  ┌──────────────┴────────────────┐
  │ Class Embedding: y -> (16,)   │
  │ Spatial Tile -> (16, 64, 64)   │
  └──────────────┬────────────────┘
                 ▼
     Concatenation -> (19, 64, 64)
                 │
      [ENCODER (SmallResNet-CVAE)]
  Stem Conv3x3 (19 -> 32, stride 1)
  Stage 1: ResBlock (32 -> 64, stride 2)   -> (64, 32, 32)
  Stage 2: ResBlock (64 -> 128, stride 2)  -> (128, 16, 16)
  Stage 3: ResBlock (128 -> 256, stride 2) -> (256, 8, 8)
  Global Adaptive Avg Pool + Flatten       -> (256,)
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  Linear (256 -> K)   Linear (256 -> K)
       mu                 logvar
       └─────────┬─────────┘
                 ▼
  Reparameterization: z = mu + exp(0.5 * logvar) * eps
                 │
                 ▼
      [DECODER (Upsample-CVAE)]
  Class Embedding: y -> (32,)
  Concatenate: [z, e_y] -> (K + 32,)
  Linear -> (256 * 8 * 8) -> Reshape (256, 8, 8)
  Stage 1: Upsample x2 + Conv3x3 + ResBlock -> (128, 16, 16)
  Stage 2: Upsample x2 + Conv3x3 + ResBlock -> (64, 32, 32)
  Stage 3: Upsample x2 + Conv3x3 + ResBlock -> (32, 64, 64)
  Final Conv3x3 (32 -> 3) + Sigmoid        -> (3, 64, 64)
```

---

## 4. Experimental Plan & Hyperparameter Sweep

### Controlled Sweep Grid (Evaluated strictly on VAL)
1. **Latent Dimensions**: $K \in \{64, 128, 256\}$
2. **Target Beta ($\beta$)**: $\beta \in \{0.5, 1.0, 2.0, 4.0\}$
3. **Loss Formulations**:
   - `sigma_vae` (calibrated Gaussian likelihood, primary model)
   - `naive_baseline` (pixel mean MSE, $\beta=1.0$, no annealing)
   - `fixed_mse` (summed pixel MSE with annealing)
   - `lpips_ablation` (calibrated ELBO + perceptual loss)

### Selection Criteria (Validation Set Only)
The winning configuration is selected strictly on the validation split by Pareto dominance across:
1. **Validation ELBO / Reconstruction PSNR & SSIM**
2. **Validation FID** (against real validation crops)
3. **Judge Conditional Accuracy** ($\ge 90\%$)
4. **Active Latent Units** ($N_{\text{active}} \ge 0.5 \times K$)
The holdout test split ($N=7,470$) is touched **once** only for the final benchmark report of the selected model.

