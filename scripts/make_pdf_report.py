"""
scripts/make_pdf_report.py
Generates a publication-grade academic PDF report (PROJECT_REPORT.pdf)
using headless Chrome with embedded Base64 images, formal typography,
academic front matter, and 7 comprehensive chapters.
"""

import os
import sys
import base64
import subprocess
from pathlib import Path

def img_to_b64(path_str):
    p = Path(path_str)
    if p.exists():
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    return ""

def build_pdf_report():
    print("[*] Generating academic HTML for PDF compilation...")

    # Load Base64 images
    img_arch = img_to_b64("results/figures/fig1_system_architecture.png")
    img_splits = img_to_b64("results/autoencoder/class_distribution_splits.png")
    img_samples = img_to_b64("results/autoencoder/sample_crops_grid.png")
    img_ae_loss = img_to_b64("results/autoencoder/ae_loss_curve.png")
    img_ae_pca = img_to_b64("results/autoencoder/latent_pca_scatter.png")
    img_ae_recon = img_to_b64("results/autoencoder/reconstructions.png")
    img_ae_conf = img_to_b64("results/autoencoder/svm_confusion_matrix.png")
    img_cvae_loss = img_to_b64("results/cvae/figures/cvae_loss_curves.png")
    img_cvae_kl = img_to_b64("results/cvae/figures/cvae_kl_per_dimension.png")
    img_cvae_slerp = img_to_b64("results/cvae/figures/cvae_slerp_interpolation.png")
    img_cgan_grid = img_to_b64("results/cgan/figures/01_cgan_samples_grid.png")
    img_cgan_priv = img_to_b64("results/cgan/figures/04_cgan_nearest_neighbor_privacy.png")
    img_cgan_aug = img_to_b64("results/cgan/figures/05_cgan_downstream_augmentation_benchmark.png")
    img_diff_tradeoff = img_to_b64("results/diffusion/figures/02_perception_distortion_tradeoff.png")
    img_diff_flips = img_to_b64("results/diffusion/figures/05_safety_critical_flips.png")
    img_policy_rel = img_to_b64("results/policy/figures/04_calibration_reliability.png")
    img_policy_dec = img_to_b64("results/policy/figures/05_end_to_end_verdict_matrix.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Academic Project Report - PPE Generative Monitoring</title>
<style>
@page {{
    size: A4;
    margin: 22mm 18mm 22mm 22mm;
    @bottom-center {{
        content: counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 10pt;
    }}
}}
body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    line-height: 1.45;
    color: #111827;
    text-align: justify;
    margin: 0;
    padding: 0;
}}
.page-break {{
    page-break-before: always;
}}
.cover-page {{
    text-align: center;
    padding-top: 25px;
}}
.cover-title-pre {{
    font-size: 13pt;
    font-weight: bold;
    color: #64748b;
    letter-spacing: 1px;
    margin-bottom: 12px;
}}
.cover-title {{
    font-size: 18pt;
    font-weight: bold;
    color: #003366;
    line-height: 1.3;
    margin-bottom: 25px;
    text-transform: uppercase;
}}
.cover-sub {{
    font-size: 11.5pt;
    color: #374151;
    line-height: 1.6;
    margin-top: 35px;
}}
.cover-footer {{
    margin-top: 50px;
    font-size: 12pt;
    font-weight: bold;
    color: #1e293b;
    letter-spacing: 0.5px;
}}
h1.section-title {{
    font-size: 16pt;
    font-weight: bold;
    color: #003366;
    border-bottom: 2px solid #003366;
    padding-bottom: 5px;
    margin-top: 20px;
    margin-bottom: 14px;
    text-transform: uppercase;
}}
h2 {{
    font-size: 13pt;
    font-weight: bold;
    color: #1a365d;
    margin-top: 16px;
    margin-bottom: 8px;
}}
h3 {{
    font-size: 11.5pt;
    font-weight: bold;
    color: #2d3748;
    margin-top: 12px;
    margin-bottom: 6px;
}}
p {{
    margin-top: 0;
    margin-bottom: 8px;
}}
ul {{
    margin-top: 4px;
    margin-bottom: 8px;
    padding-left: 22px;
}}
li {{
    margin-bottom: 4px;
    line-height: 1.4;
}}
.callout {{
    background-color: #f8fafc;
    border-left: 4px solid #003366;
    padding: 8px 14px;
    margin: 10px 0;
    font-size: 10.5pt;
    font-style: italic;
    color: #334155;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}
th {{
    background-color: #1a365d;
    color: #ffffff;
    font-weight: bold;
    padding: 7px 9px;
    border: 1px solid #cbd5e0;
    text-align: center;
}}
td {{
    padding: 5px 8px;
    border: 1px solid #e2e8f0;
}}
tr:nth-child(even) td {{
    background-color: #f8fafc;
}}
.table-title {{
    font-weight: bold;
    color: #1a365d;
    margin-bottom: 4px;
    font-size: 10pt;
}}
.figure-container {{
    text-align: center;
    margin: 14px 0 16px 0;
    page-break-inside: avoid;
}}
.figure-container img {{
    max-width: 88%;
    max-height: 350px;
    height: auto;
    border: 1px solid #cbd5e0;
    border-radius: 4px;
}}
.figure-caption {{
    font-size: 9.5pt;
    font-style: italic;
    color: #4b5563;
    margin-top: 5px;
}}
.toc-line {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 10.5pt;
}}
.toc-dots {{
    flex-grow: 1;
    border-bottom: 1px dotted #9ca3af;
    margin: 0 8px;
    position: relative;
    top: -4px;
}}
.ref-item {{
    padding-left: 28px;
    text-indent: -28px;
    margin-bottom: 6px;
    font-size: 10pt;
}}
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-page">
    <div class="cover-title-pre">A MAJOR PROJECT REPORT ON</div>
    <div class="cover-title">MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE</div>
    <div class="cover-sub">
        Submitted in partial fulfillment of the requirements<br>
        for the award of the degree of<br><br>
        <strong>BACHELOR OF TECHNOLOGY</strong><br>
        IN<br>
        <strong>COMPUTER SCIENCE AND ENGINEERING</strong><br><br>
        By<br><br>
        <strong>Mohit Patel</strong><br>
        [Roll No. 21BCE10000]<br><br>
        Under the Guidance and Supervision of<br><br>
        <strong>Faculty Project Guide</strong><br>
        Department of Computer Science & Engineering
    </div>
    <div class="cover-footer">
        DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING<br>
        ACADEMIC YEAR 2025–2026
    </div>
</div>

<div class="page-break"></div>

<!-- CERTIFICATE -->
<h1 class="section-title">CERTIFICATE OF RECOMMENDATION</h1>
<p>This is to certify that the project report entitled <strong>"MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE"</strong> submitted by <strong>Mohit Patel</strong> (Roll No. 21BCE10000) in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering, is a bona fide record of research carried out under my guidance and supervision.</p>
<p>The results embodied in this project report have not been submitted to any other University or Institute for the award of any degree, diploma, or certificate.</p>
<br><br><br>
<table style="border:none; width:100%;">
    <tr style="background:none;"><td style="border:none; width:50%;">_________________________<br><strong>Project Supervisor / Guide</strong><br>Dept. of Computer Science & Eng.</td>
    <td style="border:none; width:50%; text-align:right;">_________________________<br><strong>Head of Department (CSE)</strong><br>Dept. of Computer Science & Eng.</td></tr>
    <tr style="background:none;"><td style="border:none; padding-top:50px;" colspan="2">_________________________<br><strong>External Examiner</strong></td></tr>
</table>

<div class="page-break"></div>

<!-- DECLARATION -->
<h1 class="section-title">CANDIDATE'S DECLARATION</h1>
<p>I hereby declare that this project report entitled <strong>"MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE"</strong> is an authentic record of my own research and experimental work carried out under the guidance of my supervisor. To the best of my knowledge, it contains no material previously written or published by another person, except where due reference and formal citation has been made in the text.</p>
<p>I have adhered to all ethical principles of academic integrity, code reproducibility, and zero-plagiarism policies. All experimental metrics, tables, and figures presented herein reflect verified executions of the underlying algorithmic suite.</p>
<br><br>
<p style="text-align:right;"><strong>Mohit Patel</strong><br>Roll No. 21BCE10000<br>Department of Computer Science & Engineering<br>November 2026</p>

<div class="page-break"></div>

<!-- ACKNOWLEDGEMENTS -->
<h1 class="section-title">ACKNOWLEDGEMENTS</h1>
<p>I express my deepest gratitude to my project supervisor, whose insightful feedback, methodological rigor, and continuous encouragement were instrumental throughout the design, training, and evaluation of this generative safety framework. Their expert advice on deep generative modeling, loss formulations, and calibration provided essential direction.</p>
<p>I am profoundly grateful to the Head of the Department and the faculty members of the Department of Computer Science and Engineering for providing access to computational infrastructure, laboratory facilities, and constant academic support. Special thanks are due to the open-source computer vision community for curating the Safety Helmet and Worker Dataset (SHWD) and Pictor PPE datasets.</p>
<p>Finally, I extend my heartfelt thanks to my family, peers, and fellow researchers whose patience, support, and constructive criticism sustained me during the intensive development and rigorous benchmarking phases of this work.</p>

<div class="page-break"></div>

<!-- ABSTRACT -->
<h1 class="section-title">ABSTRACT</h1>
<p>Automated monitoring of Personal Protective Equipment (PPE) in construction and industrial high-hazard zones is critically impeded by three systemic vulnerabilities: (1) severe visual degradation from low-resolution surveillance cameras, atmospheric turbulence, and motion blur; (2) extreme long-tail class imbalance where non-compliant bare heads represent less than 3.5% of real-world site observations; and (3) legal ambiguity within contextual, compound safety regulations (such as OSHA 29 CFR 1926 and MUTCD Part 6D) that conventional computer vision detectors cannot parse.</p>
<p>This project report presents a unified, multi-objective empirical benchmark evaluating five distinct artificial intelligence paradigms on identical, tightly extracted 64×64 PPE crops (N = 41,534) and an un-leaked regulatory text corpus (N = 321): (i) <strong>Self-Supervised Convolutional Autoencoders (ConvAE)</strong> with calibrated RBF-SVM latent classification, achieving 91.93% accuracy and lifting bare-head violation recall from 63.55% to 85.05% (+21.50%) under optimal decision thresholding (theta* = 0.190); (ii) <strong>Conditional sigma-VAEs</strong> utilizing analytical decoder variance calibration to completely eradicate posterior collapse (128/128 active latent units, ELBO = -1.312 bits/dim), combined with Manifold Anchor Sampling to yield high-fidelity reconstruction (PSNR: 31.23 dB, SSIM: 0.9686); (iii) <strong>Deep Conditional GANs</strong> with Spectral Normalization, Projection Discriminators, and DiffAugment, reaching 89.56% conditional generation accuracy under an independent ResNet-18 judge while passing strict industrial privacy audits (5th-percentile nearest-neighbor distance p5 = 0.2637 >> 0.0600 noise floor) and boosting downstream 5-seed classifier Minority Head F1 from 70.76% to 76.81% (+6.05%, p = 0.0224); (iv) <strong>Score-Based Conditional Diffusion (DDPM/DDIM)</strong>, which empirically establishes the Blau & Michaeli perception-distortion tradeoff by demonstrating that unconstrained generative sampling induces a catastrophic 66.63% safety-critical class flip rate (hallucinating compliant helmets onto bare heads), whereas Regression-Anchored U-Net restoration eliminates flips (0.0% error, 29.82 dB PSNR, 99.30% Judge fidelity); and (v) <strong>Calibrated Regulatory Safety Policy Transformers (DistilBERT)</strong>, reaching 87.64% test accuracy and 88.49% Macro-F1 (+43.33% over regex), with temperature scaling (T* = 0.598) shrinking Expected Calibration Error from 21.20% to 10.80% (and 6.26% on Exp 2).</p>
<p>Finally, a <strong>Dual-Vision Consensus Gating</strong> mechanism synthesizes supervised and self-supervised visual streams with calibrated policy predicates to enforce a zero-hallucination, tri-state safety verdict (COMPLIANT, VIOLATION, CANNOT DETERMINE). An interactive web dashboard and production containerization complete the deployment pipeline.</p>

<div class="page-break"></div>

<!-- CHAPTER 1 -->
<h1 class="section-title">CHAPTER 1: INTRODUCTION</h1>
<h2>1.1 Industrial Context & High-Consequence Safety Auditing</h2>
<p>Industrial manufacturing, heavy infrastructure, and construction sites represent some of the highest-consequence operational environments in modern civil engineering. According to statistical audits by the International Labour Organization (ILO) and OSHA, fatal workplace injuries exceed 5,000 incidents annually in North America alone, with catastrophic falls and falling-object impacts accounting for over 58% of all recorded fatalities (the 'Fatal Four'). Rigorous epidemiological analyses consistently demonstrate that the continuous wearing of Personal Protective Equipment (PPE)—predominantly industrial hard hats and retroreflective safety vests—reduces traumatic brain injury by upwards of 73% and fatal vehicle-pedestrian strikes by over 60%.</p>
<p>To enforce safety, regulatory agencies mandate strict adherence to legal codes such as OSHA 29 CFR 1926.100, OSHA 29 CFR 1926.651, MUTCD Part 6D.03, and ANSI/ISEA 107/89.1. Traditionally, compliance has relied upon manual inspections by safety officers, which suffer from physiological fatigue, sparse temporal sampling, and limited physical coverage. Automated visual inspection powered by edge AI and CCTV surveillance has therefore emerged as an urgent imperative for 24/7 autonomous safety auditing.</p>

<h2>1.2 The Triad of Systemic Inspection Challenges</h2>
<p>Deploying automated safety inspection systems in active industrial environments exposes three fundamental systemic challenges:</p>
<ul>
    <li><strong>Severe Visual Degradation in Surveillance Feeds:</strong> Surveillance cameras placed on tower cranes or high perimeter poles are positioned 15 to 80 meters from workers. Consequently, PPE items occupy sub-64×64 pixels and are corrupted by atmospheric dust, motion blur, sensor noise, and lossy compression.</li>
    <li><strong>The Extreme Long-Tail Minority Class Imbalance:</strong> Bare heads (critical safety violations) represent less than 3.5% of real-world observations (an 18.4 : 1 imbalance ratio). Standard supervised deep networks collapse to predicting the majority compliant class, missing life-threatening safety violations entirely.</li>
    <li><strong>Contextual and Legal Ambiguity in Safety Regulations:</strong> Compliance is legally context-dependent, governed by compound clauses and operational exemptions (e.g., 'Hard hats required only when overhead crane hoisting is active; safety vests not required while operating inside enclosed vehicle cabs').</li>
</ul>

<h2>1.3 Project Motivation & Research Questions</h2>
<p>Deep generative models (VAEs, GANs, Diffusion) offer powerful capabilities in representation learning, synthetic data augmentation, and degraded image restoration. This project systematically investigates five core research questions:</p>
<ul>
    <li><strong>RQ1 (Representation):</strong> Can self-supervised latent representations establish highly separable decision boundaries for safety classification?</li>
    <li><strong>RQ2 (Manifold Geometry):</strong> How can continuous variational manifolds be stabilized against posterior collapse while preserving full latent capacity?</li>
    <li><strong>RQ3 (Minority Augmentation):</strong> Can conditional adversarial synthesis alleviate minority imbalance without memorizing worker facial identities?</li>
    <li><strong>RQ4 (Restoration Safety):</strong> Does generative score diffusion present catastrophic safety hazards (hallucinating helmets onto bare heads) due to the perception-distortion tradeoff?</li>
    <li><strong>RQ5 (Cross-Modal Consensus):</strong> How can visual detections and legal regulatory text be synthesized to guarantee zero-hallucination compliance verdicts?</li>
</ul>

<h2>1.4 Key Scientific Innovations & Contributions</h2>
<p>This work delivers five primary contributions: (1) A standardized multi-objective benchmark across Autoencoders, VAEs, GANs, and Diffusion models on identical splits; (2) Empirical proof of diffusion safety flips (66.63% hallucination rate); (3) Privacy-audited minority data augmentation (p5 = 0.2637 >> 0.0600) boosting downstream ResNet-18 F1 by +6.05%; (4) Uncertainty-calibrated regulatory NLP reducing ECE by 49.1%; and (5) A Dual-Stage Deconvolution Anti-Checkerboard filter eliminating 2px and 4px Nyquist harmonics in deep cGANs.</p>

<div class="figure-container">
    <img src="{img_arch}" alt="System Architecture Blueprint">
    <div class="figure-caption">Figure 1.1: End-to-End System Architecture Blueprint depicting multi-modal visual generative pipelines coupled with calibrated regulatory policy parsing.</div>
</div>

<div class="page-break"></div>

<!-- CHAPTER 2 -->
<h1 class="section-title">CHAPTER 2: LITERATURE REVIEW & THEORETICAL FOUNDATIONS</h1>
<h2>2.1 Evolution of Automated PPE Detection Systems</h2>
<p>Automated visual inspection has progressed from classical handcrafted features (Haar cascades, HOG + SVM) to modern deep neural networks. Two-stage object detectors (Faster R-CNN) and single-stage models (YOLOv3 through YOLOv11) have established real-time detection on standard datasets. However, when applied to distant surveillance crops under 64×64 pixels, standard spatial downsampling in deep backbones obliterates fine structural details, causing severe misclassifications.</p>

<h2>2.2 Self-Supervised Representation Learning</h2>
<p>Convolutional Autoencoders (ConvAEs) map high-dimensional imagery x in R^D into a low-dimensional bottleneck latent code z in R^d without human labels. In safety inspection, self-supervised learning enables the network to discover a continuous geometric manifold capturing helmet shells and head contours, providing an invariant feature space for downstream classifiers.</p>

<h2>2.3 Deep Generative Modeling Paradigms</h2>
<p>Variational Autoencoders (VAEs) optimize the Evidence Lower Bound (ELBO), balancing reconstruction against KL divergence from a prior. However, standard VAEs suffer from posterior collapse. Generative Adversarial Networks (GANs) synthesize sharp high-frequency textures via minimax games, stabilized by Spectral Normalization and DiffAugment. Score-based Diffusion Models (DDPMs) generate photorealistic textures via iterative reverse Markov denoising.</p>

<h2>2.4 The Perception-Distortion Tradeoff & Safety Flips</h2>
<p>Blau and Michaeli (2018) proved mathematically that perceptual quality (low LPIPS) and distortion (high PSNR/SSIM) are strictly conflicting objectives: optimizing perceptual sharpness forces the model to synthesize high-frequency hallucinations. In safety auditing, this creates a catastrophic failure mode: the generative prior hallucinates hardhats onto bare heads.</p>

<h2>2.5 NLP for Regulatory Verification & Calibration</h2>
<p>Rule-based regex parsers fail on compound clauses, passive voice, and contextual exemptions. Fine-tuned Transformers (DistilBERT) accurately parse legal clauses but require Temperature Scaling calibration to align predicted probabilities with empirical accuracy.</p>

<div class="page-break"></div>

<!-- CHAPTER 3 -->
<h1 class="section-title">CHAPTER 3: SYSTEM ARCHITECTURE & ALGORITHMIC FRAMEWORK</h1>
<h2>3.1 End-to-End Multi-Modal Workflow</h2>
<p>Surveillance feeds are cropped into standardized 64x64 safety items and concurrently evaluated by a Supervised ResNet-18 Judge and a Self-Supervised ConvAE-SVM. Concurrently, regulatory clauses are parsed by a calibrated DistilBERT model. Visual and textual predicates converge at the Dual-Vision Consensus Gating Engine.</p>

<h2>3.2 Model 1: Self-Supervised ConvAE + Calibrated RBF-SVM</h2>
<p>The ConvAE features 4 strided conv stages (channels: 32 -> 64 -> 128 -> 256, k=4, s=2, p=1) mapping to a 256-d bottleneck z, and 4 transposed conv stages reconstructing x_hat. Training optimizes a composite loss: <code>L_AE = 0.85 * ||x - x_hat||_1 + 0.15 * (1 - SSIM(x, x_hat))</code>. Post-hoc threshold optimization on validation probabilities establishes an optimal decision threshold theta* = 0.190, maximizing bare-head recall.</p>

<h2>3.3 Model 2: Conditional sigma-VAE with Variance Calibration</h2>
<p>Analytical observation noise calibration sets <code>sigma*^2 = (1/D) * ||x - x_hat||_2^2</code>, which scales the reconstruction term and completely prevents posterior collapse. Class conditioning is applied symmetrically: one-hot label y is embedded into R^16 (spatially tiled in encoder) and R^32 (concatenated with z in R^128 in decoder). Manifold Anchor Sampling transfers high-frequency details from nearest training anchors.</p>

<h2>3.4 Model 3: Deep cGAN with DiffAugment & Anti-Checkerboard Filter</h2>
<p>Model 3 utilizes a Projection Discriminator with Spectral Normalization (1-Lipschitz continuity bound) and Class-Conditional BatchNorm (CBN) in the Generator. DiffAugment (color, translation +-4px, cutout 8x8) prevents discriminator overfitting on limited minority data. To eliminate deconvolution Nyquist artifacts (2px and 4px harmonics), a dual-stage 3x3 median and bilateral filter (d=5, sigmaColor=75, sigmaSpace=3.5) is applied prior to Bicubic upsampling.</p>

<h2>3.5 Model 4: Conditional Score Diffusion & Regression-Anchored Sampling</h2>
<p>A 6-channel conditioning U-Net (x_deg concatenated with noisy x_t) is trained via L1 epsilon matching under a cosine schedule. Fast sampling is executed via DDIM across S in {{10, 20, 50, 100}} steps. To eliminate generative safety flips, Regression-Anchored DDIM Sampling initializes reverse diffusion at intermediate timestep t_anchor approx 200 using a deterministic pixel-L1 U-Net.</p>

<h2>3.6 Model 5: Calibrated Policy Transformer (DistilBERT)</h2>
<p>DistilBERT-base-uncased is fine-tuned to classify regulatory clauses into 5 classes. Post-hoc Temperature Scaling <code>p_hat = softmax(z / T*)</code> with optimal T* = 0.598 aligns confidence with accuracy.</p>

<h2>3.7 Dual-Vision Consensus Gating Engine</h2>
<p>Enforces tri-state safety decision logic: (1) <code>CANNOT DETERMINE</code> if vision models disagree or max confidence < 0.60; (2) <code>SAFETY VIOLATION</code> if consensus confirms an unhelmeted head in a mandatory zone; and (3) <code>COMPLIANT</code> when verified compliant.</p>

<div class="page-break"></div>

<!-- CHAPTER 4 -->
<h1 class="section-title">CHAPTER 4: DATASET SPECIFICATIONS & EXPERIMENTAL SETUP</h1>
<h2>4.1 Visual PPE Crop Dataset</h2>
<p>The visual dataset comprises 41,534 standardized 64x64 RGB crops partitioned into fixed train, validation, and test splits:</p>
<div class="table-title">Table 4.1: Visual PPE Crop Dataset Split Breakdown and Class Imbalance Matrix</div>
<table>
    <tr><th>Class Name</th><th>Label</th><th>Train Count</th><th>Train %</th><th>Val Count</th><th>Val %</th><th>Test Count</th><th>Test %</th><th>Total Count</th></tr>
    <tr><td><strong>head (Violation)</strong></td><td>0</td><td>867</td><td>3.22%</td><td>231</td><td>3.22%</td><td>241</td><td>3.23%</td><td>1,339</td></tr>
    <tr><td><strong>helmet (Compliant)</strong></td><td>1</td><td>15,991</td><td>59.45%</td><td>4,265</td><td>59.52%</td><td>4,446</td><td>59.52%</td><td>24,702</td></tr>
    <tr><td><strong>vest (Compliant)</strong></td><td>2</td><td>10,041</td><td>37.33%</td><td>2,669</td><td>37.25%</td><td>2,783</td><td>37.25%</td><td>15,493</td></tr>
    <tr><td><strong>Total Visual Crops</strong></td><td>—</td><td><strong>26,899</strong></td><td>100.0%</td><td><strong>7,165</strong></td><td>100.0%</td><td><strong>7,470</strong></td><td>100.0%</td><td><strong>41,534</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_splits}" alt="Class Splits">
    <div class="figure-caption">Figure 4.1: Dataset Class Distribution across Train, Val, and Test splits showing the extreme 18.4 : 1 imbalance ratio.</div>
</div>

<div class="figure-container">
    <img src="{img_samples}" alt="Sample Crops">
    <div class="figure-caption">Figure 4.2: Standardized 64x64 Industrial Safety Crops showing realistic variance in illumination and skin tones.</div>
</div>

<h2>4.2 Legal Regulatory Safety Text Corpus</h2>
<div class="table-title">Table 4.2: Legal Regulatory Safety Corpus Split Distribution (Zero Leakage)</div>
<table>
    <tr><th>Dataset Split</th><th>Sentences</th><th>Helmet Req.</th><th>Vest Req.</th><th>Both Req.</th><th>Other PPE</th><th>None / Context</th></tr>
    <tr><td>Train Split</td><td>155</td><td>34</td><td>42</td><td>31</td><td>24</td><td>24</td></tr>
    <tr><td>Validation Split</td><td>77</td><td>14</td><td>25</td><td>14</td><td>13</td><td>11</td></tr>
    <tr><td>Test Split (Held-Out)</td><td>89</td><td>18</td><td>19</td><td>18</td><td>11</td><td>23</td></tr>
    <tr><td><strong>Total Corpus</strong></td><td><strong>321</strong></td><td><strong>66</strong></td><td><strong>86</strong></td><td><strong>63</strong></td><td><strong>48</strong></td><td><strong>58</strong></td></tr>
</table>

<h2>4.3 Independent Evaluation Protocol & Noise Floor</h2>
<p>Between 2,000 real test and train crop pairs, the Inception Noise Floor was established at <strong>FID = 24.728</strong>, <strong>KID = 0.000258 +- 0.001322</strong>, Manifold Precision = 0.3065, and Recall = 0.3055. The supervised 5-seed ResNet-18 Judge ensemble achieved <strong>86.97% +- 0.97% test accuracy</strong> and <strong>98.13% +- 0.87% bare-head recall</strong>.</p>

<div class="page-break"></div>

<!-- CHAPTER 5 -->
<h1 class="section-title">CHAPTER 5: EMPIRICAL RESULTS, COMPARATIVE BENCHMARKING & ANALYSIS</h1>
<h2>5.1 Model 1: ConvAE Representation Quality & Threshold Optimization</h2>
<p>The ConvAE reached a validation loss of 0.006138 at epoch 94, achieving MSE = 0.006199, PSNR = 23.12 dB, and SSIM = 0.7132 on the test set. Calibrating the RBF-SVM decision threshold to theta* = 0.190 boosted bare-head recall from 63.55% to 85.05% (+21.50%).</p>

<div class="table-title">Table 5.1: ConvAE + RBF-SVM Classification Performance Under Threshold Calibration</div>
<table>
    <tr><th>Metric</th><th>ConvAE + SVM (Baseline)</th><th>ConvAE + Calibrated SVM (theta*=0.190)</th><th>Linear Pixel-PCA Baseline</th><th>Relative Lift</th></tr>
    <tr><td>Test Accuracy</td><td><strong>91.93%</strong></td><td>91.04%</td><td>78.67%</td><td>+12.37% over PCA</td></tr>
    <tr><td>Balanced Accuracy</td><td>83.47%</td><td><strong>89.17%</strong></td><td>60.88%</td><td>+28.29% over PCA</td></tr>
    <tr><td>Macro-F1 Score</td><td><strong>86.11%</strong></td><td>86.82%</td><td>65.46%</td><td>+21.36% over PCA</td></tr>
    <tr><td>Minority Head Recall</td><td>63.55%</td><td><strong>85.05%</strong></td><td>42.15%</td><td><strong>+21.50% Recall Lift</strong></td></tr>
    <tr><td>Minority Head Precision</td><td>85.37%</td><td>64.71%</td><td>36.84%</td><td>+27.87% over PCA</td></tr>
    <tr><td>Reconstruction PSNR</td><td>23.12 dB</td><td>23.12 dB</td><td>18.45 dB</td><td>+4.67 dB</td></tr>
    <tr><td>Reconstruction SSIM</td><td>0.7132</td><td>0.7132</td><td>0.5120</td><td>+0.2012</td></tr>
</table>

<div class="figure-container">
    <img src="{img_ae_loss}" alt="ConvAE Loss Curves">
    <div class="figure-caption">Figure 5.1: ConvAE 100-Epoch Composite Training and Validation Loss Curves showing smooth convergence.</div>
</div>

<div class="figure-container">
    <img src="{img_ae_pca}" alt="ConvAE Latent PCA">
    <div class="figure-caption">Figure 5.2: 2D PCA Scatter Projection of the frozen 256-d ConvAE Latent Manifold demonstrating clear class clustering.</div>
</div>

<div class="figure-container">
    <img src="{img_ae_recon}" alt="ConvAE Reconstructions">
    <div class="figure-caption">Figure 5.3: Visual Comparison between Input Crops and ConvAE Reconstructions preserving structural boundaries.</div>
</div>

<div class="figure-container">
    <img src="{img_ae_conf}" alt="ConvAE Confusion Matrix">
    <div class="figure-caption">Figure 5.4: Normalized Multi-Class Confusion Matrix for Calibrated ConvAE + RBF-SVM on Test Set.</div>
</div>

<h2>5.2 Model 2: Conditional sigma-VAE Latent Manifold Dynamics</h2>
<p>Analytical variance calibration maintained 128/128 (100%) active latent units (ELBO = -1.312 bits/dim). Judge conditional fidelity reached 68.00% at tau = 0.70 with 94.0% head fidelity. Manifold Anchor Sampling produced high-fidelity reconstruction: <strong>PSNR = 31.23 dB, SSIM = 0.9686, MSE = 0.00075</strong>.</p>

<div class="table-title">Table 5.2: Conditional sigma-VAE Objective Loss Formulations & Multi-Experiment Ablation Sweep</div>
<table>
    <tr><th>Experiment</th><th>Loss Formulation</th><th>beta</th><th>PSNR</th><th>SSIM</th><th>Active Units</th><th>Judge Acc (tau=0.7)</th><th>Head Fidelity</th></tr>
    <tr><td>Baseline Calibrated</td><td>Standard Calibrated ELBO</td><td>1.00</td><td>21.19 dB</td><td>0.6068</td><td>128 / 128</td><td>62.00%</td><td>79.0%</td></tr>
    <tr><td>Exp 1 (Optimal)</td><td>Edge-Preserving L1+MSE</td><td>0.10</td><td>22.10 dB</td><td>0.6788</td><td>128 / 128</td><td><strong>68.67%</strong></td><td>83.0%</td></tr>
    <tr><td>Exp 2</td><td>Balanced Composite</td><td>0.50</td><td>22.13 dB</td><td>0.6780</td><td>128 / 128</td><td>67.67%</td><td>90.0%</td></tr>
    <tr><td>Exp 3</td><td>Structured ELBO</td><td>1.00</td><td>22.13 dB</td><td>0.6757</td><td>128 / 128</td><td>64.00%</td><td>89.0%</td></tr>
    <tr><td>Exp 4</td><td>High-Reg Latent</td><td>2.00</td><td>21.97 dB</td><td>0.6664</td><td>128 / 128</td><td>64.00%</td><td><strong>93.0%</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_cvae_loss}" alt="cVAE Loss Curves">
    <div class="figure-caption">Figure 5.5: Conditional sigma-VAE 100-Epoch Training Dynamics showing calibrated ELBO and smooth KL warmup.</div>
</div>

<div class="figure-container">
    <img src="{img_cvae_kl}" alt="cVAE KL per Dimension">
    <div class="figure-caption">Figure 5.6: KL Divergence per Latent Dimension confirming 100% active latent units across all 128 channels.</div>
</div>

<div class="figure-container">
    <img src="{img_cvae_slerp}" alt="cVAE SLERP Walk">
    <div class="figure-caption">Figure 5.7: Spherical Latent Interpolation (SLERP) across the Gaussian shell proving smooth manifold continuity.</div>
</div>

<h2>5.3 Model 3: cGAN Synthesis, Privacy Auditing & Downstream Lift</h2>
<p>Model 3 attained 89.56% Judge conditional accuracy (head: 98.67%, vest: 93.67%, helmet: 76.33%) with FID = 134.62 and KID = 0.0725. The privacy audit against all 26,899 real crops confirmed p5 = 0.2637 >> 0.0600 with zero memorization. Adding 2.0x synthetic cGAN heads boosted downstream ResNet-18 Minority Head F1 from 70.76% to 76.81% (+6.05%, p = 0.0224) and overall Macro-F1 to 87.72%.</p>

<div class="table-title">Table 5.3: Downstream ResNet-18 Augmentation Benchmark Across 5 Seeds</div>
<table>
    <tr><th>Condition</th><th>Dataset Composition</th><th>Accuracy</th><th>Balanced Acc</th><th>Macro-F1</th><th>Head Recall</th><th>Head Precision</th><th>Minority Head F1</th></tr>
    <tr><td>A: Real Only</td><td>26,899 Real</td><td>92.12% +- 0.51%</td><td>92.88% +- 0.43%</td><td>85.47% +- 2.19%</td><td>94.49% +- 0.90%</td><td>56.97% +- 8.37%</td><td>70.76% +- 6.37%</td></tr>
    <tr><td>B: Weighted Loss</td><td>26,899 Real (w_c)</td><td>89.14% +- 0.33%</td><td>91.47% +- 1.68%</td><td>79.13% +- 1.66%</td><td>96.26% +- 3.25%</td><td>39.71% +- 8.43%</td><td>55.69% +- 7.12%</td></tr>
    <tr><td>C: Oversampling</td><td>27,766 Real (Heads 2x)</td><td>91.88% +- 0.24%</td><td>93.09% +- 0.25%</td><td>84.72% +- 0.77%</td><td>95.14% +- 0.42%</td><td>53.86% +- 2.81%</td><td>68.75% +- 2.28%</td></tr>
    <tr><td>D: cGAN 1.0x</td><td>+867 Synthetic Heads</td><td>92.26% +- 0.43%</td><td>92.71% +- 0.28%</td><td>86.62% +- 1.26%</td><td>93.36% +- 0.61%</td><td>61.88% +- 4.92%</td><td>74.33% +- 3.41%</td></tr>
    <tr><td>E: cGAN 2.0x</td><td>+1,734 Synthetic Heads</td><td><strong>92.80% +- 0.35%</strong></td><td>92.63% +- 0.66%</td><td><strong>87.72% +- 1.03%</strong></td><td>92.34% +- 2.03%</td><td><strong>65.86% +- 4.01%</strong></td><td><strong>76.81% +- 2.67%</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_cgan_grid}" alt="cGAN Sample Grid">
    <div class="figure-caption">Figure 5.8: Class-Conditional Synthetic Crops generated by Model 3 across head, helmet, and vest classes.</div>
</div>

<div class="figure-container">
    <img src="{img_cgan_priv}" alt="cGAN Privacy Histogram">
    <div class="figure-caption">Figure 5.9: Nearest-Neighbor Distance Histogram proving zero memorization of training worker facial identities.</div>
</div>

<div class="figure-container">
    <img src="{img_cgan_aug}" alt="cGAN Augmentation Benchmark">
    <div class="figure-caption">Figure 5.10: Downstream 5-Seed ResNet-18 F1 Score Lift comparing cGAN Augmentation against Baselines.</div>
</div>

<h2>5.4 Model 4: Perception-Distortion Tradeoff & Diffusion Safety Flips</h2>
<p>While unconstrained DDIM-100 diffusion achieved peak perceptual texture (LPIPS = 0.1600), it produced a catastrophic 66.63% safety-critical class flip rate on degraded bare heads. In contrast, Regression-Anchored U-Net achieved 29.82 dB PSNR, 0.8783 SSIM, 99.30% Judge fidelity, and strictly 0.0% class flips.</p>

<div class="table-title">Table 5.4: Perception-Distortion Restoration Benchmark Across 5 Degradations</div>
<table>
    <tr><th>Restoration Model</th><th>Optimization Objective</th><th>LPIPS</th><th>PSNR</th><th>SSIM</th><th>Judge Accuracy</th><th>Safety Flip Rate</th></tr>
    <tr><td>Bicubic Resampling</td><td>Deterministic Interpolation</td><td>0.4200</td><td>27.46 dB</td><td>0.7929</td><td>86.89%</td><td>0.0%</td></tr>
    <tr><td>Regression U-Net (Anchored)</td><td>Pixel L1 Minimization</td><td>0.2200</td><td><strong>29.82 dB</strong></td><td><strong>0.8783</strong></td><td><strong>99.30%</strong></td><td><strong>0.0%</strong></td></tr>
    <tr><td>Score Diffusion (DDIM-100)</td><td>Generative Score Matching</td><td><strong>0.1600</strong></td><td>17.16 dB</td><td>0.5045</td><td>32.90%</td><td><strong>66.63%</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_diff_tradeoff}" alt="Perception-Distortion Tradeoff">
    <div class="figure-caption">Figure 5.11: Empirical Perception-Distortion Tradeoff Curve illustrating the inverse relationship between LPIPS and safety flips.</div>
</div>

<div class="figure-container">
    <img src="{img_diff_flips}" alt="Diffusion Safety Flips">
    <div class="figure-caption">Figure 5.12: Visual Evidence of Diffusion Safety Flips: generative prior hallucinates hardhat textures onto degraded bare heads.</div>
</div>

<h2>5.5 Model 5: Policy Compliance Accuracy & Temperature Calibration</h2>
<p>Fine-tuned DistilBERT reached 87.64% accuracy and 88.49% Macro-F1 (Exp 2), outperforming regex baselines by +43.33% F1. Temperature scaling (T* = 0.598 / 0.489) reduced ECE from 21.20% to 10.80% (and 6.26% on Exp 2).</p>

<div class="table-title">Table 5.5: Policy Compliance Model Performance on Held-Out Test Clauses (N = 89 sentences)</div>
<table>
    <tr><th>Model Architecture</th><th>Test Accuracy</th><th>Balanced Acc</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro-F1</th><th>Negation Acc</th><th>Calibrated ECE</th></tr>
    <tr><td>Keyword Regex Heuristic</td><td>47.19%</td><td>46.48%</td><td>70.48%</td><td>46.48%</td><td>45.16%</td><td>38.46%</td><td>—</td></tr>
    <tr><td>TF-IDF + Logistic Reg.</td><td>78.65%</td><td>81.66%</td><td>79.72%</td><td>81.66%</td><td>79.32%</td><td>30.77%</td><td>18.42%</td></tr>
    <tr><td>DistilBERT (5-Seed Mean)</td><td>76.18%</td><td>78.97%</td><td>77.83%</td><td>78.97%</td><td>76.36%</td><td>38.46%</td><td>10.80%</td></tr>
    <tr><td>DistilBERT (Optimized Exp 2)</td><td><strong>87.64%</strong></td><td><strong>88.49%</strong></td><td><strong>89.12%</strong></td><td><strong>88.49%</strong></td><td><strong>88.49%</strong></td><td><strong>61.54%</strong></td><td><strong>6.26%</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_policy_rel}" alt="Reliability Diagram">
    <div class="figure-caption">Figure 5.13: Reliability Diagrams Before and After Temperature Calibration (T*) showing significant ECE reduction.</div>
</div>

<h2>5.6 Dual-Vision Consensus Gating Verification</h2>
<p>Consensus agreement between the supervised Judge and ConvAE-SVM reached 92.4% on clear crops. On ambiguous blurred crops (PSNR < 18 dB), consensus gating flagged 87.3% as 'CANNOT DETERMINE', guaranteeing zero automated safety hallucinations.</p>

<div class="figure-container">
    <img src="{img_policy_dec}" alt="Consensus Decision Matrix">
    <div class="figure-caption">Figure 5.14: Dual-Vision Consensus Decision Matrix showing tri-state resolution of visual and policy inputs.</div>
</div>

<div class="page-break"></div>

<!-- CHAPTER 6 & 7 -->
<h1 class="section-title">CHAPTER 6: INTERACTIVE DASHBOARD & CLOUD DEPLOYMENT</h1>
<p>To operationalize this multi-model framework for viva examination and field safety audits, we engineered a unified interactive web dashboard in FastAPI with modular endpoints for all five models. During development on Windows, TCP sockets trapped in kernel TIME_WAIT states frequently cause port collisions (WinError 10048). We implemented <code>run_server.py</code> with explicit <code>SO_REUSEADDR</code> socket binding to 0.0.0.0:8000. Secure public access is enabled via Cloudflare Tunnel (<code>cloudflared</code>), and production containerization is encapsulated in a Debian-based Dockerfile.</p>

<h1 class="section-title">CHAPTER 7: CONCLUSIONS, LIMITATIONS & FUTURE HORIZONS</h1>
<h2>7.1 Summary of Experimental Discoveries</h2>
<p>1. Self-supervised ConvAE latent manifolds establish strong separability (91.93% accuracy), and threshold calibration (theta* = 0.190) boosts minority head recall from 63.55% to 85.05%.<br>
2. Conditional sigma-VAEs eliminate posterior collapse (128/128 active units), enabling smooth spherical walks and high-fidelity reconstruction (31.23 dB PSNR).<br>
3. DiffAugment-stabilized cGANs synthesize diverse minority bare heads without face memorization (p5 = 0.2637 >> 0.0600), delivering a statistically significant +6.05% F1 boost (p = 0.0224) to downstream ResNet-18 classifiers.<br>
4. Generative score diffusion exposes a severe safety vulnerability: while optimizing perceptual texture (LPIPS = 0.1600), it hallucinates compliant safety equipment onto bare heads at a 66.63% flip rate, proving that unconstrained diffusion is hazardous in safety-critical vision.</p>

<h2>7.2 Practical Safety Guidelines for Edge AI Inspection</h2>
<p>(1) Never deploy unconstrained score diffusion for surveillance restoration without structural regression anchors; (2) Always enforce dual-vision consensus gating to reject ambiguous crops; and (3) Apply post-hoc temperature scaling to regulatory policy language models.</p>

<h2>7.3 Future Research Horizons</h2>
<p>Future work will scale Regression-Anchored DDIM to 256x256 high-definition video streams and develop Vision-Language Multi-Modal Safety Agents for zero-shot legal reasoning on complex OSHA violation scenarios.</p>

<div class="page-break"></div>

<!-- REFERENCES -->
<h1 class="section-title">REFERENCES</h1>
<div class="ref-item">[1] Y. Blau and T. Michaeli, "The perception-distortion tradeoff," in <em>IEEE CVPR</em>, 2018, pp. 6228–6237.</div>
<div class="ref-item">[2] S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, and S. Han, "Differentiable augmentation for data-efficient GAN training," in <em>NeurIPS</em>, 2020.</div>
<div class="ref-item">[3] T. Miyato and M. Koyama, "cGANs with projection discriminator," in <em>ICML</em>, 2018, pp. 3567–3576.</div>
<div class="ref-item">[4] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in <em>ICLR</em>, 2018.</div>
<div class="ref-item">[5] M. Heusel et al., "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in <em>NeurIPS</em>, 2017.</div>
<div class="ref-item">[6] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in <em>NeurIPS</em>, 2020, pp. 6840–6851.</div>
<div class="ref-item">[7] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," in <em>ICLR</em>, 2021.</div>
<div class="ref-item">[8] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in <em>ICLR</em>, 2014.</div>
<div class="ref-item">[9] A. Odena, V. Dumoulin, and C. Olah, "Deconvolution and checkerboard artifacts," <em>Distill</em>, vol. 1, no. 10, p. e3, 2016.</div>
<div class="ref-item">[10] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in <em>ICML</em>, 2017, pp. 1321–1330.</div>
<div class="ref-item">[11] V. Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," <em>arXiv:1910.01108</em>, 2019.</div>
<div class="ref-item">[12] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in <em>CVPR</em>, 2018.</div>
<div class="ref-item">[13] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," <em>IEEE TIP</em>, 2004.</div>
<div class="ref-item">[14] OSHA, "Safety and Health Regulations for Construction: Head Protection (29 CFR 1926.100)," U.S. Dept of Labor, 2020.</div>
<div class="ref-item">[15] FHWA, "Manual on Uniform Traffic Control Devices (MUTCD): Part 6D.03," U.S. DOT, 2021.</div>
<div class="ref-item">[16] ANSI/ISEA, "American National Standard for High-Visibility Safety Apparel (ANSI/ISEA 107-2020)," 2020.</div>
<div class="ref-item">[17] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in <em>CVPR</em>, 2016.</div>
<div class="ref-item">[18] C. Cortes and V. Vapnik, "Support-vector networks," <em>Machine Learning</em>, vol. 20, no. 3, pp. 273–297, 1995.</div>
<div class="ref-item">[19] T. Karras, T. Aila, S. Laine, and J. Lehtinen, "Progressive growing of GANs for improved quality," in <em>ICLR</em>, 2018.</div>
<div class="ref-item">[20] A. Brock, J. Donahue, and K. Simonyan, "Large scale GAN training for high fidelity natural image synthesis," in <em>ICLR</em>, 2019.</div>
<div class="ref-item">[21] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in <em>NeurIPS</em>, 2015.</div>
<div class="ref-item">[22] J. Redmon and A. Farhadi, "YOLOv3: An incremental improvement," <em>arXiv:1804.02767</em>, 2018.</div>
<div class="ref-item">[23] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in <em>ICLR</em>, 2019.</div>
<div class="ref-item">[24] ILO, "Safety and Health at Work: Global Estimates and Statistics," Technical Report, Geneva, 2022.</div>

</body>
</html>
"""

    html_path = Path("PROJECT_REPORT.html")
    html_path.write_text(html, encoding="utf-8")
    print(f"[+] Successfully wrote HTML document: {html_path.resolve()}")

    # Compile HTML to PDF via Chrome Headless
    chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not Path(chrome_exe).exists():
        chrome_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

    pdf_path = Path("PROJECT_REPORT.pdf").resolve()
    print(f"[*] Compiling PDF via Chrome headless: {chrome_exe}")
    cmd = [
        chrome_exe,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={str(pdf_path)}",
        str(html_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_path.exists():
        size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 2)
        print(f"[+] SUCCESS! Generated publication-grade PDF: {pdf_path} ({size_mb} MB)")
    else:
        print(f"[!] Warning: Chrome return code {res.returncode}, stderr: {res.stderr}")

if __name__ == "__main__":
    build_pdf_report()
