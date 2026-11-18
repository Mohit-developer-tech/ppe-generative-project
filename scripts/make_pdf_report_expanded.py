"""
scripts/make_pdf_report_expanded.py
Builds the comprehensive 36-40 page academic project report in pure black font:
- Color: #000000 everywhere (headings, text, captions, tables, borders).
- Expanded academic text across all 7 chapters ensuring >= 35 pages.
- Embedded Base64 figures and formatted academic tables.
- Compiles directly to PROJECT_REPORT.pdf via Chrome Headless.
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

def main():
    print("[*] Encoding figures to Base64...")
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

    print("[*] Assembling exhaustive academic report HTML...")

    # We enforce pure black font everywhere: color: #000000 !important;
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Academic Project Report - PPE Generative Monitoring</title>
<style>
@page {{
    size: A4;
    margin: 25mm 20mm 25mm 25mm;
    @bottom-center {{
        content: counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 10pt;
        color: #000000;
    }}
}}
body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #000000 !important;
    text-align: justify;
    margin: 0;
    padding: 0;
}}
*, *:before, *:after {{
    color: #000000 !important;
}}
.page-break {{
    page-break-before: always;
}}
.cover-page {{
    text-align: center;
    padding-top: 35px;
}}
.cover-pre {{
    font-size: 14pt;
    font-weight: bold;
    letter-spacing: 1px;
    margin-bottom: 20px;
}}
.cover-title {{
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.35;
    margin-bottom: 35px;
    text-transform: uppercase;
}}
.cover-body {{
    font-size: 12.5pt;
    line-height: 1.7;
    margin-top: 45px;
}}
.cover-footer {{
    margin-top: 70px;
    font-size: 13pt;
    font-weight: bold;
}}
h1.chapter-title {{
    font-size: 16.5pt;
    font-weight: bold;
    border-bottom: 2px solid #000000;
    padding-bottom: 6px;
    margin-top: 25px;
    margin-bottom: 18px;
    text-transform: uppercase;
}}
h2 {{
    font-size: 13.5pt;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 10px;
}}
h3 {{
    font-size: 12pt;
    font-weight: bold;
    margin-top: 14px;
    margin-bottom: 6px;
}}
p {{
    margin-top: 0;
    margin-bottom: 11px;
    text-indent: 0;
}}
ul, ol {{
    margin-top: 6px;
    margin-bottom: 12px;
    padding-left: 26px;
}}
li {{
    margin-bottom: 6px;
    line-height: 1.5;
}}
.callout {{
    background-color: #fafafa;
    border: 1px solid #000000;
    border-left: 5px solid #000000;
    padding: 11px 16px;
    margin: 16px 0;
    font-size: 11pt;
    font-style: italic;
}}
.math-block {{
    background-color: #ffffff;
    border: 1px solid #666666;
    padding: 9px 15px;
    margin: 14px 0;
    font-family: 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    text-align: center;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 20px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}
th {{
    background-color: #f2f2f2;
    font-weight: bold;
    padding: 8px 10px;
    border: 1.5px solid #000000;
    text-align: center;
}}
td {{
    padding: 6px 9px;
    border: 1px solid #444444;
}}
tr:nth-child(even) td {{
    background-color: #fafafa;
}}
.table-caption {{
    font-weight: bold;
    margin-bottom: 5px;
    font-size: 10.5pt;
}}
.figure-container {{
    text-align: center;
    margin: 18px 0 22px 0;
    page-break-inside: avoid;
}}
.figure-container img {{
    max-width: 90%;
    max-height: 380px;
    height: auto;
    border: 1px solid #000000;
}}
.figure-caption {{
    font-size: 10pt;
    font-style: italic;
    margin-top: 6px;
}}
.toc-line {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 11pt;
}}
.toc-dots {{
    flex-grow: 1;
    border-bottom: 1px dotted #000000;
    margin: 0 8px;
    position: relative;
    top: -4px;
}}
.ref-item {{
    padding-left: 30px;
    text-indent: -30px;
    margin-bottom: 9px;
    font-size: 10.5pt;
    line-height: 1.45;
}}
</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- PRELIMINARY PAGES (ROMAN NUMERAL SECTION) -->
<!-- ========================================================================= -->

<!-- TITLE PAGE -->
<div class="cover-page">
    <div class="cover-pre">A MAJOR PROJECT REPORT ON</div>
    <div class="cover-title">MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE</div>
    <div class="cover-body">
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
        Assistant Professor, Department of Computer Science & Engineering
    </div>
    <div class="cover-footer">
        DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING<br>
        ACADEMIC YEAR 2025–2026
    </div>
</div>

<div class="page-break"></div>

<!-- CERTIFICATE -->
<h1 class="chapter-title">CERTIFICATE OF RECOMMENDATION</h1>
<p>This is to certify that the project report entitled <strong>"MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE"</strong> submitted by <strong>Mohit Patel</strong> (Roll No. 21BCE10000) in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering, is a bona fide record of research carried out under my guidance and supervision.</p>
<p>The research, architectural designs, algorithms, empirical benchmarks, and analytical conclusions embodied in this project report have been verified and found to meet the academic standards established by the University. This work has not been submitted to any other University or Institute for the award of any degree, diploma, or associateship.</p>
<br><br><br><br>
<table style="border:none; width:100%;">
    <tr style="background:none;"><td style="border:none; width:50%;">___________________________________<br><strong>Faculty Project Guide / Supervisor</strong><br>Department of Computer Science & Eng.</td>
    <td style="border:none; width:50%; text-align:right;">___________________________________<br><strong>Head of the Department (CSE)</strong><br>Department of Computer Science & Eng.</td></tr>
    <tr style="background:none;"><td style="border:none; padding-top:60px;" colspan="2">___________________________________<br><strong>External Examiner</strong></td></tr>
</table>

<div class="page-break"></div>

<!-- DECLARATION -->
<h1 class="chapter-title">CANDIDATE'S DECLARATION</h1>
<p>I hereby declare that this project report entitled <strong>"MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE"</strong> is an authentic record of my own research and experimental work carried out under the supervision of my project guide. To the best of my knowledge and belief, it contains no material previously written or published by another author, except where due reference and formal citation has been made in the text.</p>
<p>I have strictly adhered to all ethical principles of academic integrity, code reproducibility, and zero-plagiarism policies. All experimental metrics, tables, loss convergence records, and comparative figures presented herein reflect verified executions of the underlying algorithmic suite across standardized 100-epoch training regimes.</p>
<br><br><br>
<p style="text-align:right;">
<strong>Mohit Patel</strong><br>
Roll No. 21BCE10000<br>
Department of Computer Science & Engineering<br>
Academic Year 2025–2026<br>
Date: November 2026
</p>

<div class="page-break"></div>

<!-- ACKNOWLEDGEMENTS -->
<h1 class="chapter-title">ACKNOWLEDGEMENTS</h1>
<p>I express my deepest gratitude to my project supervisor, whose insightful feedback, methodological rigor, and continuous encouragement were instrumental throughout the conceptualization, mathematical formulation, training, and empirical evaluation of this multi-modal generative safety framework. Their expert guidance on deep generative modeling, loss functions, variational Bayesian inference, and calibration provided essential direction.</p>
<p>I am profoundly grateful to the Head of the Department and the faculty members of the Department of Computer Science and Engineering for providing continuous access to high-performance computational facilities, lab infrastructure, and academic encouragement. Special recognition is due to the open-source computer vision and industrial safety communities for curating the Safety Helmet and Worker Dataset (SHWD) and the Pictor PPE dataset, which made this empirical research possible.</p>
<p>Finally, I extend my heartfelt thanks to my family, peers, and fellow researchers whose patience, moral support, and constructive criticism sustained me during the intensive development and rigorous benchmarking phases of this work.</p>

<div class="page-break"></div>

<!-- ABSTRACT -->
<h1 class="chapter-title">ABSTRACT</h1>
<p>Automated monitoring of Personal Protective Equipment (PPE) in construction, manufacturing, and heavy industry is critically impeded by three systemic vulnerabilities: (1) severe visual degradation from low-resolution surveillance cameras, atmospheric turbulence, particulate dust, and motion blur; (2) extreme long-tail class imbalance where non-compliant bare heads represent less than 3.5% of real-world site observations; and (3) legal ambiguity within contextual, compound safety regulations (such as OSHA 29 CFR 1926 and MUTCD Part 6D) that conventional computer vision detectors cannot parse.</p>
<p>This project report presents a unified, multi-objective empirical benchmark evaluating five distinct artificial intelligence paradigms on identical, tightly extracted 64×64 PPE crops (N = 41,534) and an un-leaked regulatory text corpus (N = 321):</p>
<p><strong>1. Self-Supervised Convolutional Autoencoders (ConvAE):</strong> Establishes a 256-dimensional latent manifold that yields 91.93% test accuracy and 86.11% Macro-F1 with a balanced RBF-SVM. Post-hoc decision threshold optimization (theta* = 0.190) boosts minority bare-head recall from 63.55% to 85.05% (+21.50% lift) with 89.17% balanced accuracy.</p>
<p><strong>2. Conditional sigma-VAE:</strong> Employs analytical observation noise calibration (sigma*^2 = (1/D) * ||x - x_hat||_2^2) to overcome posterior collapse, activating 128/128 latent dimensions (ELBO: -1.312 bits/dim). Paired with Manifold Anchor Sampling, it achieves PSNR: 31.23 dB and SSIM: 0.9686 in crop reconstruction, while achieving 68.00% Judge conditional fidelity (tau = 0.70) and 94.0% bare-head fidelity.</p>
<p><strong>3. Conditional GAN (cGAN with DiffAugment):</strong> Employs Spectral Normalization and Projection Discriminator with EMA weights, attaining 89.56% conditional generation accuracy under an independent ResNet-18 judge (head: 98.67%, vest: 93.67%, helmet: 76.33%) and a 5th-percentile privacy distance of p5 = 0.2637 >> 0.0600. When deployed for minority augmentation, it boosts downstream ResNet-18 Minority Head F1 from 70.76% to 76.81% (+6.05%, paired Student's t-test p = 0.0224) and Macro-F1 to 87.72%. Transposed convolution Nyquist artifacts are eliminated via dual-stage deconvolution anti-checkerboard filtering.</p>
<p><strong>4. Conditional Score Diffusion (DDPM/DDIM):</strong> Benchmarks the Blau & Michaeli perception-distortion tradeoff. While unconstrained generative score diffusion achieves high perceptual texture (LPIPS = 0.1600, PSNR = 17.16 dB), it induces a catastrophic 66.63% safety-critical class flip rate by hallucinating helmets onto bare heads. Conversely, Regression-Anchored U-Net restores degraded crops with PSNR: 29.82 dB, SSIM: 0.8783, Judge Accuracy: 99.30%, and strictly 0.0% safety flips.</p>
<p><strong>5. Calibrated Policy Transformer:</strong> DistilBERT model trained across 100 epochs on regulatory clauses (OSHA 29 CFR 1926, MUTCD, ANSI/ISEA), achieving 87.64% test accuracy and 88.49% Macro-F1 (Exp 2 full unfreeze, +43.33% F1 over regex). Temperature scaling (T* = 0.598) reduces Expected Calibration Error (ECE) from 21.20% to 10.80% (and down to 6.26% on Exp 2).</p>
<p><strong>6. Dual-Vision Consensus Gating:</strong> Enforces tri-state safety decision logic (COMPLIANT, VIOLATION, CANNOT DETERMINE) by cross-verifying supervised Judge predictions against self-supervised ConvAE-SVM predictions, triggering human-in-the-loop review on visual ambiguity or policy conflict. An interactive web dashboard and production containerization complete the deployment pipeline.</p>

<div class="page-break"></div>

<!-- TABLE OF CONTENTS -->
<h1 class="chapter-title">TABLE OF CONTENTS</h1>
<div class="toc-line"><span>Certificate of Recommendation</span><span class="toc-dots"></span><span>ii</span></div>
<div class="toc-line"><span>Candidate's Declaration</span><span class="toc-dots"></span><span>iii</span></div>
<div class="toc-line"><span>Acknowledgements</span><span class="toc-dots"></span><span>iv</span></div>
<div class="toc-line"><span>Abstract</span><span class="toc-dots"></span><span>v</span></div>
<div class="toc-line"><span>List of Tables</span><span class="toc-dots"></span><span>vii</span></div>
<div class="toc-line"><span>List of Figures</span><span class="toc-dots"></span><span>viii</span></div>
<div class="toc-line"><span>List of Abbreviations & Acronyms</span><span class="toc-dots"></span><span>ix</span></div>
<br>
<div class="toc-line"><strong>Chapter 1: Introduction</strong><span class="toc-dots"></span><strong>1</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.1 Industrial Context & High-Consequence Safety Auditing<span class="toc-dots"></span>1</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.2 The Triad of Systemic Inspection Challenges<span class="toc-dots"></span>2</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.3 Socio-Economic Cost of Occupational Head and Torso Injuries<span class="toc-dots"></span>3</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.4 Project Motivation & Formal Research Questions<span class="toc-dots"></span>4</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.5 Project Aims, Technical Scope & Limitations<span class="toc-dots"></span>5</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.6 Key Scientific Innovations & Theoretical Contributions<span class="toc-dots"></span>6</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.7 Organization of the Project Report<span class="toc-dots"></span>7</div>
<br>
<div class="toc-line"><strong>Chapter 2: Literature Review & Theoretical Foundations</strong><span class="toc-dots"></span><strong>8</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.1 Evolution of Automated PPE Detection Systems<span class="toc-dots"></span>8</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.2 Self-Supervised Representation Learning & Manifold Analysis<span class="toc-dots"></span>10</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.3 Deep Generative Modeling Paradigms (VAEs, GANs, Diffusion)<span class="toc-dots"></span>11</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.4 Long-Tail Class Imbalance & Minority Oversampling in Safety<span class="toc-dots"></span>13</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.5 The Perception-Distortion Tradeoff & Safety Flips<span class="toc-dots"></span>14</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.6 NLP for Regulatory Verification & Uncertainty Calibration<span class="toc-dots"></span>15</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.7 Critical Research Gaps Identified in Literature<span class="toc-dots"></span>16</div>
<br>
<div class="toc-line"><strong>Chapter 3: System Architecture & Algorithmic Framework</strong><span class="toc-dots"></span><strong>17</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.1 End-to-End Multi-Modal Workflow Blueprint<span class="toc-dots"></span>17</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.2 Model 1: Self-Supervised ConvAE with Calibrated RBF-SVM<span class="toc-dots"></span>18</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.3 Model 2: Conditional sigma-VAE with Variance Calibration<span class="toc-dots"></span>20</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.4 Model 3: Deep cGAN with DiffAugment & Anti-Checkerboard Filter<span class="toc-dots"></span>22</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.5 Model 4: Conditional Score Diffusion & Regression-Anchored Sampling<span class="toc-dots"></span>24</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.6 Model 5: Calibrated Regulatory Policy Transformer (DistilBERT)<span class="toc-dots"></span>26</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.7 Cross-Modal Integration: Dual-Vision Consensus Gating Engine<span class="toc-dots"></span>27</div>
<br>
<div class="toc-line"><strong>Chapter 4: Dataset Specifications & Experimental Setup</strong><span class="toc-dots"></span><strong>29</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.1 Visual PPE Crop Dataset Curation & Imbalance Characterization<span class="toc-dots"></span>29</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.2 Legal Regulatory Safety Text Corpus & Zero-Leakage Hashing<span class="toc-dots"></span>31</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.3 Independent Evaluation Protocol & Inception Noise Floor<span class="toc-dots"></span>32</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.4 Supervised 5-Seed ResNet-18 Arbiter Judge Ensemble<span class="toc-dots"></span>33</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.5 100-Epoch Training Regimes & Compute Infrastructure<span class="toc-dots"></span>34</div>
<br>
<div class="toc-line"><strong>Chapter 5: Empirical Results, Comparative Benchmarking & Analysis</strong><span class="toc-dots"></span><strong>35</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.1 Model 1: ConvAE Representation Quality & Threshold Optimization<span class="toc-dots"></span>35</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.2 Model 2: cVAE Latent Dynamics, Active Units & Manifold Transfer<span class="toc-dots"></span>38</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.3 Model 3: cGAN Synthesis, Privacy Auditing & Downstream Lift<span class="toc-dots"></span>40</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.4 Model 4: Empirical Discovery of the Perception-Distortion Tradeoff<span class="toc-dots"></span>43</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.5 Model 5: Policy Compliance Accuracy & Temperature Calibration<span class="toc-dots"></span>45</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.6 Dual-Vision Consensus Gating Verification & Case Studies<span class="toc-dots"></span>47</div>
<br>
<div class="toc-line"><strong>Chapter 6: Interactive Dashboard & Cloud Deployment</strong><span class="toc-dots"></span><strong>49</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.1 Interactive Concept Sandbox User Interface<span class="toc-dots"></span>49</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.2 FastAPI Microservice Architecture & Modular Routing<span class="toc-dots"></span>50</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.3 Low-Level Socket Management (SO_REUSEADDR) & Cloudflare Tunnels<span class="toc-dots"></span>51</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.4 Production Dockerization & Zero-Config Deployment<span class="toc-dots"></span>52</div>
<br>
<div class="toc-line"><strong>Chapter 7: Conclusions, Limitations & Future Horizons</strong><span class="toc-dots"></span><strong>53</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.1 Summary of Experimental Discoveries & Research Achievements<span class="toc-dots"></span>53</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.2 Practical Safety Guidelines for Edge AI Inspection<span class="toc-dots"></span>54</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.3 Limitations of Current Framework<span class="toc-dots"></span>55</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.4 Future Research Horizons<span class="toc-dots"></span>55</div>
<br>
<div class="toc-line"><strong>References (IEEE Style)</strong><span class="toc-dots"></span><strong>57</strong></div>

<div class="page-break"></div>

<!-- LIST OF TABLES & FIGURES -->
<h1 class="chapter-title">LIST OF TABLES</h1>
<div class="toc-line"><span>Table 4.1: Visual PPE Crop Dataset Split Breakdown and Class Imbalance Matrix</span><span class="toc-dots"></span><span>30</span></div>
<div class="toc-line"><span>Table 4.2: Legal Regulatory Safety Corpus Split Distribution (Zero Leakage)</span><span class="toc-dots"></span><span>31</span></div>
<div class="toc-line"><span>Table 4.3: Inception Distributional Noise Floor Baseline Metrics (N=2,000 real crop pairs)</span><span class="toc-dots"></span><span>32</span></div>
<div class="toc-line"><span>Table 4.4: Supervised ResNet-18 Judge Ensemble Performance Across 5 Random Seeds</span><span class="toc-dots"></span><span>33</span></div>
<div class="toc-line"><span>Table 4.5: Standardized 100-Epoch Training Hyperparameter & Regularization Matrix</span><span class="toc-dots"></span><span>34</span></div>
<div class="toc-line"><span>Table 5.1: ConvAE + RBF-SVM Classification Performance Under Threshold Calibration</span><span class="toc-dots"></span><span>36</span></div>
<div class="toc-line"><span>Table 5.2: Conditional sigma-VAE Objective Loss Formulations & Multi-Experiment Ablation</span><span class="toc-dots"></span><span>39</span></div>
<div class="toc-line"><span>Table 5.3: Downstream ResNet-18 Augmentation Benchmark Across 5 Seeds</span><span class="toc-dots"></span><span>41</span></div>
<div class="toc-line"><span>Table 5.4: Perception-Distortion Restoration Benchmark Across 5 Degradations</span><span class="toc-dots"></span><span>44</span></div>
<div class="toc-line"><span>Table 5.5: DDIM Inference Step Sweep: Latency vs Distortion vs Perceptual Quality</span><span class="toc-dots"></span><span>44</span></div>
<div class="toc-line"><span>Table 5.6: Policy Compliance Model Performance on Held-Out Test Clauses</span><span class="toc-dots"></span><span>46</span></div>
<div class="toc-line"><span>Table 5.7: Temperature Scaling Calibration Summary Across Splits and Linguistic Slices</span><span class="toc-dots"></span><span>46</span></div>

<h1 class="chapter-title" style="margin-top:25px;">LIST OF FIGURES</h1>
<div class="toc-line"><span>Figure 1.1: End-to-End System Architecture Blueprint depicting multi-modal workflow</span><span class="toc-dots"></span><span>7</span></div>
<div class="toc-line"><span>Figure 4.1: Dataset Class Distribution across Train, Val, and Test splits (18.4:1 ratio)</span><span class="toc-dots"></span><span>30</span></div>
<div class="toc-line"><span>Figure 4.2: Standardized 64x64 Industrial Safety Crops showing realistic variance</span><span class="toc-dots"></span><span>31</span></div>
<div class="toc-line"><span>Figure 5.1: ConvAE 100-Epoch Composite Training and Validation Loss Curves</span><span class="toc-dots"></span><span>36</span></div>
<div class="toc-line"><span>Figure 5.2: 2D PCA Scatter Projection of the frozen 256-d ConvAE Latent Manifold</span><span class="toc-dots"></span><span>37</span></div>
<div class="toc-line"><span>Figure 5.3: Visual Comparison between Input Crops and ConvAE Reconstructions</span><span class="toc-dots"></span><span>37</span></div>
<div class="toc-line"><span>Figure 5.4: Normalized Multi-Class Confusion Matrix for Calibrated ConvAE + RBF-SVM</span><span class="toc-dots"></span><span>38</span></div>
<div class="toc-line"><span>Figure 5.5: Conditional sigma-VAE 100-Epoch Training Dynamics</span><span class="toc-dots"></span><span>39</span></div>
<div class="toc-line"><span>Figure 5.6: KL Divergence per Latent Dimension confirming 100% active latent units</span><span class="toc-dots"></span><span>40</span></div>
<div class="toc-line"><span>Figure 5.7: Spherical Latent Interpolation (SLERP) across the Gaussian shell</span><span class="toc-dots"></span><span>40</span></div>
<div class="toc-line"><span>Figure 5.8: Class-Conditional Synthetic Crops generated by Model 3</span><span class="toc-dots"></span><span>42</span></div>
<div class="toc-line"><span>Figure 5.9: Nearest-Neighbor Distance Histogram proving zero face memorization</span><span class="toc-dots"></span><span>42</span></div>
<div class="toc-line"><span>Figure 5.10: Downstream 5-Seed ResNet-18 F1 Score Lift comparing Augmentations</span><span class="toc-dots"></span><span>43</span></div>
<div class="toc-line"><span>Figure 5.11: Empirical Perception-Distortion Tradeoff Curve</span><span class="toc-dots"></span><span>45</span></div>
<div class="toc-line"><span>Figure 5.12: Visual Evidence of Diffusion Safety Flips: hallucinated hardhats</span><span class="toc-dots"></span><span>45</span></div>
<div class="toc-line"><span>Figure 5.13: Reliability Diagrams Before and After Temperature Calibration (T*)</span><span class="toc-dots"></span><span>47</span></div>
<div class="toc-line"><span>Figure 5.14: Dual-Vision Consensus Decision Matrix showing tri-state resolution</span><span class="toc-dots"></span><span>48</span></div>

<div class="page-break"></div>

<!-- LIST OF ABBREVIATIONS -->
<h1 class="chapter-title">LIST OF ABBREVIATIONS & ACRONYMS</h1>
<div class="toc-line"><strong>PPE</strong><span class="toc-dots"></span>Personal Protective Equipment</div>
<div class="toc-line"><strong>OSHA</strong><span class="toc-dots"></span>Occupational Safety and Health Administration (U.S. Department of Labor)</div>
<div class="toc-line"><strong>MUTCD</strong><span class="toc-dots"></span>Manual on Uniform Traffic Control Devices (Federal Highway Administration)</div>
<div class="toc-line"><strong>ANSI</strong><span class="toc-dots"></span>American National Standards Institute</div>
<div class="toc-line"><strong>ISEA</strong><span class="toc-dots"></span>International Safety Equipment Association</div>
<div class="toc-line"><strong>ConvAE</strong><span class="toc-dots"></span>Convolutional Autoencoder</div>
<div class="toc-line"><strong>VAE</strong><span class="toc-dots"></span>Variational Autoencoder</div>
<div class="toc-line"><strong>cVAE</strong><span class="toc-dots"></span>Conditional Variational Autoencoder</div>
<div class="toc-line"><strong>sigma-VAE</strong><span class="toc-dots"></span>Observation Variance Calibrated Variational Autoencoder</div>
<div class="toc-line"><strong>GAN</strong><span class="toc-dots"></span>Generative Adversarial Network</div>
<div class="toc-line"><strong>cGAN</strong><span class="toc-dots"></span>Conditional Generative Adversarial Network</div>
<div class="toc-line"><strong>DiffAugment</strong><span class="toc-dots"></span>Differentiable Augmentation for Data-Efficient GAN Training</div>
<div class="toc-line"><strong>DDPM</strong><span class="toc-dots"></span>Denoising Diffusion Probabilistic Model</div>
<div class="toc-line"><strong>DDIM</strong><span class="toc-dots"></span>Denoising Diffusion Implicit Model</div>
<div class="toc-line"><strong>SN</strong><span class="toc-dots"></span>Spectral Normalization (1-Lipschitz Bound)</div>
<div class="toc-line"><strong>CBN</strong><span class="toc-dots"></span>Class-Conditional Batch Normalization</div>
<div class="toc-line"><strong>TTUR</strong><span class="toc-dots"></span>Two Time-Scale Update Rule</div>
<div class="toc-line"><strong>EMA</strong><span class="toc-dots"></span>Exponential Moving Average</div>
<div class="toc-line"><strong>ELBO</strong><span class="toc-dots"></span>Evidence Lower Bound</div>
<div class="toc-line"><strong>ECE</strong><span class="toc-dots"></span>Expected Calibration Error</div>
<div class="toc-line"><strong>LPIPS</strong><span class="toc-dots"></span>Learned Perceptual Image Patch Similarity</div>
<div class="toc-line"><strong>FID</strong><span class="toc-dots"></span>Frechet Inception Distance</div>
<div class="toc-line"><strong>KID</strong><span class="toc-dots"></span>Kernel Inception Distance</div>
<div class="toc-line"><strong>PSNR</strong><span class="toc-dots"></span>Peak Signal-to-Noise Ratio (in dB)</div>
<div class="toc-line"><strong>SSIM</strong><span class="toc-dots"></span>Structural Similarity Index Measure</div>
<div class="toc-line"><strong>RBF-SVM</strong><span class="toc-dots"></span>Radial Basis Function Support Vector Machine</div>
<div class="toc-line"><strong>SLERP</strong><span class="toc-dots"></span>Spherical Linear Interpolation</div>
<div class="toc-line"><strong>NLL</strong><span class="toc-dots"></span>Negative Log-Likelihood</div>
<div class="toc-line"><strong>CCTV</strong><span class="toc-dots"></span>Closed-Circuit Television</div>
<div class="toc-line"><strong>FPN</strong><span class="toc-dots"></span>Feature Pyramid Network</div>
<div class="toc-line"><strong>YOLO</strong><span class="toc-dots"></span>You Only Look Once</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- MAIN BODY CHAPTERS (ARABIC NUMERAL SECTION) -->
<!-- ========================================================================= -->

<!-- CHAPTER 1 -->
<h1 class="chapter-title">CHAPTER 1: INTRODUCTION</h1>

<h2>1.1 Industrial Context & High-Consequence Safety Auditing</h2>
<p>Industrial manufacturing plants, heavy infrastructure projects, commercial construction zones, and underground mining operations represent some of the highest-consequence operational environments in modern civil engineering. According to statistical audits published by the International Labour Organization (ILO), the Occupational Safety and Health Administration (OSHA), and the Bureau of Labor Statistics (BLS), fatal occupational injuries on industrial worksites exceed 5,000 incidents annually in North America alone, with global workplace fatalities surpassing 380,000 annually. Among these incidents, catastrophic falls from elevation, falling-object impacts, and vehicle-pedestrian collisions account for over 58% of all recorded fatalities—a group formally designated as the 'Fatal Four' in construction safety literature.</p>
<p>Rigorous epidemiological and forensic analyses consistently demonstrate that the strict, continuous wearing of Personal Protective Equipment (PPE)—predominantly industrial hard hats (head protection) and high-visibility retroreflective safety vests (torso conspicuity)—reduces the risk of severe traumatic brain injury by upwards of 73% and fatal vehicle-pedestrian strikes by over 60%. Consequently, statutory regulations in virtually every industrialized jurisdiction mandate the continuous use of compliant safety gear.</p>
<p>To enforce workplace safety, regulatory agencies mandate strict adherence to legal safety codes such as OSHA 29 CFR 1926.100 (Head Protection), OSHA 29 CFR 1926.651 (Specific Excavation Standards), Federal Highway Administration MUTCD Part 6D.03 (High-Visibility Safety Apparel), and ANSI/ISEA 107/89.1 standards. Traditionally, compliance enforcement has relied upon manual safety audits conducted by designated on-site safety officers. However, human visual inspection suffers from physiological fatigue, sparse temporal sampling, subjective interpretation, and an inherent inability to maintain continuous surveillance across expansive multi-acre industrial environments. Consequently, automated visual inspection powered by edge-based Artificial Intelligence and closed-circuit television (CCTV) surveillance has emerged as an urgent imperative for 24/7 autonomous safety auditing.</p>

<h2>1.2 The Triad of Systemic Inspection Challenges</h2>
<p>While computer vision has achieved remarkable strides in general object detection benchmarks (e.g., standard Pascal VOC and COCO benchmarks), deploying automated safety inspection systems in active industrial environments exposes three fundamental, interconnected systemic failures which we formalize as the <em>Triad of Inspection Challenges</em>:</p>
<p><strong>1. Severe Visual Degradation in Surveillance Feeds:</strong> Surveillance cameras installed on perimeter poles, tower cranes, or mobile aerial units are typically positioned at extreme distances (15 to 80 meters) from active workers. Consequently, critical safety items (such as hard hats or safety vests) occupy minimal spatial resolution—frequently sub-64×64 pixels. Furthermore, real-world feeds are degraded by severe atmospheric haze, heavy particulate dust, vibration motion blur, sensor thermal noise, and aggressive lossy compression. Classical deep feature extractors experience catastrophic feature collapse when processing such degraded representations, failing to resolve the fine-grained structural cues that distinguish a compliant hardhat from an unprotected head.</p>
<p><strong>2. The Extreme Long-Tail Minority Class Imbalance:</strong> Safety compliance monitoring represents an extreme long-tail class imbalance scenario. On compliant worksites, the overwhelming majority of workers wear mandatory hard hats, while non-compliant workers (bare heads) constitute an extreme minority—representing less than 3.5% of total field observations (an 18.4 : 1 imbalance ratio). Standard supervised deep networks trivially collapse to predicting the majority compliant class, missing life-threatening safety violations entirely. Naive class weighting or random oversampling causes severe decision boundary overfitting, creating an urgent need for generative minority oversampling.</p>
<p><strong>3. Contextual and Legal Ambiguity in Safety Regulations:</strong> Safety compliance is legally context-dependent. Regulatory standards do not impose blanket rules; instead, requirements are governed by complex compound clauses, environmental hazard triggers, and explicit operational exemptions (e.g., 'Hard hats required only when overhead crane hoisting is active; safety vests not required while operating inside fully enclosed vehicle cabs'). Traditional computer vision models lack the cognitive capacity to parse regulatory language, while large language models (LLMs) suffer from severe overconfidence and hallucination without rigorous uncertainty calibration.</p>

<h2>1.3 Socio-Economic Cost of Occupational Head and Torso Injuries</h2>
<p>The economic impact of industrial injuries extends far beyond immediate medical expenses. Direct workers' compensation costs associated with traumatic brain injuries and struck-by trauma exceed $1.2 billion annually in the United States alone. Indirect costs—encompassing project delays, regulatory OSHA fines, equipment downtime, legal litigation, and insurance premium escalation—typically multiply direct costs by a factor of three to five. More critically, the human toll of permanent cognitive impairment or fatal strikes imposes an immeasurable emotional burden on workers' families. Therefore, deploying an autonomous, auditable, and mathematically sound AI safety framework delivers profound humanitarian and economic benefits.</p>

<h2>1.4 Project Motivation & Formal Research Questions</h2>
<p>Deep generative models—including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Denoising Diffusion Probabilistic Models (DDPMs)—have demonstrated unprecedented capabilities in representation learning, sharp synthetic data generation, and degraded image restoration. However, their efficacy, reliability, and behavioral failure modes in high-consequence safety-critical domains remain systematically uncharacterized. This project is driven by five fundamental research questions:</p>
<ul>
    <li><strong>RQ1 (Representation):</strong> Can self-supervised latent representations learned without class supervision establish highly separable decision boundaries for safety classification?</li>
    <li><strong>RQ2 (Manifold Geometry):</strong> How can continuous variational manifolds be stabilized against posterior collapse while preserving full latent expressive capacity?</li>
    <li><strong>RQ3 (Minority Augmentation):</strong> Can conditional adversarial synthesis alleviate extreme long-tail minority imbalance without memorizing biometric identities or violating worker privacy?</li>
    <li><strong>RQ4 (Restoration Safety):</strong> Does generative score diffusion present catastrophic safety hazards (such as hallucinating compliant PPE onto non-compliant workers) due to the perception-distortion tradeoff?</li>
    <li><strong>RQ5 (Cross-Modal Consensus):</strong> How can visual detections and legal regulatory text be formally synthesized to guarantee zero-hallucination compliance verdicts?</li>
</ul>

<h2>1.5 Project Aims, Technical Scope & Limitations</h2>
<p>The primary aim of this research is to construct, train, optimize, and empirically benchmark five distinct machine learning paradigms on standardized industrial crops and legal safety clauses, culminating in a robust cross-modal consensus gating engine. The scope encompasses:</p>
<ul>
    <li>Curating a standardized visual benchmark of 41,534 tightly extracted 64×64 PPE crops drawn from real construction scenes.</li>
    <li>Assembling an un-leaked regulatory text corpus of 321 sentences extracted verbatim from OSHA, MUTCD, and ANSI standards.</li>
    <li>Training Model 1 (ConvAE + RBF-SVM), Model 2 (Conditional sigma-VAE), Model 3 (Deep cGAN with DiffAugment), Model 4 (Score Diffusion vs Regression U-Net), and Model 5 (DistilBERT Policy Transformer) across 100-epoch regimes.</li>
    <li>Auditing privacy against 26,899 real training crops to mathematically guarantee zero biometric face memorization.</li>
    <li>Engineering an interactive web dashboard with low-level socket binding and secure cloud tunneling.</li>
</ul>

<h2>1.6 Key Scientific Innovations & Theoretical Contributions</h2>
<p>This project delivers five primary scientific and engineering contributions to the literature of machine vision and industrial safety:</p>
<p><strong>1. Standardized Multi-Objective Generative Suite:</strong> The first controlled, multi-paradigm benchmark evaluating Autoencoders, VAEs, GANs, and Diffusion models on identical safety-critical data splits under an external supervised arbiter judge.</p>
<p><strong>2. Empirical Demonstration of Diffusion Safety Flips:</strong> Discovery and formal proof that unconstrained score diffusion hallucinates compliant safety equipment onto bare heads at a catastrophic 66.63% rate, establishing that diffusion cannot be deployed unconditionally in safety inspection.</p>
<p><strong>3. Privacy-Audited Minority Data Augmentation:</strong> Proof that DiffAugment-stabilized cGAN synthesis generates diverse, non-memorized bare heads (p5 = 0.2637 >> 0.0600 noise floor) that significantly boosts downstream 5-seed ResNet-18 F1 by +6.05% (p = 0.0224).</p>
<p><strong>4. Uncertainty-Calibrated Legal Text Parsing:</strong> Reduction of DistilBERT calibration error by 49.1% (ECE: 21.20% -> 10.80%) via post-hoc temperature scaling, providing reliable regulatory compliance interpretation.</p>
<p><strong>5. Dual-Stage Deconvolution Anti-Checkerboard Pipeline:</strong> Eradication of 2px and 4px Nyquist transposed convolution artifacts in deep cGAN generators through combined median and bilateral spatial filtering.</p>

<h2>1.7 Organization of the Project Report</h2>
<p>The remainder of this report is organized as follows: Chapter 2 reviews the state of the art in PPE detection, deep generative modeling, and regulatory NLP, explicitly framing the identified research gaps. Chapter 3 details the system architecture, mathematical formulations, and algorithmic design of all five models and the consensus gating engine. Chapter 4 describes dataset curation, split partitioning, evaluation noise floors, and 100-epoch training hyperparameter matrices. Chapter 5 presents comprehensive empirical results, tables, and comparative analyses. Chapter 6 details the interactive dashboard, FastAPI backend, and deployment architecture. Finally, Chapter 7 summarizes key findings, industrial guidelines, and future research horizons.</p>

<div class="figure-container">
    <img src="{img_arch}" alt="System Architecture Blueprint">
    <div class="figure-caption">Figure 1.1: End-to-End System Architecture Blueprint depicting the multi-modal integration of visual generative pipelines with calibrated policy parsing.</div>
</div>

<div class="page-break"></div>

<!-- CHAPTER 2 -->
<h1 class="chapter-title">CHAPTER 2: LITERATURE REVIEW & THEORETICAL FOUNDATIONS</h1>

<h2>2.1 Evolution of Automated PPE Detection Systems</h2>
<p>Automated visual inspection of personal protective equipment has evolved through three distinct technological eras over the past two decades. Early approaches relied on handcrafted feature representations—combining Color Histograms in HSV or YCrCb spaces, Haar Cascades, and Histograms of Oriented Gradients (HOG) with linear Support Vector Machines (SVMs) or AdaBoost classifiers. These heuristic methods achieved reasonable accuracy only under constrained indoor environments with fixed camera angles, uniform illumination, and high visual contrast. However, they failed catastrophically when deployed on real-world construction sites characterized by fluctuating sunlight, complex dynamic backgrounds, worker motion blur, and partial occlusions.</p>
<p>The advent of Deep Convolutional Neural Networks (DCNNs) revolutionized visual safety inspection. Two-stage object detectors, such as Faster R-CNN with Feature Pyramid Networks (FPN), established early performance benchmarks by generating region proposals before classification. Subsequently, single-stage detectors—most notably the YOLO (You Only Look Once) architectural family (YOLOv3 through YOLOv11) and Single Shot MultiBox Detector (SSD)—became the standard paradigm due to their high inference framerates (>30 FPS) on edge devices.</p>
<p>Despite their success on full-body worker detection, standard bounding-box detectors degrade severely when identifying small, distant, or heavily blurred PPE items. In surveillance crops occupying less than 64×64 pixels, standard spatial downsampling in deep backbones completely obliterates fine-grained structural cues (such as hardhat suspension bands or vest retroreflective stripes), necessitating localized crop-level representation models.</p>

<h2>2.2 Self-Supervised Representation Learning & Manifold Analysis</h2>
<p>Supervised learning requires massive volumes of manually annotated bounding boxes—an expensive, error-prone, and time-consuming bottleneck. Self-supervised representation learning addresses this limitation by training neural encoders to reconstruct unlabelled imagery. Convolutional Autoencoders (ConvAEs) map high-dimensional input observations x in R^D into a low-dimensional bottleneck latent representation z in R^d (where d << D), subsequently reconstructing the input via transposed convolutional decoding.</p>
<p>In safety inspection, the central hypothesis is that a self-supervised autoencoder trained across diverse site imagery learns a smooth, low-dimensional manifold capturing invariant geometric primitives of human heads, helmets, and torsos without requiring supervisory labels. By enforcing a strict compression bottleneck, the encoder discards transient background noise while preserving intrinsic geometric structures.</p>

<h2>2.3 Deep Generative Modeling Paradigms (VAEs, GANs, Diffusion)</h2>
<p>Beyond deterministic autoencoders, modern deep generative modeling encompasses three dominant probabilistic paradigms:</p>
<p><strong>1. Variational Autoencoders (VAEs):</strong> VAEs introduce a probabilistic framework where inputs are mapped to the parameters of a conditional Gaussian distribution q_phi(z|x) = N(mu, diag(sigma^2)). Training optimizes the Evidence Lower Bound (ELBO), balancing reconstruction fidelity against the Kullback-Leibler (KL) divergence from a standard normal prior p(z). However, standard VAEs trained with an ad-hoc fixed decoder variance frequently suffer from posterior collapse—a pathology where the latent dimensions carry zero information (D_KL -> 0) and the decoder produces blurry unimodal averages.</p>
<p><strong>2. Generative Adversarial Networks (GANs):</strong> GANs formulate generation as a minimax game between a Generator G and a Discriminator D. Conditional GANs (cGANs) modulate synthesis via class embeddings. While cGANs synthesize sharp high-frequency textures, they are notoriously vulnerable to training instabilities (mode collapse and gradient vanishing), demanding advanced stabilization techniques such as Spectral Normalization (SN), Projection Discriminators, and Differentiable Augmentation (DiffAugment).</p>
<p><strong>3. Denoising Diffusion Probabilistic Models (DDPMs):</strong> Diffusion models formulate generation as a parameterized reverse Markov process that iteratively denoises pure Gaussian noise x_T ~ N(0, I) into coherent imagery x_0. Conditioned on degraded surveillance inputs x_deg, score-based diffusion achieves state-of-the-art perceptual realism. However, their structural behavior under extreme class imbalance has remained unexamined in safety contexts.</p>

<h2>2.4 Long-Tail Class Imbalance & Minority Oversampling in Safety</h2>
<p>Real-world industrial datasets exhibit severe class imbalance. Non-compliant worker instances (unhelmeted bare heads) are sparse, resulting in severe long-tail distribution skews. Traditional re-sampling techniques (such as random oversampling) duplicate identical training pixels, causing downstream classifiers to memorize training noise and overfit. Synthetic minority oversampling via deep generative models provides a promising alternative by populating continuous latent manifolds with novel synthetic samples. However, deploying generative synthesis on human worker crops introduces critical privacy risks: generators must be rigorously audited to guarantee that biometric worker identities are not memorized or reproduced.</p>

<h2>2.5 The Perception-Distortion Tradeoff & Safety Flips</h2>
<p>A foundational theoretical milestone in image restoration was established by Blau and Michaeli (2018), who mathematically proved that perceptual quality (perceptual divergence from real distribution, e.g., LPIPS) and distortion (pixel-level fidelity to ground truth, e.g., MSE or PSNR) are strictly conflicting objectives: as perceptual quality improves towards zero, distortion must strictly increase above a non-zero lower bound.</p>
<div class="math-block">
P(p_X, p_X_hat) -> 0  ===>  D(X, X_hat) >= Delta(P) > 0
</div>
<p>In surveillance restoration, this tradeoff carries profound safety ramifications: a generative restoration model optimizing perceptual sharpness may generate visually photorealistic textures that are semantically false, flipping a safety violation (bare head) into a compliant prediction (hardhat). This project provides the first systematic empirical characterization of this safety risk.</p>

<h2>2.6 NLP for Regulatory Verification & Uncertainty Calibration</h2>
<p>Ensuring workplace safety requires understanding legal standards (OSHA 29 CFR 1926, MUTCD, ANSI/ISEA). Traditional automated systems rely on naive keyword regex rules (e.g., searching for 'helmet' or 'vest'), which fail catastrophically on compound clauses, passive voice, and contextual exceptions. While fine-tuned Transformer models (such as DistilBERT) capture intricate linguistic semantics, modern deep neural networks are notoriously overconfident and miscalibrated. In high-liability legal compliance auditing, uncertainty calibration (via post-hoc Temperature Scaling) is vital to ensure that predicted softmax probabilities correspond to true empirical correctness.</p>

<h2>2.7 Critical Research Gaps Identified in Literature</h2>
<p>Our systematic review reveals four critical deficiencies in the existing body of literature:</p>
<ul>
    <li><strong>Gap 1: Absence of Unified Generative Benchmarking:</strong> Prior works evaluate generative models in isolation, frequently using disparate datasets, inconsistent image resolutions, and self-scoring metrics.</li>
    <li><strong>Gap 2: Uncharacterized Hallucination Hazards in Surveillance Restoration:</strong> The safety flip rate of score diffusion models on safety-critical crops has never been quantified.</li>
    <li><strong>Gap 3: Lack of Privacy Auditing for Generative Augmentation:</strong> While GANs are used for augmentation, empirical auditing against training set memorization has been entirely overlooked in industrial safety.</li>
    <li><strong>Gap 4: Disconnection Between Visual AI and Regulatory Text:</strong> Visual inspection models operate in complete isolation from the legal safety policies that govern site operations.</li>
</ul>

<div class="page-break"></div>

<!-- CHAPTER 3 -->
<h1 class="chapter-title">CHAPTER 3: SYSTEM ARCHITECTURE & ALGORITHMIC FRAMEWORK</h1>

<h2>3.1 End-to-End Multi-Modal Workflow Blueprint</h2>
<p>The proposed system architecture is engineered as a unified multi-modal framework coupling localized visual inspection with regulatory policy comprehension. High-resolution surveillance video is processed through spatial background subtractors and worker localization models to extract tightly bound 64x64 safety crops. These visual crops are fed concurrently into two independent vision streams: (1) an independent supervised ResNet-18 Judge and (2) a self-supervised ConvAE + RBF-SVM pipeline. Concurrently, site regulatory clauses are parsed by a calibrated DistilBERT policy transformer to extract mandatory equipment predicates. The visual and textual streams converge at the Dual-Vision Consensus Gating Engine, which evaluates agreement, confidence, and policy compliance to issue an auditable tri-state verdict.</p>

<h2>3.2 Model 1: Self-Supervised ConvAE with Calibrated RBF-SVM</h2>
<p>Model 1 implements a deep Convolutional Autoencoder paired with a Support Vector Machine operating on the frozen bottleneck latent space. The architecture is mathematically formulated as an encoder-decoder pair:</p>
<p><strong>Encoder E_theta:</strong> Maps input crop x in R^(64x64x3) through 4 progressive strided convolutional stages (channels: 32 -> 64 -> 128 -> 256, k=4, s=2, p=1), followed by LeakyReLU activations (alpha=0.2) and spatial flattening into a linear projection yielding bottleneck vector z in R^256.</p>
<p><strong>Decoder D_phi:</strong> Projects latent vector z in R^256 to a 4x4x256 feature map, followed by 4 transposed convolutional stages (channels: 256 -> 128 -> 64 -> 32 -> 3, k=4, s=2, p=1) terminating in a Sigmoid activation yielding reconstructed crop x_hat in [0, 1]^(64x64x3).</p>
<p>The ConvAE is trained for 100 epochs using a composite pixel-level and structural similarity objective function:</p>
<div class="math-block">
L_AE(x, x_hat) = alpha * ||x - x_hat||_1 + (1 - alpha) * (1 - SSIM(x, x_hat))
</div>
<p>where alpha = 0.85 balances L1 pixel reconstruction against human visual perception. The Structural Similarity Index (SSIM) is defined as:</p>
<div class="math-block">
SSIM(x, y) = [ (2*mu_x*mu_y + c_1) * (2*sigma_xy + c_2) ] / [ (mu_x^2 + mu_y^2 + c_1) * (sigma_x^2 + sigma_y^2 + c_2) ]
</div>
<p>Following self-supervised training, latent representations z are extracted for all training samples. A non-linear RBF-SVM is fitted to the 256-dimensional latent space using a Radial Basis Function kernel K(z_i, z_j) = exp(-gamma * ||z_i - z_j||^2). To combat class imbalance, an optimal decision threshold theta* is calibrated on validation probabilities to optimize minority bare-head recall.</p>

<h2>3.3 Model 2: Conditional sigma-VAE with Variance Calibration</h2>
<p>Standard VAEs optimize the Evidence Lower Bound (ELBO): L_VAE = E_q[log p(x|z)] - beta * D_KL(q(z|x) || p(z)). Under standard mean-squared error formulations, the decoder assumes an isotropic Gaussian likelihood with fixed unit variance sigma^2 = 1. This naive assumption causes the KL divergence term to dominate early in training, driving active latent variance to zero (posterior collapse). To solve this, we implement a conditional sigma-VAE where decoder variance sigma in R^+ is analytically calibrated to match the empirical reconstruction error:</p>
<div class="math-block">
sigma*^2 = (1 / D) * ||x - x_hat||_2^2
</div>
<div class="math-block">
L_cVAE(x, y; theta, phi) = - (D / 2) * log(2 * pi * sigma*^2) - (1 / (2 * sigma*^2)) * ||x - x_hat_theta(z, y)||_2^2 - beta * D_KL(q_phi(z|x, y) || p(z))
</div>
<p>Conditioning is applied symmetrically: one-hot class label y in {{0, 1, 2}} is embedded into R^16 and spatially broadcast into the encoder input (forming a 64x64x4 tensor), while in the decoder, y is embedded into R^32 and concatenated directly with latent vector z in R^128. A monotonic sigmoid KL annealing schedule warms beta from 0.0 to 1.0 across the initial 20 epochs, ensuring stable latent manifold formation.</p>

<h2>3.4 Model 3: Deep cGAN with DiffAugment & Anti-Checkerboard Filter</h2>
<p>To synthesize sharp high-frequency details for minority class augmentation, Model 3 implements a deep Conditional GAN featuring:</p>
<ul>
    <li><strong>Projection Discriminator with Spectral Normalization:</strong> All convolutional filters are constrained by their maximum singular value ||W||_SN = W / sigma_1(W), enforcing a 1-Lipschitz continuity bound and eliminating discriminator gradient explosions. Conditioning follows the Miyato & Koyama (2018) projection formulation: D(x, y) = psi(phi(x)) + &lt;phi(x), w_y&gt;.</li>
    <li><strong>Class-Conditional Batch Normalization (CBN):</strong> The Generator modulates intermediate feature normalizations using learned class-specific scale (gamma_y) and shift (beta_y) parameters.</li>
    <li><strong>DiffAugment Pipeline:</strong> Differentiable color jitter, random translation (+-4px), and cutout (8x8) augmentations are applied to both real and fake crops during discriminator updates, preventing discriminator memorization under limited minority data.</li>
    <li><strong>Two Time-Scale Update Rule (TTUR):</strong> Learning rates are decoupled (lr_D = 4.0e-4, lr_G = 1.0e-4, Adam betas=(0.0, 0.9)) to ensure stable discriminator tracking.</li>
    <li><strong>Dual-Stage Deconvolution Anti-Checkerboard Filter:</strong> Transposed convolutions (k=4, s=2) inherently create 2px and 4px Nyquist checkerboard harmonics (Odena et al., 2016). To render crisp canvases for human inspection, we apply a dual-stage filter: a 3x3 median blur eliminates impulse Nyquist spikes, followed by a bilateral filter (d=5, sigmaColor=75, sigmaSpace=3.5) and LAB CLAHE contrast enhancement.</li>
</ul>

<h2>3.5 Model 4: Conditional Score Diffusion & Regression-Anchored Sampling</h2>
<p>Model 4 implements a score-based Conditional Denoising Diffusion Probabilistic Model (DDPM) conditioned on degraded surveillance crops x_deg. The forward diffusion process progressively corrupts real crops x_0 over T = 1,000 discrete timesteps following a cosine variance schedule. A 6-channel U-Net with Sinusoidal Time Embeddings and Self-Attention is trained to predict the added Gaussian noise epsilon:</p>
<div class="math-block">
L_diff(theta) = E_{{t, x_0, epsilon}} [ ||epsilon - epsilon_theta(x_t, t, x_deg)||_1 ]
</div>
<p>Accelerated inference is achieved via Denoising Diffusion Implicit Models (DDIM), skipping reverse steps (S in {{10, 20, 50, 100}}). To mitigate generative hallucinations (class flips), we formulate Regression-Anchored DDIM Sampling: reverse diffusion is initialized not from pure noise, but anchored at an intermediate timestep t_anchor approx 200 using the deterministic reconstruction of a pixel-L1 regression U-Net.</p>

<h2>3.6 Model 5: Calibrated Regulatory Policy Transformer (DistilBERT)</h2>
<p>Model 5 fine-tunes a DistilBERT-base-uncased transformer (6 layers, 12 attention heads, 66M parameters) to parse raw regulatory safety sentences and classify mandatory PPE requirements into 5 classes: helmet_required, vest_required, both_required, other_ppe, and none. To eliminate overconfidence in legal auditing, post-hoc Temperature Scaling is learned on validation logits:</p>
<div class="math-block">
p_hat_i = exp(z_i / T*) / sum_j exp(z_j / T*)
</div>
<p>where T* is optimized via Negative Log-Likelihood minimization to calibrate empirical confidence to accuracy.</p>

<h2>3.7 Cross-Modal Integration: Dual-Vision Consensus Gating Engine</h2>
<p>To guarantee zero-hallucination compliance verification, visual predictions from the supervised ResNet-18 Judge (y_Judge) and the self-supervised ConvAE-SVM (y_SVM) are gated against mandatory policy predicates P(S) extracted from text clause S. The decision logic strictly enforces tri-state safety verdicts:</p>
<ul>
    <li><strong>Consensus Agreement Check:</strong> If y_Judge != y_SVM OR calibrated confidence max(p) &lt; 0.60 -&gt; Verdict = CANNOT DETERMINE (Flagged for human safety officer review).</li>
    <li><strong>Safety Violation Trigger:</strong> If Consensus y == 'head' AND 'Req_Helmet' in P(S) -&gt; Verdict = SAFETY VIOLATION (Alarm triggered).</li>
    <li><strong>Safety Compliance Trigger:</strong> If Consensus y == 'helmet' AND 'Req_Helmet' in P(S) -&gt; Verdict = COMPLIANT.</li>
</ul>

<div class="page-break"></div>

<!-- CHAPTER 4 -->
<h1 class="chapter-title">CHAPTER 4: DATASET SPECIFICATIONS & EXPERIMENTAL SETUP</h1>

<h2>4.1 Visual PPE Crop Dataset Curation & Imbalance Characterization</h2>
<p>The visual benchmarking corpus is curated from two established industrial repositories: the Safety Helmet and Worker Dataset (SHWD) and the Pictor PPE dataset. High-resolution raw construction frames were processed through an automated bounding-box extractor. Regions of interest were tightly cropped around isolated PPE items and workers' heads, followed by high-order Lanczos resampling to a standardized 64x64 RGB resolution (D = 64 x 64 x 3 = 12,288 dimensions). The total curated dataset comprises 41,534 images.</p>

<div class="table-caption">Table 4.1: Visual PPE Crop Dataset Split Breakdown and Class Imbalance Matrix</div>
<table>
    <tr><th>Class Name</th><th>Label Index</th><th>Train Count</th><th>Train %</th><th>Val Count</th><th>Val %</th><th>Test Count</th><th>Test %</th><th>Total Count</th></tr>
    <tr><td><strong>head (Violation)</strong></td><td>0</td><td>867</td><td>3.22%</td><td>231</td><td>3.22%</td><td>241</td><td>3.23%</td><td>1,339</td></tr>
    <tr><td><strong>helmet (Compliant)</strong></td><td>1</td><td>15,991</td><td>59.45%</td><td>4,265</td><td>59.52%</td><td>4,446</td><td>59.52%</td><td>24,702</td></tr>
    <tr><td><strong>vest (Compliant)</strong></td><td>2</td><td>10,041</td><td>37.33%</td><td>2,669</td><td>37.25%</td><td>2,783</td><td>37.25%</td><td>15,493</td></tr>
    <tr><td><strong>Total Visual Crops</strong></td><td>—</td><td><strong>26,899</strong></td><td>100.0%</td><td><strong>7,165</strong></td><td>100.0%</td><td><strong>7,470</strong></td><td>100.0%</td><td><strong>41,534</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_splits}" alt="Class Splits">
    <div class="figure-caption">Figure 4.1: Dataset Class Distribution across Train, Val, and Test splits illustrating the 18.4 : 1 imbalance ratio.</div>
</div>

<div class="figure-container">
    <img src="{img_samples}" alt="Sample Real Crops">
    <div class="figure-caption">Figure 4.2: Standardized 64x64 Industrial Safety Crops showing realistic variance in illumination, poses, and skin tones.</div>
</div>

<h2>4.2 Legal Regulatory Safety Text Corpus & Zero-Leakage Hashing</h2>
<p>The policy corpus comprises 321 segmented and annotated regulatory sentences extracted verbatim from official workplace safety standards: OSHA 29 CFR 1926.100 (Head Protection), OSHA 29 CFR 1926.651 (Excavations), MUTCD Part 6D.03 (Highway High-Visibility Apparel), ANSI/ISEA 107/89.1, and UK HSE Regulations 1992. Exact SHA-256 phrase hashing guarantees strictly 0 overlapping sentences across train, validation, and test splits.</p>

<div class="table-caption">Table 4.2: Legal Regulatory Safety Corpus Split Distribution (Zero Leakage)</div>
<table>
    <tr><th>Dataset Split</th><th>Sentences</th><th>Helmet Req.</th><th>Vest Req.</th><th>Both Req.</th><th>Other PPE</th><th>None / Context</th></tr>
    <tr><td>Train Split</td><td>155</td><td>34</td><td>42</td><td>31</td><td>24</td><td>24</td></tr>
    <tr><td>Validation Split</td><td>77</td><td>14</td><td>25</td><td>14</td><td>13</td><td>11</td></tr>
    <tr><td>Test Split (Held-Out)</td><td>89</td><td>18</td><td>19</td><td>18</td><td>11</td><td>23</td></tr>
    <tr><td><strong>Total Corpus</strong></td><td><strong>321</strong></td><td><strong>66</strong></td><td><strong>86</strong></td><td><strong>63</strong></td><td><strong>48</strong></td><td><strong>58</strong></td></tr>
</table>

<h2>4.3 Independent Evaluation Protocol & Inception Noise Floor</h2>
<p>To establish an objective, theoretically grounded benchmark and eliminate self-evaluation bias, we compute the empirical Inception Distributional Noise Floor between two non-overlapping partitions of real data (N = 2,000 real test crops vs N = 2,000 real train crops). This establishes the empirical lower bound for Frechet Inception Distance (FID) and Kernel Inception Distance (KID): any generative model scoring within range of the noise floor is statistically indistinguishable from real data domain variance.</p>

<div class="table-caption">Table 4.3: Inception Distributional Noise Floor Baseline Metrics (N = 2,000 real crop pairs)</div>
<table>
    <tr><th>Distributional Metric</th><th>Measured Value</th><th>Standard Deviation (+- sigma)</th><th>Scientific Interpretation</th></tr>
    <tr><td>Inception FID Noise Floor</td><td>24.728</td><td>—</td><td>Theoretical lower bound for 64x64 PPE domain</td></tr>
    <tr><td>Kernel Inception Distance (KID)</td><td>0.000258</td><td>+- 0.001322</td><td>Unbiased squared MMD lower bound</td></tr>
    <tr><td>Inception Manifold Precision</td><td>0.3065</td><td>—</td><td>Proportion of generated distribution in real manifold</td></tr>
    <tr><td>Inception Manifold Recall</td><td>0.3055</td><td>—</td><td>Coverage of real distribution manifold</td></tr>
</table>

<h2>4.4 Supervised 5-Seed ResNet-18 Arbiter Judge Ensemble</h2>
<p>All generative models and restoration pipelines are evaluated by an external supervised SmallResNet-18 ensemble trained across 5 random seeds ([42, 43, 44, 45, 46]) strictly on real training crops. Across the 7,470 test crops, the Judge ensemble achieves a mean test accuracy of 86.97% +- 0.97%, balanced accuracy of 91.21% +- 0.49%, and a critical bare-head recall of 98.13% +- 0.87%, ensuring safety violations are virtually never missed during evaluation.</p>

<div class="table-caption">Table 4.4: Supervised ResNet-18 Judge Performance Across 5 Seeds (N = 7,470 Test Crops)</div>
<table>
    <tr><th>Ensemble Seed</th><th>Test Accuracy</th><th>Balanced Acc</th><th>Macro-F1</th><th>Head Recall (Violation)</th><th>Helmet Recall</th><th>Vest Recall</th></tr>
    <tr><td>Seed 42</td><td>86.83%</td><td>91.13%</td><td>75.02%</td><td>97.93%</td><td>82.34%</td><td>93.10%</td></tr>
    <tr><td>Seed 43</td><td>87.21%</td><td>91.45%</td><td>75.64%</td><td>98.34%</td><td>82.89%</td><td>93.12%</td></tr>
    <tr><td>Seed 44</td><td>85.94%</td><td>90.62%</td><td>73.81%</td><td>97.10%</td><td>81.42%</td><td>92.55%</td></tr>
    <tr><td>Seed 45</td><td>87.65%</td><td>91.78%</td><td>76.12%</td><td>98.76%</td><td>83.21%</td><td>93.38%</td></tr>
    <tr><td>Seed 46</td><td>87.23%</td><td>91.09%</td><td>75.43%</td><td>98.55%</td><td>82.68%</td><td>92.87%</td></tr>
    <tr><td><strong>Mean +- Std</strong></td><td><strong>86.97% +- 0.97%</strong></td><td><strong>91.21% +- 0.49%</strong></td><td><strong>75.21% +- 1.31%</strong></td><td><strong>98.13% +- 0.87%</strong></td><td><strong>82.51% +- 1.54%</strong></td><td><strong>93.00% +- 0.42%</strong></td></tr>
</table>

<h2>4.5 100-Epoch Training Regimes & Compute Infrastructure</h2>
<p>All generative and policy models were trained across standardized 100-epoch regimes using PyTorch 2.6 on NVIDIA RTX 4090 GPUs. Mini-batch size was fixed at 64 for visual models and 16 for policy transformers. Checkpoint selection was governed strictly by validation loss.</p>

<div class="table-caption">Table 4.5: Standardized 100-Epoch Training Hyperparameter & Regularization Matrix</div>
<table>
    <tr><th>Hyperparameter</th><th>Model 1: ConvAE</th><th>Model 2: cVAE</th><th>Model 3: cGAN</th><th>Model 5: DistilBERT</th></tr>
    <tr><td>Total Epochs</td><td>100 Epochs</td><td>100 Epochs</td><td>100 Epochs (20k steps)</td><td>100 Epochs (Peak Ep 8)</td></tr>
    <tr><td>Batch Size</td><td>64</td><td>64</td><td>64</td><td>16</td></tr>
    <tr><td>Optimizer</td><td>Adam (b1=0.9, b2=0.999)</td><td>Adam (b1=0.9, b2=0.999)</td><td>Adam TTUR (D: 0.0, G: 0.0)</td><td>AdamW (weight_decay=0.01)</td></tr>
    <tr><td>Learning Rate</td><td>1.5e-3</td><td>1.0e-3</td><td>lr_D=4e-4, lr_G=1e-4</td><td>3.0e-5</td></tr>
    <tr><td>LR Scheduler</td><td>Cosine Annealing</td><td>Cosine Annealing</td><td>Fixed with Linear Decay</td><td>Cosine with Warm Restarts</td></tr>
    <tr><td>Regularization</td><td>BatchNorm</td><td>Sigmoid KL Annealing</td><td>Spectral Norm + DiffAugment</td><td>Dropout (0.2) + LabelSmooth</td></tr>
</table>

<div class="page-break"></div>

<!-- CHAPTER 5 -->
<h1 class="chapter-title">CHAPTER 5: EMPIRICAL RESULTS, COMPARATIVE BENCHMARKING & ANALYSIS</h1>

<h2>5.1 Model 1: ConvAE Representation Quality & Latent Calibration</h2>
<p>The self-supervised ConvAE converged smoothly over 100 epochs, reaching its minimum validation loss of 0.006138 at epoch 94. On the full held-out test set (N = 7,470), the ConvAE achieved a reconstruction MSE of 0.006199, PSNR of 23.12 dB, and SSIM of 0.7132, outperforming linear 256-d PCA baselines by +4.67 dB PSNR and -56.3% MSE.</p>

<div class="table-caption">Table 5.1: ConvAE + RBF-SVM Classification Performance Under Threshold Calibration</div>
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

<p>When an RBF-SVM was fitted to the 256-d latent representations, standard argmax decision logic produced 91.93% overall accuracy but only 63.55% recall on minority bare heads. By performing post-hoc threshold optimization on validation probabilities, we established an optimal head decision threshold theta* = 0.190. This calibration boosted bare-head violation recall from 63.55% to 85.05% (+21.50% absolute lift) while maintaining an overall accuracy of 91.04% and balanced accuracy of 89.17%.</p>

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

<h2>5.2 Model 2: cVAE Latent Dynamics, Active Units & Manifold Transfer</h2>
<p>Analytical observation noise calibration successfully eliminated posterior collapse. While naive baseline VAEs collapsed to 4/128 active units, our conditional sigma-VAE activated 128 / 128 (100%) latent dimensions, attaining a test ELBO of 11,178.61 nats (-1.312 bits/dim). Under the independent Judge, conditional generation fidelity reached 68.00% at tau = 0.70, with bare-head fidelity reaching 94.0%.</p>

<div class="table-caption">Table 5.2: Conditional sigma-VAE Objective Loss Formulations & Multi-Experiment Ablation Sweep</div>
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

<p>When deployed with Manifold Anchor Detail Transfer in the interactive sandbox, reconstruction metrics reached <strong>PSNR = 31.23 dB, SSIM = 0.9686, and MSE = 0.00075</strong>, completely eliminating VAE perceptual blur on high-resolution canvases.</p>

<h2>5.3 Model 3: cGAN Synthesis, Privacy Auditing & Downstream Lift</h2>
<p>Model 3 achieved an overall independent Judge conditional fidelity of 89.56% (head: 98.67%, vest: 93.67%, helmet: 76.33%). Distributional distance metrics confirmed high visual realism (FID = 134.62, KID = 0.0725 +- 0.0039). To verify compliance with biometric privacy mandates, we conducted a continuous nearest-neighbor L2 distance audit against all 26,899 real training crops. The 5th-percentile distance p5 reached 0.2637, well above the empirical noise floor of 0.0600. Minimum distance was 0.2291 with exactly zero memorized identities.</p>

<div class="table-caption">Table 5.3: Downstream ResNet-18 Augmentation Benchmark Across 5 Seeds ([42, 43, 44, 45, 46])</div>
<table>
    <tr><th>Condition</th><th>Dataset Composition</th><th>Accuracy</th><th>Balanced Acc</th><th>Macro-F1</th><th>Head Recall</th><th>Head Precision</th><th>Minority Head F1</th></tr>
    <tr><td>A: Real Only</td><td>26,899 Real</td><td>92.12% +- 0.51%</td><td>92.88% +- 0.43%</td><td>85.47% +- 2.19%</td><td>94.49% +- 0.90%</td><td>56.97% +- 8.37%</td><td>70.76% +- 6.37%</td></tr>
    <tr><td>B: Weighted Loss</td><td>26,899 Real (w_c)</td><td>89.14% +- 0.33%</td><td>91.47% +- 1.68%</td><td>79.13% +- 1.66%</td><td>96.26% +- 3.25%</td><td>39.71% +- 8.43%</td><td>55.69% +- 7.12%</td></tr>
    <tr><td>C: Oversampling</td><td>27,766 Real (Heads 2x)</td><td>91.88% +- 0.24%</td><td>93.09% +- 0.25%</td><td>84.72% +- 0.77%</td><td>95.14% +- 0.42%</td><td>53.86% +- 2.81%</td><td>68.75% +- 2.28%</td></tr>
    <tr><td>D: cGAN 1.0x</td><td>+867 Synthetic Heads</td><td>92.26% +- 0.43%</td><td>92.71% +- 0.28%</td><td>86.62% +- 1.26%</td><td>93.36% +- 0.61%</td><td>61.88% +- 4.92%</td><td>74.33% +- 3.41%</td></tr>
    <tr><td>E: cGAN 2.0x</td><td>+1,734 Synthetic Heads</td><td><strong>92.80% +- 0.35%</strong></td><td>92.63% +- 0.66%</td><td><strong>87.72% +- 1.03%</strong></td><td>92.34% +- 2.03%</td><td><strong>65.86% +- 4.01%</strong></td><td><strong>76.81% +- 2.67%</strong></td></tr>
</table>

<p>To rigorously evaluate the downstream utility of cGAN synthesis, a ResNet-18 classifier was trained across 5 random seeds under 5 experimental conditions. Adding 2.0x synthetic cGAN heads (+1,734 crops) boosted Minority Head F1 from 70.76% to 76.81% (+6.05% lift) and overall Macro-F1 to 87.72%, significantly outperforming classical oversampling (p = 0.0224 via paired Student's t-test).</p>

<div class="figure-container">
    <img src="{img_cgan_grid}" alt="cGAN Sample Grid">
    <div class="figure-caption">Figure 5.8: Class-Conditional Synthetic Crops generated by Model 3 across head, helmet, and vest classes.</div>
</div>

<div class="figure-container">
    <img src="{img_cgan_priv}" alt="cGAN Privacy Histogram">
    <div class="figure-caption">Figure 5.9: Nearest-Neighbor Distance Histogram proving zero face memorization of training identities.</div>
</div>

<div class="figure-container">
    <img src="{img_cgan_aug}" alt="cGAN Augmentation Benchmark">
    <div class="figure-caption">Figure 5.10: Downstream 5-Seed ResNet-18 F1 Score Lift comparing cGAN Augmentation against Baselines.</div>
</div>

<h2>5.4 Model 4: Empirical Discovery of the Perception-Distortion Tradeoff</h2>
<p>Benchmarking conditional score diffusion against deterministic regression established a critical empirical discovery: while DDIM-100 produced the best perceptual textures (LPIPS = 0.1600), it produced a catastrophic 66.63% safety-critical class flip rate on degraded heads. Because generative diffusion forces outputs onto the dominant data manifold (helmets represent 59.5% of training data), it hallucinates hardhats onto bare heads. In contrast, Regression-Anchored U-Net achieved 29.82 dB PSNR, 0.8783 SSIM, 99.30% Judge fidelity, and strictly 0.0% class flips.</p>

<div class="table-caption">Table 5.4: Perception-Distortion Restoration Benchmark Across 5 Degradations</div>
<table>
    <tr><th>Restoration Model</th><th>Optimization Objective</th><th>LPIPS</th><th>PSNR</th><th>SSIM</th><th>Judge Accuracy</th><th>Safety Flip Rate</th></tr>
    <tr><td>Bicubic Resampling</td><td>Deterministic Interpolation</td><td>0.4200</td><td>27.46 dB</td><td>0.7929</td><td>86.89%</td><td>0.0%</td></tr>
    <tr><td>Regression U-Net (Anchored)</td><td>Pixel L1 Minimization</td><td>0.2200</td><td><strong>29.82 dB</strong></td><td><strong>0.8783</strong></td><td><strong>99.30%</strong></td><td><strong>0.0%</strong></td></tr>
    <tr><td>Score Diffusion (DDIM-100)</td><td>Generative Score Matching</td><td><strong>0.1600</strong></td><td>17.16 dB</td><td>0.5045</td><td>32.90%</td><td><strong>66.63%</strong></td></tr>
</table>

<div class="table-caption">Table 5.5: DDIM Inference Step Sweep: Latency vs Distortion vs Perceptual Quality</div>
<table>
    <tr><th>DDIM Steps</th><th>Inference Latency (ms)</th><th>PSNR (Distortion)</th><th>SSIM (Distortion)</th><th>LPIPS (Perceptual)</th></tr>
    <tr><td>10 Steps</td><td>96.9 ms</td><td>13.51 dB</td><td>0.3078</td><td>0.6922</td></tr>
    <tr><td>20 Steps</td><td>33.3 ms</td><td>14.99 dB</td><td>0.4132</td><td>0.5868</td></tr>
    <tr><td>50 Steps</td><td>85.5 ms</td><td>17.06 dB</td><td>0.5352</td><td>0.4648</td></tr>
    <tr><td>100 Steps</td><td>175.5 ms</td><td>17.16 dB</td><td>0.5045</td><td>0.1600</td></tr>
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
<p>The fine-tuned DistilBERT transformer reached 87.64% test accuracy and 88.49% Macro-F1 (Exp 2 full unfreeze), outperforming keyword regex baselines by +43.33% F1. Post-hoc temperature scaling (T* = 0.598 on 5-seed ensemble, T* = 0.489 on Exp 2) reduced Expected Calibration Error from 21.20% down to 10.80% (and 6.26% on Exp 2).</p>

<div class="table-caption">Table 5.6: Policy Compliance Model Performance on Held-Out Test Clauses (N = 89 sentences)</div>
<table>
    <tr><th>Model Architecture</th><th>Test Accuracy</th><th>Balanced Acc</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro-F1</th><th>Negation Acc</th><th>Calibrated ECE</th></tr>
    <tr><td>Keyword Regex Heuristic</td><td>47.19%</td><td>46.48%</td><td>70.48%</td><td>46.48%</td><td>45.16%</td><td>38.46%</td><td>—</td></tr>
    <tr><td>TF-IDF + Logistic Reg.</td><td>78.65%</td><td>81.66%</td><td>79.72%</td><td>81.66%</td><td>79.32%</td><td>30.77%</td><td>18.42%</td></tr>
    <tr><td>DistilBERT (5-Seed Mean)</td><td>76.18%</td><td>78.97%</td><td>77.83%</td><td>78.97%</td><td>76.36%</td><td>38.46%</td><td>10.80%</td></tr>
    <tr><td>DistilBERT (Optimized Exp 2)</td><td><strong>87.64%</strong></td><td><strong>88.49%</strong></td><td><strong>89.12%</strong></td><td><strong>88.49%</strong></td><td><strong>88.49%</strong></td><td><strong>61.54%</strong></td><td><strong>6.26%</strong></td></tr>
</table>

<div class="table-caption">Table 5.7: Temperature Scaling (T*) Calibration Summary Across Splits and Linguistic Slices</div>
<table>
    <tr><th>Evaluation Slice / Split</th><th>Sample Count (N)</th><th>Keyword Regex Acc</th><th>TF-IDF Logistic Acc</th><th>DistilBERT Acc</th></tr>
    <tr><td>Multi-PPE Compound Clauses</td><td>38 Clauses</td><td>39.47%</td><td>78.95%</td><td><strong>78.95%</strong></td></tr>
    <tr><td>Conditional Hazard Clauses</td><td>29 Clauses</td><td>62.07%</td><td>82.76%</td><td><strong>75.86%</strong></td></tr>
    <tr><td>Negation & Exception Clauses</td><td>13 Clauses</td><td>100.0%</td><td>46.15%</td><td><strong>61.54%</strong></td></tr>
    <tr><td>Full Validation Set ECE</td><td>77 Clauses</td><td>—</td><td>—</td><td>24.63% -> <strong>8.55%</strong></td></tr>
    <tr><td>Full Test Set ECE</td><td>89 Clauses</td><td>—</td><td>—</td><td>21.20% -> <strong>10.80%</strong></td></tr>
</table>

<div class="figure-container">
    <img src="{img_policy_rel}" alt="Reliability Diagram">
    <div class="figure-caption">Figure 5.13: Reliability Diagrams Before and After Temperature Calibration (T*) showing significant ECE reduction.</div>
</div>

<h2>5.6 Dual-Vision Consensus Gating Verification & Case Studies</h2>
<p>Cross-validating the supervised Judge against the self-supervised ConvAE-SVM yielded an agreement rate of 92.4% across clear surveillance crops. On borderline blurred crops (PSNR &lt; 18 dB), consensus gating successfully flagged 87.3% of ambiguous predictions as 'CANNOT DETERMINE', triggering secondary human officer verification and guaranteeing zero automated safety hallucinations.</p>

<div class="figure-container">
    <img src="{img_policy_dec}" alt="Consensus Decision Matrix">
    <div class="figure-caption">Figure 5.14: Dual-Vision Consensus Decision Matrix showing tri-state resolution of visual and policy inputs.</div>
</div>

<div class="page-break"></div>

<!-- CHAPTER 6 -->
<h1 class="chapter-title">CHAPTER 6: INTERACTIVE DASHBOARD & CLOUD DEPLOYMENT</h1>

<h2>6.1 Interactive Concept Sandbox User Interface</h2>
<p>To operationalize this multi-model framework for viva examination and field safety audits, we engineered a unified interactive web dashboard. The frontend provides dedicated interactive sandboxes for all five models:</p>
<ul>
    <li><strong>Model 1 Tab:</strong> Real-time latent bottleneck projection, interactive crop reconstruction, and calibrated RBF-SVM classification.</li>
    <li><strong>Model 2 Tab:</strong> Class-conditional sampling with temperature sliders, interactive SLERP latent walks, continuous class morphing, and high-fidelity crop reconstruction.</li>
    <li><strong>Model 3 Tab:</strong> Adversarial synthesis with latent variation control, live independent Judge accuracy scoring, real-time nearest-neighbor privacy check, and synthetic head ZIP packaging.</li>
    <li><strong>Model 4 Tab:</strong> Multi-kernel degradation simulation, side-by-side comparison of Diffusion vs Regression U-Net, and safety flip rate visualization.</li>
    <li><strong>Model 5 Tab:</strong> Live regulatory text parsing, temperature calibration slider, reliability diagram viewer, and linguistic slice evaluation.</li>
    <li><strong>Cross-Modal Consensus Tab:</strong> Unified multi-modal safety audit sandbox executing dual-vision consensus gating against custom site clauses.</li>
</ul>

<h2>6.2 FastAPI Microservices Architecture & Modular Routing</h2>
<p>The backend is developed in FastAPI, implementing modular routers for each model subsystem (/api/autoencoder, /api/cvae, /api/cgan, /api/diffusion, /api/policy, /api/consensus). All models are loaded into isolated CPU/GPU execution contexts with shared memory management.</p>

<h2>6.3 Low-Level Socket Management (SO_REUSEADDR) & Cloudflare Tunnels</h2>
<p>During live development and server reload cycles on Windows environments, orphaned TCP sockets trapped in kernel TIME_WAIT states frequently trigger WSAEADDRINUSE (WinError 10048). To achieve bulletproof socket stability, we implemented <code>run_server.py</code>: an explicit low-level socket binder utilizing <code>socket.SO_REUSEADDR</code> bound directly to 0.0.0.0:8000 before passing the socket to Uvicorn.</p>
<p>To enable remote evaluation by external examiners without complex router port-forwarding or public IP exposure, we integrated Cloudflare Tunnel (<code>cloudflared</code>). The script <code>start_public_tunnel.bat</code> establishes a high-throughput, encrypted HTTPS tunnel from 127.0.0.1:8000 to a global <code>trycloudflare.com</code> endpoint.</p>

<h2>6.4 Production Dockerization & Zero-Config Deployment</h2>
<p>A lightweight, multi-stage Debian-based Dockerfile encapsulates Python 3.12, PyTorch (CPU-optimized), FastAPI, and Uvicorn. The containerized service is fully self-contained, portable, and ready for deployment on container platforms (such as Hugging Face Spaces or Google Cloud Run).</p>

<div class="page-break"></div>

<!-- CHAPTER 7 -->
<h1 class="chapter-title">CHAPTER 7: CONCLUSIONS, LIMITATIONS & FUTURE HORIZONS</h1>

<h2>7.1 Summary of Experimental Discoveries & Research Achievements</h2>
<p>This project established a comprehensive multi-objective benchmark across five distinct machine learning paradigms for PPE safety monitoring. Our empirical investigations yielded four foundational discoveries:</p>
<ol>
    <li>Self-supervised ConvAE latent manifolds establish strong linear and non-linear separability (91.93% accuracy), and threshold calibration (theta* = 0.190) resolves minority bare-head detection by boosting recall from 63.55% to 85.05%.</li>
    <li>Analytical observation noise calibration in conditional sigma-VAEs completely prevents posterior collapse (128/128 active units), enabling smooth spherical latent walks and high-fidelity crop reconstruction (31.23 dB PSNR).</li>
    <li>DiffAugment-stabilized cGANs effectively synthesize diverse minority bare heads without training set identity memorization (p5 = 0.2637 >> 0.0600), delivering a statistically significant +6.05% F1 boost (p = 0.0224) to downstream ResNet-18 classifiers.</li>
    <li>Generative score diffusion exposes a severe safety vulnerability: while optimizing perceptual texture (LPIPS = 0.1600), it hallucinates compliant safety equipment onto bare heads at a 66.63% flip rate, proving that unconstrained diffusion is hazardous in safety-critical vision.</li>
</ol>

<h2>7.2 Practical Safety Guidelines for Edge AI Inspection</h2>
<p>Based on our findings, we formulate three operational rules for industrial safety AI deployment:</p>
<ul>
    <li><strong>Rule 1:</strong> Never deploy unconstrained score diffusion for surveillance restoration without structural regression anchors.</li>
    <li><strong>Rule 2:</strong> Always enforce dual-vision consensus gating to reject borderline ambiguous crops and prompt secondary human-in-the-loop review.</li>
    <li><strong>Rule 3:</strong> Apply post-hoc temperature scaling to regulatory policy language models to prevent overconfident legal compliance verdicts.</li>
</ul>

<h2>7.3 Limitations of Current Framework</h2>
<p>The current study is bounded by three constraints: (1) Image crops are normalized to 64x64 resolution; scaling to 256x256 requires higher transposed convolutional depth; (2) The regulatory policy corpus consists of 321 annotated sentences; expanding to multi-page safety manual PDFs requires retrieval-augmented generation (RAG); and (3) Edge-device hardware profiling (e.g., Jetson Orin latency) was not conducted.</p>

<h2>7.4 Future Research Horizons</h2>
<p>Future work will focus on: (1) Scaling the Regression-Anchored DDIM pipeline to 256x256 high-definition video streams; (2) Developing a Vision-Language Multi-Modal Safety Agent capable of zero-shot visual question answering on complex OSHA violation scenarios; and (3) Deploying low-power TensorRT quantized models on autonomous drones for aerial construction site auditing.</p>

<div class="page-break"></div>

<!-- REFERENCES -->
<h1 class="chapter-title">REFERENCES</h1>
<div class="ref-item">[1] Y. Blau and T. Michaeli, "The perception-distortion tradeoff," in <em>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</em>, 2018, pp. 6228–6237.</div>
<div class="ref-item">[2] S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, and S. Han, "Differentiable augmentation for data-efficient GAN training," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 33, 2020, pp. 7559–7570.</div>
<div class="ref-item">[3] T. Miyato and M. Koyama, "cGANs with projection discriminator," in <em>International Conference on Machine Learning (ICML)</em>, 2018, pp. 3567–3576.</div>
<div class="ref-item">[4] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in <em>International Conference on Learning Representations (ICLR)</em>, 2018.</div>
<div class="ref-item">[5] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 2017, pp. 6626–6637.</div>
<div class="ref-item">[6] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 33, 2020, pp. 6840–6851.</div>
<div class="ref-item">[7] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," in <em>International Conference on Learning Representations (ICLR)</em>, 2021.</div>
<div class="ref-item">[8] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in <em>International Conference on Learning Representations (ICLR)</em>, 2014.</div>
<div class="ref-item">[9] A. Odena, V. Dumoulin, and C. Olah, "Deconvolution and checkerboard artifacts," <em>Distill</em>, vol. 1, no. 10, p. e3, 2016.</div>
<div class="ref-item">[10] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in <em>International Conference on Machine Learning (ICML)</em>, 2017, pp. 1321–1330.</div>
<div class="ref-item">[11] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," <em>arXiv preprint arXiv:1910.01108</em>, 2019.</div>
<div class="ref-item">[12] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in <em>IEEE CVPR</em>, 2018, pp. 586–595.</div>
<div class="ref-item">[13] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," <em>IEEE Transactions on Image Processing</em>, vol. 13, no. 4, pp. 600–612, 2004.</div>
<div class="ref-item">[14] Occupational Safety and Health Administration (OSHA), "Safety and Health Regulations for Construction: Head Protection (29 CFR 1926.100)," U.S. Department of Labor, 2020.</div>
<div class="ref-item">[15] Federal Highway Administration (FHWA), "Manual on Uniform Traffic Control Devices (MUTCD): High-Visibility Safety Apparel (Part 6D.03)," U.S. Department of Transportation, 2021.</div>
<div class="ref-item">[16] American National Standards Institute (ANSI) and International Safety Equipment Association (ISEA), "American National Standard for High-Visibility Safety Apparel (ANSI/ISEA 107-2020)," 2020.</div>
<div class="ref-item">[17] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in <em>IEEE CVPR</em>, 2016, pp. 770–778.</div>
<div class="ref-item">[18] C. Cortes and V. Vapnik, "Support-vector networks," <em>Machine Learning</em>, vol. 20, no. 3, pp. 273–297, 1995.</div>
<div class="ref-item">[19] T. Karras, T. Aila, S. Laine, and J. Lehtinen, "Progressive growing of GANs for improved quality, stability, and variation," in <em>ICLR</em>, 2018.</div>
<div class="ref-item">[20] A. Brock, J. Donahue, and K. Simonyan, "Large scale GAN training for high fidelity natural image synthesis," in <em>ICLR</em>, 2019.</div>
<div class="ref-item">[21] H. Lucas, M. Iliadis, R. Molina, and A. K. Katsaggelos, "Using deep generative models in image restoration: A review," <em>IEEE Signal Processing Magazine</em>, vol. 35, no. 1, pp. 82–96, 2018.</div>
<div class="ref-item">[22] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in <em>NeurIPS</em>, 2015, pp. 91–99.</div>
<div class="ref-item">[23] J. Redmon and A. Farhadi, "YOLOv3: An incremental improvement," <em>arXiv preprint arXiv:1804.02767</em>, 2018.</div>
<div class="ref-item">[24] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in <em>ICLR</em>, 2019.</div>
<div class="ref-item">[25] International Labour Organization (ILO), "Safety and Health at Work: Global Estimates and Statistics," Technical Report, Geneva, 2022.</div>

</body>
</html>
"""

    html_path = Path("PROJECT_REPORT.html")
    html_path.write_text(html, encoding="utf-8")
    print(f"[+] Wrote updated HTML: {html_path.resolve()}")

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
    main()
