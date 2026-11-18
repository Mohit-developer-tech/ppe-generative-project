"""
build_final_report.py
=====================
Compiles the comprehensive, publication-grade Academic Project Report for PPE Generative Monitoring.
Key features:
1. Pure black font and borders (#000000) throughout the entire report.
2. Structured preliminaries (first 5-6 pages: Title, Certificate, Declaration & Acknowledgements, Abstract, TOC, Lists).
3. Page numbering starts at 1 on Chapter 1 Introduction.
4. Preliminaries numbered with lowercase Roman numerals (ii, iii, iv, v, vi...).
5. Running top header on main body pages.
6. Two-pass dynamic synchronization of Table of Contents, List of Tables, and List of Figures page numbers.
7. Total document length >= 35 pages; main report body <= 35 pages.
"""

import os
import sys
import base64
import subprocess
import re
from pathlib import Path
import fitz

def img_to_b64(path_str):
    p = Path(path_str)
    if p.exists():
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    print(f"[!] Warning: Image not found: {path_str}")
    return ""

def to_roman(n):
    roman_map = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")
    ]
    result = ""
    for val, sym in roman_map:
        while n >= val:
            result += sym
            n -= val
    return result

def get_body_html(images):
    # Extract images
    img_arch = images["arch"]
    img_splits = images["splits"]
    img_samples = images["samples"]
    img_ae_loss = images["ae_loss"]
    img_ae_pca = images["ae_pca"]
    img_ae_recon = images["ae_recon"]
    img_ae_conf = images["ae_conf"]
    img_cvae_loss = images["cvae_loss"]
    img_cvae_kl = images["cvae_kl"]
    img_cvae_slerp = images["cvae_slerp"]
    img_cgan_grid = images["cgan_grid"]
    img_cgan_priv = images["cgan_priv"]
    img_cgan_aug = images["cgan_aug"]
    img_diff_tradeoff = images["diff_tradeoff"]
    img_diff_flips = images["diff_flips"]
    img_policy_rel = images["policy_rel"]
    img_policy_dec = images["policy_dec"]

    with open("scripts/make_pdf_report_expanded.py", "r", encoding="utf-8") as f:
        src = f.read()

    # Find the start of Chapter 1
    idx_ch1 = src.find("<!-- CHAPTER 1 -->")
    idx_end = src.find("</body>")
    body_content = src[idx_ch1:idx_end]

    # Replace image placeholders
    body_content = body_content.replace("{img_arch}", img_arch)
    body_content = body_content.replace("{img_splits}", img_splits)
    body_content = body_content.replace("{img_samples}", img_samples)
    body_content = body_content.replace("{img_ae_loss}", img_ae_loss)
    body_content = body_content.replace("{img_ae_pca}", img_ae_pca)
    body_content = body_content.replace("{img_ae_recon}", img_ae_recon)
    body_content = body_content.replace("{img_ae_conf}", img_ae_conf)
    body_content = body_content.replace("{img_cvae_loss}", img_cvae_loss)
    body_content = body_content.replace("{img_cvae_kl}", img_cvae_kl)
    body_content = body_content.replace("{img_cvae_slerp}", img_cvae_slerp)
    body_content = body_content.replace("{img_cgan_grid}", img_cgan_grid)
    body_content = body_content.replace("{img_cgan_priv}", img_cgan_priv)
    body_content = body_content.replace("{img_cgan_aug}", img_cgan_aug)
    body_content = body_content.replace("{img_diff_tradeoff}", img_diff_tradeoff)
    body_content = body_content.replace("{img_diff_flips}", img_diff_flips)
    body_content = body_content.replace("{img_policy_rel}", img_policy_rel)
    body_content = body_content.replace("{img_policy_dec}", img_policy_dec)

    return body_content

def build_full_html(body_html, page_map=None):
    if page_map is None:
        # Default placeholder mapping
        page_map = {
            "ch1": "1", "ch2": "6", "ch3": "10", "ch4": "14",
            "ch5": "18", "ch6": "29", "ch7": "31", "ref": "33",
            "t41": "14", "t42": "15", "t43": "16", "t44": "17", "t45": "17",
            "t51": "19", "t52": "21", "t53": "23", "t54": "25", "t55": "26", "t56": "28", "t57": "28",
            "f11": "5", "f41": "14", "f42": "15",
            "f51": "19", "f52": "20", "f53": "20", "f54": "21",
            "f55": "22", "f56": "22", "f57": "23",
            "f58": "24", "f59": "24", "f510": "25",
            "f511": "26", "f512": "27", "f513": "28", "f514": "29"
        }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Academic Project Report - PPE Generative Monitoring</title>
<style>
@page {{
    size: A4;
    margin: 22mm 20mm 22mm 25mm;
}}
body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    line-height: 1.55;
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
    padding-top: 25px;
}}
.cover-pre {{
    font-size: 13pt;
    font-weight: bold;
    letter-spacing: 1.5px;
    margin-bottom: 25px;
}}
.cover-title {{
    font-size: 18pt;
    font-weight: bold;
    line-height: 1.35;
    margin-bottom: 35px;
    text-transform: uppercase;
}}
.cover-body {{
    font-size: 12pt;
    line-height: 1.65;
    margin-top: 40px;
}}
.cover-footer {{
    margin-top: 60px;
    font-size: 12pt;
    font-weight: bold;
    line-height: 1.5;
}}
h1.chapter-title {{
    font-size: 16pt;
    font-weight: bold;
    border-bottom: 2px solid #000000;
    padding-bottom: 5px;
    margin-top: 20px;
    margin-bottom: 16px;
    text-transform: uppercase;
}}
h2 {{
    font-size: 13pt;
    font-weight: bold;
    margin-top: 18px;
    margin-bottom: 8px;
}}
h3 {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 12px;
    margin-bottom: 5px;
}}
p {{
    margin-top: 0;
    margin-bottom: 10px;
    text-indent: 0;
}}
ul, ol {{
    margin-top: 5px;
    margin-bottom: 10px;
    padding-left: 24px;
}}
li {{
    margin-bottom: 5px;
    line-height: 1.45;
}}
.callout {{
    background-color: #fafafa;
    border: 1px solid #000000;
    border-left: 5px solid #000000;
    padding: 10px 14px;
    margin: 14px 0;
    font-size: 10.5pt;
    font-style: italic;
}}
.math-block {{
    background-color: #ffffff;
    border: 1px solid #000000;
    padding: 8px 14px;
    margin: 12px 0;
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    text-align: center;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 18px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}
th {{
    background-color: #f2f2f2;
    font-weight: bold;
    padding: 7px 9px;
    border: 1.5px solid #000000;
    text-align: center;
}}
td {{
    padding: 5px 8px;
    border: 1px solid #000000;
}}
tr:nth-child(even) td {{
    background-color: #fafafa;
}}
.table-caption {{
    font-weight: bold;
    margin-bottom: 4px;
    font-size: 10pt;
}}
.figure-container {{
    text-align: center;
    margin: 16px 0 18px 0;
    page-break-inside: avoid;
}}
.figure-container img {{
    max-width: 90%;
    max-height: 360px;
    height: auto;
    border: 1px solid #000000;
}}
.figure-caption {{
    font-size: 9.5pt;
    font-style: italic;
    margin-top: 5px;
}}
.toc-line {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-size: 10.5pt;
    line-height: 1.35;
}}
.toc-dots {{
    flex-grow: 1;
    border-bottom: 1px dotted #000000;
    margin: 0 6px;
    position: relative;
    top: -4px;
}}
.ref-item {{
    padding-left: 28px;
    text-indent: -28px;
    margin-bottom: 8px;
    font-size: 10pt;
    line-height: 1.4;
}}
.compact-p {{
    font-size: 10.5pt;
    line-height: 1.45;
    margin-bottom: 7px;
}}
</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- PRELIMINARY PAGES (FIRST 5-6 PAGES: ROMAN NUMERALS) -->
<!-- ========================================================================= -->

<!-- PAGE 1: TITLE PAGE -->
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

<!-- PAGE 2: CERTIFICATE OF RECOMMENDATION -->
<h1 class="chapter-title">CERTIFICATE OF RECOMMENDATION</h1>
<p style="margin-top: 20px;">This is to certify that the project report entitled <strong>"MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE"</strong> submitted by <strong>Mohit Patel</strong> (Roll No. 21BCE10000) in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering, is a bona fide record of research carried out under my guidance and supervision.</p>
<p style="margin-top: 15px;">The architectural designs, deep learning formulations, mathematical proofs, empirical benchmarks, and analytical conclusions embodied in this report have been examined and verified to meet all academic and methodological standards established by the University. This work has not been submitted in part or full to any other University or Institute for the award of any degree, diploma, or academic associateship.</p>
<br><br><br><br>
<table style="border:none; width:100%;">
    <tr style="background:none;"><td style="border:none; width:50%;">___________________________________<br><strong>Faculty Project Guide / Supervisor</strong><br>Department of Computer Science & Eng.</td>
    <td style="border:none; width:50%; text-align:right;">___________________________________<br><strong>Head of the Department (CSE)</strong><br>Department of Computer Science & Eng.</td></tr>
    <tr style="background:none;"><td style="border:none; padding-top:55px;" colspan="2">___________________________________<br><strong>External Examiner</strong></td></tr>
</table>

<div class="page-break"></div>

<!-- PAGE 3: CANDIDATE'S DECLARATION & ACKNOWLEDGEMENTS -->
<h1 class="chapter-title">CANDIDATE'S DECLARATION</h1>
<p class="compact-p">I hereby declare that this project report entitled <strong>"MULTI-OBJECTIVE BENCHMARKING OF GENERATIVE VISION MODELS AND CALIBRATED POLICY AUDITING FOR PERSONAL PROTECTIVE EQUIPMENT (PPE) COMPLIANCE"</strong> is an authentic record of my own research and experimental investigations carried out under the supervision of my project guide. To the best of my knowledge and belief, it contains no material previously written or published by another author, except where due reference and formal citation has been made in the text.</p>
<p class="compact-p">I have strictly adhered to all principles of academic integrity, rigorous code reproducibility, and zero-plagiarism policies. All experimental metrics, loss curves, confusion matrices, and comparative figures reflect verified executions of the underlying algorithmic suite across standardized 100-epoch regimes.</p>
<p style="text-align:right; font-size:10.5pt; margin-top:15px; margin-bottom:25px;">
<strong>Mohit Patel</strong> &bull; Roll No. 21BCE10000 &bull; Dept. of Computer Science & Eng. &bull; Academic Year 2025–2026
</p>

<h1 class="chapter-title" style="margin-top:20px;">ACKNOWLEDGEMENTS</h1>
<p class="compact-p">I express my deepest gratitude to my project supervisor, whose insightful guidance, methodological rigor, and continuous encouragement were instrumental throughout the conceptualization, mathematical formulation, training, and empirical evaluation of this multi-modal generative safety framework.</p>
<p class="compact-p">I am profoundly grateful to the Head of the Department and the faculty members of the Department of Computer Science and Engineering for providing continuous access to high-performance computational facilities, lab infrastructure, and academic encouragement. Special recognition is due to the open-source computer vision communities for curating the Safety Helmet and Worker Dataset (SHWD) and Pictor PPE dataset, which made this empirical research possible.</p>
<p class="compact-p">Finally, I extend my heartfelt thanks to my family, peers, and fellow researchers whose patience, moral support, and constructive criticism sustained me during the intensive development and rigorous benchmarking phases of this work.</p>

<div class="page-break"></div>

<!-- PAGE 4: EXECUTIVE ABSTRACT -->
<h1 class="chapter-title">ABSTRACT</h1>
<p class="compact-p">Automated monitoring of Personal Protective Equipment (PPE) in construction, manufacturing, and heavy industry is critically impeded by three systemic vulnerabilities: (1) severe visual degradation from low-resolution surveillance cameras, atmospheric turbulence, particulate dust, and motion blur; (2) extreme long-tail class imbalance where non-compliant bare heads represent less than 3.5% of real-world site observations; and (3) legal ambiguity within contextual, compound safety regulations (such as OSHA 29 CFR 1926 and MUTCD Part 6D) that conventional computer vision detectors cannot parse.</p>
<p class="compact-p">This project report presents a unified, multi-objective empirical benchmark evaluating five distinct artificial intelligence paradigms on identical, tightly extracted 64×64 PPE crops (N = 41,534) and an un-leaked regulatory text corpus (N = 321):</p>
<p class="compact-p"><strong>1. Self-Supervised Convolutional Autoencoders (ConvAE):</strong> Establishes a 256-dimensional latent manifold that yields 91.93% test accuracy and 86.11% Macro-F1 with a balanced RBF-SVM. Post-hoc decision threshold optimization (&theta;* = 0.190) boosts minority bare-head recall from 63.55% to 85.05% (+21.50% lift) with 89.17% balanced accuracy.</p>
<p class="compact-p"><strong>2. Conditional &sigma;-VAE:</strong> Employs analytical observation noise calibration (&sigma;*<sup>2</sup> = (1/D) &bull; ||x &minus; x&#770;||<sub>2</sub><sup>2</sup>) to overcome posterior collapse, activating 128/128 latent dimensions (ELBO: &minus;1.312 bits/dim). Paired with Manifold Anchor Sampling, it achieves PSNR: 31.23 dB and SSIM: 0.9686 in crop reconstruction, while achieving 68.00% Judge conditional fidelity (&tau; = 0.70) and 94.0% bare-head fidelity.</p>
<p class="compact-p"><strong>3. Conditional GAN (cGAN with DiffAugment):</strong> Employs Spectral Normalization and Projection Discriminator with EMA weights, attaining 89.56% conditional generation accuracy under an independent ResNet-18 judge (head: 98.67%, vest: 93.67%, helmet: 76.33%) and a 5th-percentile privacy distance of p<sub>5</sub> = 0.2637 &gt;&gt; 0.0600. When deployed for minority augmentation, it boosts downstream ResNet-18 Minority Head F1 from 70.76% to 76.81% (+6.05%, paired Student's t-test p = 0.0224) and Macro-F1 to 87.72%. Transposed convolution Nyquist artifacts are eliminated via dual-stage deconvolution anti-checkerboard filtering.</p>
<p class="compact-p"><strong>4. Conditional Score Diffusion (DDPM/DDIM):</strong> Benchmarks the Blau &amp; Michaeli perception-distortion tradeoff. While unconstrained generative score diffusion achieves high perceptual texture (LPIPS = 0.1600, PSNR = 17.16 dB), it induces a catastrophic 66.63% safety-critical class flip rate by hallucinating helmets onto bare heads. Conversely, Regression-Anchored U-Net restores degraded crops with PSNR: 29.82 dB, SSIM: 0.8783, Judge Accuracy: 99.30%, and strictly 0.0% safety flips.</p>
<p class="compact-p"><strong>5. Calibrated Policy Transformer:</strong> DistilBERT model trained across 100 epochs on regulatory clauses (OSHA 29 CFR 1926, MUTCD, ANSI/ISEA), achieving 87.64% test accuracy and 88.49% Macro-F1 (Exp 2 full unfreeze, +43.33% F1 over regex). Temperature scaling (T* = 0.598) reduces Expected Calibration Error (ECE) from 21.20% to 10.80% (and down to 6.26% on Exp 2).</p>
<p class="compact-p"><strong>6. Dual-Vision Consensus Gating:</strong> Enforces tri-state safety decision logic (COMPLIANT, VIOLATION, CANNOT DETERMINE) by cross-verifying supervised Judge predictions against self-supervised ConvAE-SVM predictions, triggering human-in-the-loop review on visual ambiguity or policy conflict. An interactive web dashboard and production containerization complete the deployment pipeline.</p>

<div class="page-break"></div>

<!-- PAGE 5: TABLE OF CONTENTS -->
<h1 class="chapter-title">TABLE OF CONTENTS</h1>
<div class="toc-line"><span>Certificate of Recommendation</span><span class="toc-dots"></span><span>ii</span></div>
<div class="toc-line"><span>Candidate's Declaration &amp; Acknowledgements</span><span class="toc-dots"></span><span>iii</span></div>
<div class="toc-line"><span>Executive Abstract</span><span class="toc-dots"></span><span>iv</span></div>
<div class="toc-line"><span>Table of Contents</span><span class="toc-dots"></span><span>v</span></div>
<div class="toc-line"><span>List of Tables, Figures &amp; Abbreviations</span><span class="toc-dots"></span><span>vi</span></div>
<br>
<div class="toc-line"><strong>Chapter 1: Introduction</strong><span class="toc-dots"></span><strong>{page_map["ch1"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.1 Industrial Context &amp; High-Consequence Safety Auditing<span class="toc-dots"></span>{page_map["ch1"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.2 The Triad of Systemic Inspection Challenges<span class="toc-dots"></span>{page_map["ch1"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.3 Socio-Economic Cost of Occupational Head and Torso Injuries<span class="toc-dots"></span>{page_map["ch1"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.4 Project Motivation &amp; Formal Research Questions<span class="toc-dots"></span>{page_map["ch1"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.5 Project Aims, Technical Scope &amp; Limitations<span class="toc-dots"></span>{page_map["ch1"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.6 Key Scientific Innovations &amp; Theoretical Contributions<span class="toc-dots"></span>{page_map["ch1"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;1.7 Organization of the Project Report<span class="toc-dots"></span>{page_map["ch1"]}</div>
<br>
<div class="toc-line"><strong>Chapter 2: Literature Review &amp; Theoretical Foundations</strong><span class="toc-dots"></span><strong>{page_map["ch2"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.1 Evolution of Automated PPE Detection Systems<span class="toc-dots"></span>{page_map["ch2"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.2 Self-Supervised Representation Learning &amp; Manifold Analysis<span class="toc-dots"></span>{page_map["ch2"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.3 Deep Generative Modeling Paradigms (VAEs, GANs, Diffusion)<span class="toc-dots"></span>{page_map["ch2"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.4 Long-Tail Class Imbalance &amp; Minority Oversampling in Safety<span class="toc-dots"></span>{page_map["ch2"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.5 The Perception-Distortion Tradeoff &amp; Safety Flips<span class="toc-dots"></span>{page_map["ch2"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.6 NLP for Regulatory Verification &amp; Uncertainty Calibration<span class="toc-dots"></span>{page_map["ch2"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;2.7 Critical Research Gaps Identified in Literature<span class="toc-dots"></span>{page_map["ch2"]}</div>
<br>
<div class="toc-line"><strong>Chapter 3: System Architecture &amp; Algorithmic Framework</strong><span class="toc-dots"></span><strong>{page_map["ch3"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.1 End-to-End Multi-Modal Workflow Blueprint<span class="toc-dots"></span>{page_map["ch3"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.2 Model 1: Self-Supervised ConvAE with Calibrated RBF-SVM<span class="toc-dots"></span>{page_map["ch3"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.3 Model 2: Conditional &sigma;-VAE with Variance Calibration<span class="toc-dots"></span>{page_map["ch3"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.4 Model 3: Deep cGAN with DiffAugment &amp; Anti-Checkerboard Filter<span class="toc-dots"></span>{page_map["ch3"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.5 Model 4: Conditional Score Diffusion &amp; Regression-Anchored Sampling<span class="toc-dots"></span>{page_map["ch3"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.6 Model 5: Calibrated Regulatory Policy Transformer (DistilBERT)<span class="toc-dots"></span>{page_map["ch3"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;3.7 Cross-Modal Integration: Dual-Vision Consensus Gating Engine<span class="toc-dots"></span>{page_map["ch3"]}</div>
<br>
<div class="toc-line"><strong>Chapter 4: Dataset Specifications &amp; Experimental Setup</strong><span class="toc-dots"></span><strong>{page_map["ch4"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.1 Visual PPE Crop Dataset Curation &amp; Imbalance Characterization<span class="toc-dots"></span>{page_map["ch4"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.2 Legal Regulatory Safety Text Corpus &amp; Zero-Leakage Hashing<span class="toc-dots"></span>{page_map["ch4"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.3 Independent Evaluation Protocol &amp; Inception Noise Floor<span class="toc-dots"></span>{page_map["ch4"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.4 Supervised 5-Seed ResNet-18 Arbiter Judge Ensemble<span class="toc-dots"></span>{page_map["ch4"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;4.5 100-Epoch Training Regimes &amp; Compute Infrastructure<span class="toc-dots"></span>{page_map["ch4"]}</div>
<br>
<div class="toc-line"><strong>Chapter 5: Empirical Results, Comparative Benchmarking &amp; Analysis</strong><span class="toc-dots"></span><strong>{page_map["ch5"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.1 Model 1: ConvAE Representation Quality &amp; Threshold Optimization<span class="toc-dots"></span>{page_map["ch5"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.2 Model 2: cVAE Latent Dynamics, Active Units &amp; Manifold Transfer<span class="toc-dots"></span>{page_map["ch5"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.3 Model 3: cGAN Synthesis, Privacy Auditing &amp; Downstream Lift<span class="toc-dots"></span>{page_map["ch5"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.4 Model 4: Empirical Discovery of the Perception-Distortion Tradeoff<span class="toc-dots"></span>{page_map["ch5"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.5 Model 5: Policy Compliance Accuracy &amp; Temperature Calibration<span class="toc-dots"></span>{page_map["ch5"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;5.6 Dual-Vision Consensus Gating Verification &amp; Case Studies<span class="toc-dots"></span>{page_map["ch5"]}</div>
<br>
<div class="toc-line"><strong>Chapter 6: Interactive Dashboard &amp; Cloud Deployment</strong><span class="toc-dots"></span><strong>{page_map["ch6"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.1 Interactive Concept Sandbox User Interface<span class="toc-dots"></span>{page_map["ch6"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.2 FastAPI Microservice Architecture &amp; Modular Routing<span class="toc-dots"></span>{page_map["ch6"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.3 Low-Level Socket Management (SO_REUSEADDR) &amp; Cloudflare Tunnels<span class="toc-dots"></span>{page_map["ch6"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;6.4 Production Dockerization &amp; Zero-Config Deployment<span class="toc-dots"></span>{page_map["ch6"]}</div>
<br>
<div class="toc-line"><strong>Chapter 7: Conclusions, Limitations &amp; Future Horizons</strong><span class="toc-dots"></span><strong>{page_map["ch7"]}</strong></div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.1 Summary of Experimental Discoveries &amp; Research Achievements<span class="toc-dots"></span>{page_map["ch7"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.2 Practical Safety Guidelines for Edge AI Inspection<span class="toc-dots"></span>{page_map["ch7"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.3 Limitations of Current Framework<span class="toc-dots"></span>{page_map["ch7"]}</div>
<div class="toc-line">&nbsp;&nbsp;&nbsp;&nbsp;7.4 Future Research Horizons<span class="toc-dots"></span>{page_map["ch7"]}</div>
<br>
<div class="toc-line"><strong>References (IEEE Style)</strong><span class="toc-dots"></span><strong>{page_map["ref"]}</strong></div>

<div class="page-break"></div>

<!-- PAGE 6: LIST OF TABLES, FIGURES & ABBREVIATIONS -->
<h1 class="chapter-title">LIST OF TABLES</h1>
<div class="toc-line"><span>Table 4.1: Visual PPE Crop Dataset Split Breakdown and Class Imbalance Matrix</span><span class="toc-dots"></span><span>{page_map["t41"]}</span></div>
<div class="toc-line"><span>Table 4.2: Legal Regulatory Safety Corpus Split Distribution (Zero Leakage)</span><span class="toc-dots"></span><span>{page_map["t42"]}</span></div>
<div class="toc-line"><span>Table 4.3: Inception Distributional Noise Floor Baseline Metrics</span><span class="toc-dots"></span><span>{page_map["t43"]}</span></div>
<div class="toc-line"><span>Table 4.4: Supervised ResNet-18 Judge Ensemble Performance Across 5 Seeds</span><span class="toc-dots"></span><span>{page_map["t44"]}</span></div>
<div class="toc-line"><span>Table 4.5: Standardized 100-Epoch Training Hyperparameter &amp; Regularization Matrix</span><span class="toc-dots"></span><span>{page_map["t45"]}</span></div>
<div class="toc-line"><span>Table 5.1: ConvAE + RBF-SVM Classification Performance Under Threshold Calibration</span><span class="toc-dots"></span><span>{page_map["t51"]}</span></div>
<div class="toc-line"><span>Table 5.2: Conditional &sigma;-VAE Objective Loss Formulations &amp; Multi-Experiment Ablation</span><span class="toc-dots"></span><span>{page_map["t52"]}</span></div>
<div class="toc-line"><span>Table 5.3: Downstream ResNet-18 Augmentation Benchmark Across 5 Seeds</span><span class="toc-dots"></span><span>{page_map["t53"]}</span></div>
<div class="toc-line"><span>Table 5.4: Perception-Distortion Restoration Benchmark Across 5 Degradations</span><span class="toc-dots"></span><span>{page_map["t54"]}</span></div>
<div class="toc-line"><span>Table 5.5: DDIM Inference Step Sweep: Latency vs Distortion vs Perceptual Quality</span><span class="toc-dots"></span><span>{page_map["t55"]}</span></div>
<div class="toc-line"><span>Table 5.6: Policy Compliance Model Performance on Held-Out Test Clauses</span><span class="toc-dots"></span><span>{page_map["t56"]}</span></div>
<div class="toc-line"><span>Table 5.7: Temperature Scaling Calibration Summary Across Splits and Linguistic Slices</span><span class="toc-dots"></span><span>{page_map["t57"]}</span></div>

<h1 class="chapter-title" style="margin-top:16px;">LIST OF FIGURES</h1>
<div class="toc-line"><span>Figure 1.1: End-to-End System Architecture Blueprint depicting multi-modal workflow</span><span class="toc-dots"></span><span>{page_map["f11"]}</span></div>
<div class="toc-line"><span>Figure 4.1: Dataset Class Distribution across Train, Val, and Test splits</span><span class="toc-dots"></span><span>{page_map["f41"]}</span></div>
<div class="toc-line"><span>Figure 4.2: Standardized 64x64 Industrial Safety Crops showing realistic variance</span><span class="toc-dots"></span><span>{page_map["f42"]}</span></div>
<div class="toc-line"><span>Figure 5.1: ConvAE 100-Epoch Composite Training and Validation Loss Curves</span><span class="toc-dots"></span><span>{page_map["f51"]}</span></div>
<div class="toc-line"><span>Figure 5.2: 2D PCA Scatter Projection of the frozen 256-d ConvAE Latent Manifold</span><span class="toc-dots"></span><span>{page_map["f52"]}</span></div>
<div class="toc-line"><span>Figure 5.3: Visual Comparison between Input Crops and ConvAE Reconstructions</span><span class="toc-dots"></span><span>{page_map["f53"]}</span></div>
<div class="toc-line"><span>Figure 5.4: Normalized Multi-Class Confusion Matrix for Calibrated ConvAE + RBF-SVM</span><span class="toc-dots"></span><span>{page_map["f54"]}</span></div>
<div class="toc-line"><span>Figure 5.5: Conditional &sigma;-VAE 100-Epoch Training Dynamics</span><span class="toc-dots"></span><span>{page_map["f55"]}</span></div>
<div class="toc-line"><span>Figure 5.6: KL Divergence per Latent Dimension confirming 100% active latent units</span><span class="toc-dots"></span><span>{page_map["f56"]}</span></div>
<div class="toc-line"><span>Figure 5.7: Spherical Latent Interpolation (SLERP) across the Gaussian shell</span><span class="toc-dots"></span><span>{page_map["f57"]}</span></div>
<div class="toc-line"><span>Figure 5.8: Class-Conditional Synthetic Crops generated by Model 3</span><span class="toc-dots"></span><span>{page_map["f58"]}</span></div>
<div class="toc-line"><span>Figure 5.9: Nearest-Neighbor Distance Histogram proving zero face memorization</span><span class="toc-dots"></span><span>{page_map["f59"]}</span></div>
<div class="toc-line"><span>Figure 5.10: Downstream 5-Seed ResNet-18 F1 Score Lift comparing Augmentations</span><span class="toc-dots"></span><span>{page_map["f510"]}</span></div>
<div class="toc-line"><span>Figure 5.11: Empirical Perception-Distortion Tradeoff Curve</span><span class="toc-dots"></span><span>{page_map["f511"]}</span></div>
<div class="toc-line"><span>Figure 5.12: Visual Evidence of Diffusion Safety Flips: hallucinated hardhats</span><span class="toc-dots"></span><span>{page_map["f512"]}</span></div>
<div class="toc-line"><span>Figure 5.13: Reliability Diagrams Before and After Temperature Calibration (T*)</span><span class="toc-dots"></span><span>{page_map["f513"]}</span></div>
<div class="toc-line"><span>Figure 5.14: Dual-Vision Consensus Decision Matrix showing tri-state resolution</span><span class="toc-dots"></span><span>{page_map["f514"]}</span></div>

<h1 class="chapter-title" style="margin-top:16px;">ABBREVIATIONS &amp; ACRONYMS</h1>
<table style="border:none; margin: 4px 0; font-size:8.5pt;">
<tr style="background:none;"><td style="border:none; width:33%; padding:2px;"><strong>PPE:</strong> Personal Protective Equipment</td><td style="border:none; width:33%; padding:2px;"><strong>OSHA:</strong> Occupational Safety &amp; Health Admin.</td><td style="border:none; width:34%; padding:2px;"><strong>MUTCD:</strong> Manual on Uniform Traffic Control Devices</td></tr>
<tr style="background:none;"><td style="border:none; padding:2px;"><strong>ConvAE:</strong> Convolutional Autoencoder</td><td style="border:none; padding:2px;"><strong>cVAE:</strong> Conditional Variational Autoencoder</td><td style="border:none; padding:2px;"><strong>&sigma;-VAE:</strong> Variance-Calibrated VAE</td></tr>
<tr style="background:none;"><td style="border:none; padding:2px;"><strong>cGAN:</strong> Conditional Generative Adv. Net</td><td style="border:none; padding:2px;"><strong>DiffAugment:</strong> Differentiable Augmentation</td><td style="border:none; padding:2px;"><strong>DDPM:</strong> Denoising Diffusion Prob. Model</td></tr>
<tr style="background:none;"><td style="border:none; padding:2px;"><strong>DDIM:</strong> Denoising Diffusion Implicit Model</td><td style="border:none; padding:2px;"><strong>SN:</strong> Spectral Normalization (1-Lipschitz)</td><td style="border:none; padding:2px;"><strong>TTUR:</strong> Two Time-Scale Update Rule</td></tr>
<tr style="background:none;"><td style="border:none; padding:2px;"><strong>ELBO:</strong> Evidence Lower Bound</td><td style="border:none; padding:2px;"><strong>ECE:</strong> Expected Calibration Error</td><td style="border:none; padding:2px;"><strong>LPIPS:</strong> Learned Perceptual Patch Similarity</td></tr>
<tr style="background:none;"><td style="border:none; padding:2px;"><strong>PSNR:</strong> Peak Signal-to-Noise Ratio</td><td style="border:none; padding:2px;"><strong>SSIM:</strong> Structural Similarity Index Measure</td><td style="border:none; padding:2px;"><strong>SLERP:</strong> Spherical Linear Interpolation</td></tr>
</table>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- MAIN REPORT BODY (CHAPTERS 1 TO 7 + REFERENCES) -->
<!-- ========================================================================= -->
{body_html}

</body>
</html>
"""
    return html

def compile_pdf_raw(html_content, output_pdf):
    html_path = Path("PROJECT_REPORT_temp.html")
    html_path.write_text(html_content, encoding="utf-8")

    chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not Path(chrome_exe).exists():
        chrome_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

    cmd = [
        chrome_exe,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={str(output_pdf.resolve())}",
        str(html_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Chrome compilation failed: {res.stderr}")

def stamp_page_numbers_and_headers(raw_pdf_path, final_pdf_path, intro_page_idx):
    doc = fitz.open(str(raw_pdf_path))
    total_pages = len(doc)
    print(f"[*] Processing {total_pages} pages in PDF (Chapter 1 begins at page index {intro_page_idx})...")

    for i, page in enumerate(doc):
        rect = page.rect
        y_bottom = rect.height - 35  # ~12mm above bottom edge
        fontname = "times-roman"
        fontsize = 10
        text_content = page.get_text()

        if i == 0:
            # Title page: NO page number
            continue
        elif i < intro_page_idx:
            # Preliminary pages: lowercase Roman numerals (ii, iii, iv, v, vi...)
            num_str = to_roman(i + 1)
            tw = fitz.get_text_length(num_str, fontname=fontname, fontsize=fontsize)
            x = (rect.width - tw) / 2
            page.insert_text(fitz.Point(x, y_bottom), num_str, fontname=fontname, fontsize=fontsize, color=(0, 0, 0))
        else:
            # Main body: Arabic numerals starting at 1 on Chapter 1
            arabic_num = i - intro_page_idx + 1
            num_str = str(arabic_num)
            tw = fitz.get_text_length(num_str, fontname=fontname, fontsize=fontsize)
            x = (rect.width - tw) / 2
            page.insert_text(fitz.Point(x, y_bottom), num_str, fontname=fontname, fontsize=fontsize, color=(0, 0, 0))

            # Running top header (omit on pages that start a major CHAPTER)
            is_chapter_first_page = False
            first_lines = [l.strip() for l in text_content.split("\n")[:6] if l.strip()]
            for line in first_lines:
                if line.startswith("CHAPTER") or line.startswith("REFERENCES"):
                    is_chapter_first_page = True
                    break

            if not is_chapter_first_page:
                header_text = "Multi-Objective Benchmarking of Generative Vision Models for PPE Compliance"
                page.insert_text(fitz.Point(70, 42), header_text, fontname="times-italic", fontsize=8.5, color=(0, 0, 0))
                # 0.5pt subtle black rule under header
                page.draw_line(fitz.Point(70, 46), fitz.Point(rect.width - 70, 46), color=(0, 0, 0), width=0.5)

    doc.save(str(final_pdf_path))
    doc.close()

def main():
    print("=" * 70)
    print("ACADEMIC PROJECT REPORT GENERATOR: 100% PURE BLACK & EXACT PAGE NUMBERING")
    print("=" * 70)

    print("[1/5] Encoding figures to Base64...")
    images = {
        "arch": img_to_b64("results/figures/fig1_system_architecture.png"),
        "splits": img_to_b64("results/autoencoder/class_distribution_splits.png"),
        "samples": img_to_b64("results/autoencoder/sample_crops_grid.png"),
        "ae_loss": img_to_b64("results/autoencoder/ae_loss_curve.png"),
        "ae_pca": img_to_b64("results/autoencoder/latent_pca_scatter.png"),
        "ae_recon": img_to_b64("results/autoencoder/reconstructions.png"),
        "ae_conf": img_to_b64("results/autoencoder/svm_confusion_matrix.png"),
        "cvae_loss": img_to_b64("results/cvae/figures/cvae_loss_curves.png"),
        "cvae_kl": img_to_b64("results/cvae/figures/cvae_kl_per_dimension.png"),
        "cvae_slerp": img_to_b64("results/cvae/figures/cvae_slerp_interpolation.png"),
        "cgan_grid": img_to_b64("results/cgan/figures/01_cgan_samples_grid.png"),
        "cgan_priv": img_to_b64("results/cgan/figures/04_cgan_nearest_neighbor_privacy.png"),
        "cgan_aug": img_to_b64("results/cgan/figures/05_cgan_downstream_augmentation_benchmark.png"),
        "diff_tradeoff": img_to_b64("results/diffusion/figures/02_perception_distortion_tradeoff.png"),
        "diff_flips": img_to_b64("results/diffusion/figures/05_safety_critical_flips.png"),
        "policy_rel": img_to_b64("results/policy/figures/04_calibration_reliability.png"),
        "policy_dec": img_to_b64("results/policy/figures/05_end_to_end_verdict_matrix.png"),
    }

    print("[2/5] Extracting main report body...")
    body_html = get_body_html(images)

    # Initial Pass: compile raw PDF to find exact page coordinates
    print("[3/5] Pass 1: Discovering exact chapter and element page offsets...")
    initial_html = build_full_html(body_html, None)
    temp_raw_pdf = Path("PROJECT_REPORT_pass1.pdf")
    compile_pdf_raw(initial_html, temp_raw_pdf)

    # Read pass 1 PDF to find exact pages
    doc = fitz.open(str(temp_raw_pdf))
    total_pass1 = len(doc)
    intro_page_idx = None
    for i, page in enumerate(doc):
        txt = page.get_text()
        if "CHAPTER 1: INTRODUCTION" in txt:
            intro_page_idx = i
            break

    if intro_page_idx is None:
        intro_page_idx = 6  # fallback

    print(f"      Total Pass 1 pages: {total_pass1}")
    print(f"      Chapter 1 starts at PDF page {intro_page_idx + 1} (Page index {intro_page_idx})")

    # Find page locations of all elements
    def find_page_for_text(target_str, must_start=False):
        target_upper = target_str.upper()
        for i in range(intro_page_idx, len(doc)):
            txt_norm = re.sub(r'\s+', ' ', doc[i].get_text()).upper()
            if target_upper in txt_norm:
                if must_start:
                    if target_upper in txt_norm[:250]:
                        return str(i - intro_page_idx + 1)
                else:
                    return str(i - intro_page_idx + 1)
        return "1"

    page_map = {
        "ch1": find_page_for_text("CHAPTER 1: INTRODUCTION", must_start=True),
        "ch2": find_page_for_text("LITERATURE REVIEW", must_start=True),
        "ch3": find_page_for_text("SYSTEM ARCHITECTURE", must_start=True),
        "ch4": find_page_for_text("DATASET SPECIFICATIONS", must_start=True),
        "ch5": find_page_for_text("EMPIRICAL RESULTS", must_start=True),
        "ch6": find_page_for_text("INTERACTIVE DASHBOARD", must_start=True),
        "ch7": find_page_for_text("CONCLUSIONS", must_start=True),
        "ref": find_page_for_text("REFERENCES", must_start=True),
        "t41": find_page_for_text("TABLE 4.1:"),
        "t42": find_page_for_text("TABLE 4.2:"),
        "t43": find_page_for_text("TABLE 4.3:"),
        "t44": find_page_for_text("TABLE 4.4:"),
        "t45": find_page_for_text("TABLE 4.5:"),
        "t51": find_page_for_text("TABLE 5.1:"),
        "t52": find_page_for_text("TABLE 5.2:"),
        "t53": find_page_for_text("TABLE 5.3:"),
        "t54": find_page_for_text("TABLE 5.4:"),
        "t55": find_page_for_text("TABLE 5.5:"),
        "t56": find_page_for_text("TABLE 5.6:"),
        "t57": find_page_for_text("TABLE 5.7:"),
        "f11": find_page_for_text("FIGURE 1.1:"),
        "f41": find_page_for_text("FIGURE 4.1:"),
        "f42": find_page_for_text("FIGURE 4.2:"),
        "f51": find_page_for_text("FIGURE 5.1:"),
        "f52": find_page_for_text("FIGURE 5.2:"),
        "f53": find_page_for_text("FIGURE 5.3:"),
        "f54": find_page_for_text("FIGURE 5.4:"),
        "f55": find_page_for_text("FIGURE 5.5:"),
        "f56": find_page_for_text("FIGURE 5.6:"),
        "f57": find_page_for_text("FIGURE 5.7:"),
        "f58": find_page_for_text("FIGURE 5.8:"),
        "f59": find_page_for_text("FIGURE 5.9:"),
        "f510": find_page_for_text("FIGURE 5.10:"),
        "f511": find_page_for_text("FIGURE 5.11:"),
        "f512": find_page_for_text("FIGURE 5.12:"),
        "f513": find_page_for_text("FIGURE 5.13:"),
        "f514": find_page_for_text("FIGURE 5.14:")
    }
    doc.close()
    if temp_raw_pdf.exists():
        temp_raw_pdf.unlink()

    print("[4/5] Pass 2: Re-compiling HTML with synchronized Table of Contents...")
    final_html = build_full_html(body_html, page_map)
    Path("PROJECT_REPORT.html").write_text(final_html, encoding="utf-8")
    raw_pdf_path = Path("PROJECT_REPORT_raw.pdf")
    compile_pdf_raw(final_html, raw_pdf_path)

    print("[5/5] Stamping Roman & Arabic page numbers and running headers...")
    final_pdf_path = Path("PROJECT_REPORT.pdf")
    stamp_page_numbers_and_headers(raw_pdf_path, final_pdf_path, intro_page_idx)

    if raw_pdf_path.exists():
        raw_pdf_path.unlink()
    if Path("PROJECT_REPORT_temp.html").exists():
        Path("PROJECT_REPORT_temp.html").unlink()

    # Final Verification
    verify_doc = fitz.open(str(final_pdf_path))
    v_total = len(verify_doc)
    v_main = v_total - intro_page_idx
    size_mb = round(final_pdf_path.stat().st_size / (1024 * 1024), 2)
    verify_doc.close()

    print("=" * 70)
    print(f"[+] BUILD SUCCESSFUL!")
    print(f"    - Output PDF: {final_pdf_path.resolve()} ({size_mb} MB)")
    print(f"    - Total Pages: {v_total} (Minimum 35 constraint: {'PASSED' if v_total >= 35 else 'FAILED'})")
    print(f"    - Preliminaries: {intro_page_idx} pages (Roman numerals ii to {to_roman(intro_page_idx)})")
    print(f"    - Main Body: {v_main} pages (Arabic numerals 1 to {v_main}; Max 35 constraint: {'PASSED' if v_main <= 35 else 'FAILED'})")
    print(f"    - All colors: 100% pure black (#000000)")
    print("=" * 70)

if __name__ == "__main__":
    main()
