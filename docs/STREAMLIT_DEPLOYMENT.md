# Streamlit Deployment Guide: PPE-VisionAI 🦺

This guide details how to launch and deploy the interactive **PPE-VisionAI 5-Model Generative Benchmarking Platform** using **Streamlit**.

---

## 1. Running Locally

Ensure your virtual environment is active and dependencies are installed:

```bash
# 1. Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 2. Run the Streamlit application
streamlit run streamlit_app.py
```

Streamlit will launch local browser preview at:
`http://localhost:8501`

### Running on a Custom Port or Headless Mode
```bash
streamlit run streamlit_app.py --server.port 8505 --server.headless true
```

---

## 2. Deploying to Streamlit Community Cloud (Free & Instant)

Streamlit Community Cloud is the standard, free hosting platform for Streamlit applications directly linked to GitHub.

### Step 1: Push Repository to GitHub
Ensure your changes are committed and pushed:
```bash
git add requirements.txt .streamlit/ streamlit_app.py Dockerfile.streamlit docs/
git commit -m "Add Streamlit deployment app and configuration"
git push origin main
```

> [!NOTE]
> **Large File Storage (Git LFS) / Checkpoint Files:**
> GitHub has a 100 MB per-file limit. The policy transformer checkpoint (`artifacts/policy/policy_classifier.pt`) is ~255 MB.
> - If you use Git LFS: `git lfs track "*.pt"` ensures all weights upload cleanly.
> - If Git LFS is not configured: `streamlit_app.py` has built-in graceful fallback to the interpretable regex engine, ensuring your cloud app will boot without crashing.

### Step 2: Deploy on Streamlit Community Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Fill in the deployment form:
   - **Repository:** `<your-username>/ppe-generative-project`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** Customize if desired (e.g. `ppe-visionai.streamlit.app`)
4. In **Advanced settings** (optional):
   - Python Version: Select `3.10` or `3.11`.
5. Click **Deploy!**

Streamlit will read `requirements.txt`, install all libraries, apply the custom theme from `.streamlit/config.toml`, and provide a public HTTPS URL.

---

## 3. Deploying via Docker

For private cloud servers, AWS EC2, GCP Compute Engine, or Azure Container Instances:

```bash
# 1. Build Docker image
docker build -t ppe-visionai-streamlit -f Dockerfile.streamlit .

# 2. Run container
docker run -d -p 8501:8501 --name ppe-streamlit-app ppe-visionai-streamlit
```

Access the application at `http://<your-server-ip>:8501`.

---

## 4. Architectural Summary in Streamlit

| Tab | Model Architecture | Key Interactive Capabilities |
| :--- | :--- | :--- |
| **🦺 1. Autoencoder + SVM** | 256-d ConvAE + RBF SVM | Test crop selector, 256-d bottleneck reconstruction, MSE/PSNR/SSIM, error heatmap, SVM confidence bars. |
| **🎲 2. Conditional VAE** | 128-d cVAE + $\beta$-ELBO | Class conditioning ($y \in \{0, 1, 2\}$), sampling temperature $\tau$, seed generator, SLERP interpolation walk ($\alpha \in [0, 1]$). |
| **⚡ 3. Conditional GAN** | cGAN + DiffAugment + EMA | Adversarial sample generation, latent variance scale $\sigma$, ResNet Judge evaluation, Nearest-Neighbor Privacy Audit ($p_5 = 0.2637$). |
| **🌊 4. Diffusion Restoration** | Conditional UNet + DDIM | 5 degradation types, severity 1-5, DDIM sampling steps (10-100), Regression baseline comparison, zero-hallucination audit. |
| **📜 5. Policy Checker** | DistilBERT + Temp Scaling | OSHA 1926.100 & ANSI 107 clause classification, dual-vision consensus gating, 3-state compliance verdicts. |
| **📊 6. Master Benchmark** | Master Table 1 Comparison | Comprehensive metrics, domain winner analysis, honest limitations, and report download links. |
