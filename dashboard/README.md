# PPE Generative Autoencoder & Downstream Classification Web Dashboard

Research Project: **"Empirical Multi-Objective Benchmarking of Generative Models on PPE Safety Crops"** (Model 1: Autoencoder + SVM).

## Overview
This interactive web dashboard serves the trained Convolutional Autoencoder (ConvAE) and downstream Support Vector Machine (RBF kernel) for safety violation detection on PPE crops (`head`, `helmet`, `vest`).

### Key Capabilities
1. **Interactive Crop & Inference:**
   - Upload any PPE crop directly or upload a full construction site photo.
   - Use the built-in visual crop selector to drag and resize an interactive bounding box.
   - Run inference matching the training pipeline: 64x64 Lanczos resampling $\to$ normalization in $[0, 1] \to$ ConvAE encoding & reconstruction $\to$ StandardScaler $\to$ RBF SVM classification.
2. **Multi-Objective Visual Feedback:**
   - **Original Crop vs. Reconstruction:** View the $64\times 64$ crop alongside its autoencoder reconstruction.
   - **Perceptual Error Heatmap:** Absolute pixel error heatmap ($|x - \hat{x}|$) color-coded using the `inferno` colormap.
   - **Per-Image Metrics:** Real-time computation of MSE, MAE, PSNR (dB), and SSIM.
   - **Downstream Classification Badge:** Highlights compliance (`helmet`, `vest`) or safety violation notice (`head` = no helmet).
   - **Relative Scores:** Softmax over the SVM decision function (labeled *"relative score, uncalibrated"*).
   - **Latent Space Visualization:** Bar representation of the 256-dimensional bottleneck vector and a 2D PCA scatter plot showing test-set clusters with the current sample highlighted.
3. **Model Performance Hub (`#model-performance`):**
   - Live tables populated directly from evaluation CSVs in `results/autoencoder/`.
   - Side-by-side comparison against the linear raw-pixel PCA(256) baseline.
   - High-resolution evaluation figures (Loss curves, Confusion Matrix, Latent Scatter).
   - Transparent research limitations card.

---

## Quick Start (Single Command)

From the project root directory:

```bash
uvicorn dashboard.app:app --reload
```

Open your browser at:
```
http://127.0.0.1:8000
```

---

## API Endpoints

- `GET /` : Serves the single-page dashboard.
- `GET /api/health` : Health check with model status, device, and class list.
- `POST /api/predict` : Multipart form upload accepting `file` and optional `crop_x`, `crop_y`, `crop_w`, `crop_h`.
- `GET /api/metrics` : Reads dynamic evaluation metrics and returns structured JSON for the performance hub.
- `GET /api/samples` : Returns up to 12 random test crops with labels for one-click testing.
- `GET /results/{figure_name}` : Serves evaluation figures from `results/autoencoder/`.

