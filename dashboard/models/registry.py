"""
dashboard/models/registry.py
Model Registry for all 5 generative-AI PPE benchmark models:
1. Autoencoder + SVM (Active)
2. Conditional VAE (cVAE) (Coming Soon)
3. Conditional GAN (cGAN) (Coming Soon)
4. Diffusion Restoration (DDPM) (Coming Soon)
5. Safety Policy Checker (Coming Soon)
"""

from typing import Any, Dict, List, Optional


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "autoencoder": {
        "id": "autoencoder",
        "name": "Autoencoder + SVM",
        "tab_label": "Autoencoder",
        "status": "ready",
        "badge": "Active (Model 1)",
        "badge_color": "green",
        "description": "Convolutional Autoencoder compressing 64x64 PPE crops into 256-dim latent vectors with an RBF Support Vector Machine for PPE violation detection.",
        "architecture": "ResBlock-enhanced ConvAE (Zero Skip-Connections, Bottleneck D=256) + RBF Kernel SVM",
        "loss_objective": "L1 Pixel Loss + 0.20 * (1 - SSIM) Structural Loss",
        "endpoint_prefix": "/api/autoencoder",
        "features": [
            "Interactive 64x64 crop bounding box tool",
            "Real-time generative reconstruction inspection",
            "Per-image PSNR, SSIM, MSE, MAE metrics",
            "256-dim latent space representation & 2D PCA scatter",
            "SVM safety violation detection (head vs helmet vs vest)"
        ]
    },
    "cvae": {
        "id": "cvae",
        "name": "Conditional VAE (cVAE)",
        "tab_label": "Conditional VAE",
        "status": "ready",
        "badge": "Active (Model 2)",
        "badge_color": "indigo",
        "description": "Conditional Variational Autoencoder modeling class-conditional distributions p(x|y) for synthetic PPE crop synthesis and probabilistic generation.",
        "architecture": "Conditional Conv-VAE with Gaussian Latent Bottleneck (D=256) conditioned on one-hot class embedding",
        "loss_objective": "Beta-ELBO: Reconstruction Loss (L1+SSIM) + beta * KL Divergence (q(z|x,y) || p(z))",
        "endpoint_prefix": "/api/cvae",
        "features": [
            "Controllable class sampling (generate head, helmet, or vest on demand)",
            "Latent space interpolation between PPE classes",
            "Intra-class diversity assessment (pairwise LPIPS)",
            "Distributional fidelity (FID, KID) vs Noise Floor"
        ]
    },
    "cgan": {
        "id": "cgan",
        "name": "Conditional GAN (cGAN)",
        "tab_label": "Conditional GAN",
        "status": "ready",
        "badge": "Active (Model 3)",
        "badge_color": "purple",
        "description": "Class-Conditional DCGAN backbone with Spectral Normalization, Projection Discriminator, Class-Conditional BatchNorm, Hinge Loss, and DiffAugment for rare-class data augmentation.",
        "architecture": "Modernized DCGAN Backbone + Spectral Normalization (Lipschitz <= 1) + Projection Discriminator + Class-Conditional BatchNorm",
        "loss_objective": "Hinge Adversarial Loss + Differentiable Augmentation (color, translation, cutout)",
        "endpoint_prefix": "/api/cgan",
        "features": [
            "Class-conditional synthesis with latent variation slider",
            "Spherical linear interpolation (SLERP) continuous latent walks",
            "Live Independent Judge & SVM safety classification",
            "Real-time Nearest-Real-Neighbor privacy & memorization check",
            "5-condition downstream classifier augmentation benchmark with paired t-tests",
            "Filtered synthetic head crops download (ZIP)"
        ]
    },
    "diffusion": {
        "id": "diffusion",
        "name": "Diffusion Restoration (DDPM / DDIM)",
        "tab_label": "Diffusion Restoration",
        "status": "ready",
        "badge": "Active (Model 4)",
        "badge_color": "teal",
        "description": "SR3-style image-conditioned Denoising Diffusion Probabilistic Model (DDPM / DDIM) trained to reverse real-world surveillance corruptions without hallucinating compliance.",
        "architecture": "SR3 Conditional U-Net (6ch input) + Sinusoidal Time MLP + Self-Attention (16x16, 8x8) + GroupNorm + EMA",
        "loss_objective": "Epsilon-L1 Loss + Low-Noise (t < 250) abar_t-weighted LPIPS Perceptual Term",
        "endpoint_prefix": "/api/diffusion",
        "features": [
            "SR3-style image concatenation conditioning (Strictly No Label Conditioning)",
            "Deterministic DDIM fast sampling (20, 50, 100 steps) with interactive slider",
            "Regression U-Net baseline for empirical perception-distortion trade-off comparison",
            "Zero-shot generalization audit on strictly held-out directional motion blur",
            "Downstream compliance recovery tracking (Independent Judge & AE+SVM)",
            "Safety-critical class-flip audit: alert flag if unprotected head becomes compliant PPE"
        ]
    },
    "policy": {
        "id": "policy",
        "name": "Safety Policy Transformer & Compliance Checker",
        "tab_label": "Policy Transformer",
        "status": "ready",
        "badge": "Active (Model 5)",
        "badge_color": "emerald",
        "description": "Natural Language Policy Compliance Checker classifying regulatory documents and executing three-state multi-modal safety audits (Violation / Compliant / Cannot determine) with dual-vision consensus.",
        "architecture": "Fine-Tuned DistilBERT (5-Class Taxonomy, Temperature Calibrated) + Consensus Vision Gating (Judge v2 + AE+SVM)",
        "loss_objective": "Class-Weighted CrossEntropyLoss + Post-Hoc L-BFGS Temperature Scaling",
        "endpoint_prefix": "/api/policy",
        "features": [
            "Multi-format document ingestion (PDF, DOCX, TXT) with scanned document detection",
            "Sentence segmentation with legal citation and abbreviation preservation",
            "5-class regulatory classification (helmet_required, vest_required, both_required, other_ppe, none)",
            "Calibrated confidence scoring with Expected Calibration Error (ECE) reduction",
            "Three-state multi-modal decision logic citing specific regulatory clause and page number",
            "Consensus gating between Independent Judge v2 and AE+SVM to prevent false certification"
        ]
    }
}


def get_registered_models() -> List[Dict[str, Any]]:
    """Return list of all registered model dictionaries in benchmark sequence."""
    return list(MODEL_REGISTRY.values())


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Return metadata dictionary for a specific model ID."""
    return MODEL_REGISTRY.get(model_id)

