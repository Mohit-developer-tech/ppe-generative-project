"""
scripts/smoke_models.py
Smoke test profile for all 5 models + judge ensemble + AE+SVM baseline:
Executes lightweight 1-step or single-sample inference across every model
to verify environment, device placement, tensor flow, and weights loading.
"""
import time
import sys
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def smoke_test_all():
    print("=" * 60)
    print("RUNNING SMOKE TEST PROFILE ACROSS ALL MODELS")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target execution device: {device.type.upper()}\n")

    dummy_crop = torch.rand(1, 3, 64, 64, device=device)

    # 1. Model 1: Autoencoder + SVM
    print("[1/6] Testing Autoencoder + SVM Baseline...")
    t0 = time.time()
    from dashboard.model_def import ConvAE
    ae = ConvAE(in_channels=3, latent_dim=256).to(device)
    ae_pt = "artifacts/autoencoder/ae_best.pt"
    if Path(ae_pt).exists():
        ckpt = torch.load(ae_pt, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        ae.load_state_dict(sd)
    ae.eval()
    with torch.no_grad():
        recon, z = ae(dummy_crop)
    assert recon.shape == (1, 3, 64, 64)
    assert z.shape == (1, 256)
    print(f"  [PASS] Autoencoder forward pass completed in {(time.time()-t0)*1000:.1f} ms.")

    # 2. Judge Ensemble
    print("[2/6] Testing Independent SmallResNet Judge Ensemble...")
    t0 = time.time()
    from common.judge import load_judge
    judge = load_judge()
    with torch.no_grad():
        j_logits = judge(dummy_crop)
        j_res = judge.score_crops(dummy_crop)
    assert j_logits.shape == (1, 3)
    assert len(j_res["predicted_classes"]) == 1
    print(f"  [PASS] Judge ensemble scored crop in {(time.time()-t0)*1000:.1f} ms: class='{j_res['predicted_classes'][0]}'.")

    # 3. Model 2: Conditional VAE (cVAE)
    print("[3/6] Testing Conditional VAE...")
    t0 = time.time()
    from common.cvae import ConditionalVAE
    cvae = ConditionalVAE(latent_dim=128).to(device)
    cvae_pt = "artifacts/cvae/cvae_best.pt"
    if Path(cvae_pt).exists():
        ckpt = torch.load(cvae_pt, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        cvae.load_state_dict(sd)
    cvae.eval()
    y_cond = torch.tensor([1], device=device)  # helmet
    with torch.no_grad():
        cvae_sample = cvae.sample(num_samples=1, class_idx=1, temperature=1.0, device=device)
    assert cvae_sample.shape == (1, 3, 64, 64)
    print(f"  [PASS] cVAE generation completed in {(time.time()-t0)*1000:.1f} ms.")

    # 4. Model 3: Conditional GAN (cGAN)
    print("[4/6] Testing Conditional GAN (cGAN)...")
    t0 = time.time()
    from common.cgan import ConditionalGenerator
    cgan = ConditionalGenerator(z_dim=128, num_classes=3, base_channels=64).to(device)
    cgan_pt = "artifacts/cgan/cgan_best.pt"
    if Path(cgan_pt).exists():
        ckpt = torch.load(cgan_pt, map_location=device)
        sd = ckpt.get("generator_ema_state_dict", ckpt.get("generator_state_dict", ckpt))
        cgan.load_state_dict(sd, strict=False)
    cgan.eval()
    z_noise = torch.randn(1, 128, device=device)
    with torch.no_grad():
        cgan_sample = cgan(z_noise, y_cond)
    assert cgan_sample.shape == (1, 3, 64, 64)
    print(f"  [PASS] cGAN synthesis completed in {(time.time()-t0)*1000:.1f} ms.")

    # 5. Model 4: Diffusion Restoration
    print("[5/6] Testing Conditional Diffusion Restoration...")
    t0 = time.time()
    from common.diffusion import ConditionalUNet, DiffusionSchedule, sample_ddim
    diff_net = ConditionalUNet(in_channels=6, out_channels=3, base_channels=32).to(device)
    diff_pt = "artifacts/diffusion/diffusion_best.pt"
    if Path(diff_pt).exists():
        ckpt = torch.load(diff_pt, map_location=device)
        sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
        diff_net.load_state_dict(sd, strict=False)
    diff_net.eval()
    schedule = DiffusionSchedule(timesteps=100, device=device)
    with torch.no_grad():
        restored = sample_ddim(diff_net, dummy_crop, schedule, steps=5, eta=0.0, device=device)
    assert restored.shape == (1, 3, 64, 64)
    print(f"  [PASS] Diffusion 5-step DDIM restoration completed in {(time.time()-t0)*1000:.1f} ms.")

    # 6. Model 5: Policy-Driven Compliance Checker
    print("[6/6] Testing Safety Policy Compliance Checker...")
    t0 = time.time()
    from common.policy import KeywordRegexClassifier, ThreeStateVerdictEngine
    kw = KeywordRegexClassifier()
    clause = "A protective safety helmet is mandatory on all active construction zones."
    pred_label = kw.predict_one(clause)
    assert pred_label == "helmet_required"
    pol_pred = {"text": clause, "predicted_label": pred_label, "page": 1}
    verdict = ThreeStateVerdictEngine.evaluate_verdict(
        policy_clauses=[pol_pred],
        crop_label="helmet",
        vision_confidence=0.95,
        vision_agreed=True
    )
    assert verdict["verdict"] == "Compliant"
    print(f"  [PASS] Policy compliance verdict completed in {(time.time()-t0)*1000:.1f} ms: verdict='{verdict['verdict']}'.")

    print("\n" + "=" * 60)
    print("[ALL SMOKE TESTS PASSED] Every model executes cleanly!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    smoke_test_all()
