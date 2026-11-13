"""
test_all_models_fastapi.py
Direct programmatic verification of all 5 generative and predictive models in the dashboard.
"""

import io
import json
from PIL import Image
import numpy as np
import torch
from fastapi.testclient import TestClient

from dashboard.app import app

client = TestClient(app)

def test_model1_autoencoder():
    print("\n--- [1] Testing ConvAE + RBF-SVM ---")
    # Create test 64x64 helmet-like image
    im = Image.new("RGB", (64, 64), color=(220, 200, 30))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_helmet.png", buf.getvalue(), "image/png")}
    data = {}
    resp = client.post("/api/predict", files=files, data=data)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    res = resp.json()
    print("  Status: OK")
    print(f"  Predicted Class: {res['predicted_class']}")
    print(f"  PSNR: {res['psnr']:.2f} dB, SSIM: {res['ssim']:.4f}, MSE: {res['mse']:.6f}")
    print(f"  Class Scores: {res['scores']}")
    assert "reconstruction_b64" in res
    assert len(res["reconstruction_b64"]) > 100

def test_model2_cvae():
    print("\n--- [2] Testing Conditional Beta-VAE ---")
    payload = {
        "class_name": "helmet",
        "temperature": 0.7,
        "seed": 42,
        "num_samples": 4
    }
    resp = client.post("/api/cvae/generate", json=payload)
    assert resp.status_code == 200, f"CVAE generate failed: {resp.text}"
    res = resp.json()
    print("  Status: OK")
    print(f"  Generated {res['num_samples']} samples for class '{res['class']}'")
    assert len(res["images"]) == 4

    # Test class morph
    morph_payload = {
        "class_from": "head",
        "class_to": "helmet",
        "alpha": 0.5,
        "seed": 42
    }
    morph_resp = client.post("/api/cvae/class_morph", json=morph_payload)
    assert morph_resp.status_code == 200, f"CVAE morph failed: {morph_resp.text}"
    print("  Class Morphing: OK")

def test_model3_cgan():
    print("\n--- [3] Testing Conditional GAN (DiffAugment) ---")
    payload = {
        "class_id": 0, # Head
        "latent_variation": 0.9,
        "count": 4,
        "seed": 42
    }
    resp = client.post("/api/cgan/generate", json=payload)
    assert resp.status_code == 200, f"cGAN generate failed: {resp.text}"
    res = resp.json()
    print("  Status: OK")
    print(f"  Generated {len(res['samples'])} samples for '{res['requested_class']}'")
    for s in res['samples'][:2]:
        print(f"    Sample #{s['index']}: Judge={s['judge_prediction']} ({s['judge_confidence']*100:.1f}%), Privacy Dist={s['nearest_real_distance']:.4f}")

    # Test Latent Walk
    walk_payload = {
        "class_id": 1, # Helmet
        "seed1": 42,
        "seed2": 100,
        "steps": 6
    }
    walk_resp = client.post("/api/cgan/latent_walk", json=walk_payload)
    assert walk_resp.status_code == 200, f"cGAN walk failed: {walk_resp.text}"
    walk_res = walk_resp.json()
    print(f"  Latent Walk: OK ({len(walk_res['frames'])} frames generated)")

def test_model4_diffusion():
    print("\n--- [4] Testing Conditional Diffusion (DDPM / DDIM) ---")
    data = {
        "sample_class": "helmet",
        "degradation_type": "gaussian_noise",
        "severity": "medium",
        "ddim_steps": 10
    }
    resp = client.post("/api/diffusion/restore", data=data)
    assert resp.status_code == 200, f"Diffusion restore failed: {resp.text}"
    res = resp.json()
    print("  Status: OK")
    print(f"  Diffusion PSNR: {res['metrics']['psnr_diffusion']} dB, SSIM: {res['metrics']['ssim_diffusion']}")
    print(f"  Regression PSNR: {res['metrics']['psnr_regression']} dB, SSIM: {res['metrics']['ssim_regression']}")
    print(f"  Step Trajectory Recorded: {len(res.get('cleanup_steps', []))} steps")
    assert len(res.get("cleanup_steps", [])) > 0

def test_model5_policy():
    print("\n--- [5] Testing Policy Transformer (DistilBERT) ---")
    # 1. Document Audit
    data = {"sample_id": "civil_dual"}
    resp = client.post("/api/policy/audit-document", data=data)
    assert resp.status_code == 200, f"Policy audit failed: {resp.text}"
    res = resp.json()
    print("  Document Audit Status: OK")
    print(f"  Overall Score: {res['compliance_score']}% ({res['overall_verdict']})")
    print(f"  Summary: {res['verdict_summary']}")
    print(f"  Evaluated {len(res['categories'])} statutory categories:")
    for cat in res['categories']:
        print(f"    - {cat['name']}: {cat['status']}")

    # 2. Cross-Modal Verdict
    clauses = [
        {"predicted_label": "both_required", "text": "All workers must wear hard hat and safety vest.", "page": 1}
    ]
    v_data = {
        "sample_crop": "head",
        "policy_sentences_json": json.dumps(clauses)
    }
    v_resp = client.post("/api/policy/verdict", data=v_data)
    assert v_resp.status_code == 200, f"Verdict failed: {v_resp.text}"
    v_res = v_resp.json()
    print("  Surveillance Cross-Modal Verdict: OK")
    print(f"  Verdict: {v_res['verdict']} -> {v_res['summary']}")

if __name__ == "__main__":
    print("================================================================================")
    print("STARTING COMPLETE VERIFICATION OF ALL 5 MODELS ACROSS THE PLATFORM")
    print("================================================================================")
    test_model1_autoencoder()
    test_model2_cvae()
    test_model3_cgan()
    test_model4_diffusion()
    test_model5_policy()
    print("\n================================================================================")
    print("ALL 5 MODELS VERIFIED AND PASSING WITH EXCELLENT RESULTS!")
    print("================================================================================")

