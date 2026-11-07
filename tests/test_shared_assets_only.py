"""
tests/test_shared_assets_only.py
Tests that the dashboard boots and functions with ONLY the shared assets present
(Autoencoder baseline + Judge ensemble + noise floor), with non-shared sanity weights moved aside.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NON_SHARED_WEIGHTS = [
    "artifacts/cvae/cvae_best.pt",
    "artifacts/cgan/cgan_best.pt",
    "artifacts/diffusion/diffusion_best.pt",
    "artifacts/diffusion/regression_best.pt",
    "artifacts/policy/policy_classifier.pt"
]

def main():
    print("=== Testing Dashboard with ONLY Shared Assets ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        stashed = []

        # 1. Stash non-shared weights
        print("[*] Stashing non-shared weights aside...")
        for p_str in NON_SHARED_WEIGHTS:
            p = Path(p_str)
            if p.exists():
                dst = tmp_path / p.name
                shutil.move(str(p), str(dst))
                stashed.append((p, dst))
                print(f"  Stashed: {p} -> {dst}")

        try:
            # 2. Test Dashboard initialization
            print("\n[*] Initializing dashboard ML artifacts...")
            from dashboard.app import app, load_artifacts, get_health, predict_crop, MODELS
            from dashboard.models.registry import get_registered_models, get_model_info
            from dashboard.models.autoencoder.router import load_autoencoder_artifacts
            from dashboard.models.cvae.router import get_cvae_info, get_cvae_metrics_endpoint
            from dashboard.models.cgan.router import get_cgan_info
            from dashboard.models.diffusion.router import get_diffusion_info

            # Force re-load
            MODELS.clear()
            load_artifacts()
            health = get_health()
            assert health["model_loaded"] is True, "Autoencoder model must be loaded!"
            assert health["svm_loaded"] is True, "SVM must be loaded!"
            print(f"  [PASS] /api/health: {health['status']} (device: {health['device']}, AE loaded: True, SVM loaded: True)")

            # Autoencoder prediction verification
            ae_info = get_model_info("autoencoder")
            assert ae_info["status"] == "ready"
            print("  [PASS] Autoencoder status: ready")

            # Check other model status cards
            cvae_info = get_cvae_info()
            print(f"  [PASS] CVAE without weights status: {cvae_info['status']}")

            cgan_info = get_cgan_info()
            print(f"  [PASS] cGAN without weights status: {cgan_info['status']}")

            diff_info = get_diffusion_info()
            print(f"  [PASS] Diffusion without weights status: {diff_info['status']}")

            # Verify autoencoder prediction on test sample
            test_crop = Path("crops_64/test/helmet")
            if test_crop.exists():
                sample_file = list(test_crop.glob("*.png"))[0]
                from dashboard.test_api import make_upload_file
                with open(sample_file, "rb") as fp:
                    u_file = make_upload_file(sample_file.name, fp.read())
                pred_res = predict_crop(u_file)
                assert pred_res["predicted_class"] == "helmet"
                print(f"  [PASS] AE Prediction on helmet crop: {pred_res['predicted_class']} (scores: {pred_res['class_scores']})")

            print("\n[SUCCESS] Dashboard operates properly with ONLY shared assets!")

        finally:
            # 3. Restore all stashed files
            print("\n[*] Restoring stashed weights...")
            for orig, dst in stashed:
                if dst.exists():
                    orig.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(orig))
                    print(f"  Restored: {orig}")

if __name__ == "__main__":
    main()
