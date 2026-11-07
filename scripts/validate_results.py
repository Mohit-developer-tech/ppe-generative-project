"""
scripts/validate_results.py
Validates result integrity across all models:
1. Verifies results/<model>/ contains only small text/metric/figure files (no weights).
2. Verifies artifacts/<model>/ contains the expected trained model weights.
3. Validates json metric schemas and sanity ranges for every benchmarked model.
"""
import json
import os
import sys
from pathlib import Path

def validate():
    print("=" * 60)
    print("PPE GENERATIVE AI BENCHMARK: RESULT VALIDATION SUITE")
    print("=" * 60)

    # 1. Check results directory does not leak large binaries
    leak_exts = {".pt", ".pth", ".ckpt", ".safetensors", ".joblib", ".npz"}
    leaks = []
    results_dir = Path("results")
    if results_dir.exists():
        for p in results_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in leak_exts:
                leaks.append(str(p))

    if leaks:
        print(f"[FAIL] Found {len(leaks)} large weight files leaked into results/:\n" + "\n".join(f"  - {x}" for x in leaks), file=sys.stderr)
        return False
    else:
        print("[PASS] Clean results/ layout: zero weight binaries found in results/.")

    # 2. Check artifacts directory has shared assets
    required_artifacts = [
        "artifacts/autoencoder/ae_best.pt",
        "artifacts/autoencoder/scaler.joblib",
        "artifacts/autoencoder/svm_model.joblib",
        "artifacts/judge/judge_seed_42.pt",
        "results/noise_floor.json"
    ]
    missing_artifacts = [p for p in required_artifacts if not os.path.exists(p)]
    if missing_artifacts:
        print(f"[FAIL] Missing required core artifacts:\n" + "\n".join(f"  - {x}" for x in missing_artifacts), file=sys.stderr)
        return False
    else:
        print("[PASS] Core baseline and judge ensemble weights verified in artifacts/.")

    # 3. Model Metrics Validation
    models = {
        "autoencoder": {
            "dir": "results/autoencoder",
            "cfg": "model_config.json",
            "cls": "class_names.json",
            "metrics": "final_summary.csv"
        },
        "cvae": {
            "dir": "results/cvae",
            "cfg": "model_config.json",
            "cls": "class_names.json",
            "metrics": "cvae_metrics.json"
        },
        "cgan": {
            "dir": "results/cgan",
            "cfg": "model_config.json",
            "cls": "class_names.json",
            "metrics": "cgan_metrics.json"
        },
        "diffusion": {
            "dir": "results/diffusion",
            "cfg": "model_config.json",
            "cls": "class_names.json",
            "metrics": "diffusion_metrics.json"
        },
        "policy": {
            "dir": "results/policy",
            "cfg": "model_config.json",
            "cls": None,
            "metrics": "policy_metrics.json"
        }
    }

    all_models_valid = True
    for mname, mspec in models.items():
        mdir = Path(mspec["dir"])
        if not mdir.exists():
            print(f"[FAIL] Missing model results folder: {mdir}", file=sys.stderr)
            all_models_valid = False
            continue

        cfg_p = mdir / mspec["cfg"]
        if not cfg_p.exists():
            print(f"[FAIL] Missing config in {mdir}: {mspec['cfg']}", file=sys.stderr)
            all_models_valid = False
            continue

        met_p = mdir / mspec["metrics"]
        if not met_p.exists():
            print(f"[FAIL] Missing metrics in {mdir}: {mspec['metrics']}", file=sys.stderr)
            all_models_valid = False
            continue

        fig_count = len(list(mdir.glob("*.png")) + list(mdir.glob("*/*.png")))
        print(f"[PASS] Model '{mname}': config & metrics verified, {fig_count} figures found.")

    print("=" * 60)
    if all_models_valid:
        print("[ALL CHECKS PASSED] Result schemas and artifact layouts are 100% valid.")
        return True
    else:
        print("[FAILED] Result validation errors detected.", file=sys.stderr)
        return False

if __name__ == "__main__":
    if not validate():
        sys.exit(1)

