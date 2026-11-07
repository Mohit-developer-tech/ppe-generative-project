"""
scripts/prepare_release_assets.py
Prepares the release_assets/ directory containing copies of only shared assets:
- Autoencoder v1 weights, scaler, svm, pca
- Autoencoder v2 weights
- Judge v2 5-seed ensemble weights
- FID reference statistics (noise_floor.json)

Computes SHA256 checksums and writes assets_manifest.json.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

SHARED_FILES = [
    # Autoencoder v1
    ("artifacts/autoencoder/ae_best.pt", "release_assets/autoencoder/ae_best.pt"),
    ("artifacts/autoencoder/scaler.joblib", "release_assets/autoencoder/scaler.joblib"),
    ("artifacts/autoencoder/svm_model.joblib", "release_assets/autoencoder/svm_model.joblib"),
    ("artifacts/autoencoder/latent_pca2d.joblib", "release_assets/autoencoder/latent_pca2d.joblib"),
    ("artifacts/autoencoder/latent_features.npz", "release_assets/autoencoder/latent_features.npz"),

    # Autoencoder v2
    ("artifacts/autoencoder_v2/v2_loss_combined_256/ae_best.pt", "release_assets/autoencoder_v2/ae_best.pt"),

    # Judge v2 (Ensemble seeds 42..46)
    ("artifacts/judge/judge_seed_42.pt", "release_assets/judge/judge_seed_42.pt"),
    ("artifacts/judge/judge_seed_43.pt", "release_assets/judge/judge_seed_43.pt"),
    ("artifacts/judge/judge_seed_44.pt", "release_assets/judge/judge_seed_44.pt"),
    ("artifacts/judge/judge_seed_45.pt", "release_assets/judge/judge_seed_45.pt"),
    ("artifacts/judge/judge_seed_46.pt", "release_assets/judge/judge_seed_46.pt"),

    # FID Reference Noise Floor
    ("results/noise_floor.json", "release_assets/noise_floor.json"),
]

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    rel_dir = Path("release_assets")
    rel_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "release_tag": "v0.1",
        "description": "Shared baseline weights, judge ensemble, and reference statistics for PPE Generative AI Benchmark",
        "judge_version": "judge_v2_small_resnet_ensemble_5seed",
        "assets": {}
    }

    for src_str, dst_str in SHARED_FILES:
        src = Path(src_str)
        dst = Path(dst_str)
        if not src.exists():
            print(f"[!] Warning: Source file missing: {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        checksum = sha256_file(dst)
        rel_key = dst.relative_to(rel_dir).as_posix()
        manifest["assets"][rel_key] = {
            "target_path": src_str,
            "size_bytes": size,
            "sha256": checksum
        }
        print(f"[+] Staged {rel_key} ({size:,} bytes) - SHA256: {checksum[:12]}...")

    with open("assets_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully created release_assets/ and assets_manifest.json with {len(manifest['assets'])} files.")

if __name__ == "__main__":
    main()

