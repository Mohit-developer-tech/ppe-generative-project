"""
scripts/move_weights_to_artifacts.py
Moves all model weights, checkpoints, and joblib/npz files from results/ to artifacts/
leaving only small files (metrics.json, csv, figures) in results/.
"""
import os
import shutil
from pathlib import Path

MOVES = [
    # Autoencoder
    ("results/autoencoder/ae_best.pt", "artifacts/autoencoder/ae_best.pt"),
    ("results/autoencoder/latent_features.npz", "artifacts/autoencoder/latent_features.npz"),
    ("results/autoencoder/latent_pca2d.joblib", "artifacts/autoencoder/latent_pca2d.joblib"),
    ("results/autoencoder/scaler.joblib", "artifacts/autoencoder/scaler.joblib"),
    ("results/autoencoder/svm_model.joblib", "artifacts/autoencoder/svm_model.joblib"),
    ("results/autoencoder/latent_256/ae_best.pt", "artifacts/autoencoder/latent_256/ae_best.pt"),
    ("results/autoencoder_v2/v2_loss_combined_256/ae_best.pt", "artifacts/autoencoder_v2/v2_loss_combined_256/ae_best.pt"),

    # CVAE
    ("results/cvae/cvae_best.pt", "artifacts/cvae/cvae_best.pt"),

    # CGAN
    ("results/cgan/cgan_best.pt", "artifacts/cgan/cgan_best.pt"),

    # Diffusion
    ("results/diffusion/diffusion_best.pt", "artifacts/diffusion/diffusion_best.pt"),
    ("results/diffusion/regression_best.pt", "artifacts/diffusion/regression_best.pt"),

    # Policy
    ("results/policy/policy_classifier.pt", "artifacts/policy/policy_classifier.pt"),
]

# CVAE Checkpoints
cvae_ckpt_dir = Path("results/cvae/checkpoints")
if cvae_ckpt_dir.exists():
    for f in cvae_ckpt_dir.glob("checkpoint_epoch_*.pt"):
        MOVES.append((str(f), f"artifacts/cvae/checkpoints/{f.name}"))

# Checkpoints
ckpt_dir = Path("results/autoencoder/latent_256")
if ckpt_dir.exists():
    for f in ckpt_dir.glob("checkpoint_epoch_*.pt"):
        MOVES.append((str(f), f"artifacts/autoencoder/checkpoints/{f.name}"))

# Judge v2
judge_dir = Path("results/judge")
if judge_dir.exists():
    for f in judge_dir.glob("judge_seed_*.pt"):
        MOVES.append((str(f), f"artifacts/judge/{f.name}"))

# Judge v1
judge_v1_dir = Path("results/judge_v1")
if judge_v1_dir.exists():
    for f in judge_v1_dir.glob("judge_seed_*.pt"):
        MOVES.append((str(f), f"artifacts/judge_v1/{f.name}"))

print(f"Total files to move: {len(MOVES)}")
moved = []
for src_str, dst_str in MOVES:
    src = Path(src_str)
    dst = Path(dst_str)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append((src_str, dst_str))
        print(f"MOVED: {src_str} -> {dst_str}")
    else:
        print(f"SKIPPED (not found): {src_str}")

print(f"\nSuccessfully moved {len(moved)} files.")

