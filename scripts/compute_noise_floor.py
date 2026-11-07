"""
scripts/compute_noise_floor.py
Computes the empirical FID noise floor: FID(real test vs real train)
using PyTorch InceptionV3 (2048-d pool3 features, 64x64 upsampled to 299x299).
Saves results to results/noise_floor.json.
"""

import os
import sys
import json
import time
import torch
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath("."))

from common.data import set_seed
from common.metrics import InceptionFeatureExtractor, compute_fid_noise_floor


def load_cached_split(split_name: str, cache_dir: str = "data_cache"):
    path = os.path.join(cache_dir, f"{split_name}_crops.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached split {path}")
    data = torch.load(path, map_location="cpu")
    # Float in [0, 1]
    return data["X"].float() / 255.0, data["y"]


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("COMPUTING EMPIRICAL FID NOISE FLOOR: FID(Real Test vs Real Train)")
    print(f"Device: {device}")
    print("=" * 70)

    t0 = time.time()
    print("[*] Loading cached datasets...")
    X_train, _ = load_cached_split("train")
    X_test, _ = load_cached_split("test")
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    # Standardized evaluation sample size: 2000 train vs 2000 test (or full test 7470)
    # Using 2000 for standard fixed-N reference benchmark
    N = 2000
    rng = np.random.RandomState(42)
    idx_train = rng.choice(len(X_train), size=min(N, len(X_train)), replace=False)
    idx_test = rng.choice(len(X_test), size=min(N, len(X_test)), replace=False)

    sub_train = X_train[idx_train]
    sub_test = X_test[idx_test]

    print(f"[*] Initializing InceptionV3 feature extractor...")
    extractor = InceptionFeatureExtractor(device=device)

    print(f"[*] Extracting Inception pool3 features for {len(sub_train)} real train crops...")
    t_feat0 = time.time()
    feat_train = extractor.extract_features(sub_train)
    print(f"    Train features: {feat_train.shape} in {time.time()-t_feat0:.1f}s")

    print(f"[*] Extracting Inception pool3 features for {len(sub_test)} real test crops...")
    t_feat1 = time.time()
    feat_test = extractor.extract_features(sub_test)
    print(f"    Test features: {feat_test.shape} in {time.time()-t_feat1:.1f}s")

    print("[*] Computing empirical distribution metrics (FID, KID, Precision, Recall)...")
    noise_floor_results = compute_fid_noise_floor(feat_train, feat_test)
    noise_floor_results["evaluation_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    noise_floor_results["duration_seconds"] = round(time.time() - t0, 2)

    os.makedirs("results", exist_ok=True)
    out_path = "results/noise_floor.json"
    with open(out_path, "w") as f:
        json.dump(noise_floor_results, f, indent=2)

    print("=" * 70)
    print("FID NOISE FLOOR BENCHMARK RESULTS:")
    print("=" * 70)
    print(f"FID Noise Floor:         {noise_floor_results['noise_floor_fid']:.4f}")
    print(f"KID Noise Floor (Mean):  {noise_floor_results['noise_floor_kid_mean']:.6f} +/- {noise_floor_results['noise_floor_kid_std']:.6f}")
    print(f"Precision (Coverage):    {noise_floor_results['noise_floor_precision']:.4f}")
    print(f"Recall (Density):        {noise_floor_results['noise_floor_recall']:.4f}")
    print(f"Sample Sizes:            Train={noise_floor_results['n_train']}, Test={noise_floor_results['n_test']}")
    print(f"Saved to:                {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

