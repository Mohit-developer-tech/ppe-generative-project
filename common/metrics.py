"""
common/metrics.py
Unified benchmarking metrics suite for generative and restoration PPE models.
Provides:
  - Per-image fidelity: PSNR, SSIM, MAE, MSE
  - Perceptual similarity: LPIPS (AlexNet)
  - Distributional fidelity: Fréchet Inception Distance (FID) and Kernel Inception Distance (KID)
  - Distributional coverage: Improved Precision & Recall (k-NN manifold estimation)
  - Diversity: Intra-class pairwise LPIPS diversity
  - Privacy / Memorization check: Nearest-real-neighbor distance in feature space
  - Noise floor calculation: FID(real test vs real train)
All implementations are deterministic, fixed-seed, and report sample sizes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.linalg
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from skimage.metrics import structural_similarity as sk_ssim, peak_signal_noise_ratio as sk_psnr
try:
    import lpips
except ImportError:
    lpips = None


# ==============================================================================
# Per-Image Reconstruction Metrics
# ==============================================================================

def _to_numpy_hwc(img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or ndarray to numpy [H, W, C] in [0, 1]."""
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.min() < -0.1:  # Range [-1, 1]
            t = (t + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        return t.permute(1, 2, 0).numpy()
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 1.0)


def compute_mse(img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute Mean Squared Error between two images."""
    a1 = _to_numpy_hwc(img1)
    a2 = _to_numpy_hwc(img2)
    return float(np.mean((a1 - a2) ** 2))


def compute_mae(img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute Mean Absolute Error between two images."""
    a1 = _to_numpy_hwc(img1)
    a2 = _to_numpy_hwc(img2)
    return float(np.mean(np.abs(a1 - a2)))


def compute_psnr(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    max_psnr: float = 100.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (dB).
    Gracefully handles identical images without division by zero or crashing.
    """
    a1 = _to_numpy_hwc(img1)
    a2 = _to_numpy_hwc(img2)
    mse_val = np.mean((a1 - a2) ** 2)
    if mse_val <= 1e-12:
        return float(max_psnr)
    return float(sk_psnr(a1, a2, data_range=data_range))


def compute_ssim(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0
) -> float:
    """Compute Structural Similarity Index (SSIM). Identical images return exactly 1.0."""
    a1 = _to_numpy_hwc(img1)
    a2 = _to_numpy_hwc(img2)
    if np.array_equal(a1, a2):
        return 1.0
    return float(sk_ssim(a1, a2, data_range=data_range, channel_axis=2))


# ==============================================================================
# Perceptual Metric: LPIPS
# ==============================================================================

class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity (LPIPS) using pretrained AlexNet."""
    def __init__(self, device: Optional[torch.device] = None):
        if lpips is None:
            raise ImportError("lpips package is required for perceptual loss/metric. Install via 'pip install lpips'.")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device)
        self.loss_fn.eval()

    def _prepare_tensor(self, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert input to torch tensor [1, 3, H, W] in [-1, 1]."""
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.from_numpy(arr).unsqueeze(0).float()
        elif isinstance(img, torch.Tensor):
            t = img.detach().clone().float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Rescale [0, 1] to [-1, 1] if needed
        if t.min() >= -0.01 and t.max() <= 1.01:
            t = (t * 2.0) - 1.0
        return t.to(self.device)

    def compute(self, img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray]) -> float:
        """Compute pairwise LPIPS distance between two images."""
        t1 = self._prepare_tensor(img1)
        t2 = self._prepare_tensor(img2)
        with torch.no_grad():
            dist = self.loss_fn(t1, t2)
        return float(dist.item())

    def compute_batch(self, batch1: torch.Tensor, batch2: torch.Tensor) -> np.ndarray:
        """Compute per-pair LPIPS distance for batches [B, 3, H, W]."""
        t1 = batch1.to(self.device).float()
        t2 = batch2.to(self.device).float()
        if t1.min() >= -0.01 and t1.max() <= 1.01:
            t1 = (t1 * 2.0) - 1.0
        if t2.min() >= -0.01 and t2.max() <= 1.01:
            t2 = (t2 * 2.0) - 1.0
        with torch.no_grad():
            dist = self.loss_fn(t1, t2)
        return dist.view(-1).cpu().numpy()


# Global cached LPIPS instance
_CACHED_LPIPS: Optional[LPIPSMetric] = None

def get_lpips_metric(device: Optional[torch.device] = None) -> LPIPSMetric:
    """Return a singleton LPIPS metric instance."""
    global _CACHED_LPIPS
    if _CACHED_LPIPS is None:
        _CACHED_LPIPS = LPIPSMetric(device=device)
    return _CACHED_LPIPS


def compute_lpips(img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray], device: Optional[torch.device] = None) -> float:
    """Convenience function computing pairwise LPIPS."""
    metric = get_lpips_metric(device=device)
    return metric.compute(img1, img2)


def compute_intra_class_diversity(
    images: Union[List[torch.Tensor], torch.Tensor, np.ndarray],
    max_pairs: int = 500,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Compute intra-class generative diversity as the mean pairwise LPIPS distance.
    Higher diversity = model generates varied samples rather than mode collapse.
    """
    metric = get_lpips_metric(device=device)
    num_samples = len(images)
    if num_samples < 2:
        return {"mean_diversity": 0.0, "std_diversity": 0.0, "num_pairs": 0, "sample_size": num_samples}

    rng = np.random.RandomState(seed)
    pairs = []
    # If small, compute all combinations; otherwise sample uniformly
    total_possible = (num_samples * (num_samples - 1)) // 2
    if total_possible <= max_pairs:
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                pairs.append((i, j))
    else:
        seen = set()
        while len(pairs) < max_pairs:
            i, j = rng.choice(num_samples, size=2, replace=False)
            if i > j:
                i, j = j, i
            if (i, j) not in seen:
                seen.add((i, j))
                pairs.append((i, j))

    dists = []
    for i, j in pairs:
        im1 = images[i]
        im2 = images[j]
        d = metric.compute(im1, im2)
        dists.append(d)

    dists = np.array(dists)
    return {
        "mean_diversity": float(np.mean(dists)),
        "std_diversity": float(np.std(dists)),
        "num_pairs": len(dists),
        "sample_size": num_samples
    }


# ==============================================================================
# InceptionV3 Feature Extractor (Upsampling 64x64 to 299x299)
# ==============================================================================

class InceptionFeatureExtractor:
    """
    Extracts 2048-dimensional pool3 features using PyTorch InceptionV3.
    Standardized across all generative models (cVAE, cGAN, Diffusion).
    """
    def __init__(self, device: Optional[torch.device] = None):
        if lpips is None:
            raise ImportError("lpips package is required for perceptual loss/metric. Install via 'pip install lpips'.")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = tv_models.inception_v3(weights="DEFAULT")
        model.fc = nn.Identity()  # Remove final 1000-class linear projection
        model.eval()
        self.model = model.to(self.device)

        # Inception normalization and upsampling pipeline
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def extract_features(
        self,
        dataloader_or_tensors: Union[torch.utils.data.DataLoader, torch.Tensor],
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract Inception 2048-d features.
        Input images are upsampled from 64x64 to 299x299 as mathematically required.
        """
        features_list = []
        collected = 0

        def process_batch(batch_x: torch.Tensor):
            nonlocal collected
            t = batch_x.to(self.device).float()
            # If in [-1, 1], convert to [0, 1]
            if t.min() < -0.1:
                t = torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)
            else:
                t = torch.clamp(t, 0.0, 1.0)

            # Upsample to 299x299 with bilinear interpolation
            t_299 = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
            t_norm = self.normalize(t_299)

            with torch.no_grad():
                feats = self.model(t_norm)  # (B, 2048)
            features_list.append(feats.cpu().numpy())
            collected += t.size(0)

        if isinstance(dataloader_or_tensors, torch.Tensor):
            batch_size = 64
            for i in range(0, dataloader_or_tensors.size(0), batch_size):
                if max_samples and collected >= max_samples:
                    break
                batch = dataloader_or_tensors[i:i + batch_size]
                process_batch(batch)
        else:
            for batch in dataloader_or_tensors:
                if max_samples and collected >= max_samples:
                    break
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                process_batch(batch_x)

        all_feats = np.concatenate(features_list, axis=0)
        if max_samples:
            all_feats = all_feats[:max_samples]
        return all_feats


# Global cached Inception extractor
_CACHED_INCEPTION: Optional[InceptionFeatureExtractor] = None

def get_inception_extractor(device: Optional[torch.device] = None) -> InceptionFeatureExtractor:
    """Return a singleton InceptionFeatureExtractor instance."""
    global _CACHED_INCEPTION
    if _CACHED_INCEPTION is None:
        _CACHED_INCEPTION = InceptionFeatureExtractor(device=device)
    return _CACHED_INCEPTION


# ==============================================================================
# Generative Distribution Metrics: FID & KID
# ==============================================================================

def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Fréchet Inception Distance (FID) between real and generated feature distributions.
    ONE exact, standard implementation used everywhere.
    
    Formula:
        FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 * (Sigma_r * Sigma_f)^(1/2))
    """
    if np.array_equal(real_features, fake_features):
        return 0.0

    mu_r = np.mean(real_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)

    mu_f = np.mean(fake_features, axis=0)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f

    # Matrix square root of covariance product
    try:
        covmean = scipy.linalg.sqrtm(sigma_r.dot(sigma_f))
    except TypeError:
        covmean, _ = scipy.linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_r + offset).dot(sigma_f + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    # Numerical imaginary artifact elimination
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_f) - 2 * tr_covmean)
    return max(0.0, fid)


def compute_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    subset_size: int = 100,
    num_subsets: int = 50,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Kernel Inception Distance (KID) using polynomial kernel MMD:
        k(x, y) = (1/d * x^T y + 1)^3
    Unbiased, fixed-N, fixed-seed.
    
    Returns:
        (mean_kid, std_kid)
    """
    if np.array_equal(real_features, fake_features):
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    m = min(subset_size, n_real, n_fake)

    if m < 2:
        return 0.0, 0.0

    d = real_features.shape[1]

    def polynomial_kernel(x, y):
        return ((x @ y.T) / d + 1.0) ** 3

    mmd_scores = []
    for _ in range(num_subsets):
        idx_r = rng.choice(n_real, size=m, replace=False)
        idx_f = rng.choice(n_fake, size=m, replace=False)

        x = real_features[idx_r]
        y = fake_features[idx_f]

        k_xx = polynomial_kernel(x, x)
        k_yy = polynomial_kernel(y, y)
        k_xy = polynomial_kernel(x, y)

        # Unbiased estimator: zero out diagonals for k_xx and k_yy
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        mmd = (k_xx.sum() / (m * (m - 1))) + (k_yy.sum() / (m * (m - 1))) - (2.0 * k_xy.sum() / (m * m))
        mmd_scores.append(mmd)

    return float(np.mean(mmd_scores)), float(np.std(mmd_scores))


# ==============================================================================
# Distributional Coverage: Improved Precision and Recall
# ==============================================================================

def compute_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3
) -> Dict[str, float]:
    """
    Improved Precision and Recall for generative model evaluation
    (Sajjadi et al. 2018; Kynkäänniemi et al. NeurIPS 2019).
    
    - Precision: Fraction of generated samples falling within the real data manifold.
    - Recall: Fraction of real samples falling within the generated data manifold.
    """
    if np.array_equal(real_features, fake_features):
        return {"precision": 1.0, "recall": 1.0, "k": k, "n_real": len(real_features), "n_fake": len(fake_features)}

    # Distance to k-th nearest neighbor in real features
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(real_features)
    dist_r, _ = nn_real.kneighbors(real_features)
    radii_real = dist_r[:, k]  # (N_real,)

    # Distance to k-th nearest neighbor in fake features
    nn_fake = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(fake_features)
    dist_f, _ = nn_fake.kneighbors(fake_features)
    radii_fake = dist_f[:, k]  # (N_fake,)

    # Precision: For each fake, is it within radius of ANY real?
    # Find distance from each fake to closest real sample
    dists_fake_to_real, idx_closest_real = nn_real.kneighbors(fake_features, n_neighbors=1)
    precision = float(np.mean(dists_fake_to_real.squeeze(1) <= radii_real[idx_closest_real.squeeze(1)]))

    # Recall: For each real, is it within radius of ANY fake?
    dists_real_to_fake, idx_closest_fake = nn_fake.kneighbors(real_features, n_neighbors=1)
    recall = float(np.mean(dists_real_to_fake.squeeze(1) <= radii_fake[idx_closest_fake.squeeze(1)]))

    return {
        "precision": precision,
        "recall": recall,
        "k": k,
        "n_real": len(real_features),
        "n_fake": len(fake_features)
    }


# ==============================================================================
# Memorization Check: Nearest-Real-Neighbor
# ==============================================================================

def check_memorization(
    generated_features: np.ndarray,
    real_features: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Check whether generative models are memorizing / copy-pasting training crops.
    Computes minimum Euclidean distance to real training crops in feature space.
    """
    # Normalize features to unit sphere for cosine distance
    norm_gen = generated_features / np.linalg.norm(generated_features, axis=1, keepdims=True)
    norm_real = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(norm_real)
    dists, indices = nn.kneighbors(norm_gen)
    dists = dists.squeeze(1)
    indices = indices.squeeze(1)

    min_dist = float(np.min(dists))
    mean_dist = float(np.mean(dists))
    p5_dist = float(np.percentile(dists, 5))
    memorized_count = int(np.sum(dists < threshold))

    return {
        "min_distance": min_dist,
        "mean_distance": mean_dist,
        "p5_distance": p5_dist,
        "memorized_count": memorized_count,
        "total_generated": len(generated_features),
        "is_memorizing": (memorized_count > 0 or p5_dist < threshold),
        "nearest_real_indices": indices.tolist()
    }


# ==============================================================================
# FID Noise Floor (Real Test vs Real Train)
# ==============================================================================

def compute_fid_noise_floor(
    train_features: np.ndarray,
    test_features: np.ndarray
) -> Dict[str, Any]:
    """
    Compute the empirical FID noise floor: FID(real test vs real train).
    Provides the empirical lower bound against which all generative models are judged.
    """
    floor_fid = compute_fid(train_features, test_features)
    floor_kid_mean, floor_kid_std = compute_kid(train_features, test_features)
    pr_stats = compute_precision_recall(train_features, test_features)

    return {
        "noise_floor_fid": floor_fid,
        "noise_floor_kid_mean": floor_kid_mean,
        "noise_floor_kid_std": floor_kid_std,
        "noise_floor_precision": pr_stats["precision"],
        "noise_floor_recall": pr_stats["recall"],
        "n_train": len(train_features),
        "n_test": len(test_features)
    }

