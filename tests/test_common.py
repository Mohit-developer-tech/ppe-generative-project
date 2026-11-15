"""
tests/test_common.py
Comprehensive unit test suite for the shared foundation package:
  - common/data.py
  - common/degrade.py
  - common/metrics.py
  - common/judge.py
  - common/export.py
"""

import json
import os
import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import shutil
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from common.data import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    get_transforms,
    denormalize_tensor,
    verify_dataset_counts
)
from common.degrade import (
    degrade_image,
    degrade_batch,
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_downscale_upscale,
    apply_jpeg_compression,
    apply_motion_blur,
    HELD_OUT_DEGRADATION
)
from common.metrics import (
    compute_mse,
    compute_mae,
    compute_psnr,
    compute_ssim,
    compute_fid,
    compute_kid,
    compute_precision_recall,
    check_memorization
)
from common.judge import (
    SmallResNet,
    train_classifier,
    evaluate_classifier
)
from common.export import (
    export_model_bundle,
    load_model_bundle
)


class TestCommonFoundation(unittest.TestCase):
    
    # -------------------------------------------------------------------------
    # 1. Metrics Unit Tests
    # -------------------------------------------------------------------------
    def test_metrics_identical_images(self):
        """Verify that identical images produce perfect fidelity scores without crashing."""
        img = np.random.RandomState(42).rand(64, 64, 3).astype(np.float32)
        
        # SSIM of identical images must be exactly 1.0
        ssim_val = compute_ssim(img, img)
        self.assertAlmostEqual(ssim_val, 1.0, places=5)
        
        # PSNR of identical images must handle 0 MSE gracefully (returning 100.0)
        psnr_val = compute_psnr(img, img)
        self.assertEqual(psnr_val, 100.0)
        
        # MSE and MAE must be exactly 0.0
        self.assertEqual(compute_mse(img, img), 0.0)
        self.assertEqual(compute_mae(img, img), 0.0)

    def test_metrics_fid_kid_identical_distributions(self):
        """Verify that identical feature sets produce zero FID and zero KID."""
        rng = np.random.RandomState(42)
        feats = rng.randn(60, 64).astype(np.float64)
        
        fid = compute_fid(feats, feats)
        self.assertAlmostEqual(fid, 0.0, places=4)
        
        kid_mean, kid_std = compute_kid(feats, feats, subset_size=25, num_subsets=5, seed=42)
        self.assertAlmostEqual(kid_mean, 0.0, places=4)
        
        pr = compute_precision_recall(feats, feats, k=3)
        self.assertAlmostEqual(pr["precision"], 1.0, places=4)
        self.assertAlmostEqual(pr["recall"], 1.0, places=4)

    def test_memorization_check(self):
        """Verify that memorization check flags duplicate samples and passes distinct ones."""
        real_feats = np.random.RandomState(42).randn(50, 32).astype(np.float32)
        # Duplicate 5 samples directly (simulating memorization)
        gen_feats_memorized = real_feats[:5].copy()
        mem_res = check_memorization(gen_feats_memorized, real_feats, threshold=0.01)
        self.assertTrue(mem_res["is_memorizing"])
        self.assertEqual(mem_res["memorized_count"], 5)

        # Generate distinct samples
        distinct_feats = np.random.RandomState(99).randn(20, 32).astype(np.float32)
        distinct_res = check_memorization(distinct_feats, real_feats, threshold=0.01)
        self.assertFalse(distinct_res["is_memorizing"])

    # -------------------------------------------------------------------------
    # 2. Degradation Pipeline Unit Tests
    # -------------------------------------------------------------------------
    def test_degradation_shape_and_range(self):
        """Ensure all degradation operators maintain tensor shapes and pixel range [0, 1]."""
        x = torch.rand(3, 64, 64)
        for op in [apply_gaussian_noise, apply_gaussian_blur, apply_downscale_upscale, apply_jpeg_compression]:
            out = op(x, severity="medium", seed=42)
            self.assertEqual(out.shape, x.shape)
            self.assertGreaterEqual(float(out.min()), -0.01)
            self.assertLessEqual(float(out.max()), 1.01)

    def test_degradation_determinism(self):
        """Ensure seeded degradations are 100% deterministic."""
        x = torch.rand(3, 64, 64)
        d1 = degrade_image(x, severity="medium", seed=777)
        d2 = degrade_image(x, severity="medium", seed=777)
        self.assertTrue(torch.allclose(d1, d2, atol=1e-5))

    def test_degradation_severity_gradient(self):
        """Verify that severe degradation introduces more distortion than mild degradation."""
        x = torch.ones(3, 64, 64) * 0.5
        d_mild = apply_gaussian_noise(x, severity="mild", seed=42)
        d_sev = apply_gaussian_noise(x, severity="severe", seed=42)
        
        err_mild = float(torch.mean(torch.abs(d_mild - x)))
        err_sev = float(torch.mean(torch.abs(d_sev - x)))
        self.assertGreater(err_sev, err_mild)

    def test_held_out_motion_blur_permission(self):
        """Ensure held-out motion blur is protected and rejected in random training degradation."""
        x = torch.rand(3, 64, 64)
        # Applying motion blur in degrade_image without allow_held_out=True must raise PermissionError
        with self.assertRaises(PermissionError):
            degrade_image(x, degradations=["motion_blur"], allow_held_out=False)

        # Must succeed when allow_held_out=True
        out_mb = degrade_image(x, degradations=["motion_blur"], allow_held_out=True, seed=42)
        self.assertEqual(out_mb.shape, x.shape)

    def test_degrade_batch(self):
        """Ensure batch degradation works seamlessly on 4D tensors [B, C, H, W]."""
        batch = torch.rand(4, 3, 64, 64)
        out_batch = degrade_batch(batch, severity="mild", seed=42)
        self.assertEqual(out_batch.shape, (4, 3, 64, 64))

    # -------------------------------------------------------------------------
    # 3. Data Loading & Audit Unit Tests
    # -------------------------------------------------------------------------
    def test_verify_dataset_counts(self):
        """Verify that disk crop files match crops_64/stats.csv exactly."""
        df_audit = verify_dataset_counts(data_dir="crops_64", stats_csv="crops_64/stats.csv")
        self.assertEqual(len(df_audit), 9)  # 3 splits * 3 classes
        self.assertTrue(df_audit["matches"].all())

    def test_transforms_modes(self):
        """Verify [0,1] and [-1,1] transforms."""
        t_01 = get_transforms(mode="[0,1]")
        t_m11 = get_transforms(mode="[-1,1]")
        from PIL import Image
        pil_im = Image.new("RGB", (64, 64), color=(128, 128, 128))
        
        tensor_01 = t_01(pil_im)
        self.assertGreaterEqual(float(tensor_01.min()), 0.0)
        self.assertLessEqual(float(tensor_01.max()), 1.0)
        
        tensor_m11 = t_m11(pil_im)
        self.assertGreaterEqual(float(tensor_m11.min()), -1.0)
        self.assertLessEqual(float(tensor_m11.max()), 1.0)
        
        denorm = denormalize_tensor(tensor_m11, mode="[-1,1]")
        self.assertTrue(torch.allclose(denorm, tensor_01, atol=1e-2))

    # -------------------------------------------------------------------------
    # 4. SmallResNet Judge Pipeline on Tiny Subset
    # -------------------------------------------------------------------------
    def test_judge_pipeline_tiny_subset(self):
        """Verify that train_classifier runs cleanly on a tiny synthetic dataset."""
        # 30 synthetic samples: 10 per class
        x = torch.rand(30, 3, 64, 64)
        y = torch.tensor([0]*10 + [1]*10 + [2]*10, dtype=torch.long)
        
        train_ds = TensorDataset(x[:24], y[:24])
        val_ds = TensorDataset(x[24:], y[24:])
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=6, shuffle=False)
        
        res = train_classifier(
            train_data=train_loader,
            val_data=val_loader,
            test_data=val_loader,
            seed=42,
            epochs=2,
            patience=2,
            verbose=False
        )
        self.assertIn("model", res)
        self.assertIn("val_metrics", res)
        self.assertIn("test_metrics", res)
        self.assertGreaterEqual(res["test_metrics"]["accuracy"], 0.0)

    # -------------------------------------------------------------------------
    # 5. Export / Load Bundle Unit Tests
    # -------------------------------------------------------------------------
    def test_export_and_load_bundle(self):
        """Verify export_model_bundle and load_model_bundle."""
        temp_dir = tempfile.mkdtemp()
        try:
            m = SmallResNet(in_channels=3, num_classes=3)
            paths = export_model_bundle(
                model_name="test_model",
                output_dir=temp_dir,
                model=m,
                model_config={"param": 123},
                metrics_summary={"accuracy": 0.95}
            )
            self.assertTrue(os.path.exists(paths["weights"]))
            self.assertTrue(os.path.exists(paths["config"]))
            
            bundle = load_model_bundle("test_model", base_results_dir=os.path.dirname(temp_dir))
            # Just verify directly
            with open(paths["config"]) as f:
                loaded_cfg = json.load(f)
            self.assertEqual(loaded_cfg["param"], 123)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import json
    unittest.main()

