"""
Quick sanity unit test for common/diffusion.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import torch
from common.diffusion import (
    DiffusionSchedule,
    ConditionalUNet,
    RegressionUNet,
    EMA,
    sample_ddim,
    compute_diffusion_loss
)

def test_all():
    print("[*] Testing DiffusionSchedule...")
    sched = DiffusionSchedule(timesteps=100)
    assert sched.alphas_cumprod.shape[0] == 100
    assert sched.betas.min() > 0.0

    print("[*] Testing ConditionalUNet forward pass...")
    net = ConditionalUNet(in_channels=6, out_channels=3, base_channels=32, channel_mults=(1, 2, 2))
    x_t = torch.randn(2, 3, 64, 64)
    deg = torch.randn(2, 3, 64, 64)
    t = torch.tensor([10, 50], dtype=torch.long)
    out = net(x_t, deg, t)
    assert out.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64), got {out.shape}"
    print(f"  ✓ ConditionalUNet shape: {out.shape}")

    print("[*] Testing RegressionUNet forward pass...")
    reg_net = RegressionUNet(base_channels=32)
    reg_out = reg_net(deg)
    assert reg_out.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64), got {reg_out.shape}"
    print(f"  ✓ RegressionUNet shape: {reg_out.shape}")

    print("[*] Testing compute_diffusion_loss...")
    loss_dict = compute_diffusion_loss(net, x_t, deg, t, sched, loss_type="eps_l1")
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0
    print(f"  ✓ Loss L1: {loss_dict['loss'].item():.4f}")

    loss_dict_mse = compute_diffusion_loss(net, x_t, deg, t, sched, loss_type="eps_mse")
    assert "loss" in loss_dict_mse
    print(f"  ✓ Loss MSE: {loss_dict_mse['loss'].item():.4f}")

    print("[*] Testing DDIM 2-step sampling...")
    restored = sample_ddim(net, deg, sched, steps=2, eta=0.0)
    assert restored.shape == (2, 3, 64, 64)
    print(f"  ✓ DDIM restored shape: {restored.shape}")

    print("\nALL DIFFUSION UNIT TESTS PASSED!")

if __name__ == "__main__":
    test_all()
