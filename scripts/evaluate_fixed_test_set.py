"""
scripts/evaluate_fixed_test_set.py
Standardized Evaluation of Fixed Construction Safety PPE Test Crops
across Baseline vs Optimized Checkpoints.
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim

from dashboard.model_def import ConvAE

os.makedirs("results/evaluation_set", exist_ok=True)
os.makedirs("artifacts/evaluation_set", exist_ok=True)

test_samples = [
    {"class": "helmet", "path": "crops_64/test/helmet/ds1_hard_hat_workers0_0.png", "desc": "Standard Yellow Hardhat Worker"},
    {"class": "helmet", "path": "crops_64/test/helmet/ds1_hard_hat_workers1000_0.png", "desc": "White Construction Helmet"},
    {"class": "head", "path": "crops_64/test/head/ds1_hard_hat_workers1017_2.png", "desc": "Bare Head (Safety Violation - Worker Headshot)"},
    {"class": "head", "path": "crops_64/test/head/ds1_hard_hat_workers1018_2.png", "desc": "Unprotected Worker Face/Head Hazard"},
    {"class": "vest", "path": "crops_64/test/vest/ds2_helmet_jacket_00003_1.png", "desc": "High-Visibility Safety Vest"},
    {"class": "vest", "path": "crops_64/test/vest/ds2_helmet_jacket_00007_0.png", "desc": "Fluorescent Yellow/Green Vest"}
]

print("=" * 80)
print("FIXED TEST CROP RECONSTRUCTION EVALUATION ACROSS MODEL CHECKPOINTS")
print("=" * 80)

# Load Baseline ConvAE
ae_base = ConvAE(in_channels=3, latent_dim=256)
ckpt_base = torch.load("checkpoints/baseline/autoencoder/ae_best.pt", map_location="cpu")
ae_base.load_state_dict(ckpt_base["model_state_dict"] if "model_state_dict" in ckpt_base else ckpt_base)
ae_base.eval()

eval_records = []

for sample in test_samples:
    p = sample["path"]
    if not os.path.exists(p):
        print(f"[-] Missing: {p}")
        continue
    
    img = Image.open(p).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0)
    
    with torch.no_grad():
        recon_base, z_base = ae_base(x)
        
    orig_np = x[0].permute(1, 2, 0).numpy()
    recon_base_np = np.clip(recon_base[0].permute(1, 2, 0).numpy(), 0, 1)
    
    # Dual-Band high-frequency edge restoration on 256x256 display
    recon_u8 = (recon_base_np * 255).astype(np.uint8)
    recon_256 = cv2.resize(recon_u8, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    blur1 = cv2.GaussianBlur(recon_256, (0, 0), sigmaX=1.0)
    blur2 = cv2.GaussianBlur(recon_256, (0, 0), sigmaX=3.0)
    fine = cv2.subtract(recon_256, blur1)
    coarse = cv2.subtract(blur1, blur2)
    sharp_256 = cv2.addWeighted(recon_256, 1.0, fine, 2.2, 0)
    sharp_256 = np.clip(cv2.addWeighted(sharp_256, 1.0, coarse, 1.1, 0), 0, 255).astype(np.uint8)
    
    orig_256 = np.array(img.resize((256, 256), Image.Resampling.LANCZOS))
    
    # Metrics
    mse_raw = float(np.mean((orig_np - recon_base_np)**2))
    psnr_raw = float(sk_psnr(orig_np, recon_base_np, data_range=1.0))
    ssim_raw = float(sk_ssim(orig_np, recon_base_np, data_range=1.0, channel_axis=2))
    
    mse_sharp = float(np.mean((orig_256/255.0 - sharp_256/255.0)**2))
    psnr_sharp = float(sk_psnr(orig_256/255.0, sharp_256/255.0, data_range=1.0))
    ssim_sharp = float(sk_ssim(orig_256/255.0, sharp_256/255.0, data_range=1.0, channel_axis=2))
    
    diff_256 = cv2.absdiff(orig_256, sharp_256)
    diff_heatmap = cv2.applyColorMap(diff_256, cv2.COLORMAP_INFERNO)
    
    # Save visual comparison strip: [Original | Raw 64x64 | Enhanced 256x256 | Error Heatmap]
    raw_disp = cv2.resize(recon_u8, (256, 256), interpolation=cv2.INTER_NEAREST)
    orig_bgr = cv2.cvtColor(orig_256, cv2.COLOR_RGB2BGR)
    raw_bgr = cv2.cvtColor(raw_disp, cv2.COLOR_RGB2BGR)
    sharp_bgr = cv2.cvtColor(sharp_256, cv2.COLOR_RGB2BGR)
    
    # Add text labels
    cv2.putText(orig_bgr, "Original Target", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(raw_bgr, f"Raw Model (PSNR:{psnr_raw:.1f}dB)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(sharp_bgr, f"Enhanced (PSNR:{psnr_sharp:.1f}dB)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(diff_heatmap, "Absolute Error Residual", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    strip = np.hstack([orig_bgr, raw_bgr, sharp_bgr, diff_heatmap])
    fname = os.path.basename(p)
    out_strip_path = os.path.join("results/evaluation_set", f"strip_{fname}")
    cv2.imwrite(out_strip_path, strip)
    
    record = {
        "file": fname,
        "class": sample["class"],
        "description": sample["desc"],
        "mse_raw_64": mse_raw,
        "psnr_raw_64": psnr_raw,
        "ssim_raw_64": ssim_raw,
        "mse_enhanced_256": mse_sharp,
        "psnr_enhanced_256": psnr_sharp,
        "ssim_enhanced_256": ssim_sharp,
        "strip_saved": out_strip_path
    }
    eval_records.append(record)
    print(f"[{sample['class'].upper()}] {sample['desc']}")
    print(f"  Raw 64x64:      MSE={mse_raw:.6f}, PSNR={psnr_raw:.2f} dB, SSIM={ssim_raw:.4f}")
    print(f"  Enhanced 256:   MSE={mse_sharp:.6f}, PSNR={psnr_sharp:.2f} dB, SSIM={ssim_sharp:.4f}")
    print(f"  Visual Strip:   {out_strip_path}")

out_json = "results/evaluation_set/evaluation_summary.json"
with open(out_json, "w") as f:
    json.dump(eval_records, f, indent=2)

print("\n[+] Fixed evaluation set audit complete. Saved to results/evaluation_set/evaluation_summary.json")
