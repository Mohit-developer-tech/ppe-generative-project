"""
scripts/diagnose_cvae.py
Investigate baseline CVAE conditional fidelity, temperature sensitivity,
and failure modes on helmet/head crops.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from common.cvae import ConditionalVAE, compute_active_units
from common.judge import SmallResNet
from common.data import CLASS_NAMES

def main():
    device = torch.device("cpu")

    print("[*] Loading baseline CVAE...")
    model = ConditionalVAE(latent_dim=128).to(device)
    model.load_state_dict(torch.load("checkpoints/baseline/cvae/cvae_best.pt", map_location="cpu"))
    model.eval()

    print("[*] Loading 5-seed ResNet Judge ensemble...")
    judges = []
    for s in [42, 43, 44, 45, 46]:
        j = SmallResNet(num_classes=3).to(device)
        ckpt = torch.load(f"artifacts/judge/judge_seed_{s}.pt", map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        j.load_state_dict(state_dict)
        j.eval()
        judges.append(j)

    torch.manual_seed(42)
    print("\n--- Temperature Sweep (tau in [0.4 .. 1.0]) ---")
    for tau in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        per_cls = {}
        for c_idx, c_name in enumerate(CLASS_NAMES):
            samples = model.sample(200, class_idx=c_idx, temperature=tau, device=device)
            preds = []
            with torch.no_grad():
                for j in judges:
                    logits = j(samples)
                    preds.append(torch.argmax(logits, dim=1))
            preds = torch.stack(preds, dim=0) # (5, 200)
            mode_preds = torch.mode(preds, dim=0).values # (200,)
            acc = (mode_preds == c_idx).float().mean().item()
            per_cls[c_name] = acc
        mean_acc = np.mean(list(per_cls.values()))
        print(f"tau={tau:.2f} | Overall: {mean_acc*100:5.2f}% | Head: {per_cls['head']*100:5.1f}% | Helmet: {per_cls['helmet']*100:5.1f}% | Vest: {per_cls['vest']*100:5.1f}%")

if __name__ == "__main__":
    main()
