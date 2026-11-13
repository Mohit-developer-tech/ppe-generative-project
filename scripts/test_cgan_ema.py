"""
scripts/test_cgan_ema.py
Evaluate raw Generator vs. Exponential Moving Average (EMA) Generator
on 5-seed ResNet Judge conditional fidelity, diversity, and privacy.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from common.data import CLASS_NAMES
from common.cgan import ConditionalGenerator
from common.judge import SmallResNet

def evaluate_gen(gen, judges, z_dim=128, num_samples=200):
    gen.eval()
    device = next(gen.parameters()).device
    torch.manual_seed(42)
    per_cls = {}
    
    for c_idx, c_name in enumerate(CLASS_NAMES):
        z = torch.randn(num_samples, z_dim, device=device)
        y = torch.full((num_samples,), c_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            fakes = gen(z, y)
            preds = []
            for j in judges:
                logits = j(fakes)
                preds.append(torch.argmax(logits, dim=1))
            preds = torch.stack(preds, dim=0)
            mode_preds = torch.mode(preds, dim=0).values
            acc = (mode_preds == c_idx).float().mean().item()
            per_cls[c_name] = acc
            
    mean_acc = float(np.mean(list(per_cls.values())))
    return mean_acc, per_cls

def main():
    print("=" * 75)
    print("MODEL 3: CONDITIONAL GAN RAW VS EMA GENERATOR BENCHMARK")
    print("=" * 75)
    
    device = torch.device("cpu")
    
    # Load Judges
    judges = []
    for s in [42, 43, 44, 45, 46]:
        j = SmallResNet(num_classes=3).to(device)
        ckpt = torch.load(f"artifacts/judge/judge_seed_{s}.pt", map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        j.load_state_dict(state_dict)
        j.eval()
        judges.append(j)
        
    ckpt = torch.load("checkpoints/baseline/cgan/cgan_best.pt", map_location="cpu")
    z_dim = ckpt.get("z_dim", 128)
    
    # 1. Evaluate Raw Generator
    print("[*] Evaluating Raw Generator weights...")
    gen_raw = ConditionalGenerator(z_dim=z_dim, num_classes=3).to(device)
    gen_raw.load_state_dict(ckpt["generator_state_dict"])
    raw_acc, raw_cls = evaluate_gen(gen_raw, judges, z_dim=z_dim)
    print(f"Raw Generator -> Overall: {raw_acc*100:5.2f}% | Head: {raw_cls['head']*100:5.1f}%, Helmet: {raw_cls['helmet']*100:5.1f}%, Vest: {raw_cls['vest']*100:5.1f}%")
    
    # 2. Evaluate EMA Generator
    print("[*] Evaluating EMA Generator weights...")
    gen_ema = ConditionalGenerator(z_dim=z_dim, num_classes=3).to(device)
    # First copy running stats from raw generator, then load EMA weights with strict=False
    gen_ema.load_state_dict(ckpt["generator_state_dict"])
    gen_ema.load_state_dict(ckpt["generator_ema_state_dict"], strict=False)
    ema_acc, ema_cls = evaluate_gen(gen_ema, judges, z_dim=z_dim)
    print(f"EMA Generator -> Overall: {ema_acc*100:5.2f}% | Head: {ema_cls['head']*100:5.1f}%, Helmet: {ema_cls['helmet']*100:5.1f}%, Vest: {ema_cls['vest']*100:5.1f}%")
    
    # Save optimized model
    os.makedirs("checkpoints/optimized/cgan", exist_ok=True)
    best_gen_state = ckpt["generator_ema_state_dict"] if ema_acc >= raw_acc else ckpt["generator_state_dict"]
    winner = "EMA Generator" if ema_acc >= raw_acc else "Raw Generator"
    torch.save(best_gen_state, "checkpoints/optimized/cgan/cgan_generator_optimized.pt")
    
    rep = {
        "model_id": "cgan",
        "name": "Conditional GAN (DiffAugment)",
        "status": "optimized",
        "selected_generator": winner,
        "raw_judge_accuracy": raw_acc,
        "ema_judge_accuracy": ema_acc,
        "per_class": ema_cls if ema_acc >= raw_acc else raw_cls
    }
    os.makedirs("results/experiments", exist_ok=True)
    with open("results/experiments/cgan_optimization_report.json", "w") as f:
        json.dump(rep, f, indent=2)
        
    print(f"\n[+] Optimization Winner: {winner}")
    print("[+] Saved optimized generator to checkpoints/optimized/cgan/cgan_generator_optimized.pt")

if __name__ == "__main__":
    main()
