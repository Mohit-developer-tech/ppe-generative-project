"""Deploy optimized policy weights into artifacts/policy/policy_classifier.pt"""
import os
import json
import torch

os.makedirs("artifacts/policy", exist_ok=True)
sd = torch.load("checkpoints/optimized/policy/policy_classifier.pt", map_location="cpu")
with open("results/experiments/policy_optimization_report.json", "r") as f:
    rep = json.load(f)

t = float(rep.get("optimal_temperature", 0.598))
bundle = {
    "state_dict": sd,
    "temperature": t,
    "metrics": rep
}
torch.save(bundle, "artifacts/policy/policy_classifier.pt")
print(f"[+] Successfully deployed bundled policy checkpoint with T*={t:.4f}")

