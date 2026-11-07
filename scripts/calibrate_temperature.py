"""
scripts/calibrate_temperature.py
Implements post-hoc Temperature Scaling and Expected Calibration Error (ECE)
strictly calibrated on the VALIDATION split.

Temperature scaling learns a single positive scalar T:
    p_i = softmax(z_i / T)
optimizing cross-entropy loss with L-BFGS.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath("."))


class ModelWithTemperature(nn.Module):
    """
    A thin decorator around a trained classifier that applies temperature scaling
    to its logits at inference time.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature parameter T."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.
    probs: (N, C) predicted probabilities
    targets: (N,) ground truth class indices
    """
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (preds == targets).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(corrects[in_bin]))
            conf_in_bin = float(np.mean(confs[in_bin]))
            abs_diff = abs(acc_in_bin - conf_in_bin)
            ece += abs_diff * prop_in_bin
            bin_details.append({
                "bin": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": int(np.sum(in_bin)),
                "accuracy": acc_in_bin,
                "confidence": conf_in_bin,
                "abs_diff": abs_diff
            })

    return float(ece), {"ece": float(ece), "bins": bin_details}


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01
) -> float:
    """
    Tune temperature T > 0 on validation logits using L-BFGS optimizer.
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.CrossEntropyLoss()

    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return float(temperature.item())

