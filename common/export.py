"""
common/export.py
Standardized model artifact packaging and layout manager for all PPE generative models.
Ensures every model exports into results/<model_name>/ with consistent naming conventions,
configuration dictionaries, class names, metrics tables, and figure assets.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from common.data import CLASS_NAMES


def get_model_result_dir(model_name: str, base_results_dir: str = "results") -> Path:
    """Return Path to results/<model_name>/, creating the directory if absent."""
    p = Path(base_results_dir) / model_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_model_bundle(
    model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[torch.nn.Module] = None,
    weights_state_dict: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    metrics_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    figures_to_copy: Optional[Dict[str, str]] = None,
    extra_files: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Export a complete standardized model bundle to results/<model_name>/.
    
    Returns:
        Dictionary mapping artifact type to exported file path.
    """
    if model_config is None and "config" in kwargs:
        model_config = kwargs["config"]
    if metrics_summary is None and "metrics_dict" in kwargs:
        metrics_summary = kwargs["metrics_dict"]
    if extra_files is None and "extra_files" in kwargs:
        extra_files = kwargs["extra_files"]
    if output_dir is None:
        out_path = get_model_result_dir(model_name)
    else:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    exported_paths = {}

    # 1. Weights
    weights_path = out_path / f"{model_name}_best.pt"
    if weights_state_dict is not None:
        torch.save(weights_state_dict, weights_path)
        exported_paths["weights"] = str(weights_path)
    elif model is not None:
        torch.save(model.state_dict(), weights_path)
        exported_paths["weights"] = str(weights_path)

    # 2. Config
    if model_config is not None:
        cfg_path = out_path / "model_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)
        exported_paths["config"] = str(cfg_path)

    # 3. Class names
    names = class_names or CLASS_NAMES
    classes_path = out_path / "class_names.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({str(i): name for i, name in enumerate(names)}, f, indent=2)
    exported_paths["class_names"] = str(classes_path)

    # 4. Metrics Summary JSON
    if metrics_summary is not None:
        metrics_json_path = out_path / f"{model_name}_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        exported_paths["metrics_json"] = str(metrics_json_path)

    # 5. DataFrames (CSVs)
    if metrics_dfs is not None:
        for fname, df in metrics_dfs.items():
            if not fname.endswith(".csv"):
                fname += ".csv"
            csv_path = out_path / fname
            df.to_csv(csv_path, index=False)
            exported_paths[f"csv_{fname}"] = str(csv_path)

    # 6. Copy Figures
    if figures_to_copy is not None:
        for target_name, src_path in figures_to_copy.items():
            if os.path.exists(src_path):
                dst_path = out_path / target_name
                shutil.copy2(src_path, dst_path)
                exported_paths[f"fig_{target_name}"] = str(dst_path)

    print(f"[*] Successfully exported model bundle '{model_name}' -> {out_path}")
    return exported_paths


def load_model_bundle(
    model_name: str,
    base_results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Read model bundle configuration, metrics, and figure inventory from results/<model_name>/.
    Returns a structured dictionary without hardcoded values.
    """
    dir_path = Path(base_results_dir) / model_name
    if not dir_path.exists():
        return {
            "status": "not_found",
            "model_name": model_name,
            "message": f"Results directory '{dir_path}' does not exist."
        }

    bundle: Dict[str, Any] = {
        "status": "ready",
        "model_name": model_name,
        "dir": str(dir_path),
        "weights_path": None,
        "config": None,
        "class_names": CLASS_NAMES,
        "metrics_summary": None,
        "metrics_tables": {},
        "figures": []
    }

    # Find weights
    pt_files = list(dir_path.glob("*.pt"))
    if pt_files:
        bundle["weights_path"] = str(pt_files[0])

    # Find config
    cfg_file = dir_path / "model_config.json"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            bundle["config"] = json.load(f)

    # Find class names
    classes_file = dir_path / "class_names.json"
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            bundle["class_names"] = json.load(f)

    # Find metrics JSON
    for mf in dir_path.glob("*metrics*.json"):
        with open(mf, "r", encoding="utf-8") as f:
            bundle["metrics_summary"] = json.load(f)
            break

    # Find CSVs
    for cf in dir_path.glob("*.csv"):
        try:
            df = pd.read_csv(cf)
            bundle["metrics_tables"][cf.stem] = df.to_dict(orient="records")
        except Exception:
            pass

    # Find figures
    for fig in dir_path.glob("*.png"):
        bundle["figures"].append(fig.name)

    return bundle

