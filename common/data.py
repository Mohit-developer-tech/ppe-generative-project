"""
common/data.py
Shared data loading foundation for generative PPE models.
Provides ImageFolder loaders with [0,1] and [-1,1] normalization,
class-balanced sampling, fixed seeds, and dataset verification against stats.csv.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

# Strictly fixed class names in ImageFolder alphabetical order
CLASS_NAMES: List[str] = ["head", "helmet", "vest"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def set_seed(seed: int = 42) -> None:
    """Set global random seeds across random, numpy, and torch for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(
    mode: str = "[0,1]",
    augment: bool = False,
    img_size: int = 64
) -> transforms.Compose:
    """
    Build transform pipeline.
    
    Args:
        mode: Pixel intensity range, either "[0,1]" (standard ToTensor) or
              "[-1,1]" (ToTensor + Normalize((0.5,), (0.5,)) for GANs/Diffusion).
        augment: If True, applies RandomHorizontalFlip(p=0.5).
        img_size: Target square image dimension (default 64).
    """
    transform_list: List[transforms.Transform] = [
        transforms.Resize((img_size, img_size))
    ]
    if augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.append(transforms.ToTensor())

    if mode == "[-1,1]":
        transform_list.append(
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        )
    elif mode != "[0,1]":
        raise ValueError(f"Invalid mode '{mode}'. Expected '[0,1]' or '[-1,1]'.")

    return transforms.Compose(transform_list)


def denormalize_tensor(tensor: torch.Tensor, mode: str = "[-1,1]") -> torch.Tensor:
    """Convert tensor from [-1, 1] range back to [0, 1] for metric computation and visualization."""
    if mode == "[-1,1]":
        return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
    return torch.clamp(tensor, 0.0, 1.0)


def get_dataset(
    split: str = "train",
    data_dir: Union[str, Path] = "crops_64",
    mode: str = "[0,1]",
    augment: bool = False,
    img_size: int = 64
) -> datasets.ImageFolder:
    """
    Instantiate ImageFolder dataset for a specific split.
    
    Args:
        split: One of 'train', 'val', 'test'.
        data_dir: Path to directory containing train, val, test subfolders.
        mode: Pixel normalization range '[0,1]' or '[-1,1]'.
        augment: Whether to apply horizontal flip augmentation.
        img_size: Square image resolution (default 64).
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    transform = get_transforms(mode=mode, augment=augment, img_size=img_size)
    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)

    # Enforce strict class ordering
    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Dataset class names {dataset.classes} do not match expected ImageFolder order {CLASS_NAMES}"
        )
    return dataset


def get_balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Create a class-balanced WeightedRandomSampler to handle extreme PPE class imbalance.
    Head (n=867) vs Helmet (n=15991) vs Vest (n=10041).
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    
    # Weight per class is inversely proportional to class frequency
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    # Weight per sample
    sample_weights = class_weights[targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloader(
    split: str = "train",
    data_dir: Union[str, Path] = "crops_64",
    mode: str = "[0,1]",
    batch_size: int = 128,
    balanced: bool = False,
    augment: Optional[bool] = None,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    High-level factory function returning a configured DataLoader.
    
    Args:
        split: 'train', 'val', or 'test'.
        data_dir: Dataset root directory.
        mode: '[0,1]' or '[-1,1]'.
        batch_size: Samples per batch.
        balanced: If True, uses WeightedRandomSampler (train split only).
        augment: If None, defaults to True for train split and False otherwise.
        num_workers: Data loading workers (0 recommended on Windows).
        seed: Seed for reproducible loading.
        shuffle: If None, defaults to True for train split without sampler.
    """
    set_seed(seed)
    if augment is None:
        augment = (split == "train")

    dataset = get_dataset(
        split=split,
        data_dir=data_dir,
        mode=mode,
        augment=augment
    )

    sampler = None
    should_shuffle = False

    if split == "train":
        if balanced:
            sampler = get_balanced_sampler(dataset)
            should_shuffle = False
        else:
            should_shuffle = True if shuffle is None else shuffle
    else:
        should_shuffle = False if shuffle is None else shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def verify_dataset_counts(
    data_dir: Union[str, Path] = "crops_64",
    stats_csv: Union[str, Path] = "crops_64/stats.csv"
) -> pd.DataFrame:
    """
    Scan disk directories and verify actual crop counts against crops_64/stats.csv.
    Raises AssertionError if any discrepancy is detected.
    """
    data_path = Path(data_dir)
    stats_path = Path(stats_csv)

    if not stats_path.exists():
        raise FileNotFoundError(f"stats.csv not found at: {stats_path}")

    df_expected = pd.read_csv(stats_path)
    df_expected.set_index("split", inplace=True)

    records = []
    splits = ["train", "val", "test"]

    for split in splits:
        split_path = data_path / split
        actual_counts = {}
        actual_total = 0

        for cls_name in CLASS_NAMES:
            cls_path = split_path / cls_name
            if cls_path.exists():
                count = len(list(cls_path.glob("*.png")))
            else:
                count = 0
            actual_counts[cls_name] = count
            actual_total += count

        exp_row = df_expected.loc[split]
        for cls_name in CLASS_NAMES:
            exp_val = int(exp_row[cls_name])
            act_val = actual_counts[cls_name]
            match = (act_val == exp_val)
            records.append({
                "split": split,
                "class": cls_name,
                "actual_count": act_val,
                "expected_count": exp_val,
                "matches": match
            })
            assert match, (
                f"Count mismatch in split '{split}' class '{cls_name}': "
                f"actual={act_val} vs expected={exp_val}"
            )

        assert actual_total == int(exp_row["total"]), (
            f"Total mismatch in split '{split}': "
            f"actual={actual_total} vs expected={int(exp_row['total'])}"
        )

    df_result = pd.DataFrame(records)
    print("=" * 65)
    print("DATASET INTEGRITY AUDIT: crops_64/ matches stats.csv exactly!")
    print("=" * 65)
    for split in splits:
        split_df = df_result[df_result["split"] == split]
        parts = [f"{r['class']}={r['actual_count']}" for _, r in split_df.iterrows()]
        print(f"  {split.capitalize():<5}: {', '.join(parts)} (Total={df_expected.loc[split, 'total']}) [OK]")
    print("=" * 65)
    return df_result

