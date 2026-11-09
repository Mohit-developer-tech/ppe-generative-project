"""
scripts/cache_dataset.py
Pre-caches crops_64 into fast binary .pt tensor files (uint8) in data_cache/.
Eliminates 100% of filesystem open/close overhead for rapid multi-seed training.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch

CLASS_NAMES = ["head", "helmet", "vest"]

def _load_single_image(args):
    fpath, cls_idx = args
    with Image.open(fpath) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr, cls_idx

def cache_split(split_name: str, data_dir: str = "crops_64", out_dir: str = "data_cache", num_workers: int = 16):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{split_name}_crops.pt")
    if os.path.exists(out_file):
        print(f"[*] Already cached: {out_file}")
        return

    print(f"[*] Caching {split_name} split from {data_dir}/{split_name}...", flush=True)
    t0 = time.time()
    file_list = []

    split_dir = os.path.join(data_dir, split_name)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))
        for fname in files:
            if fname.lower().endswith(".png"):
                file_list.append((os.path.join(cls_dir, fname), cls_idx))

    print(f"    Found {len(file_list)} images, loading in parallel with {num_workers} threads...", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_load_single_image, file_list, chunksize=64))

    images = [r[0] for r in results]
    labels = [r[1] for r in results]

    # Convert to contiguous uint8 tensors [N, 3, 64, 64]
    X = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(labels, dtype=torch.long)

    torch.save({"X": X, "y": y}, out_file)
    dur = time.time() - t0
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"    [+] Saved {len(X)} crops to {out_file} in {dur:.1f}s ({size_mb:.1f} MB)", flush=True)

if __name__ == "__main__":
    os.makedirs("data_cache", exist_ok=True)
    cache_split("train")
    cache_split("val")
    cache_split("test")
    print("[*] All splits cached successfully!")

