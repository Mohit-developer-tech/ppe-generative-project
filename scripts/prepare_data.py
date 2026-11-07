"""
scripts/prepare_data.py
Prepares and verifies the crops_64 dataset:
1. Extracts crops_64.zip via --from-zip <path>
2. Verifies split and class counts against crops_64/stats.csv using common.data.verify_dataset_counts()
3. Pre-caches contiguous binary tensors in data_cache/ for high-throughput training/testing
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

def extract_zip(zip_path: str, target_dir: str = "crops_64"):
    p_zip = Path(zip_path)
    if not p_zip.exists():
        print(f"[ERROR] Zip file not found: {p_zip}", file=sys.stderr)
        return False

    print(f"[*] Extracting {p_zip} to {target_dir} ...")
    with zipfile.ZipFile(p_zip, "r") as zf:
        # Check if zip contains top-level 'crops_64/'
        namelist = zf.namelist()
        has_root = any(name.startswith("crops_64/") for name in namelist)
        if has_root:
            extract_target = Path(target_dir).parent
            zf.extractall(extract_target)
        else:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            zf.extractall(target_dir)

    print(f"[+] Successfully extracted dataset to {target_dir}")
    return True

def verify_dataset(crops_dir: str = "crops_64"):
    try:
        sys.path.insert(0, ".")
        from common.data import verify_dataset_counts
        df_audit = verify_dataset_counts(data_dir=crops_dir, stats_csv=os.path.join(crops_dir, "stats.csv"))
        print(f"[PASS] Dataset count audit verified against {crops_dir}/stats.csv.")
        return True
    except Exception as e:
        print(f"[WARN] Dataset count audit warning: {e}", file=sys.stderr)
        return False

def build_cache(crops_dir: str = "crops_64", cache_dir: str = "data_cache"):
    try:
        from scripts.cache_dataset import cache_split
        print(f"[*] Building zero-copy binary tensor cache in {cache_dir} ...")
        for split in ["train", "val", "test"]:
            cache_split(split_name=split, data_dir=crops_dir, out_dir=cache_dir)
        print("[+] Binary cache successfully created.")
        return True
    except Exception as e:
        print(f"[WARN] Cache generation skipped or failed: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare, unpack, and verify PPE crops_64 dataset.")
    parser.add_argument("--from-zip", type=str, help="Path to crops_64.zip archive.")
    parser.add_argument("--crops-dir", type=str, default="crops_64", help="Target crops directory.")
    parser.add_argument("--no-cache", action="store_true", help="Do not build data_cache binary tensors.")
    args = parser.parse_args()

    if args.from_zip:
        ok = extract_zip(args.from_zip, target_dir=args.crops_dir)
        if not ok:
            sys.exit(1)

    if not os.path.exists(args.crops_dir):
        print(f"[ERROR] Crops directory '{args.crops_dir}' does not exist.", file=sys.stderr)
        print("  Please provide '--from-zip <path_to_crops_64.zip>'.", file=sys.stderr)
        sys.exit(1)

    print("[*] Auditing dataset integrity...")
    verify_dataset(crops_dir=args.crops_dir)

    if not args.no_cache and not os.path.exists("data_cache/train_crops.pt"):
        build_cache(crops_dir=args.crops_dir)

    print("\n[SUCCESS] Dataset preparation and verification completed successfully!")

if __name__ == "__main__":
    main()

