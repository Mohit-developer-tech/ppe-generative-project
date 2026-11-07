"""
extract_ppe_crops.py
Download the merged PPE dataset (Hugging Face: jhboyo/ppe-dataset) and extract
class-wise square crops from YOLO labels, ready for AE / VAE / GAN / DDPM / VQ-VAE+GPT.

Install:
    pip install huggingface_hub pillow tqdm matplotlib

Usage:
    # download + extract 64x64 crops
    python extract_ppe_crops.py --size 64 --out crops_64

    # use an already downloaded dataset folder (has train/ val/ test/)
    python extract_ppe_crops.py --root path/to/ppe-dataset --size 128 --out crops_128

Output:
    crops_64/<split>/<class_name>/<image>_<idx>.png
    crops_64/stats.csv          (crop counts per split and class)
    crops_64/class_distribution.png   (graph for the paper's dataset section)
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

CLASS_NAMES = {0: "helmet", 1: "head", 2: "vest"}  # head = no helmet (violation)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
REPO_ID = "jhboyo/ppe-dataset"


def download_dataset(local_dir):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=local_dir))


def square_box(cx, cy, w, h, pad, W, H):
    """Square box around the object, expanded by `pad`, shifted to stay inside the image."""
    s = max(w, h) * (1 + pad)
    s = min(s, W, H)
    x0 = min(max(cx - s / 2, 0), W - s)
    y0 = min(max(cy - s / 2, 0), H - s)
    return int(round(x0)), int(round(y0)), int(round(x0 + s)), int(round(y0 + s))


def process_split(root, split, out, size, min_px, pad, keep):
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    counts, skipped = Counter(), Counter()
    if not img_dir.exists():
        print(f"[warn] {img_dir} not found, skipping")
        return counts, skipped

    for img_path in tqdm(sorted(img_dir.iterdir()), desc=split):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped["bad_image"] += 1
            continue
        W, H = img.size

        for i, line in enumerate(lbl_path.read_text().strip().splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls not in CLASS_NAMES or CLASS_NAMES[cls] not in keep:
                continue
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            cx, cy, w, h = xc * W, yc * H, bw * W, bh * H
            if min(w, h) < min_px:          # drop tiny, noisy objects
                skipped["too_small"] += 1
                continue
            box = square_box(cx, cy, w, h, pad, W, H)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                skipped["degenerate"] += 1
                continue
            crop = img.crop(box).resize((size, size), Image.LANCZOS)
            name = CLASS_NAMES[cls]
            dest = out / split / name
            dest.mkdir(parents=True, exist_ok=True)
            crop.save(dest / f"{img_path.stem}_{i}.png")
            counts[name] += 1
    return counts, skipped


def save_stats(out, all_counts):
    classes = sorted({c for cnt in all_counts.values() for c in cnt})
    with open(out / "stats.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split"] + classes + ["total"])
        for split, cnt in all_counts.items():
            wr.writerow([split] + [cnt[c] for c in classes] + [sum(cnt.values())])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.8 / max(len(all_counts), 1)
        for k, (split, cnt) in enumerate(all_counts.items()):
            xs = [j + k * width for j in range(len(classes))]
            ax.bar(xs, [cnt[c] for c in classes], width, label=split)
        ax.set_xticks([j + width * (len(all_counts) - 1) / 2 for j in range(len(classes))])
        ax.set_xticklabels(classes)
        ax.set_ylabel("Number of crops")
        ax.set_title("Class distribution of extracted crops")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "class_distribution.png", dpi=300)
    except ImportError:
        print("[info] matplotlib not installed, skipped the distribution graph")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="existing dataset folder; if omitted it is downloaded")
    ap.add_argument("--download-dir", default="ppe-dataset")
    ap.add_argument("--out", default="crops_64")
    ap.add_argument("--size", type=int, default=64, help="output crop size (64 or 128)")
    ap.add_argument("--min-px", type=int, default=32, help="minimum box side in original pixels")
    ap.add_argument("--pad", type=float, default=0.15, help="context padding around the box")
    ap.add_argument("--classes", nargs="+", default=list(CLASS_NAMES.values()))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    root = Path(args.root) if args.root else download_dataset(args.download_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_counts = {}
    for split in args.splits:
        counts, skipped = process_split(root, split, out, args.size, args.min_px, args.pad, set(args.classes))
        all_counts[split] = counts
        print(f"{split}: {dict(counts)} | skipped: {dict(skipped)}")
    save_stats(out, all_counts)
    print(f"Done. Crops saved in {out}/")


if __name__ == "__main__":
    main()