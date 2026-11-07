"""
scripts/fetch_assets.py
Retrieves shared project assets into artifacts/ from:
  1. Local folder / Drive dump: --from-folder <path>
  2. GitHub CLI (private release): --from-gh [--tag <tag>] [--repo <repo>]
  3. Public URL / CDN: --from-url <base_url>

Verifies SHA256 checksums of every downloaded asset against assets_manifest.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MANIFEST_FILE = "assets_manifest.json"

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_and_place(src_file: Path, target_path_str: str, expected_sha: str, expected_size: int) -> bool:
    if not src_file.exists():
        print(f"[ERROR] Source file not found: {src_file}", file=sys.stderr)
        return False

    actual_size = src_file.stat().st_size
    actual_sha = sha256_file(src_file)

    if actual_sha != expected_sha:
        print(
            f"[ERROR] Checksum mismatch for {src_file.name}!\n"
            f"  Expected SHA256: {expected_sha}\n"
            f"  Actual SHA256:   {actual_sha}\n"
            f"  Expected Size:   {expected_size} bytes\n"
            f"  Actual Size:     {actual_size} bytes",
            file=sys.stderr
        )
        return False

    target = Path(target_path_str)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, target)
    print(f"  [OK] Verified & placed -> {target} ({actual_size:,} bytes)")
    return True

def fetch_from_folder(folder_path: str, manifest: dict) -> bool:
    src_dir = Path(folder_path)
    if not src_dir.exists():
        print(f"[ERROR] Folder does not exist: {src_dir}", file=sys.stderr)
        return False

    print(f"[*] Importing assets from local folder: {src_dir}")
    all_ok = True
    for rel_key, meta in manifest["assets"].items():
        # Try both direct relative path or flattened name
        cand1 = src_dir / rel_key
        cand2 = src_dir / Path(rel_key).name
        chosen = cand1 if cand1.exists() else cand2

        if not chosen.exists():
            print(f"[ERROR] Required asset '{rel_key}' not found in {src_dir}", file=sys.stderr)
            all_ok = False
            continue

        ok = verify_and_place(chosen, meta["target_path"], meta["sha256"], meta["size_bytes"])
        if not ok:
            all_ok = False

    return all_ok

def fetch_from_gh(tag: str, repo: str, manifest: dict) -> bool:
    print(f"[*] Fetching assets from GitHub Release '{tag}' via GitHub CLI (gh)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "gh", "release", "download", tag,
            "--repo", repo,
            "--dir", tmpdir
        ]
        try:
            print(f"  Running: {' '.join(cmd)}")
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Download successful:\n{res.stdout}")
        except FileNotFoundError:
            print("[ERROR] GitHub CLI ('gh') is not installed or not in PATH.", file=sys.stderr)
            print("  Please install GitHub CLI or use '--from-folder <path>'.", file=sys.stderr)
            return False
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 'gh release download' failed with exit code {e.returncode}:\n{e.stderr}", file=sys.stderr)
            return False

        return fetch_from_folder(tmpdir, manifest)

def fetch_from_url(base_url: str, manifest: dict) -> bool:
    base_url = base_url.rstrip("/")
    print(f"[*] Downloading assets from URL: {base_url} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        all_ok = True
        for rel_key, meta in manifest["assets"].items():
            url = f"{base_url}/{rel_key}"
            local_target = Path(tmpdir) / rel_key
            local_target.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, str(local_target))
            except Exception as e:
                print(f"[ERROR] Failed to download {url}: {e}", file=sys.stderr)
                all_ok = False
                continue

            ok = verify_and_place(local_target, meta["target_path"], meta["sha256"], meta["size_bytes"])
            if not ok:
                all_ok = False

        return all_ok

def main():
    parser = argparse.ArgumentParser(description="Fetch and verify PPE project shared assets.")
    parser.add_argument("--from-folder", type=str, help="Import assets from a local folder or Drive dump.")
    parser.add_argument("--from-gh", action="store_true", help="Download assets using GitHub CLI ('gh release download').")
    parser.add_argument("--tag", type=str, default="v0.1", help="GitHub Release tag (default: v0.1).")
    parser.add_argument("--repo", type=str, default="Mohit-developer-tech/ppe-generative-project", help="GitHub repo.")
    parser.add_argument("--from-url", type=str, help="Download assets from a public URL.")
    parser.add_argument("--manifest", type=str, default=MANIFEST_FILE, help="Path to assets_manifest.json.")

    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest file '{args.manifest}' not found. Cannot verify assets.", file=sys.stderr)
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    success = False
    if args.from_folder:
        success = fetch_from_folder(args.from_folder, manifest)
    elif args.from_gh:
        success = fetch_from_gh(args.tag, args.repo, manifest)
    elif args.from_url:
        success = fetch_from_url(args.from_url, manifest)
    else:
        # Default convenience fallback: check if release_assets/ exists locally
        if os.path.exists("release_assets"):
            print("[*] No flag specified. Using existing 'release_assets/' folder...")
            success = fetch_from_folder("release_assets", manifest)
        else:
            parser.print_help()
            sys.exit(1)

    if success:
        print("\n[SUCCESS] All shared model assets successfully fetched, verified, and placed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] One or more assets failed download or checksum verification.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

