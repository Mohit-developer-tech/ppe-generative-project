r"""
scripts/scan_repo_privacy.py
Scans repository for:
1. Hardcoded absolute personal paths (C:\Users, C:\Desktop, mrpat)
2. Potential secrets / tokens / API keys
3. Image inventory with face presence flags
4. Unresolved placeholders (<...>, TODO_)
"""
import os
import re
import subprocess
from pathlib import Path

# Get all tracked files
cmd = ["git", "ls-files"]
res = subprocess.run(cmd, capture_output=True, text=True, check=True)
tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]

print(f"Total tracked files: {len(tracked_files)}")

# 1. Personal / Absolute path patterns
path_patterns = [
    re.compile(r"C:[\\/]Users", re.IGNORECASE),
    re.compile(r"C:[\\/]Desktop", re.IGNORECASE),
    re.compile(r"mrpat", re.IGNORECASE),
]

path_hits = {}
placeholder_hits = {}
secret_hits = {}
tracked_images = []

secret_patterns = [
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|bearer\s+[a-zA-Z0-9_\-\.]{20,})\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{15,}['\"]"),
    re.compile(r"(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{82})"),
    re.compile(r"(sk-[a-zA-Z0-9]{32,})"),
]

placeholder_patterns = [
    re.compile(r"TODO_[A-Z0-9_]+"),
    re.compile(r"<[A-Za-z0-9_\-]+>"),
]

FACE_IMAGE_HINTS = [
    "sample", "reconstruction", "morph", "grid", "montage", "strip",
    "true_head", "true_helmet", "true_vest", "walk", "interpolation"
]

for fpath in tracked_files:
    p = Path(fpath)
    ext = p.suffix.lower()

    # Check images
    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        size = p.stat().st_size if p.exists() else 0
        name_lower = p.name.lower()
        parent_lower = str(p.parent).lower()

        # Flag if likely containing human faces
        is_face = any(h in name_lower or h in parent_lower for h in FACE_IMAGE_HINTS)
        # Check if it's a pure curve / chart vs crop montage
        if any(c in name_lower for c in ["loss", "curve", "confusion", "tradeoff", "flips", "steps", "benchmark", "accuracy", "calibration", "slices", "matrix", "kl_per", "distribution", "history"]):
            is_face = False

        tracked_images.append({
            "path": fpath,
            "size": size,
            "flagged_face": is_face
        })
        continue

    # Text scan
    if not p.exists():
        continue

    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    # Check personal paths
    for pat in path_patterns:
        matches = pat.findall(content)
        if matches:
            if fpath not in path_hits:
                path_hits[fpath] = []
            path_hits[fpath].append(matches[0])

    # Check secrets
    for pat in secret_patterns:
        m = pat.findall(content)
        if m:
            secret_hits[fpath] = m

    # Check placeholders
    for pat in placeholder_patterns:
        m = pat.findall(content)
        if m:
            placeholder_hits[fpath] = m

print("\n=== 1. PERSONAL / ABSOLUTE PATH HITS ===")
for f, hits in path_hits.items():
    print(f"  {f}: {hits}")

print("\n=== 2. SECRETS SCAN ===")
if not secret_hits:
    print("  [CLEAN] No API keys, passwords, or tokens detected.")
else:
    for f, hits in secret_hits.items():
        print(f"  {f}: {hits}")

print("\n=== 3. UNRESOLVED PLACEHOLDERS ===")
for f, hits in placeholder_hits.items():
    print(f"  {f}: {hits}")

print(f"\n=== 4. TRACKED IMAGES ({len(tracked_images)} total) ===")
face_count = sum(1 for img in tracked_images if img["flagged_face"])
print(f"  Images flagged as potentially depicting human faces: {face_count}")
for img in tracked_images:
    flag_str = "[FLAGGED FACE]" if img["flagged_face"] else "[CHART/PLOT]"
    print(f"  {flag_str:14} {img['size']:>8} bytes  {img['path']}")
