#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample a subset from MPIIGaze Data/Original into host/data/raw/mpiigaze_subset.

- Keeps relative folder structure (pXX/dayXX/...)
- Writes a manifest CSV for traceability
- Default: sample 10,000 images from p00..p04 (5 participants) evenly
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def guess_participant_and_session(rel: Path) -> Tuple[str, str]:
    """
    rel is path relative to src_root, typically: p00/day01/xxx.png or p00/xxx.png
    """
    parts = rel.parts
    participant = parts[0] if len(parts) >= 1 else "unknown"
    # session/day folder if exists
    session = parts[1] if len(parts) >= 2 and parts[1].lower().startswith(("day", "session")) else "nosession"
    if session == "nosession" and len(parts) >= 2 and parts[1].lower().startswith("d"):
        # some datasets use d01, d02...
        session = parts[1]
    return participant, session


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MPIIGaze Data/Original root, e.g. D:/.../MPIIGaze/Data/Original")
    ap.add_argument("--dst", default="host/data/raw/mpiigaze_subset", help="Destination subset directory (inside repo)")
    ap.add_argument("--n", type=int, default=10000, help="Total number of images to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--participants", default="p00,p01,p02,p03,p04",
                    help="Comma-separated participants. Default: p00..p04")
    args = ap.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.exists():
        raise SystemExit(f"[ERR] src not found: {src_root}")

    participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    for p in participants:
        if not (src_root / p).exists():
            raise SystemExit(f"[ERR] participant folder not found: {src_root / p}")

    random.seed(args.seed)

    # Collect image lists per participant
    per_part_imgs: Dict[str, List[Path]] = {}
    total_available = 0
    for p in participants:
        imgs = iter_images(src_root / p)
        # deterministic shuffle per participant
        rnd = random.Random(args.seed + hash(p) % 100000)
        rnd.shuffle(imgs)
        per_part_imgs[p] = imgs
        total_available += len(imgs)

    if total_available == 0:
        raise SystemExit("[ERR] no images found under selected participants")

    target_n = min(args.n, total_available)
    per_target = int(math.ceil(target_n / len(participants)))

    selected: List[Path] = []
    # First pass: take per_target each
    for p in participants:
        selected.extend(per_part_imgs[p][:per_target])

    # If too many, trim; if too few (unlikely), fill from remaining
    if len(selected) > target_n:
        selected = selected[:target_n]
    elif len(selected) < target_n:
        # fill from remaining pools
        remain = target_n - len(selected)
        pools = []
        for p in participants:
            pools.extend(per_part_imgs[p][per_target:])
        random.shuffle(pools)
        selected.extend(pools[:remain])

    # Copy & write manifest
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_root / "manifest.csv"
    copied = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset_relpath", "participant", "session", "src_abs"])

        for src_file in selected:
            rel = src_file.relative_to(src_root)
            participant, session = guess_participant_and_session(rel)
            dst_file = dst_root / rel
            safe_copy(src_file, dst_file)

            w.writerow([str(rel).replace("\\", "/"), participant, session, str(src_file)])
            copied += 1

    print("[OK] subset ready")
    print("  src:", src_root)
    print("  dst:", dst_root)
    print("  copied:", copied)
    print("  manifest:", manifest_path)


if __name__ == "__main__":
    main()