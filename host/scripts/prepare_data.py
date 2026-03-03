#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# -------------------------
# utils
# -------------------------
def load_yaml(path: Path) -> dict:
    """Prefer ruamel.yaml, fallback to PyYAML."""
    try:
        from ruamel.yaml import YAML  # type: ignore
        y = YAML(typ="safe")
        with path.open("r", encoding="utf-8") as f:
            return y.load(f)
    except Exception:
        try:
            import yaml  # type: ignore
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load YAML: {path}. Install ruamel.yaml or pyyaml."
            ) from e


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_path(base: Path, p: str) -> Path:
    """Resolve a possibly-relative path against repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def iter_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))


def resize_to(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = max(3, int(6 * sigma) | 1)  # odd
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def jpeg_reencode(img: np.ndarray, q: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def guess_session_id_from_raw_rel(raw_rel: Path) -> str:
    """
    MPIIGaze Data/Original layout: p00/day04/0842.jpg
    We use pXX/dayYY as session_id to avoid split leakage and filename collision.
    """
    parts = raw_rel.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return "unknown"


def session_path(session_id: str) -> Path:
    """Turn 'p00/day04' into Path('p00','day04') (cross-platform safe)."""
    return Path(*session_id.split("/"))


# -------------------------
# data record
# -------------------------
@dataclass
class Item:
    id: str
    relpath: str            # relative to processed_dir
    split: str              # train/val/test
    label_bin: int          # 0..4
    label_type: str         # synthetic / real
    session_id: str         # pXX/dayYY
    src_rel_raw: str        # relative to raw_dir
    sigma: float            # blur sigma used (synthetic only; -1 for real)
    baseline_lapvar: float
    baseline_tenengrad: float


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to dataset.yaml (e.g. host/src/configs/dataset.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: only process first N raw images (0=all)")
    args = ap.parse_args()

    # repo_root = .../host/scripts/prepare_data.py -> parents[2] = repo root
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else (repo_root / cfg_path)
    cfg = load_yaml(cfg_path)

    # seed
    seed = int(cfg.get("project", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # paths
    paths = cfg["paths"]
    raw_dir = resolve_path(repo_root, paths["raw_dir"])
    processed_dir = resolve_path(repo_root, paths["processed_dir"])
    manifests_dir = resolve_path(repo_root, paths["manifests_dir"])
    splits_dir = resolve_path(repo_root, paths["splits_dir"])
    calib_dir = resolve_path(repo_root, paths["calib_dir"])

    for p in [processed_dir, manifests_dir, splits_dir, calib_dir]:
        ensure_dir(p)

    # data spec
    data_spec = cfg["data_spec"]
    roi_mode = str(data_spec.get("roi_mode", "eye_roi"))
    image_size = tuple(data_spec["image_size"])  # (w,h)
    out_ext = str(data_spec.get("out_ext", ".jpg"))
    jpeg_q = int(data_spec.get("jpeg_quality", 90))

    # baselines
    baselines = cfg.get("baselines", {})
    use_lap = bool(baselines.get("laplacian_var", True))
    use_ten = bool(baselines.get("tenengrad", True))

    # augmentation
    aug_cfg = cfg["augmentation"]
    enable_syn = bool(aug_cfg.get("enable_synthetic", True))
    per_image = int(aug_cfg.get("per_image", 1))
    sigma_map: Dict[str, List[float]] = aug_cfg["gaussian_blur_sigma"]

    # label spec
    classes = cfg["label_spec"]["classes"]  # {"0":"very_blurry",...}
    id_to_name = {int(k): str(v) for k, v in classes.items()}
    class_ids = sorted(id_to_name.keys())
    if not class_ids:
        raise RuntimeError("label_spec.classes is empty")
    max_id = max(class_ids)

    # build bins (by label_spec order)
    label_bins: List[Tuple[str, Tuple[float, float], int]] = []
    for cid in class_ids:
        cname = id_to_name[cid]
        if cname not in sigma_map:
            raise RuntimeError(f"augmentation.gaussian_blur_sigma missing key: {cname}")
        rr = sigma_map[cname]
        if not isinstance(rr, (list, tuple)) or len(rr) != 2:
            raise RuntimeError(f"gaussian_blur_sigma[{cname}] must be [min,max], got: {rr}")
        s0, s1 = float(rr[0]), float(rr[1])
        label_bins.append((cname, (s0, s1), cid))

    # split spec
    split_spec = cfg["split_spec"]
    group_by = str(split_spec.get("group_by", "session_id"))
    if group_by != "session_id":
        raise RuntimeError("This pipeline expects split_spec.group_by == session_id")

    frac_train = float(split_spec.get("train", 0.8))
    frac_val = float(split_spec.get("val", 0.1))
    if frac_train < 0 or frac_val < 0 or frac_train + frac_val >= 1.0:
        raise RuntimeError(f"Invalid split ratios: train={frac_train}, val={frac_val} (train+val must be < 1)")

    overwrite_processed = bool(cfg.get("project", {}).get("overwrite_processed", True))

    # -------------------------
    # prints
    # -------------------------
    print("=== PREPARE_DATA START ===", flush=True)
    print("config:", cfg_path.resolve(), flush=True)
    print("raw_dir:", raw_dir.resolve(), flush=True)
    print("processed_dir:", processed_dir.resolve(), flush=True)
    print("manifests_dir:", manifests_dir.resolve(), flush=True)
    print("splits_dir:", splits_dir.resolve(), flush=True)
    print("calib_dir:", calib_dir.resolve(), flush=True)
    print("roi_mode:", roi_mode, "image_size:", image_size,
          "enable_syn:", enable_syn, "per_image:", per_image, flush=True)

    # -------------------------
    # 0) enumerate raw
    # -------------------------
    imgs = iter_images(raw_dir)
    if not imgs:
        raise RuntimeError(f"No images found under {raw_dir.resolve()}")

    imgs.sort()
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]
        print(f"[DEBUG] limit raw imgs to {len(imgs)}", flush=True)

    print("raw images:", len(imgs), flush=True)

    # -------------------------
    # 1) preprocess + synthetic
    # -------------------------
    out_items: List[Item] = []
    relpath_seen = set()

    processed_written = 0
    processed_seen = 0

    for idx, p in enumerate(imgs, start=1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # TODO: integrate eye crop when roi_mode == eye_roi
        img = resize_to(img, image_size)

        rel_in_raw = p.relative_to(raw_dir)
        rel_in_raw_str = str(rel_in_raw).replace("\\", "/")
        session_id = guess_session_id_from_raw_rel(rel_in_raw)

        if enable_syn:
            for cname, (s0, s1), label_id in label_bins:
                for k in range(per_image):
                    sigma = random.uniform(s0, s1)
                    x = gaussian_blur(img, sigma)
                    x = jpeg_reencode(x, jpeg_q)

                    # ✅ FIX: include session_id in output path to avoid collisions/overwrites
                    # also use sigma with 4 decimals to reduce accidental collisions
                    out_name = f"{p.stem}_s{sigma:.4f}_k{k}{out_ext}"
                    rel = Path("syn") / session_path(session_id) / cname / out_name
                    rel_str = str(rel).replace("\\", "/")

                    if rel_str in relpath_seen:
                        raise RuntimeError(f"RELATIVE PATH COLLISION detected: {rel_str}")
                    relpath_seen.add(rel_str)

                    out_path = processed_dir / rel
                    ensure_dir(out_path.parent)

                    if overwrite_processed or (not out_path.exists()):
                        cv2.imwrite(str(out_path), x, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                        processed_written += 1
                    processed_seen += 1

                    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                    lv = laplacian_var(gray) if use_lap else -1.0
                    tg = tenengrad(gray) if use_ten else -1.0

                    out_items.append(Item(
                        id=f"{session_id}-{cname}-{p.stem}-{k}-{random.randint(0, 10**9)}",
                        relpath=rel_str,
                        split="",
                        label_bin=label_id,
                        label_type="synthetic",
                        session_id=session_id,
                        src_rel_raw=rel_in_raw_str,
                        sigma=float(sigma),
                        baseline_lapvar=lv,
                        baseline_tenengrad=tg,
                    ))
        else:
            # real (no synthetic)
            out_name = f"{p.stem}{out_ext}"
            rel = Path("real") / session_path(session_id) / out_name
            rel_str = str(rel).replace("\\", "/")

            if rel_str in relpath_seen:
                raise RuntimeError(f"RELATIVE PATH COLLISION detected: {rel_str}")
            relpath_seen.add(rel_str)

            out_path = processed_dir / rel
            ensure_dir(out_path.parent)
            if overwrite_processed or (not out_path.exists()):
                cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                processed_written += 1
            processed_seen += 1

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lv = laplacian_var(gray) if use_lap else -1.0
            tg = tenengrad(gray) if use_ten else -1.0

            out_items.append(Item(
                id=f"{session_id}-real-{p.stem}-{random.randint(0, 10**9)}",
                relpath=rel_str,
                split="",
                label_bin=-1,
                label_type="real",
                session_id=session_id,
                src_rel_raw=rel_in_raw_str,
                sigma=-1.0,
                baseline_lapvar=lv,
                baseline_tenengrad=tg,
            ))

        if idx % 200 == 0:
            print(f"[{idx}/{len(imgs)}] items={len(out_items)} written={processed_written}", flush=True)

    print("processed items:", len(out_items), flush=True)
    print("processed images (seen):", processed_seen, "written:", processed_written, flush=True)

    if not out_items:
        raise RuntimeError("No output items produced.")

    # -------------------------
    # 2) group split by session_id
    # -------------------------
    sessions = sorted(list({it.session_id for it in out_items}))
    random.shuffle(sessions)

    n = len(sessions)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)

    train_s = set(sessions[:n_train])
    val_s = set(sessions[n_train:n_train + n_val])
    test_s = set(sessions[n_train + n_val:])

    for it in out_items:
        if it.session_id in train_s:
            it.split = "train"
        elif it.session_id in val_s:
            it.split = "val"
        else:
            it.split = "test"

    (splits_dir / "splits.json").write_text(json.dumps({
        "group_by": "session_id",
        "train_sessions": sorted(list(train_s)),
        "val_sessions": sorted(list(val_s)),
        "test_sessions": sorted(list(test_s)),
        "num_sessions": n,
        "num_train_sessions": len(train_s),
        "num_val_sessions": len(val_s),
        "num_test_sessions": len(test_s),
    }, indent=2), encoding="utf-8")

    # -------------------------
    # 3) export manifests
    # -------------------------
    by_split: Dict[str, List[Item]] = {"train": [], "val": [], "test": []}
    for it in out_items:
        by_split[it.split].append(it)

    denom = float(max_id) if max_id > 0 else 1.0  # for label_score in [0,1]

    def write_manifest(name: str, items: List[Item]) -> Path:
        out = manifests_dir / f"manifest_{name}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "relpath", "split",
                "label_bin", "label_score", "label_type",
                "session_id", "src_rel_raw", "sigma",
                "baseline_lapvar", "baseline_tenengrad"
            ])
            for it in items:
                label_score = (it.label_bin / denom) if it.label_bin >= 0 else -1.0
                w.writerow([
                    it.id, it.relpath, it.split,
                    it.label_bin, f"{label_score:.6f}", it.label_type,
                    it.session_id, it.src_rel_raw, f"{it.sigma:.6f}",
                    f"{it.baseline_lapvar:.6f}", f"{it.baseline_tenengrad:.6f}"
                ])
        return out

    m_train = write_manifest("train", by_split["train"])
    m_val = write_manifest("val", by_split["val"])
    m_test = write_manifest("test", by_split["test"])

    # -------------------------
    # 4) export calib list
    # -------------------------
    # Use RELPATH for portability; also write an abs version for convenience.
    calib_n = int(cfg.get("rknn", {}).get("calib_num", 300))

    train_rel = [it.relpath for it in by_split["train"]]
    # dedup while preserving order
    seen = set()
    train_rel_unique = []
    for r in train_rel:
        if r not in seen:
            seen.add(r)
            train_rel_unique.append(r)

    random.shuffle(train_rel_unique)
    calib_rel = train_rel_unique[:calib_n] if len(train_rel_unique) > calib_n else train_rel_unique

    calib_rel_path = calib_dir / "calib_list.txt"           # relative paths
    calib_abs_path = calib_dir / "calib_list_abs.txt"       # absolute paths (current OS)

    with calib_rel_path.open("w", encoding="utf-8") as f:
        for r in calib_rel:
            f.write(r + "\n")

    with calib_abs_path.open("w", encoding="utf-8") as f:
        for r in calib_rel:
            f.write(str((processed_dir / Path(r)).resolve()) + "\n")

    # -------------------------
    # 5) stats + sanity
    # -------------------------
    unique_relpaths = len({it.relpath for it in out_items})
    if unique_relpaths != len(out_items):
        # Should not happen with collision check, but keep as guard.
        raise RuntimeError(f"Manifest relpath not unique: unique={unique_relpaths} total={len(out_items)}")

    stats = {
        "raw_images": len(imgs),
        "processed_items": len(out_items),
        "unique_relpaths": unique_relpaths,
        "splits": {k: len(v) for k, v in by_split.items()},
        "num_sessions": len(sessions),
        "labels": {
            "min": min([it.label_bin for it in out_items if it.label_bin >= 0], default=None),
            "max": max([it.label_bin for it in out_items if it.label_bin >= 0], default=None),
        },
        "paths": {
            "raw_dir": str(raw_dir.resolve()),
            "processed_dir": str(processed_dir.resolve()),
            "manifests_dir": str(manifests_dir.resolve()),
            "splits_dir": str(splits_dir.resolve()),
            "calib_dir": str(calib_dir.resolve()),
        }
    }
    stats_path = manifests_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("DONE", flush=True)
    print(" - manifest_train:", m_train.resolve(), flush=True)
    print(" - manifest_val  :", m_val.resolve(), flush=True)
    print(" - manifest_test :", m_test.resolve(), flush=True)
    print(" - splits        :", (splits_dir / "splits.json").resolve(), flush=True)
    print(" - calib_rel     :", calib_rel_path.resolve(), flush=True)
    print(" - calib_abs     :", calib_abs_path.resolve(), flush=True)
    print(" - stats         :", stats_path.resolve(), flush=True)
    print("=== PREPARE_DATA END ===", flush=True)


if __name__ == "__main__":
    main()