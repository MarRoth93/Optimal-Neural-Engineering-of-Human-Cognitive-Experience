#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute PixCorr for VDVAE & Versatile Diffusion images across ALL subjects.

Folder layout expected (auto-detected, tolerant to naming):
  {BASE_RESULTS}/vdvae/{subjXX}/{network}/{alpha_folder}/*.png
  {BASE_RESULTS}/versatile_diffusion/{subjXX}/{network}/{alpha_folder}/*.png

Also compares α=0 against:
  - Original images at: {BASE_DATA}/nsddata_stimuli/test_images/{idx}.png
  - VDVAE image-to-image at: {BASE_RESULTS}/vdvae/image_to_image/{idx}.png

Outputs (per subject):
  {OUT_DIR}/subjXX/per_image.json            # nested dict of lists (PixCorr per image)
  {OUT_DIR}/subjXX/summary.json              # means/stdevs per comparison
  {OUT_DIR}/subjXX/summary.csv

Usage example:
  python compute_pixcorr_all_subjects.py \
      --base-results /home/rothermm/brain-diffuser/results \
      --base-data /home/rothermm/brain-diffuser/data \
      --subjects 1 2 5 7 \
      --models vdvae versatile_diffusion \
      --networks emonet memnet \
      --img-ext .png .jpg .jpeg \
      --n 982
"""

import argparse
from pathlib import Path
import json
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from math import atanh, tanh
from scipy import stats

# -------- Alpha normalization (tolerant to your folder naming) --------
ALPHA_CANON = {
    -4: ["alpha_-4", "alpha-4", "neg4", "negative_theta_4", "minus4", "-4", "m4"],
    -2: ["alpha_-2", "alpha-2", "neg2", "negative_theta_2", "minus2", "-2", "m2"],
     0: ["alpha_0", "alpha0", "zero", "0"],
     2: ["alpha_2", "alpha+2", "pos2", "positive_theta_2", "plus2", "+2", "p2"],
     4: ["alpha_4", "alpha+4", "pos4", "positive_theta_4", "plus4", "+4", "p4"],
}

CANON_TO_KEYS = {-4: "alpha_-4", -2: "alpha_-2", 0: "alpha_0", 2: "alpha_+2", 4: "alpha_+4"}

def guess_alpha_folder_map(model_root: Path, subj: int, network: str, img_exts: Tuple[str, ...]) -> Dict[int, Path]:
    """
    For a given model/subj/network, find paths for each canonical alpha by scanning
    subfolders and matching known alias names.
    Returns {alpha_value: folder_path} for those found.
    """
    root = model_root / f"subj{subj:02d}" / network
    if not root.exists():
        return {}

    # Build alias → alpha dict
    alias_to_alpha = {}
    for a, aliases in ALPHA_CANON.items():
        for nm in aliases:
            alias_to_alpha[nm.lower()] = a

    found: Dict[int, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        key = child.name.lower()
        # try direct match
        if key in alias_to_alpha:
            found[alias_to_alpha[key]] = child
            continue
        # try contains any alias
        for alias, a in alias_to_alpha.items():
            if alias in key:
                found[a] = child
                break

    # sanity: require that folder actually contains images
    filtered = {}
    for a, p in found.items():
        has_any = any(p.glob(f"*{ext}") for ext in img_exts)
        if has_any:
            filtered[a] = p
    return filtered

# -------- Image utils --------
def load_image_float01(path: Path) -> Optional[np.ndarray]:
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr
    except Exception:
        return None

def pixcorr(a: np.ndarray, b: np.ndarray, on_constant: str = "nan") -> float:
    """
    Pearson correlation between two images flattened to 1D.
    on_constant: what to return if either image has zero variance:
      - "nan"  -> return np.nan (default; filtered later)
      - "zero" -> return 0.0
      - "one"  -> return 1.0 if both constant and equal, else 0.0
    """
    a1 = a.reshape(-1).astype(np.float64, copy=False)
    b1 = b.reshape(-1).astype(np.float64, copy=False)
    sa = a1.std()
    sb = b1.std()
    if sa == 0.0 or sb == 0.0:
        if on_constant == "zero":
            return 0.0
        if on_constant == "one":
            return 1.0 if sa == 0.0 and sb == 0.0 and np.allclose(a1, b1) else 0.0
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(a1, b1)[0, 1]
    return float(r)


def resize_to(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize numpy HxWxC image to size_hw (width,height) using PIL bilinear."""
    h, w = img.shape[:2]
    target_w, target_h = size_hw
    if (w, h) == (target_w, target_h):
        return img
    im = Image.fromarray((img * 255.0).astype(np.uint8), mode="RGB")
    im = im.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def fisher_z_stats(r_values: np.ndarray, alpha: float = 0.05):
    """
    Compute mean/CI on correlations via Fisher z, and a two-sided
    one-sample t-test that mean(z) = 0 (i.e., mean r differs from 0).
    Returns dict with mean_r, ci_low_r, ci_high_r, mean_z, ci_low_z, ci_high_z, t_stat, p_value, n.
    """
    r = np.asarray(r_values, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"n": 0, "mean_r": None, "ci_low_r": None, "ci_high_r": None,
                "mean_z": None, "ci_low_z": None, "ci_high_z": None,
                "t_stat": None, "p_value": None}

    # clip to (-1+eps, 1-eps) for numerical stability
    eps = 1e-12
    r = np.clip(r, -1 + eps, 1 - eps)
    z = np.arctanh(r)  # Fisher transform

    mean_z = float(z.mean())
    if n > 1:
        sd_z = float(z.std(ddof=1))
        se_z = sd_z / np.sqrt(n)
        # t-based CI on z
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        half = t_crit * se_z
        ci_low_z = mean_z - half
        ci_high_z = mean_z + half
        # one-sample t-test of mean(z) vs 0
        t_stat = mean_z / se_z if se_z > 0 else np.inf
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n-1))
    else:
        ci_low_z = ci_high_z = mean_z
        t_stat = None
        p_value = None

    # back-transform z CIs to r space
    mean_r = float(np.tanh(mean_z))
    ci_low_r = float(np.tanh(ci_low_z))
    ci_high_r = float(np.tanh(ci_high_z))

    return {
        "n": int(n),
        "mean_r": mean_r,
        "ci_low_r": ci_low_r,
        "ci_high_r": ci_high_r,
        "mean_z": float(mean_z),
        "ci_low_z": float(ci_low_z),
        "ci_high_z": float(ci_high_z),
        "t_stat": None if t_stat is None else float(t_stat),
        "p_value": p_value,
    }

# -------- Pairing helpers --------
def image_path(root: Path, idx: int, img_exts: Tuple[str, ...]) -> Optional[Path]:
    for ext in img_exts:
        p = root / f"{idx}{ext}"
        if p.exists():
            return p
    return None

# -------- Main computation per subject --------
def compute_for_subject(
    subj: int,
    base_results: Path,
    base_data: Path,
    models: Tuple[str, ...],
    networks: Tuple[str, ...],
    img_exts: Tuple[str, ...],
    n_images: int,
) -> Tuple[Dict, Dict]:
    """
    Returns:
      per_image: nested dict of lists with per-image PixCorr values
      summary:   nested dict of summary stats
    """

    # roots
    model_roots = {m: base_results / m for m in models}
    originals_root = base_data / "nsddata_stimuli" / "test_images"
    i2i_root = base_results / "vdvae" / "image_to_image"

    per_image = {}  # {comparison_key: [values]}
    summary = {}    # {comparison_key: {mean, std, n, min, max}}

    def add_series(key: str, vals: List[float]):
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float32)
        per_image[key] = arr.tolist()

        if arr.size == 0:
            summary[key] = {
                "n": 0, "mean": None, "std": None, "min": None, "max": None,
                "mean_r": None, "ci_low_r": None, "ci_high_r": None,
                "mean_z": None, "ci_low_z": None, "ci_high_z": None,
                "t_stat": None, "p_value": None
            }
            return

        # classic descriptive stats
        desc = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        # Fisher-z based stats & test
        fz = fisher_z_stats(arr)

        summary[key] = {**desc, **fz}


    # --- 1) Intra-pipeline: each alpha vs alpha=0 (for both models & networks) ---
    for model in models:
        for network in networks:
            alpha_map = guess_alpha_folder_map(model_roots[model], subj, network, img_exts)
            if 0 not in alpha_map:
                # no baseline, skip all intra comparisons for this (model,network)
                continue
            base_folder = alpha_map[0]
            # Preload baseline images cache to avoid repeated IO
            base_cache: Dict[int, np.ndarray] = {}
            for idx in range(n_images):
                p = image_path(base_folder, idx, img_exts)
                if p is None:
                    continue
                im = load_image_float01(p)
                if im is not None:
                    base_cache[idx] = im

            for alpha_val, alpha_folder in sorted(alpha_map.items()):
                if alpha_val == 0:
                    continue
                key = f"intra:{model}:{network}:{CANON_TO_KEYS.get(alpha_val, str(alpha_val))}_vs_alpha_0"
                vals: List[float] = []
                for idx in range(n_images):
                    p_a = image_path(alpha_folder, idx, img_exts)
                    if p_a is None or idx not in base_cache:
                        continue
                    im_a = load_image_float01(p_a)
                    if im_a is None:
                        continue
                    im0 = base_cache[idx]
                    # resize alpha image to baseline size (defensive)
                    im_a = resize_to(im_a, (im0.shape[1], im0.shape[0]))
                    vals.append(pixcorr(im_a, im0))
                add_series(key, vals)

    # --- 2) α=0 (VDVAE & Versatile) vs originals ---
    for model in models:
        for network in networks:
            alpha_map = guess_alpha_folder_map(model_roots[model], subj, network, img_exts)
            if 0 not in alpha_map:
                continue
            base_folder = alpha_map[0]  # alpha=0
            key = f"alpha0_vs_originals:{model}:{network}"
            vals: List[float] = []
            for idx in range(n_images):
                p_gen = image_path(base_folder, idx, img_exts)
                p_gt = image_path(originals_root, idx, img_exts)
                if p_gen is None or p_gt is None:
                    continue
                im_gen = load_image_float01(p_gen)
                im_gt = load_image_float01(p_gt)
                if im_gen is None or im_gt is None:
                    continue
                # match GT size
                im_gen = resize_to(im_gen, (im_gt.shape[1], im_gt.shape[0]))
                vals.append(pixcorr(im_gen, im_gt))
            add_series(key, vals)

    # --- 3) α=0 (VDVAE & Versatile) vs VDVAE image-to-image ---
    # image_to_image is model-agnostic (lives under results/vdvae/image_to_image)
    for model in models:
        for network in networks:
            alpha_map = guess_alpha_folder_map(model_roots[model], subj, network, img_exts)
            if 0 not in alpha_map:
                continue
            base_folder = alpha_map[0]
            key = f"alpha0_vs_vdvae_i2i:{model}:{network}"
            vals: List[float] = []
            for idx in range(n_images):
                p_gen = image_path(base_folder, idx, img_exts)
                p_i2i = image_path(i2i_root, idx, img_exts)
                if p_gen is None or p_i2i is None:
                    continue
                im_gen = load_image_float01(p_gen)
                im_i2i = load_image_float01(p_i2i)
                if im_gen is None or im_i2i is None:
                    continue
                im_gen = resize_to(im_gen, (im_i2i.shape[1], im_i2i.shape[0]))
                vals.append(pixcorr(im_gen, im_i2i))
            add_series(key, vals)

    return per_image, summary

# -------- I/O helpers --------
def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def save_summary_csv(path: Path, summary: Dict[str, Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "n", "mean", "std", "min", "max",
        "mean_r", "ci_low_r", "ci_high_r",
        "mean_z", "ci_low_z", "ci_high_z",
        "t_stat", "p_value",
    ]
    rows = []
    for key, stats in summary.items():
        rows.append({
            "comparison": key,
            "n": stats.get("n"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean_r": stats.get("mean_r"),
            "ci_low_r": stats.get("ci_low_r"),
            "ci_high_r": stats.get("ci_high_r"),
            "mean_z": stats.get("mean_z"),
            "ci_low_z": stats.get("ci_low_z"),
            "ci_high_z": stats.get("ci_high_z"),
            "t_stat": stats.get("t_stat"),
            "p_value": stats.get("p_value"),
        })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser(description="Compute PixCorr across subjects with alpha vs alpha=0, and alpha0 vs originals/i2i.")
    p.add_argument("--base-results", required=True, help="Results root (e.g., /home/rothermm/brain-diffuser/results)")
    p.add_argument("--base-data", required=True, help="Data root (e.g., /home/rothermm/brain-diffuser/data)")
    p.add_argument("--subjects", nargs="+", type=int, default=[1,2,5,7], choices=[1,2,5,7])
    p.add_argument("--models", nargs="+", default=["vdvae", "versatile_diffusion"], choices=["vdvae", "versatile_diffusion"])
    p.add_argument("--networks", nargs="+", default=["emonet", "memnet"], choices=["emonet", "memnet"])
    p.add_argument("--img-ext", nargs="+", default=[".png", ".jpg", ".jpeg"])
    p.add_argument("--n", type=int, default=982, help="Number of test images (indices start at 0)")
    p.add_argument("--out-dir", default=None, help="Override output dir (default: {base-results}/metrics/pixcorr)")
    return p.parse_args()

def main():
    args = parse_args()
    base_results = Path(args.base_results)
    base_data = Path(args.base_data)
    models = tuple(args.models)
    networks = tuple(args.networks)
    img_exts = tuple(args.img_ext)
    n_images = int(args.n)
    out_dir = Path(args.out_dir) if args.out_dir else (base_results / "metrics" / "pixcorr")

    for subj in args.subjects:
        per_image, summary = compute_for_subject(
            subj=subj,
            base_results=base_results,
            base_data=base_data,
            models=models,
            networks=networks,
            img_exts=img_exts,
            n_images=n_images,
        )
        subj_dir = out_dir / f"subj{subj:02d}"
        save_json(subj_dir / "per_image.json", per_image)
        save_json(subj_dir / "summary.json", summary)
        save_summary_csv(subj_dir / "summary.csv", summary)
        print(f"[subj{subj:02d}] saved: {subj_dir}/per_image.json, summary.json, summary.csv")

if __name__ == "__main__":
    main()
