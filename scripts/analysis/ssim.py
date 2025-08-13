#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute SSIM (and optional MS-SSIM) for VDVAE & Versatile Diffusion images across ALL subjects,
for both EmoNet and MemNet, comparing each α condition against α=0.

Folder layout expected (auto-detected, tolerant to naming):
  {BASE}/vdvae/{subjXX}/{network}/{alpha_folder}/*.png
  {BASE}/versatile_diffusion/{subjXX}/{network}/{alpha_folder}/*.png

Examples of accepted alpha folder names (all mapped to canonical keys):
  α=0:   ["alpha_0", "alpha0", "zero", "0"]
  α=-2:  ["alpha_-2", "alpha-2", "neg2", "negative_theta_2", "minus2"]
  α=-4:  ["alpha_-4", "alpha-4", "neg4", "negative_theta_4", "minus4"]
  α=+2:  ["alpha_2", "alpha+2", "pos2", "positive_theta_2", "plus2"]
  α=+4:  ["alpha_4", "alpha+4", "pos4", "positive_theta_4", "plus4"]

Outputs:
  - results/metrics/ssim/{model}/per_subject_{network}.json           (lists of per-image SSIM)
  - results/metrics/ssim/{model}/summary_{network}.json               (means/stdevs per subject & overall)
  - results/metrics/ssim/{model}/summary_{network}.csv
  - results/metrics/ssim/{model}/plot_{network}.png                   (bar plot with ±1 SD)

Usage:
  python compute_ssim_all_subjects.py \
      --base /home/rothermm/brain-diffuser/results \
      --models vdvae versatile_diffusion \
      --networks emonet memnet \
      --img-ext .png .jpg .jpeg \
      --device auto
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Metrics imports (SSIM required, MS-SSIM optional) ----------
try:
    from piq import ssim as piq_ssim
    HAVE_PIQ = True
except Exception:
    HAVE_PIQ = False

try:
    from pytorch_msssim import ms_ssim as vainf_ms_ssim
    HAVE_MSSSIM = True
except Exception:
    HAVE_MSSSIM = False

# ---------- Alpha mapping ----------
ALPHA_CANONICAL = ["alpha -4", "alpha -2", "alpha 0", "alpha 2", "alpha 4"]
ALPHA_FOLDER_CANDIDATES: Dict[str, List[str]] = {
    "alpha 0":  ["alpha_0", "alpha0", "zero", "0"],
    "alpha -2": ["alpha_-2", "alpha-2", "neg2", "negative_theta_2", "minus2"],
    "alpha -4": ["alpha_-4", "alpha-4", "neg4", "negative_theta_4", "minus4"],
    "alpha 2":  ["alpha_2", "alpha+2", "pos2", "positive_theta_2", "plus2"],
    "alpha 4":  ["alpha_4", "alpha+4", "pos4", "positive_theta_4", "plus4"],
}

# ---------- Utilities ----------
def list_subjects(root: Path) -> List[str]:
    subs = []
    if not root.is_dir():
        return subs
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.lower().startswith("subj"):
            subs.append(p.name)
    return subs

def find_existing_alpha_folder(network_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = network_dir / name
        if p.is_dir():
            return p
    return None

def discover_alpha_dirs(model_dir: Path, subject: str, network: str) -> Dict[str, Optional[Path]]:
    """
    Returns mapping canonical_alpha -> Path or None if not found.
    """
    net_dir = model_dir / subject / network
    out = {}
    for alpha_key, candidates in ALPHA_FOLDER_CANDIDATES.items():
        out[alpha_key] = find_existing_alpha_folder(net_dir, candidates)
    return out

def load_img_as_tensor(path: Path, device: torch.device) -> Optional[torch.Tensor]:
    try:
        img = Image.open(path).convert("RGB")
        tensor = T.ToTensor()(img).unsqueeze(0).to(device)  # (1,C,H,W), [0,1]
        return tensor
    except Exception as e:
        print(f"[WARN] Could not load image {path}: {e}")
        return None

def resize_to(t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    # target_hw: (H, W)
    _, C, H, W = t.shape
    if (H, W) == target_hw:
        return t
    pil = T.ToPILImage()(t.squeeze(0).cpu())
    pil = pil.resize((target_hw[1], target_hw[0]), Image.Resampling.BICUBIC)
    return T.ToTensor()(pil).unsqueeze(0).to(t.device)

def compute_ssim(ref: torch.Tensor, x: torch.Tensor) -> float:
    """
    Uses piq.ssim if available; otherwise raises an informative error.
    Both inputs (1,C,H,W) in [0,1].
    """
    if not HAVE_PIQ:
        raise RuntimeError("piq is not installed. Install with: pip install piq")
    with torch.no_grad():
        val = piq_ssim(x, ref, data_range=1.0)  # symmetric
    return float(val.item())

def compute_msssim(ref: torch.Tensor, x: torch.Tensor) -> Optional[float]:
    if not HAVE_MSSSIM:
        return None
    with torch.no_grad():
        val = vainf_ms_ssim(x, ref, data_range=1.0, size_average=True)
    return float(val.item())

def intersect_filenames(a: Path, b: Path, exts: Tuple[str, ...]) -> List[str]:
    fa = {f.name for f in a.iterdir() if f.is_file() and f.suffix.lower() in exts}
    fb = {f.name for f in b.iterdir() if f.is_file() and f.suffix.lower() in exts}
    return sorted(list(fa & fb))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Core runner ----------
def process_model(
    base_dir: Path,
    model: str,
    networks: List[str],
    img_exts: Tuple[str, ...],
    device: torch.device,
    out_root: Path,
    resize_to_reference: bool = True,
):
    model_dir = base_dir / model
    subjects = list_subjects(model_dir)
    if not subjects:
        print(f"[WARN] No subjects found under {model_dir}")
        return

    print(f"\n=== {model.upper()} ===")
    print(f"Subjects: {subjects}\n")

    for network in networks:
        print(f"--- Network: {network} ---")
        per_subject_scores: Dict[str, Dict[str, List[float]]] = {}     # SSIM lists
        per_subject_mscores: Dict[str, Dict[str, List[float]]] = {}    # MS-SSIM lists (optional)

        for subj in subjects:
            alpha_dirs = discover_alpha_dirs(model_dir, subj, network)
            ref_dir = alpha_dirs.get("alpha 0")

            if ref_dir is None:
                print(f"[WARN] {model}/{subj}/{network}: missing α=0 folder. Skipping subject.")
                continue

            # Prepare target alpha comparisons (exclude alpha 0)
            target_pairs = [(k, d) for k, d in alpha_dirs.items() if k != "alpha 0" and d is not None]
            if not target_pairs:
                print(f"[WARN] {model}/{subj}/{network}: no target alpha folders found. Skipping subject.")
                continue

            ref_files = [f for f in ref_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts]
            if not ref_files:
                print(f"[WARN] {ref_dir} has no images with {img_exts}. Skipping subject.")
                continue

            # Store per-alpha lists
            per_subject_scores[subj] = {k: [] for k, _ in target_pairs}
            if HAVE_MSSSIM:
                per_subject_mscores[subj] = {k: [] for k, _ in target_pairs}

            # Reference size determined by the first ref image
            ref0 = load_img_as_tensor(ref_files[0], device)
            if ref0 is None:
                print(f"[WARN] Could not load a reference image in {ref_dir}. Skipping subject.")
                continue
            ref_H, ref_W = ref0.shape[-2], ref0.shape[-1]

            for alpha_key, alpha_path in target_pairs:
                # Intersect filenames to compare 1:1
                common = intersect_filenames(ref_dir, alpha_path, img_exts)
                if not common:
                    print(f"[WARN] No overlapping filenames between {ref_dir.name} and {alpha_path.name} for {subj} ({alpha_key}).")
                    continue

                for fname in common:
                    ref_img = load_img_as_tensor(ref_dir / fname, device)
                    tgt_img = load_img_as_tensor(alpha_path / fname, device)
                    if ref_img is None or tgt_img is None:
                        continue

                    # Match size to reference
                    if resize_to_reference and (tgt_img.shape[-2:] != (ref_H, ref_W)):
                        tgt_img = resize_to(tgt_img, (ref_H, ref_W))
                    if ref_img.shape[-2:] != (ref_H, ref_W):
                        ref_img = resize_to(ref_img, (ref_H, ref_W))

                    if ref_img.shape != tgt_img.shape:
                        # final guard
                        print(f"[WARN] Shape mismatch after resize for {fname} in {subj} {network} {alpha_key}. Skipping this pair.")
                        continue

                    try:
                        val = compute_ssim(ref_img, tgt_img)
                        per_subject_scores[subj][alpha_key].append(val)
                        if HAVE_MSSSIM:
                            mval = compute_msssim(ref_img, tgt_img)
                            if mval is not None:
                                per_subject_mscores[subj][alpha_key].append(mval)
                    except Exception as e:
                        print(f"[WARN] SSIM failed for {fname} ({subj} {network} {alpha_key}): {e}")

        # ---- Save per-subject raw lists ----
        out_dir = out_root / model
        ensure_dir(out_dir)

        per_subject_path = out_dir / f"per_subject_{network}.json"
        with open(per_subject_path, "w") as f:
            json.dump(per_subject_scores, f, indent=2)
        print(f"[OK] Saved per-subject SSIM lists -> {per_subject_path}")

        if HAVE_MSSSIM:
            per_subject_ms_path = out_dir / f"per_subject_ms_{network}.json"
            with open(per_subject_ms_path, "w") as f:
                json.dump(per_subject_mscores, f, indent=2)
            print(f"[OK] Saved per-subject MS-SSIM lists -> {per_subject_ms_path}")

        # ---- Build summary (means / stds) + overall ----
        rows = []
        for subj, alpha_dict in per_subject_scores.items():
            for alpha_key in ["alpha -4", "alpha -2", "alpha 2", "alpha 4"]:
                vals = alpha_dict.get(alpha_key, [])
                vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
                if not vals:
                    continue
                rows.append({
                    "model": model,
                    "network": network,
                    "subject": subj,
                    "alpha": alpha_key,
                    "n": len(vals),
                    "ssim_mean": float(np.mean(vals)),
                    "ssim_std": float(np.std(vals)),
                })

        df = pd.DataFrame(rows)
        summary_json = out_dir / f"summary_{network}.json"
        summary_csv  = out_dir / f"summary_{network}.csv"
        df.to_json(summary_json, orient="records", indent=2)
        df.to_csv(summary_csv, index=False)
        print(f"[OK] Saved summary -> {summary_json}")
        print(f"[OK] Saved summary -> {summary_csv}")

        # ---- Overall bar plot (mean across subjects per alpha) ----
        if not df.empty:
            plot_path = out_dir / f"plot_{network}.png"
            plot_overall_bar(df, title=f"{model.upper()} – {network} SSIM vs α (ref: α=0)", out_path=plot_path)
            print(f"[OK] Saved plot -> {plot_path}")
        else:
            print(f"[INFO] No data to plot for {model}/{network}.")


def plot_overall_bar(df: pd.DataFrame, title: str, out_path: Path):
    order = ["alpha -4", "alpha -2", "alpha 2", "alpha 4"]
    agg = (
        df.groupby("alpha", as_index=False)
          .agg(ssim_mean=("ssim_mean", "mean"), ssim_std=("ssim_mean", "std"), n=("n", "sum"))
    )
    # Keep order
    agg["alpha"] = pd.Categorical(agg["alpha"], categories=order, ordered=True)
    agg = agg.sort_values("alpha")

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(agg))
    bars = ax.bar(xs, agg["ssim_mean"].values, yerr=agg["ssim_std"].values, capsize=8, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(agg["alpha"].values)
    ax.set_ylabel("SSIM (mean across subjects)")
    ax.set_xlabel("α level (compared to α=0)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # annotate
    for b, m, s in zip(bars, agg["ssim_mean"].values, agg["ssim_std"].values):
        ax.text(b.get_x()+b.get_width()/2, m + 0.02, f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Compute SSIM across subjects for vdvae & versatile_diffusion (emonet/memnet).")
    p.add_argument("--base", type=Path, default=Path("/home/rothermm/brain-diffuser/results"),
                   help="Base results directory containing 'vdvae' and/or 'versatile_diffusion'.")
    p.add_argument("--models", nargs="+", default=["vdvae", "versatile_diffusion"],
                   help="Model families to process.")
    p.add_argument("--networks", nargs="+", default=["emonet", "memnet"],
                   help="Networks to process.")
    p.add_argument("--img-ext", nargs="+", default=[".png", ".jpg", ".jpeg"],
                   help="Image extensions to include.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Compute device.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output root (defaults to BASE/results/metrics/ssim).")
    return p.parse_args()

def main():
    args = parse_args()

    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    out_root = args.out
    if out_root is None:
        out_root = args.base / "metrics" / "ssim"
    ensure_dir(out_root)

    img_exts = tuple([e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.img_ext])

    if not HAVE_PIQ:
        print("[ERROR] 'piq' is not installed. Install it with: pip install piq")
        return
    if HAVE_MSSSIM:
        print("[INFO] MS-SSIM available (pytorch-msssim). Will also compute per-subject MS-SSIM JSON.")
    else:
        print("[INFO] MS-SSIM not available. Install with: pip install pytorch-msssim (optional).")

    # Process each requested model family
    for model in args.models:
        process_model(
            base_dir=args.base,
            model=model,
            networks=args.networks,
            img_exts=img_exts,
            device=device,
            out_root=out_root,
            resize_to_reference=True,
        )

if __name__ == "__main__":
    main()
