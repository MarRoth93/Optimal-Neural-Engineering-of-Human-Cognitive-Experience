#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal script to load model assessor scores and SSIM summaries and generate:
  1) new.png  (2x2 panels: rows=models, cols=tasks; y=relative/(1-SSIM), α≠0, log y)
  2) scores_averaged_penalized_by_1_minus_ssim_models_only_side_by_side.png (raw/(1-SSIM))
  3) scores_averaged_raw_models_only_side_by_side.png (raw means)
  4) scores_relative_penalized_by_1_minus_ssim_models_only_side_by_side.png (relative/(1-SSIM))
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path("/home/rothermm/brain-diffuser")
OUTPUT_DIR = BASE_DIR / "results" / "graphs"
MODEL_SCORE_DIR = BASE_DIR / "results" / "assessor_scores"
SSIM_ROOT = BASE_DIR / "results" / "metrics" / "ssim"
PIXCORR_ROOT = BASE_DIR / "results" / "metrics" / "pixcorr"


SUBJECTS = [1, 2, 5, 7]
MODELS = ['vdvae', 'versatile']
NETWORKS = ['emonet', 'memnet']
ALPHA_LEVELS_STR = ['alpha_-4', 'alpha_-2', 'alpha_0', 'alpha_2', 'alpha_4']
ALPHA_LEVELS_NUM = np.array([-4, -2, 0, 2, 4])

PALETTE = sns.color_palette("colorblind")
COLOR_MAP = {
    "VDVAE":              PALETTE[1],  # orange
    "Versatile Diffusion": PALETTE[2], # green
    "Valence":            PALETTE[0],  # blue (if you want task-based later)
    "Memorability":       PALETTE[3],  # red (or another distinct color)

# Color by TASK now (not by model)
TASK_COLOR = {
    "emonet": "#2ca02c",       # Valence
    "memnet": "#d62728",       # Memorability
}

SSIM_MODEL_DIRNAME = {"vdvae": "vdvae", "versatile": "versatile_diffusion"}

def setup_plotting_style():
    mpl.rcParams.update({
        "figure.figsize": (16, 10),
        "axes.facecolor": "#F6F7FB",
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "legend.frameon": True,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "savefig.dpi": 300,
        "axes.grid": True,
    })

def _bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

# =============================================================================
# Data loading
# =============================================================================
def load_model_data():
    model_data = {net: {model: {} for model in MODELS} for net in NETWORKS}
    for sub in SUBJECTS:
        for net in NETWORKS:
            for model in MODELS:
                filename = f"{net}_{model}_sub{sub:02d}.pkl"
                path = MODEL_SCORE_DIR / f"subj{sub:02d}" / filename
                try:
                    with open(path, "rb") as f:
                        model_data[net][model][sub] = pickle.load(f)
                except FileNotFoundError:
                    print(f"❌ Missing: {path}")
                    model_data[net][model][sub] = None
    print("✅ Model data loaded.")
    return model_data

def _load_pixcorr_summary_for_subject(subj: int) -> pd.DataFrame:
    """Read subjXX/summary.csv and return a tidy DF. Missing -> empty DF."""
    csv_path = PIXCORR_ROOT / f"subj{subj:02d}" / "summary.csv"
    if not csv_path.exists():
        print(f"⚠️ PixCorr summary not found: {csv_path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Failed to read {csv_path}: {e}")
        return pd.DataFrame()

def _build_pixcorr_lookup():
    """
    Returns pix_map[network][model][subject_int][alpha_num] -> mean_r
    Only uses INTRA comparisons ('intra:{model}:{network}:alpha_±X_vs_alpha_0').
    alpha_num ∈ {-4, -2, 2, 4}. α=0 handled as 1.0 elsewhere.
    """
    import re
    pix_map = {net: {m: {} for m in MODELS} for net in NETWORKS}

    # regex for keys like: intra:vdvae:emonet:alpha_-2_vs_alpha_0
    pat = re.compile(r"^intra:(.*?):(.*?):alpha_([+-]?\d+)_vs_alpha_0$")

    for subj in SUBJECTS:
        df = _load_pixcorr_summary_for_subject(subj)
        if df.empty or "comparison" not in df or "mean_r" not in df:
            continue

        rows = []
        for comp, mean_r in zip(df["comparison"].values, df["mean_r"].values):
            m = pat.match(str(comp))
            if not m:
                continue
            model, network, a = m.groups()
            # normalize model naming: 'versatile' vs 'versatile_diffusion' in your code
            if model == "versatile_diffusion":
                model_key = "versatile"
            else:
                model_key = model
            try:
                a_num = int(a.replace("+", ""))  # "+2" -> 2
            except Exception:
                continue
            rows.append((network, model_key, subj, a_num, float(mean_r)))

        # write into nested dict
        for network, model_key, subj_id, a_num, r in rows:
            pix_map.setdefault(network, {}).setdefault(model_key, {}).setdefault(subj_id, {})[a_num] = r

    return pix_map

def _get_pixcorr(pix_map, net: str, model: str, subj: int, alpha_num: int) -> float:
    if alpha_num == 0:
        return 1.0
    # map 'versatile' -> stored key 'versatile' (we already normalized on build)
    try:
        return pix_map[net][model].get(subj, {}).get(alpha_num, np.nan)
    except KeyError:
        return np.nan

def _load_ssim_summary_for(model_key: str, network: str) -> pd.DataFrame:
    mdir = SSIM_MODEL_DIRNAME.get(model_key, model_key)
    csv_path = SSIM_ROOT / mdir / f"summary_{network}.csv"
    if not csv_path.exists():
        print(f"⚠️ SSIM summary not found: {csv_path}")
        return pd.DataFrame(columns=["model","network","subject","alpha","n","ssim_mean","ssim_std"])
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Failed to read {csv_path}: {e}")
        return pd.DataFrame(columns=["model","network","subject","alpha","n","ssim_mean","ssim_std"])

def _build_ssim_lookup():
    """
    Returns ssim_map[network][model][subject_int][alpha_num] -> ssim_mean
    alpha_num ∈ {-4, -2, 2, 4}. α=0 handled as 1.0 elsewhere.
    """
    ssim_map = {net: {m: {} for m in MODELS} for net in NETWORKS}
    alpha_parse = {"alpha -4": -4, "alpha -2": -2, "alpha 2": 2, "alpha 4": 4}
    for net in NETWORKS:
        for model in MODELS:
            df = _load_ssim_summary_for(model, net)
            if df.empty:
                continue
            def subj_to_int(s):
                try:
                    return int(str(s).lower().replace("subj",""))
                except Exception:
                    return None
            df = df.copy()
            df["subject_int"] = df["subject"].apply(subj_to_int)
            df["alpha_num"] = df["alpha"].map(alpha_parse)
            df = df.dropna(subset=["subject_int","alpha_num","ssim_mean"])
            lut = {}
            for (subj, a), g in df.groupby(["subject_int","alpha_num"]):
                lut.setdefault(int(subj), {})[int(a)] = float(np.mean(g["ssim_mean"].values))
            ssim_map[net][model] = lut
    return ssim_map

def _get_ssim(ssim_map, net: str, model: str, subj: int, alpha_num: int) -> float:
    if alpha_num == 0:
        return 1.0
    try:
        return ssim_map[net][model].get(subj, {}).get(alpha_num, np.nan)
    except KeyError:
        return np.nan

# =============================================================================
# Core utilities (per-subject means per α)
# =============================================================================
def _subject_alpha_means(d_dict):
    """Return np.array of means per alpha ([-4,-2,0,2,4]) in ALPHA_LEVELS_STR order."""
    return np.array([np.nanmean(d_dict[a]) if (d_dict and a in d_dict) else np.nan
                     for a in ALPHA_LEVELS_STR], dtype=float)

# =============================================================================
# PLOTS
# =============================================================================
def plot_scores_averaged_raw_models_only(model_data):
    """
    Raw averaged scores per α. Side-by-side: Valence (EmoNet) | Memorability (MemNet).
    Each subplot shows two lines: VDVAE vs Versatile.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    col_tasks = [('emonet', 'Valence'), ('memnet', 'Memorability')]

    for c, (net_key, task_name) in enumerate(col_tasks):
        ax = axs[c]
        for model_key, model_name in [('vdvae','VDVAE'), ('versatile','Versatile Diffusion')]:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                subj_curves.append(_subject_alpha_means(d))
            if not subj_curves:
                continue
            avg = np.nanmean(np.vstack(subj_curves), axis=0)
            color = COLOR_MAP["VDVAE"] if model_key == 'vdvae' else COLOR_MAP["Versatile Diffusion"]
            ax.plot(ALPHA_LEVELS_NUM, avg, marker='o', label=model_name, color=color)

        ax.set_title(task_name)
        ax.set_xlabel("Alpha Level")
        if c == 0:
            ax.set_ylabel("Mean Score (raw)")
        ax.set_xticks(ALPHA_LEVELS_NUM)
        _bold_ticks(ax)
        ax.legend()

    fig.suptitle("Raw Mean Assessor Scores — Models Only", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.93])
    out = OUTPUT_DIR / "scores_averaged_raw_models_only_side_by_side.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"📈 Saved: {out}")

def plot_scores_averaged_penalized_by_1_minus_pixcorr_models_only(model_data):
    """
    Penalized raw scores: score(α) / (1 − PixCorr(α, α=0)).
    Side-by-side: Valence | Memorability.
    α=0 dropped from plots.
    """
    pix_map = _build_pixcorr_lookup()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    col_tasks = [('emonet', 'Valence'), ('memnet', 'Memorability')]

    alpha_no0 = ALPHA_LEVELS_NUM[ALPHA_LEVELS_NUM != 0]

    for c, (net_key, task_name) in enumerate(col_tasks):
        ax = axs[c]
        for model_key, model_name in [('vdvae','VDVAE'), ('versatile','Versatile Diffusion')]:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                raw = _subject_alpha_means(d)  # [-4,-2,0,2,4]
                pix_vals = np.array([_get_pixcorr(pix_map, net_key, model_key, sub, a)
                                     for a in ALPHA_LEVELS_NUM], dtype=float)
                denom = 1.0 - pix_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    penalized = raw / denom
                penalized[~np.isfinite(penalized)] = np.nan
                subj_curves.append(penalized)

            if not subj_curves:
                continue
            avg = np.nanmean(np.vstack(subj_curves), axis=0)
            avg_no0 = avg[ALPHA_LEVELS_NUM != 0]

            color = COLOR_MAP["VDVAE"] if model_key == 'vdvae' else COLOR_MAP["Versatile Diffusion"]
            ax.plot(alpha_no0, avg_no0, marker='o', label=model_name, color=color)

        ax.set_title(task_name)
        ax.set_xlabel("Alpha Level")
        if c == 0:
            ax.set_ylabel("Mean Score / (1 − PixCorr)")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        ax.legend()

    fig.suptitle("Raw Scores Penalized by (1 − PixCorr) — Models Only", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.93])
    out = OUTPUT_DIR / "scores_averaged_penalized_by_1_minus_pixcorr_models_only_side_by_side.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"📈 Saved: {out}")

def plot_scores_relative_penalized_by_1_minus_pixcorr_models_only(model_data):
    """
    Relative penalized scores: (score(α)/score(0)) / (1 − PixCorr(α, α=0)).
    Side-by-side: Valence | Memorability.
    α=0 dropped from plots.
    """
    pix_map = _build_pixcorr_lookup()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    col_tasks = [('emonet', 'Valence'), ('memnet', 'Memorability')]

    alpha_no0 = ALPHA_LEVELS_NUM[ALPHA_LEVELS_NUM != 0]

    for c, (net_key, task_name) in enumerate(col_tasks):
        ax = axs[c]
        for model_key, model_name in [('vdvae','VDVAE'), ('versatile','Versatile Diffusion')]:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                raw = _subject_alpha_means(d)
                base_idx = np.where(ALPHA_LEVELS_NUM == 0)[0][0]
                base_val = raw[base_idx]
                if np.isnan(base_val) or base_val == 0:
                    continue
                rel = raw / base_val
                pix_vals = np.array([_get_pixcorr(pix_map, net_key, model_key, sub, a)
                                     for a in ALPHA_LEVELS_NUM], dtype=float)
                denom = 1.0 - pix_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    penalized_rel = rel / denom
                penalized_rel[~np.isfinite(penalized_rel)] = np.nan
                subj_curves.append(penalized_rel)

            if not subj_curves:
                continue
            avg = np.nanmean(np.vstack(subj_curves), axis=0)
            avg_no0 = avg[ALPHA_LEVELS_NUM != 0]

            color = COLOR_MAP["VDVAE"] if model_key == 'vdvae' else COLOR_MAP["Versatile Diffusion"]
            ax.plot(alpha_no0, avg_no0, marker='o', label=model_name, color=color)

        ax.set_title(task_name)
        ax.set_xlabel("Alpha Level")
        if c == 0:
            ax.set_ylabel("Relative Score (vs α=0) / (1 − PixCorr)")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        ax.legend()

    fig.suptitle("Relative Scores Penalized by (1 − PixCorr) — Models Only", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.93])
    out = OUTPUT_DIR / "scores_relative_penalized_by_1_minus_pixcorr_models_only_side_by_side.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"📈 Saved: {out}")



def plot_scores_averaged_penalized_by_1_minus_ssim_models_only(model_data):
    """
    Penalized raw scores: score(α) / (1 − SSIM(α, α=0)).
    Side-by-side: Valence | Memorability.
    α=0 dropped from plots.
    """
    ssim_map = _build_ssim_lookup()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    col_tasks = [('emonet', 'Valence'), ('memnet', 'Memorability')]

    alpha_no0 = ALPHA_LEVELS_NUM[ALPHA_LEVELS_NUM != 0]

    for c, (net_key, task_name) in enumerate(col_tasks):
        ax = axs[c]
        for model_key, model_name in [('vdvae','VDVAE'), ('versatile','Versatile Diffusion')]:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                raw = _subject_alpha_means(d)
                ssim_vals = np.array([_get_ssim(ssim_map, net_key, model_key, sub, a)
                                      for a in ALPHA_LEVELS_NUM], dtype=float)
                denom = 1.0 - ssim_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    penalized = raw / denom
                penalized[~np.isfinite(penalized)] = np.nan
                subj_curves.append(penalized)

            if not subj_curves:
                continue
            avg = np.nanmean(np.vstack(subj_curves), axis=0)
            avg_no0 = avg[ALPHA_LEVELS_NUM != 0]

            color = COLOR_MAP["VDVAE"] if model_key == 'vdvae' else COLOR_MAP["Versatile Diffusion"]
            ax.plot(alpha_no0, avg_no0, marker='o', label=model_name, color=color)

        ax.set_title(task_name)
        ax.set_xlabel("Alpha Level")
        if c == 0:
            ax.set_ylabel("Mean Score / (1 − SSIM)")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        ax.legend()

    fig.suptitle("Raw Scores Penalized by (1 − SSIM) — Models Only", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.93])
    out = OUTPUT_DIR / "scores_averaged_penalized_by_1_minus_ssim_models_only_side_by_side.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"📈 Saved: {out}")

def new_pixcorr(model_data):
    """
    2x1 panels: top=VDVAE, bottom=Versatile Diffusion.
    Each panel shows two lines colored by TASK (Valence/EmoNet & Memorability/MemNet).
    y = (score(α)/score(0)) / (1 − PixCorr(α, α=0)), α≠0 only, linear y-axis.
    """
    pix_map = _build_pixcorr_lookup()
    alpha_no0 = ALPHA_LEVELS_NUM[ALPHA_LEVELS_NUM != 0]

    series = {model: {} for model in MODELS}
    for model_key in MODELS:
        for net_key in NETWORKS:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                raw = _subject_alpha_means(d)
                base_val = raw[ALPHA_LEVELS_NUM.tolist().index(0)]
                if not np.isfinite(base_val) or base_val == 0:
                    continue
                rel = raw / base_val
                pix_vals = np.array([_get_pixcorr(pix_map, net_key, model_key, sub, a)
                                     for a in ALPHA_LEVELS_NUM], dtype=float)
                denom = 1.0 - pix_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    penalized = rel / denom
                penalized[~np.isfinite(penalized)] = np.nan
                subj_curves.append(penalized[ALPHA_LEVELS_NUM != 0])
            series[model_key][net_key] = (np.nanmean(np.vstack(subj_curves), axis=0)
                                          if subj_curves else None)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    pan = [
        ("vdvae", "VDVAE", axs[0]),
        ("versatile", "Versatile Diffusion", axs[1]),
    ]
    for model_key, model_name, ax in pan:
        for net_key, task_name in (("emonet","Valence"), ("memnet","Memorability")):
            y = series[model_key][net_key]
            if y is None:
                continue
            ax.plot(alpha_no0, y, marker='o', linewidth=2.5,
                    label=task_name, color=TASK_COLOR[net_key])
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Alpha Level", fontweight="bold")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        ax.legend(title="Task")

    axs[0].set_ylabel("Relative Mean Score", fontweight="bold")
    axs[1].set_ylabel("Relative Mean Score", fontweight="bold")

    fig.suptitle("Relative Mean Scores (vs α=0) Penalized by (1 − PixCorr)", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUTPUT_DIR / "new_pixcorr.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"📈 Saved: {out_path}")


def plot_scores_relative_penalized_by_1_minus_ssim_models_only(model_data):
    """
    Relative penalized scores: (score(α)/score(0)) / (1 − SSIM(α, α=0)).
    Side-by-side: Valence | Memorability.
    α=0 dropped from plots.
    """
    ssim_map = _build_ssim_lookup()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    col_tasks = [('emonet', 'Valence'), ('memnet', 'Memorability')]

    alpha_no0 = ALPHA_LEVELS_NUM[ALPHA_LEVELS_NUM != 0]

    for c, (net_key, task_name) in enumerate(col_tasks):
        ax = axs[c]
        for model_key, model_name in [('vdvae','VDVAE'), ('versatile','Versatile Diffusion')]:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                raw = _subject_alpha_means(d)
                base_idx = np.where(ALPHA_LEVELS_NUM == 0)[0][0]
                base_val = raw[base_idx]
                if np.isnan(base_val) or base_val == 0:
                    continue
                rel = raw / base_val
                ssim_vals = np.array([_get_ssim(ssim_map, net_key, model_key, sub, a)
                                      for a in ALPHA_LEVELS_NUM], dtype=float)
                denom = 1.0 - ssim_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    penalized_rel = rel / denom
                penalized_rel[~np.isfinite(penalized_rel)] = np.nan
                subj_curves.append(penalized_rel)

            if not subj_curves:
                continue
            avg = np.nanmean(np.vstack(subj_curves), axis=0)
            avg_no0 = avg[ALPHA_LEVELS_NUM != 0]

            color = COLOR_MAP["VDVAE"] if model_key == 'vdvae' else COLOR_MAP["Versatile Diffusion"]
            ax.plot(alpha_no0, avg_no0, marker='o', label=model_name, color=color)

        ax.set_title(task_name)
        ax.set_xlabel("Alpha Level")
        if c == 0:
            ax.set_ylabel("Relative Score (vs α=0) / (1 − SSIM)")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        ax.legend()

    fig.suptitle("Relative Scores Penalized by (1 − SSIM) — Models Only", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.93])
    out = OUTPUT_DIR / "scores_relative_penalized_by_1_minus_ssim_models_only_side_by_side.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"📈 Saved: {out}")

def new(model_data):
    """
    2x1 panels: top=VDVAE, bottom=Versatile Diffusion.
    Each panel shows two lines colored by TASK (Valence/EmoNet & Memorability/MemNet).
    y = (score(α)/score(0)) / (1 − SSIM(α, α=0)), α≠0 only, linear y-axis.
    """
    ssim_map = _build_ssim_lookup()
    alpha_no0 = ALPHA_LEVELS_NUM[ALPHA_LEVELS_NUM != 0]

    series = {model: {} for model in MODELS}
    for model_key in MODELS:
        for net_key in NETWORKS:
            subj_curves = []
            for sub in SUBJECTS:
                d = model_data[net_key][model_key].get(sub)
                if not d:
                    continue
                raw = _subject_alpha_means(d)
                base_val = raw[ALPHA_LEVELS_NUM.tolist().index(0)]
                if not np.isfinite(base_val) or base_val == 0:
                    continue
                rel = raw / base_val
                ssim_vals = np.array([_get_ssim(ssim_map, net_key, model_key, sub, a)
                                      for a in ALPHA_LEVELS_NUM], dtype=float)
                denom = 1.0 - ssim_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    penalized = rel / denom
                penalized[~np.isfinite(penalized)] = np.nan
                subj_curves.append(penalized[ALPHA_LEVELS_NUM != 0])
            series[model_key][net_key] = (np.nanmean(np.vstack(subj_curves), axis=0)
                                          if subj_curves else None)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    pan = [
        ("vdvae", "VDVAE", axs[0]),
        ("versatile", "Versatile Diffusion", axs[1]),
    ]
    for model_key, model_name, ax in pan:
        for net_key, task_name in (("emonet","Valence"), ("memnet","Memorability")):
            y = series[model_key][net_key]
            if y is None:
                continue
            ax.plot(alpha_no0, y, marker='o', linewidth=2.5,
                    label=task_name, color=TASK_COLOR[net_key])
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Alpha Level", fontweight="bold")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        ax.legend(title="Task")

    axs[0].set_ylabel("Relative Mean Score", fontweight="bold")
    axs[1].set_ylabel("Relative Mean Score", fontweight="bold")

    fig.suptitle("Relative Mean Scores (vs α=0) Penalized by (1 − SSIM)", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUTPUT_DIR / "new.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"📈 Saved: {out_path}")

# =============================================================================
# Main
# =============================================================================
def main():
    print("--- Minimal Plot Script ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting_style()
    model_data = load_model_data()

    print("\n--- Generating scores_averaged_raw_models_only_side_by_side.png ---")
    plot_scores_averaged_raw_models_only(model_data)

    print("\n--- Generating scores_averaged_penalized_by_1_minus_ssim_models_only_side_by_side.png ---")
    plot_scores_averaged_penalized_by_1_minus_ssim_models_only(model_data)

    print("\n--- Generating scores_relative_penalized_by_1_minus_ssim_models_only_side_by_side.png ---")
    plot_scores_relative_penalized_by_1_minus_ssim_models_only(model_data)

    print("\n--- Generating new.png ---")
    new(model_data)

    print("\n--- Generating scores_averaged_penalized_by_1_minus_pixcorr_models_only_side_by_side.png ---")
    plot_scores_averaged_penalized_by_1_minus_pixcorr_models_only(model_data)

    print("\n--- Generating scores_relative_penalized_by_1_minus_pixcorr_models_only_side_by_side.png ---")
    plot_scores_relative_penalized_by_1_minus_pixcorr_models_only(model_data)

    print("\n--- Generating new_pixcorr.png ---")
    new_pixcorr(model_data)


    print("\n--- Done ---")

if __name__ == "__main__":
    main()
