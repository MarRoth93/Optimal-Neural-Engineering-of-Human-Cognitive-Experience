#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consolidated script for loading model assessor scores and human behavioral data,
performing analysis, and generating comparison plots.

Adds NEW plots:
- For each of VDVAE, CLIP (Versatile), and Human:
  stacked rows (one per alpha level) where each row shows the distribution
  of assessor scores (EmoNet vs MemNet) pooled across all subjects.

Saved as:
  results/graphs/stacked_alpha_vdvae.png
  results/graphs/stacked_alpha_clip.png
  results/graphs/stacked_alpha_human.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path

# =============================================================================
# --- Configuration ---
# =============================================================================
BASE_DIR = Path("/home/rothermm/brain-diffuser")
OUTPUT_DIR = BASE_DIR / "results" / "graphs"
MODEL_SCORE_DIR = BASE_DIR / "results" / "assessor_scores"
HUMAN_DATA_PATH = BASE_DIR / "data" / "human_data" / "human_df_detrended.csv"

SUBJECTS = [1, 2, 5, 7]
MODELS = ['vdvae', 'versatile']          # 'versatile' is labeled "CLIP (Versatile)" in new plots
NETWORKS = ['emonet', 'memnet']
ALPHA_LEVELS_STR = ['alpha_-4', 'alpha_-2', 'alpha_0', 'alpha_2', 'alpha_4']
ALPHA_LEVELS_NUM = np.array([-4, -2, 0, 2, 4])

# =============================================================================
# --- Plotting style ---
# =============================================================================
def setup_plotting_style():
    sns.set_style("whitegrid", {"grid.color": "white", "grid.linestyle": "-", "grid.linewidth": 1.2})
    sns.set_context("notebook", font_scale=1.2)
    mpl.rcParams.update({
        "figure.figsize": (16, 12),
        "axes.facecolor": "#F6F7FB",
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.2,
    })
    print("🎨 Plotting style configured.")

# =============================================================================
# --- I/O ---
# =============================================================================
def load_data():
    """Load model assessor scores and human data."""
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
                    print(f"❌ Missing model file: {path}")
                    model_data[net][model][sub] = None

    human_df = None
    try:
        human_df = pd.read_csv(HUMAN_DATA_PATH)
        alpha_map = {
            'valence-4': -4, 'valence-2': -2, 'alpha0': 0, 'valence+2': 2, 'valence+4': 4,
            'mem-4': -4, 'mem-2': -2, 'mem+2': 2, 'mem+4': 4
        }
        human_df['Alpha'] = human_df['Condition'].map(alpha_map)
    except FileNotFoundError:
        print(f"❌ Missing human data: {HUMAN_DATA_PATH}")

    print("✅ Data load finished.")
    return model_data, human_df

# =============================================================================
# --- Utilities ---
# =============================================================================
def normalize_scores(scores):
    s_arr = np.array(scores, dtype=float)
    mn, mx = s_arr.min(), s_arr.max()
    if mx == mn:
        return np.zeros_like(s_arr)
    return (s_arr - mn) / (mx - mn)

# =============================================================================
# --- (Your existing plots remain available; omitted here for brevity) ---
# If you want me to keep generating all old figures in the same run, leave the
# original plot functions in place. For now we focus on the new stacked plots.
# =============================================================================

# =============================================================================
# --- NEW: Stacked-by-alpha distributions (VDVAE, CLIP, Human) ---
# =============================================================================
def _gather_model_scores(model_data, model, net, alpha_key):
    """Concatenate assessor scores across subjects for a given model/net/alpha."""
    arrs = []
    for sub in SUBJECTS:
        d = model_data[net][model].get(sub)
        if d is None:
            continue
        if alpha_key in d:
            arrs.append(np.asarray(d[alpha_key], dtype=float))
    if len(arrs) == 0:
        return np.array([], dtype=float)
    return np.concatenate(arrs)

def _alpha_title(a):
    return f"α = {a:+d}"

def _stacked_panel(fig_title, rows_data, out_path, x_label="Assessor score", xlim=None):
    """
    Generic helper to draw a vertical stack of 5 rows (α=-4,-2,0,2,4).
    rows_data: list of dict per row with keys:
        {"alpha": int, "emonet": np.array, "memnet": np.array}
    """
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2*n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    for ax, row in zip(axes, rows_data):
        emo = row["emonet"]
        mem = row["memnet"]
        # Choose common bins per row
        combined = np.concatenate([emo, mem]) if len(emo)+len(mem) > 0 else np.array([0, 1])
        lo, hi = np.nanmin(combined), np.nanmax(combined)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        pad = 0.02*(hi-lo) if hi > lo else 0.05
        bins = np.linspace(lo - pad, hi + pad, 40)

        # Overlaid density histograms
        ax.hist(emo, bins=bins, density=True, alpha=0.55, edgecolor='black', label="EmoNet")
        ax.hist(mem, bins=bins, density=True, alpha=0.55, edgecolor='black', label="MemNet")

        ax.set_ylabel(_alpha_title(row["alpha"]))
        ax.grid(True, axis='y', alpha=0.35)

    axes[0].legend(loc="upper right", frameon=True, title="Assessor")
    axes[-1].set_xlabel(x_label)
    if xlim is not None:
        axes[-1].set_xlim(xlim)
    fig.suptitle(fig_title, y=0.995, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📊 Saved: {out_path}")

def plot_stacked_alpha_distributions(model_data, human_df):
    """
    Creates three figures:
      1) VDVAE: rows are α; each row shows EmoNet vs MemNet distributions pooled across subjects
      2) CLIP (Versatile): same as above
      3) Human: valence vs memorability distributions for the matching α
    """

    # ---------- VDVAE ----------
    rows = []
    for a_str, a_num in zip(ALPHA_LEVELS_STR, ALPHA_LEVELS_NUM):
        emo = _gather_model_scores(model_data, "vdvae", "emonet", a_str)
        mem = _gather_model_scores(model_data, "vdvae", "memnet", a_str)
        rows.append({"alpha": int(a_num), "emonet": emo, "memnet": mem})
    _stacked_panel(
        fig_title="VDVAE • Assessor score distributions pooled across subjects",
        rows_data=rows,
        out_path=OUTPUT_DIR / "stacked_alpha_vdvae.png",
        x_label="Assessor score (EmoNet / MemNet)"
    )

    # ---------- CLIP (Versatile) ----------
    rows = []
    for a_str, a_num in zip(ALPHA_LEVELS_STR, ALPHA_LEVELS_NUM):
        emo = _gather_model_scores(model_data, "versatile", "emonet", a_str)
        mem = _gather_model_scores(model_data, "versatile", "memnet", a_str)
        rows.append({"alpha": int(a_num), "emonet": emo, "memnet": mem})
    _stacked_panel(
        fig_title="CLIP (Versatile Diffusion) • Assessor score distributions pooled across subjects",
        rows_data=rows,
        out_path=OUTPUT_DIR / "stacked_alpha_clip.png",
        x_label="Assessor score (EmoNet / MemNet)"
    )

    # ---------- Human ----------
    if human_df is None:
        print("⚠️ Human data not available — skipping human stacked plot.")
        return

    # Map α to human conditions
    valence_map = {-4: 'valence-4', -2: 'valence-2', 0: 'alpha0',  2: 'valence+2', 4: 'valence+4'}
    memory_map  = {-4: 'mem-4',     -2: 'mem-2',     0: 'alpha0',  2: 'mem+2',     4: 'mem+4'}

    rows = []
    for a in ALPHA_LEVELS_NUM:
        # Pool across subjects for each α/measure
        emo_vals = human_df[human_df['Condition'] == valence_map[a]]['ValenceRating'].astype(float).values
        mem_vals = human_df[human_df['Condition'] == memory_map[a]]['MemorabilityRating'].astype(float).values
        rows.append({"alpha": int(a), "emonet": emo_vals, "memnet": mem_vals})

    _stacked_panel(
        fig_title="Human ratings • Distributions pooled across subjects",
        rows_data=rows,
        out_path=OUTPUT_DIR / "stacked_alpha_human.png",
        x_label="Human rating (Valence / Memorability)"
    )



# =============================================================================
# --- Main ---
# =============================================================================
def main():
    print("--- Starting Analysis Script ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting_style()
    model_data, human_data = load_data()

    # --- Generate ONLY the new figures for now ---
    plot_stacked_alpha_distributions(model_data, human_data)

    print("\n--- Script finished successfully ---")

if __name__ == "__main__":
    main()
