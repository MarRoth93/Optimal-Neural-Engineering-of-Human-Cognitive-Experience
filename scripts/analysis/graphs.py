#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consolidated script for loading model assessor scores and human behavioral data,
performing analysis, and generating comparison plots.

This script produces three main sets of visualizations:
1.  Normalized Mean Scores: Compares model scores (VDVAE, Versatile) against
    human ratings for 'emonet' and 'memnet' networks.
    - One plot per network, with subplots for each subject.
    - One plot per network, with model scores averaged across subjects.
2.  Slope Distributions: Compares the distribution of response slopes for each
    model, showing how scores change as the alpha parameter is modulated.
    - One plot per network, with subplots for each subject.
    - One plot per network, pooling data across all subjects.
3.  Additional Rate‑of‑Change & Comparative Plots:
    a) Per‑subject EmoNet vs MemNet ROC histograms
    b) Overall EmoNet vs MemNet ROC histograms
    c) Overall model vs bootstrap‑resampled human ROC histograms
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
from scipy.stats import ttest_1samp, ttest_ind, shapiro, linregress
from datetime import datetime

# =============================================================================
# --- Configuration ---
# =============================================================================
# --- Paths ---
BASE_DIR = Path("/home/rothermm/brain-diffuser")
OUTPUT_DIR = BASE_DIR / "results" / "graphs"
MODEL_SCORE_DIR = BASE_DIR / "results" / "assessor_scores"
HUMAN_DATA_PATH = BASE_DIR / "data" / "human_data" / "human_df_detrended.csv"

# --- Analysis Parameters ---
SUBJECTS = [1, 2, 5, 7]  # List of subject IDs to include in analysis
MODELS = ['vdvae', 'versatile']  # Model types to compare
NETWORKS = ['emonet', 'memnet']  # Network types to compare
ALPHA_LEVELS_STR = ['alpha_-4', 'alpha_-2', 'alpha_0', 'alpha_2', 'alpha_4']  # String keys for alpha levels
ALPHA_LEVELS_NUM = np.array([-4, -2, 0, 2, 4])  # Numeric alpha levels for plotting


def setup_plotting_style():
    """
    Sets the global matplotlib and seaborn plotting styles for consistent, 
    publication-quality figures.
    """
    sns.set_style(
        "whitegrid",
        {"grid.color": "white", "grid.linestyle": "-", "grid.linewidth": 1.2},
    )
    sns.set_context("notebook", font_scale=1.2)
    mpl.rcParams.update({
        "figure.figsize": (16, 12),
        "axes.facecolor": "#EAEAF2",
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1.2,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "lines.markeredgewidth": 0,
    })
    print("🎨 Plotting style configured.")


def load_data():
    """
    Loads all model scores and human data into memory.

    Returns:
        model_data (dict): Nested dictionary of model scores, indexed by network, model, and subject.
        human_df (pd.DataFrame): DataFrame of human behavioral data, or None if not found.
    """
    # --- Load model data ---
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
                    print(f"❌ Could not find file: {path}")
                    model_data[net][model][sub] = None

    # --- Load human data ---
    try:
        human_df = pd.read_csv(HUMAN_DATA_PATH)
        # Map condition strings to alpha values for easier analysis
        alpha_map = {
            'valence-4': -4, 'valence-2': -2, 'alpha0': 0, 'valence+2': 2, 'valence+4': 4,
            'mem-4': -4, 'mem-2': -2, 'mem+2': 2, 'mem+4': 4
        }
        human_df['Alpha'] = human_df['Condition'].map(alpha_map)
    except FileNotFoundError:
        print(f"❌ Could not find human data at: {HUMAN_DATA_PATH}")
        return model_data, None

    print("✅ All model and human data loaded successfully.")
    return model_data, human_df


def normalize_scores(scores):
    """
    Performs min-max normalization on a numpy array.

    Args:
        scores (array-like): Input scores to normalize.

    Returns:
        np.ndarray: Normalized scores in [0, 1].
    """
    s_arr = np.array(scores)
    min_val, max_val = s_arr.min(), s_arr.max()
    if max_val == min_val:
        return np.zeros_like(s_arr)
    return (s_arr - min_val) / (max_val - min_val)


def pearson_safe(a, b):
    """Pearson r with safety checks; returns np.nan if not computable."""
    a = np.asarray(a); b = np.asarray(b)
    n = min(len(a), len(b))
    if n < 2:
        return np.nan
    a = a[:n]; b = b[:n]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def plot_normalized_mean_scores(model_data, human_data):
    """
    Generates plots comparing normalized mean scores of models and humans.
    1) By Subject: A 2x2 grid for each subject (unchanged).
    2) Averaged: ONE figure with two side-by-side panels:
         - Left: EmoNet averaged across subjects
         - Right: MemNet averaged across subjects (human α = ±4 removed)

    Args:
        model_data (dict): Nested dictionary of model scores.
        human_data (pd.DataFrame): DataFrame of human behavioral data.
    """
    if human_data is None:
        print("Skipping mean score plots due to missing human data.")
        return

    # Will collect averaged curves for both networks to plot side-by-side later
    avg_store = {
        'emonet': {'human_x': None, 'human_y': None, 'models': {}},
        'memnet': {'human_x': None, 'human_y': None, 'models': {}},
    }

    for net in NETWORKS:
        # Select relevant conditions and human rating column for each network
        if net == 'emonet':
            conditions = ['valence-4', 'valence-2', 'alpha0', 'valence+2', 'valence+4']
            human_col = 'ValenceRating'
        else:
            conditions = ['mem-4', 'mem-2', 'alpha0', 'mem+2', 'mem+4']
            human_col = 'MemorabilityRating'

        # Filter and aggregate human data
        human_net_df = human_data[human_data['Condition'].isin(conditions)]
        human_means = human_net_df.groupby('Alpha')[human_col].mean().reindex(ALPHA_LEVELS_NUM)
        human_norm = normalize_scores(human_means)
        df_human_plot = pd.DataFrame({
            'Alpha': ALPHA_LEVELS_NUM,
            'NormalizedScore': human_norm,
            'Model': 'Human (mean)'
        })

        # -----------------------------
        # Plot 1: Per-Subject Comparison (UNCHANGED)
        # -----------------------------
        fig_sub, axs_sub = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axs_flat = axs_sub.flatten()

        for idx, sub in enumerate(SUBJECTS):
            ax = axs_flat[idx]
            # Plot human mean
            ax.plot(
                df_human_plot['Alpha'],
                df_human_plot['NormalizedScore'],
                marker='o',
                label='Human (mean)' if idx == 0 else None
            )

            # Plot each model's normalized mean scores for this subject
            for model in MODELS:
                if model_data[net][model][sub] is None:
                    continue
                means = [np.mean(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                norm_means = normalize_scores(means)
                ax.plot(
                    ALPHA_LEVELS_NUM,
                    norm_means,
                    marker='o',
                    label=model if idx == 0 else None
                )
            ax.set_title(f"Subject {sub:02d}")
            ax.set_xticks(ALPHA_LEVELS_NUM)

        fig_sub.supxlabel("Alpha Level", fontweight='bold')
        fig_sub.supylabel("Normalized Mean Score", fontweight='bold')
        fig_sub.suptitle(f"{net.capitalize()} Network: Model vs. Human Scores by Subject", fontsize=20)
        fig_sub.legend(*axs_flat[0].get_legend_handles_labels(), title="Model", loc='center right')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        out_path = OUTPUT_DIR / f"scores_{net}_by_subject.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig_sub)
        print(f"📈 Saved plot: {out_path}")

        # -----------------------------
        # Collect data for Averaged Across Subjects (to plot side-by-side later)
        # -----------------------------
        # Human series: for MemNet averaged plot ONLY, drop α = -4 and +4
        human_x = df_human_plot['Alpha'].to_numpy()
        human_y = df_human_plot['NormalizedScore'].to_numpy()
        if net == 'memnet':
            keep_mask = ~np.isin(human_x, [-4, 4])
            human_x_plot = human_x[keep_mask]
            human_y_plot = human_y[keep_mask]
        else:
            human_x_plot = human_x
            human_y_plot = human_y

        avg_store[net]['human_x'] = human_x_plot
        avg_store[net]['human_y'] = human_y_plot

        # Model curves averaged across subjects
        for model in MODELS:
            subj_norms = []
            for sub in SUBJECTS:
                if model_data[net][model][sub] is None:
                    continue
                means = [np.mean(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                subj_norms.append(normalize_scores(means))
            if subj_norms:
                avg_norm = np.mean(np.vstack(subj_norms), axis=0)  # (5,)
                avg_store[net]['models'][model] = avg_norm
            else:
                avg_store[net]['models'][model] = None  # no data for this model

    # -----------------------------
    # Plot 2: Averaged Across Subjects — SIDE BY SIDE (EmoNet | MemNet)
    # -----------------------------
    fig_avg, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left panel: EmoNet
    ax = axs[0]
    ax.plot(avg_store['emonet']['human_x'], avg_store['emonet']['human_y'],
            marker='o', label='Human (mean)')
    for model in MODELS:
        avg_norm = avg_store['emonet']['models'][model]
        if avg_norm is not None:
            ax.plot(ALPHA_LEVELS_NUM, avg_norm, marker='o', label=model)
    ax.set_title("EmoNet Network: Scores Averaged Across Subjects", fontsize=16)
    ax.set_xlabel("Alpha Level")
    ax.set_ylabel("Normalized Mean Score")
    ax.set_xticks(ALPHA_LEVELS_NUM)
    ax.legend(title="Model")

    # Right panel: MemNet (human α = ±4 removed)
    ax = axs[1]
    ax.plot(avg_store['memnet']['human_x'], avg_store['memnet']['human_y'],
            marker='o', label='Human (mean)')
    for model in MODELS:
        avg_norm = avg_store['memnet']['models'][model]
        if avg_norm is not None:
            ax.plot(ALPHA_LEVELS_NUM, avg_norm, marker='o', label=model)
    ax.set_title("MemNet Network: Scores Averaged Across Subjects", fontsize=16)
    ax.set_xlabel("Alpha Level")
    ax.set_xticks(ALPHA_LEVELS_NUM)
    ax.legend(title="Model")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "scores_averaged_side_by_side.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig_avg)
    print(f"📈 Saved side-by-side averaged plot: {out_path}")




def plot_normalized_median_scores(model_data, human_data):
    """
    Generates plots comparing normalized median scores of models and humans.
    1. By Subject: A 2x2 grid for each subject.
    2. Averaged: A single plot with model scores averaged over subjects.

    Args:
        model_data (dict): Nested dictionary of model scores.
        human_data (pd.DataFrame): DataFrame of human behavioral data.
    """
    if human_data is None:
        print("Skipping median score plots due to missing human data.")
        return

    for net in NETWORKS:
        # Select relevant conditions and human rating column for each network
        if net == 'emonet':
            conditions = ['valence-4', 'valence-2', 'alpha0', 'valence+2', 'valence+4']
            human_col = 'ValenceRating'
        else:
            conditions = ['mem-4', 'mem-2', 'alpha0', 'mem+2', 'mem+4']
            human_col = 'MemorabilityRating'

        # Filter and aggregate human data (median per alpha)
        human_net_df = human_data[human_data['Condition'].isin(conditions)]
        human_medians = human_net_df.groupby('Alpha')[human_col].median().reindex(ALPHA_LEVELS_NUM)
        human_norm = normalize_scores(human_medians)
        df_human_plot = pd.DataFrame({
            'Alpha': ALPHA_LEVELS_NUM,
            'NormalizedScore': human_norm,
            'Model': 'Human (median)'
        })

        # Plot 1: Per-Subject Comparison (medians)
        fig_sub, axs_sub = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axs_flat = axs_sub.flatten()

        for idx, sub in enumerate(SUBJECTS):
            ax = axs_flat[idx]
            # Human median curve (global across subjects for visual anchor)
            ax.plot(df_human_plot['Alpha'], df_human_plot['NormalizedScore'],
                    marker='o', label='Human (median)' if idx == 0 else None)

            # Each model's normalized median scores for this subject
            for model in MODELS:
                if model_data[net][model][sub] is None:
                    continue
                medians = [np.median(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                norm_medians = normalize_scores(medians)
                ax.plot(ALPHA_LEVELS_NUM, norm_medians, marker='o',
                        label=model if idx == 0 else None)

            ax.set_title(f"Subject {sub:02d}")
            ax.set_xticks(ALPHA_LEVELS_NUM)

        fig_sub.supxlabel("Alpha Level", fontweight='bold')
        fig_sub.supylabel("Normalized Median Score", fontweight='bold')
        fig_sub.suptitle(f"{net.capitalize()} Network: Model vs. Human (Median) by Subject", fontsize=20)
        fig_sub.legend(*axs_flat[0].get_legend_handles_labels(),
                       title="Model", loc='center right')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        out_path = OUTPUT_DIR / f"scores_{net}_median_by_subject.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig_sub)
        print(f"📈 Saved plot: {out_path}")

        # Plot 2: Averaged Across Subjects (medians)
        fig_avg, ax_avg = plt.subplots(figsize=(10, 7))
        ax_avg.plot(df_human_plot['Alpha'], df_human_plot['NormalizedScore'],
                    marker='o', label='Human (median)')

        for model in MODELS:
            subj_norms = []
            for sub in SUBJECTS:
                if model_data[net][model][sub] is None:
                    continue
                medians = [np.median(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                subj_norms.append(normalize_scores(medians))
            if subj_norms:
                avg_norm = np.mean(np.vstack(subj_norms), axis=0)  # average of normalized subject medians
                ax_avg.plot(ALPHA_LEVELS_NUM, avg_norm, marker='o', label=model)

        ax_avg.set_title(f"{net.capitalize()} Network: Median Scores Averaged Across Subjects", fontsize=20)
        ax_avg.set_xlabel("Alpha Level")
        ax_avg.set_ylabel("Normalized Median Score")
        ax_avg.set_xticks(ALPHA_LEVELS_NUM)
        ax_avg.legend(title="Model")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"scores_{net}_median_averaged.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig_avg)
        print(f"📈 Saved plot: {out_path}")


def plot_slope_histograms(model_data):
    """
    Generates histograms of response slopes for each model.
    1. By Subject: A 2x2 grid of histograms for each subject.
    2. Pooled: A single histogram with data pooled across subjects.

    Args:
        model_data (dict): Nested dictionary of model scores.
    """
    for net in NETWORKS:
        fig_sub, axs_sub = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axs_flat = axs_sub.flatten()
        pooled_slopes = {model: [] for model in MODELS}

        # Per-Subject Histograms
        for idx, sub in enumerate(SUBJECTS):
            ax = axs_flat[idx]
            subject_slopes = {}
            for model in MODELS:
                if model_data[net][model][sub] is None:
                    continue
                # Stack scores for all alpha levels, shape: (n_images, n_alphas)
                scores = np.vstack([model_data[net][model][sub][alpha] for alpha in ALPHA_LEVELS_STR]).T
                # Compute slope for each image using linear fit
                slopes = [np.polyfit(ALPHA_LEVELS_NUM, img_scores, 1)[0] for img_scores in scores]
                subject_slopes[model] = slopes
                pooled_slopes[model].extend(slopes)

            # Plot histogram for each model
            all_subj = np.concatenate(list(subject_slopes.values()))
            bins = np.linspace(all_subj.min(), all_subj.max(), 30)
            for model, slopes in subject_slopes.items():
                ax.hist(slopes, bins=bins, alpha=0.6, label=model, edgecolor='black')

            ax.set_title(f"Subject {sub:02d}")
            ax.legend()

        fig_sub.supxlabel("Response Slope (Δ Score per α‑unit)", fontweight='bold')
        fig_sub.supylabel("Number of Images", fontweight='bold')
        fig_sub.suptitle(f"{net.capitalize()} Network: Response Slope Distribution by Subject",
                         fontsize=20)
        plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])

        out_path = OUTPUT_DIR / f"slopes_{net}_by_subject.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig_sub)
        print(f"📊 Saved plot: {out_path}")

        # Pooled Histograms
        fig_pool, ax_pool = plt.subplots(figsize=(10, 7))
        all_pooled = np.concatenate(list(pooled_slopes.values()))
        bins = np.linspace(all_pooled.min(), all_pooled.max(), 40)
        for model, slopes in pooled_slopes.items():
            ax_pool.hist(slopes, bins=bins, alpha=0.6, label=model, edgecolor='black')

        ax_pool.set_title(f"{net.capitalize()} Network: Pooled Response Slope Distribution", fontsize=20)
        ax_pool.set_xlabel("Response Slope (Δ Score per α‑unit)")
        ax_pool.set_ylabel("Number of Images")
        ax_pool.legend(title="Model")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"slopes_{net}_pooled.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig_pool)
        print(f"📊 Saved plot: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# --- Additional Rate‑of‑Change & Comparative Plots ---
# ─────────────────────────────────────────────────────────────────────────────

def plot_rate_of_change_subjects():
    """
    Per‑subject rate‑of‑change histograms (EmoNet vs MemNet).
    For each subject, plots the distribution of rate-of-change for both models and both networks.
    """
    alphas = [-4, -3, -2, 2, 3, 4]
    for sub in SUBJECTS:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for ax, net in zip(axs, NETWORKS):
            # Load VDVAE model scores
            dv_path = MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_vdvae_sub{sub:02d}.pkl"
            dv = pickle.load(open(dv_path, "rb"))
            base_v = np.array(dv["alpha_0"])
            # Compute rate of change for each alpha (relative to base)
            rates_v = np.concatenate([np.array(dv[f"alpha_{a}"]) / base_v for a in alphas])

            # Load Versatile model scores
            dvs_path = MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_versatile_sub{sub:02d}.pkl"
            dvs = pickle.load(open(dvs_path, "rb"))
            base_vs = np.array(dvs["alpha_0"])
            rates_vs = np.concatenate([np.array(dvs[f"alpha_{a}"]) / base_vs for a in alphas])

            # Plot histograms for both models
            ax.hist(rates_v,  bins=50, alpha=0.6, label="VDVAE",    edgecolor="black")
            ax.hist(rates_vs, bins=50, alpha=0.6, label="Versatile", edgecolor="black")
            ax.set_title(f"{net.capitalize()} – Subject {sub:02d}")
            ax.set_xlabel("Rate of Change per α‑unit")
            if net == NETWORKS[0]:
                ax.set_ylabel("Count")
            ax.legend()

        plt.suptitle(f"Subject {sub:02d} Rate‑of‑Change: EmoNet vs MemNet", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = OUTPUT_DIR / f"roc_subject_{sub:02d}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"📈 Saved ROC per‑subject plot: {out_path}")


def plot_rate_of_change_overall():
    """
    Overall rate‑of‑change histograms across all subjects.
    For each network, pools all subjects' rate-of-change values and plots distributions for both models.
    """
    alphas = [-4, -3, -2, 2, 3, 4]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, net in zip(axs, NETWORKS):
        all_v, all_vs = [], []
        for sub in SUBJECTS:
            # Load VDVAE model scores
            dv = pickle.load(open(
                MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_vdvae_sub{sub:02d}.pkl", "rb"
            ))
            base_v = np.array(dv["alpha_0"])
            all_v.append(np.concatenate([np.array(dv[f"alpha_{a}"]) / base_v for a in alphas]))

            # Load Versatile model scores
            dvs = pickle.load(open(
                MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_versatile_sub{sub:02d}.pkl", "rb"
            ))
            base_vs = np.array(dvs["alpha_0"])
            all_vs.append(np.concatenate([np.array(dvs[f"alpha_{a}"]) / base_vs for a in alphas]))

        rates_v = np.concatenate(all_v)
        rates_vs = np.concatenate(all_vs)

        # Plot histograms for both models
        ax.hist(rates_v,  bins=60, alpha=0.6, label="VDVAE",    edgecolor="black")
        ax.hist(rates_vs, bins=60, alpha=0.6, label="Versatile", edgecolor="black")
        ax.set_title(f"Overall ({net.capitalize()})")
        ax.set_xlabel("Rate of Change per α‑unit")
        if net == NETWORKS[0]:
            ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("Overall Rate‑of‑Change Across All Subjects", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUTPUT_DIR / "roc_overall.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved overall ROC plot: {out_path}")


def plot_rate_of_change_vs_human(human_df, prob_per_bin: bool = False):
    """
    Side-by-side plots (EmoNet left, MemNet right) comparing model ROC vs human.
    - Uses real data only (no bootstrapping).
    - MemNet ONLY: removes human samples at α = -4 and α = +4.
    - prob_per_bin=True (default): each histogram's bar heights sum to 1.
      If False: use probability density (area = 1).

    Args:
        human_df (pd.DataFrame): Human behavioral data.
        prob_per_bin (bool): Normalize bars to sum to 1 (default) or use density.
    """
    # Map condition strings to alpha values
    alpha_map = {
        'valence-4': -4, 'valence-2': -2, 'alpha0': 0, 'valence+2': 2, 'valence+4': 4,
        'mem-4': -4, 'mem-2': -2, 'mem+2': 2, 'mem+4': 4
    }
    human_df = human_df.copy()
    human_df['alpha'] = human_df['Condition'].map(alpha_map)

    # --- Collect human ROC values with their alphas ---
    all_val, all_val_alpha = [], []
    all_mem, all_mem_alpha = [], []
    for sub in SUBJECTS:
        subdf = human_df[human_df['SubjectID'] == sub]

        base_v = subdf.loc[subdf['alpha'] == 0, 'ValenceRating'].mean()
        base_m = subdf.loc[subdf['alpha'] == 0, 'MemorabilityRating'].mean()

        v_rows = subdf[(subdf['alpha'] != 0) & subdf['ValenceRating'].notna()]
        if base_v and not np.isnan(base_v):
            all_val.extend((v_rows['ValenceRating'] / base_v).to_numpy())
            all_val_alpha.extend(v_rows['alpha'].to_numpy())

        m_rows = subdf[(subdf['alpha'] != 0) & subdf['MemorabilityRating'].notna()]
        if base_m and not np.isnan(base_m):
            all_mem.extend((m_rows['MemorabilityRating'] / base_m).to_numpy())
            all_mem_alpha.extend(m_rows['alpha'].to_numpy())

    human_val = np.asarray(all_val, dtype=float)
    human_val_alpha = np.asarray(all_val_alpha, dtype=int) if all_val_alpha else np.array([], dtype=int)
    human_mem = np.asarray(all_mem, dtype=float)
    human_mem_alpha = np.asarray(all_mem_alpha, dtype=int) if all_mem_alpha else np.array([], dtype=int)

    alphas = [-4, -3, -2, 2, 3, 4]

    def weights_for(x):
        return np.ones_like(x, dtype=float) / x.size if (x is not None and x.size) else np.array([])

    # --- Helper to get model ROC arrays for a network ---
    def get_model_rates(net: str):
        all_v, all_vs = [], []
        for sub in SUBJECTS:
            dv_path  = MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_vdvae_sub{sub:02d}.pkl"
            dvs_path = MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_versatile_sub{sub:02d}.pkl"
            dv  = pickle.load(open(dv_path, "rb"))
            dvs = pickle.load(open(dvs_path, "rb"))

            base_v  = np.asarray(dv["alpha_0"],  dtype=float)
            base_vs = np.asarray(dvs["alpha_0"], dtype=float)

            all_v.append(np.concatenate([np.asarray(dv[f"alpha_{a}"],  dtype=float) / base_v  for a in alphas]))
            all_vs.append(np.concatenate([np.asarray(dvs[f"alpha_{a}"], dtype=float) / base_vs for a in alphas]))
        return np.concatenate(all_v), np.concatenate(all_vs)

    # --- Gather data for both panels first (so we can share bins/xlim) ---
    rates_v_emo, rates_vs_emo = get_model_rates('emonet')
    human_pool_emo, human_label_emo = human_val, 'Human Valence'

    rates_v_mem, rates_vs_mem = get_model_rates('memnet')
    if human_mem_alpha.size:
        keep = (human_mem_alpha != -4) & (human_mem_alpha != 4)
        human_pool_mem = human_mem[keep]
    else:
        human_pool_mem = human_mem
    human_label_mem = 'Human Memorability'

    # If human pool is empty after filtering for memnet, warn but still plot models
    if human_pool_mem.size == 0:
        print("⚠️ No human samples for MemNet after filtering (±4 removed). Plotting models only.")

    # --- Global bins across both panels for comparability ---
    global_combined = np.concatenate([
        rates_v_emo, rates_vs_emo, human_pool_emo,
        rates_v_mem, rates_vs_mem,
        human_pool_mem if human_pool_mem.size else np.array([])
    ])
    bins = np.histogram_bin_edges(global_combined, bins=60)

    # --- Create side-by-side figure ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left: EmoNet
    ax = axs[0]
    if prob_per_bin:
        ax.hist(rates_v_emo,    bins=bins, weights=weights_for(rates_v_emo),    alpha=0.4,
                label="EmoNet-VDVAE", edgecolor="black")
        ax.hist(rates_vs_emo,   bins=bins, weights=weights_for(rates_vs_emo),   alpha=0.4,
                label="EmoNet-Versatile", edgecolor="black")
        ax.hist(human_pool_emo, bins=bins, weights=weights_for(human_pool_emo), alpha=0.4,
                label=human_label_emo, edgecolor="black")
        ax.set_ylabel("Probability per bin (sums to 1)")
    else:
        ax.hist(rates_v_emo,    bins=bins, density=True, alpha=0.4, label="EmoNet-VDVAE",    edgecolor="black")
        ax.hist(rates_vs_emo,   bins=bins, density=True, alpha=0.4, label="EmoNet-Versatile", edgecolor="black")
        ax.hist(human_pool_emo, bins=bins, density=True, alpha=0.4, label=human_label_emo,   edgecolor="black")
        ax.set_ylabel("Probability density")
    ax.set_title("EmoNet")
    ax.set_xlabel("Rate of Change per α-unit")
    ax.legend()

    # Right: MemNet
    ax = axs[1]
    if prob_per_bin:
        ax.hist(rates_v_mem,    bins=bins, weights=weights_for(rates_v_mem),    alpha=0.4,
                label="MemNet-VDVAE", edgecolor="black")
        ax.hist(rates_vs_mem,   bins=bins, weights=weights_for(rates_vs_mem),   alpha=0.4,
                label="MemNet-Versatile", edgecolor="black")
        if human_pool_mem.size:
            ax.hist(human_pool_mem, bins=bins, weights=weights_for(human_pool_mem), alpha=0.4,
                    label=human_label_mem + " (±4 removed)", edgecolor="black")
    else:
        ax.hist(rates_v_mem,    bins=bins, density=True, alpha=0.4, label="MemNet-VDVAE",    edgecolor="black")
        ax.hist(rates_vs_mem,   bins=bins, density=True, alpha=0.4, label="MemNet-Versatile", edgecolor="black")
        if human_pool_mem.size:
            ax.hist(human_pool_mem, bins=bins, density=True, alpha=0.4,
                    label=human_label_mem + " (±4 removed)", edgecolor="black")
    ax.set_title("MemNet")
    ax.set_xlabel("Rate of Change per α-unit")
    ax.legend()

    fig.suptitle("Overall Rate-of-Change: Models vs. Human", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = OUTPUT_DIR / "roc_vs_human_side_by_side.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved side-by-side ROC vs human plot: {out_path}")





def plot_alpha0_correlations(model_data):
    """
    Side-by-side barplots for correlations with α=0.
    Left: VDVAE; Right: Versatile Diffusion.
    For each panel, bars show mean Pearson r (± SEM) between alpha_0 and:
      - image_to_image (i2i)
      - alpha in {-4, -2, +2, +4}
    Two bars per x-tick: EmoNet vs MemNet; values averaged across subjects.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # What to compare against alpha_0 (label, key in PKL)
    comparisons = [
        ("i2i", "i2i_key"),           # will resolve per model
        ("alpha_-4", "alpha_-4"),
        ("alpha_-2", "alpha_-2"),
        ("alpha_2",  "alpha_2"),
        ("alpha_4",  "alpha_4"),
    ]
    # pretty x-tick labels
    xtick_labels = ["i2i", "α−4", "α−2", "α+2", "α+4"]

    # model-specific key for the i2i baseline
    i2i_key_for_model = {
        "vdvae": "vdvae_image_to_image",
        "versatile": "clip_image_to_image"
    }

    def compute_means_sems_for_model(model: str):
        """Return dicts: means[net], sems[net] as lists aligned with `comparisons`."""
        stats = {net: {lbl: [] for (lbl, _) in comparisons} for net in NETWORKS}

        for net in NETWORKS:  # 'emonet', 'memnet'
            for sub in SUBJECTS:
                d = model_data[net][model].get(sub)
                if not d:
                    continue
                base = d.get("alpha_0", None)
                if base is None:
                    continue
                for lbl, key in comparisons:
                    key_resolved = i2i_key_for_model[model] if key == "i2i_key" else key
                    comp = d.get(key_resolved, None)
                    if comp is None:
                        continue
                    r = pearson_safe(base, comp)
                    if not np.isnan(r):
                        stats[net][lbl].append(r)

        means = {net: [] for net in NETWORKS}
        sems  = {net: [] for net in NETWORKS}
        for net in NETWORKS:
            for lbl, _ in comparisons:
                vals = np.asarray(stats[net][lbl], dtype=float)
                if vals.size == 0:
                    means[net].append(np.nan)
                    sems[net].append(np.nan)
                else:
                    means[net].append(np.nanmean(vals))
                    sems[net].append(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals))))
        return means, sems

    # Prepare figure with two panels: left=VDVAE, right=Versatile
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    panel_defs = [("vdvae", "VDVAE"), ("versatile", "Versatile Diffusion")]

    x = np.arange(len(comparisons))
    width = 0.38

    for ax, (model_key, model_title) in zip(axs, panel_defs):
        means, sems = compute_means_sems_for_model(model_key)

        em_col = ax.bar(x - width/2, means["emonet"], width,
                        yerr=sems["emonet"], capsize=4, label="EmoNet")
        mm_col = ax.bar(x + width/2, means["memnet"], width,
                        yerr=sems["memnet"], capsize=4, label="MemNet")

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel("Pearson r (α=0 vs …)")
        ax.set_title(model_title)
        ax.set_ylim(-0.2, 1.0)
        ax.legend(title="Assessor")

        # annotate bars with the actual score (mean Pearson r)
        ymin, ymax = ax.get_ylim()
        pad = 0.02 * (ymax - ymin)
        for net, bars in zip(["emonet", "memnet"], [em_col, mm_col]):
            for idx, b in enumerate(bars):
                r_val = means[net][idx]
                label = "NA" if np.isnan(r_val) else f"{r_val:.2f}"
                ax.text(
                    b.get_x() + b.get_width()/2,
                    (0 if np.isnan(b.get_height()) else b.get_height()) + pad,
                    label,
                    ha="center", va="bottom", fontsize=9
                )

    fig.suptitle("Correlation vs α=0 (averaged across subjects)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = OUTPUT_DIR / "correlations_alpha0_side_by_side.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📊 Saved: {out_path}")



# =============================================================================
# === Statistics & Report Generation ==========================================
# =============================================================================

STAT_DIR = BASE_DIR / "results" / "statistics"

def _safe_len(x):
    return 0 if x is None else (len(x) if hasattr(x, "__len__") else 0)

def _cohen_d_one_sample(sample, mu=0.0):
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return np.nan
    return (sample.mean() - mu) / sample.std(ddof=1)

def _cohen_d_independent(a, b):
    """Hedges g (unbiased) for two independent samples (Welch t-compatible)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    na, nb = a.size, b.size
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    # Pooled SD (unweighted for Welch: use sqrt((sa2+sb2)/2))
    sp = np.sqrt((sa2 + sb2) / 2.0) if (sa2>0 or sb2>0) else np.nan
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sp
    # Small-sample correction to Hedges g
    J = 1 - (3 / (4*(na+nb) - 9))
    return d * J

def _describe_array(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, q1=np.nan,
                    median=np.nan, q3=np.nan, max=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size>1 else 0.0,
        min=float(np.min(x)),
        q1=float(np.quantile(x, 0.25)),
        median=float(np.median(x)),
        q3=float(np.quantile(x, 0.75)),
        max=float(np.max(x)),
    )

def _model_curve_across_subjects(model_data, net, model, reducer="mean"):
    """Return a 5‑length vector across ALPHA_LEVELS_STR averaged across subjects (after per‑subject normalization)."""
    subj_norms = []
    for sub in SUBJECTS:
        d = model_data[net][model].get(sub)
        if not d:
            continue
        try:
            vals = [np.mean(d[a]) for a in ALPHA_LEVELS_STR]
        except Exception:
            continue
        subj_norms.append(normalize_scores(vals))
    if not subj_norms:
        return None
    arr = np.vstack(subj_norms)
    return np.nanmean(arr, axis=0) if reducer == "mean" else np.nanmedian(arr, axis=0)

def _human_curve(human_df, net):
    if human_df is None:
        return None
    if net == 'emonet':
        conditions = ['valence-4', 'valence-2', 'alpha0', 'valence+2', 'valence+4']
        col = 'ValenceRating'
    else:
        conditions = ['mem-4', 'mem-2', 'alpha0', 'mem+2', 'mem+4']
        col = 'MemorabilityRating'
    df = human_df[human_df['Condition'].isin(conditions)]
    means = df.groupby('Alpha')[col].mean().reindex(ALPHA_LEVELS_NUM)
    return normalize_scores(means)

def _collect_image_slopes_for(model_data, net, model):
    """Return per-image slopes pooled across subjects (fit score ~ alpha)."""
    slopes_all = []
    for sub in SUBJECTS:
        d = model_data[net][model].get(sub)
        if not d:
            continue
        # Stack scores n_images x 5
        try:
            M = np.vstack([np.asarray(d[a], dtype=float) for a in ALPHA_LEVELS_STR]).T
        except Exception:
            continue
        # compute slope per image (linear fit)
        for row in M:
            if np.all(np.isfinite(row)):
                slope = np.polyfit(ALPHA_LEVELS_NUM, row, 1)[0]
                slopes_all.append(slope)
    return np.asarray(slopes_all, dtype=float) if slopes_all else np.array([], dtype=float)

def _alpha_descriptives(model_data, human_df):
    """Descriptive stats by alpha for (network × model × subject) and human means."""
    rows = []
    # Human
    if human_df is not None:
        for net in NETWORKS:
            if net == 'emonet':
                conds = ['valence-4', 'valence-2', 'alpha0', 'valence+2', 'valence+4']
                col = 'ValenceRating'
            else:
                conds = ['mem-4', 'mem-2', 'alpha0', 'mem+2', 'mem+4']
                col = 'MemorabilityRating'
            df = human_df[human_df['Condition'].isin(conds)]
            for alpha in ALPHA_LEVELS_NUM:
                vals = df[df['Alpha']==alpha][col].to_numpy()
                desc = _describe_array(vals)
                rows.append(dict(kind="human", network=net, model="human", subject="all",
                                 alpha=int(alpha), **desc))
    # Models
    for net in NETWORKS:
        for model in MODELS:
            for sub in SUBJECTS:
                d = model_data[net][model].get(sub)
                if not d:
                    continue
                for a_str, a_num in zip(ALPHA_LEVELS_STR, ALPHA_LEVELS_NUM):
                    vals = np.asarray(d.get(a_str, []), dtype=float)
                    desc = _describe_array(vals)
                    rows.append(dict(kind="model", network=net, model=model, subject=f"{sub:02d}",
                                     alpha=int(a_num), **desc))
    return pd.DataFrame(rows)

def _slope_tests(model_data):
    """One‑sample tests of slope vs 0, and model‑vs‑model slope comparisons, per network."""
    rows_1samp = []
    rows_cmp = []
    for net in NETWORKS:
        # gather slopes per model
        per_model = {m: _collect_image_slopes_for(model_data, net, m) for m in MODELS}
        # one‑sample tests
        for m, arr in per_model.items():
            arr = arr[np.isfinite(arr)]
            if arr.size >= 2:
                t, p = ttest_1samp(arr, popmean=0.0, alternative='two-sided')
                d = _cohen_d_one_sample(arr, 0.0)
                # simple normality check (cap size for shapiro due to limits)
                sw_n = min(arr.size, 5000)
                sw_p = np.nan
                try:
                    sw_p = shapiro(arr[:sw_n]).pvalue
                except Exception:
                    pass
                rows_1samp.append(dict(network=net, model=m, n=int(arr.size),
                                       mean=float(arr.mean()), std=float(arr.std(ddof=1)),
                                       t=float(t), p=float(p), cohen_d=float(d),
                                       shapiro_p=float(sw_p) if np.isfinite(sw_p) else np.nan))
            else:
                rows_1samp.append(dict(network=net, model=m, n=int(arr.size),
                                       mean=np.nan, std=np.nan, t=np.nan, p=np.nan,
                                       cohen_d=np.nan, shapiro_p=np.nan))
        # model vs model (Welch)
        if len(MODELS) == 2:
            a = per_model[MODELS[0]]
            b = per_model[MODELS[1]]
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            if a.size >= 2 and b.size >= 2:
                t, p = ttest_ind(a, b, equal_var=False)
                d = _cohen_d_independent(a, b)
            else:
                t = p = d = np.nan
            rows_cmp.append(dict(network=net, model_A=MODELS[0], model_B=MODELS[1],
                                 n_A=int(a.size), n_B=int(b.size), t=float(t) if np.isfinite(t) else np.nan,
                                 p=float(p) if np.isfinite(p) else np.nan,
                                 cohen_d=float(d) if np.isfinite(d) else np.nan))
    return pd.DataFrame(rows_1samp), pd.DataFrame(rows_cmp)

def _curve_correlations(model_data, human_df):
    """Pearson r between averaged model curves and human curves per network."""
    rows = []
    for net in NETWORKS:
        human = _human_curve(human_df, net)
        for model in MODELS:
            curve = _model_curve_across_subjects(model_data, net, model, reducer="mean")
            if curve is None or human is None:
                r = np.nan
            else:
                # align (both length 5). For memnet, also report r with ±4 removed.
                r = pearson_safe(curve, human)
            rows.append(dict(network=net, model=model, pearson_r=float(r) if np.isfinite(r) else np.nan))
        # MemNet optional: r with ±4 removed
        if net == 'memnet' and human is not None:
            keep = ~np.isin(ALPHA_LEVELS_NUM, [-4, 4])
            for model in MODELS:
                curve = _model_curve_across_subjects(model_data, net, model, reducer="mean")
                if curve is None:
                    r2 = np.nan
                else:
                    r2 = pearson_safe(curve[keep], human[keep])
                rows.append(dict(network=net, model=model+" (±4 removed)", pearson_r=float(r2) if np.isfinite(r2) else np.nan))
    return pd.DataFrame(rows)

def _alpha_linear_trend_tests(model_data):
    """
    For each (network, model, subject), regress mean score across alphas on alpha level.
    Returns slope, intercept, r, p, stderr. Also aggregates grand means across subjects.
    """
    rows = []
    for net in NETWORKS:
        for model in MODELS:
            subj_slopes = []
            for sub in SUBJECTS:
                d = model_data[net][model].get(sub)
                if not d:
                    continue
                y = []
                for a in ALPHA_LEVELS_STR:
                    arr = np.asarray(d.get(a, []), dtype=float)
                    y.append(np.nanmean(arr) if arr.size else np.nan)
                y = np.asarray(y, dtype=float)
                if np.any(np.isfinite(y)):
                    # handle potential NaNs by masking
                    mask = np.isfinite(ALPHA_LEVELS_NUM) & np.isfinite(y)
                    if mask.sum() >= 2:
                        res = linregress(ALPHA_LEVELS_NUM[mask], y[mask])
                        rows.append(dict(network=net, model=model, subject=f"{sub:02d}",
                                         slope=res.slope, intercept=res.intercept,
                                         r=res.rvalue, p=res.pvalue, stderr=res.stderr))
                        subj_slopes.append(res.slope)
            # aggregate slope across subjects
            subj_slopes = np.asarray(subj_slopes, dtype=float)
            if subj_slopes.size >= 2:
                t, p = ttest_1samp(subj_slopes, 0.0)
                rows.append(dict(network=net, model=model, subject="MEAN",
                                 slope=float(subj_slopes.mean()),
                                 intercept=np.nan, r=np.nan, p=float(p),
                                 stderr=float(subj_slopes.std(ddof=1)/np.sqrt(subj_slopes.size))))
            else:
                rows.append(dict(network=net, model=model, subject="MEAN",
                                 slope=np.nan, intercept=np.nan, r=np.nan, p=np.nan, stderr=np.nan))
    return pd.DataFrame(rows)

def generate_statistics_report(model_data, human_data):
    """
    Builds and saves a Markdown report summarizing core statistics.
    """
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Descriptive stats by alpha
    df_desc = _alpha_descriptives(model_data, human_data)

    # 2) Slopes: one-sample vs 0 and model-vs-model
    df_slope_1samp, df_slope_cmp = _slope_tests(model_data)

    # 3) Curve correlations with human
    df_corr = _curve_correlations(model_data, human_data)

    # 4) Linear trend tests on mean per subject
    df_trend = _alpha_linear_trend_tests(model_data)

    # 5) Save CSVs alongside the report for convenience
    csv_desc = STAT_DIR / "descriptives_by_alpha.csv"
    csv_1s   = STAT_DIR / "slope_one_sample.csv"
    csv_cmp  = STAT_DIR / "slope_model_vs_model.csv"
    csv_corr = STAT_DIR / "model_vs_human_curve_correlations.csv"
    csv_trnd = STAT_DIR / "alpha_linear_trend_tests.csv"
    df_desc.to_csv(csv_desc, index=False)
    df_slope_1samp.to_csv(csv_1s, index=False)
    df_slope_cmp.to_csv(csv_cmp, index=False)
    df_corr.to_csv(csv_corr, index=False)
    df_trend.to_csv(csv_trnd, index=False)

    # 6) Compose Markdown
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    md_path = STAT_DIR / f"report_statistics_{ts}.md"

    lines = []
    lines.append("# Reconstruction Statistics Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Subjects: {SUBJECTS}")
    lines.append(f"- Models: {MODELS}")
    lines.append(f"- Networks: {NETWORKS}")
    lines.append(f"- Alpha levels: {list(ALPHA_LEVELS_NUM)}")
    lines.append("")
    # Descriptives summary (aggregated over subjects per model/network/alpha)
    lines.append("## Descriptive Statistics by Alpha (Models, pooled across subjects)")
    df_desc_models = (df_desc[df_desc["kind"]=="model"]
                      .assign(subject=lambda d: d["subject"].astype(str))
                      .groupby(["network","model","alpha"], as_index=False)
                      .agg(n=("n","sum"), mean=("mean","mean"), std=("std","mean"),
                           min=("min","mean"), q1=("q1","mean"),
                           median=("median","mean"), q3=("q3","mean"), max=("max","mean")))
    lines.append(df_desc_models.to_markdown(index=False))
    lines.append("")
    if human_data is not None:
        lines.append("## Human Ratings: Descriptive Statistics by Alpha")
        df_desc_h = df_desc[df_desc["kind"]=="human"].copy()
        lines.append(df_desc_h.to_markdown(index=False))
        lines.append("")
    # Slopes one-sample
    lines.append("## Per-Image Slope Tests (One-sample vs 0)")
    lines.append(df_slope_1samp.to_markdown(index=False))
    lines.append("")
    # Model vs Model
    lines.append("## Per-Image Slope Comparison: Model A vs Model B (Welch t-test)")
    lines.append(df_slope_cmp.to_markdown(index=False))
    lines.append("")
    # Curve correlations
    lines.append("## Correlation Between Averaged Model Curves and Human Curves")
    lines.append(df_corr.to_markdown(index=False))
    lines.append("")
    # Linear trend tests
    lines.append("## Linear Trend of Mean Score vs Alpha (Per Subject)")
    lines.append(df_trend.to_markdown(index=False))
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Descriptives CSV: `{csv_desc}`")
    lines.append(f"- Slope (one-sample) CSV: `{csv_1s}`")
    lines.append(f"- Slope (model-vs-model) CSV: `{csv_cmp}`")
    lines.append(f"- Model–Human curve correlations CSV: `{csv_corr}`")
    lines.append(f"- Linear trend tests CSV: `{csv_trnd}`")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📝 Statistics report written to: {md_path}")



def main():
    """
    Main execution function.
    Loads data, sets up plotting, and generates all analysis plots.
    """
    print("--- Starting Analysis Script ---")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup and load data
    setup_plotting_style()
    model_data, human_data = load_data()

    # Generate and save plots
    print("\n--- Generating Normalized Mean Score Plots ---")
    plot_normalized_mean_scores(model_data, human_data)

    print("\n--- Generating Normalized Median Score Plots ---")
    plot_normalized_median_scores(model_data, human_data)

    print("\n--- Generating Slope Distribution Plots ---")
    plot_slope_histograms(model_data)

    print("\n--- Generating Additional ROC & Comparative Plots ---")
    plot_rate_of_change_subjects()
    plot_rate_of_change_overall()
    plot_rate_of_change_vs_human(human_data)

    print("\n--- Generating α=0 correlation barplots (i2i and other alphas) ---")
    plot_alpha0_correlations(model_data)

    print("\n--- Running statistical analyses & generating Markdown report ---")
    generate_statistics_report(model_data, human_data)




    print("\n--- Script finished successfully ---")


if __name__ == "__main__":
    main()
