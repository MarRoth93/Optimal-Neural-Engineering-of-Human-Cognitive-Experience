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


def plot_normalized_mean_scores(model_data, human_data):
    """
    Generates plots comparing normalized mean scores of models and humans.
    1. By Subject: A 2x2 grid for each subject.
    2. Averaged: A single plot with model scores averaged over subjects.

    Args:
        model_data (dict): Nested dictionary of model scores.
        human_data (pd.DataFrame): DataFrame of human behavioral data.
    """
    if human_data is None:
        print("Skipping mean score plots due to missing human data.")
        return

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

        # Plot 1: Per-Subject Comparison
        fig_sub, axs_sub = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axs_flat = axs_sub.flatten()

        for idx, sub in enumerate(SUBJECTS):
            ax = axs_flat[idx]
            # Plot human mean
            ax.plot(df_human_plot['Alpha'], df_human_plot['NormalizedScore'],
                    marker='o', label='Human (mean)' if idx == 0 else None)
            # Plot each model's normalized mean scores for this subject
            for model in MODELS:
                if model_data[net][model][sub] is None:
                    continue
                means = [np.mean(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                norm_means = normalize_scores(means)
                ax.plot(ALPHA_LEVELS_NUM, norm_means,
                        marker='o', label=model if idx == 0 else None)
            ax.set_title(f"Subject {sub:02d}")
            ax.set_xticks(ALPHA_LEVELS_NUM)

        fig_sub.supxlabel("Alpha Level", fontweight='bold')
        fig_sub.supylabel("Normalized Mean Score", fontweight='bold')
        fig_sub.suptitle(f"{net.capitalize()} Network: Model vs. Human Scores by Subject", fontsize=20)
        fig_sub.legend(*axs_flat[0].get_legend_handles_labels(),
                       title="Model", loc='center right')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        out_path = OUTPUT_DIR / f"scores_{net}_by_subject.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig_sub)
        print(f"📈 Saved plot: {out_path}")

        # Plot 2: Averaged Across Subjects
        fig_avg, ax_avg = plt.subplots(figsize=(10, 7))
        ax_avg.plot(df_human_plot['Alpha'], df_human_plot['NormalizedScore'],
                    marker='o', label='Human (mean)')

        for model in MODELS:
            subj_norms = []
            for sub in SUBJECTS:
                if model_data[net][model][sub] is None:
                    continue
                means = [np.mean(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                subj_norms.append(normalize_scores(means))
            if subj_norms:
                avg_norm = np.mean(np.vstack(subj_norms), axis=0)
                ax_avg.plot(ALPHA_LEVELS_NUM, avg_norm, marker='o', label=model)

        ax_avg.set_title(f"{net.capitalize()} Network: Scores Averaged Across Subjects", fontsize=20)
        ax_avg.set_xlabel("Alpha Level")
        ax_avg.set_ylabel("Normalized Mean Score")
        ax_avg.set_xticks(ALPHA_LEVELS_NUM)
        ax_avg.legend(title="Model")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"scores_{net}_averaged.png"
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


def plot_rate_of_change_vs_human(human_df):
    """
    Overall rate‑of‑change vs. bootstrap‑resampled human, per network.
    For each network, compares model rate-of-change distributions to a bootstrapped sample of human data.

    Args:
        human_df (pd.DataFrame): DataFrame of human behavioral data.
    """
    # Map condition strings to alpha values
    alpha_map = {
        'valence-4': -4, 'valence-2': -2, 'alpha0': 0, 'valence+2': 2, 'valence+4': 4,
        'mem-4': -4, 'mem-2': -2, 'mem+2': 2, 'mem+4': 4
    }
    human_df['alpha'] = human_df['Condition'].map(alpha_map)

    # Gather all human rate-of-change values for valence and memorability
    all_val, all_mem = [], []
    for sub in SUBJECTS:
        subdf = human_df[human_df['SubjectID'] == sub]
        base_v = subdf[subdf['alpha'] == 0]['ValenceRating'].mean()
        base_m = subdf[subdf['alpha'] == 0]['MemorabilityRating'].mean()
        all_val.extend((subdf[subdf['alpha'] != 0]['ValenceRating'] / base_v).values)
        all_mem.extend((subdf[subdf['alpha'] != 0]['MemorabilityRating'] / base_m).values)

    human_val = np.array(all_val)
    human_mem = np.array(all_mem)
    rng = np.random.default_rng(123)
    alphas = [-4, -3, -2, 2, 3, 4]

    for net in NETWORKS:
        # Gather all model rate-of-change values for this network
        all_v, all_vs = [], []
        for sub in SUBJECTS:
            dv = pickle.load(open(
                MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_vdvae_sub{sub:02d}.pkl", "rb"
            ))
            dvs = pickle.load(open(
                MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_versatile_sub{sub:02d}.pkl", "rb"
            ))
            base_v = np.array(dv["alpha_0"])
            base_vs = np.array(dvs["alpha_0"])
            all_v.append(np.concatenate([np.array(dv[f"alpha_{a}"])  / base_v  for a in alphas]))
            all_vs.append(np.concatenate([np.array(dvs[f"alpha_{a}"]) / base_vs for a in alphas]))

        rates_v  = np.concatenate(all_v)
        rates_vs = np.concatenate(all_vs)

        # Bootstrap human data to match model sample size
        if net == 'emonet':
            human_pool, human_label = human_val, 'Human Valence'
        else:
            human_pool, human_label = human_mem, 'Human Memorability'
        human_boot = rng.choice(human_pool, size=len(rates_v), replace=True)

        # Plot histograms for both models and bootstrapped human data
        plt.figure(figsize=(10, 6))
        plt.hist(rates_v,    bins=60, alpha=0.4, label=f"{net.capitalize()}‑VDVAE",    edgecolor="black")
        plt.hist(rates_vs,   bins=60, alpha=0.4, label=f"{net.capitalize()}‑Versatile", edgecolor="black")
        plt.hist(human_boot, bins=60, alpha=0.4, label=human_label,                  edgecolor="black")
        plt.xlabel("Rate of Change per α‑unit")
        plt.ylabel("Count")
        plt.title(f"Overall Rate‑of‑Change: {net.capitalize()} vs. Human")
        plt.legend()
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"roc_vs_human_{net}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📈 Saved overall ROC vs human plot: {out_path}")


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

    print("\n--- Generating Slope Distribution Plots ---")
    plot_slope_histograms(model_data)

    print("\n--- Generating Additional ROC & Comparative Plots ---")
    plot_rate_of_change_subjects()
    plot_rate_of_change_overall()
    plot_rate_of_change_vs_human(human_data)

    print("\n--- Script finished successfully ---")


if __name__ == "__main__":
    main()
