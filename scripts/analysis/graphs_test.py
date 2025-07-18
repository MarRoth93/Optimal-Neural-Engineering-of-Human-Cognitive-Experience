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
SUBJECTS = [1, 2, 5, 7]
MODELS = ['vdvae', 'versatile']
NETWORKS = ['emonet', 'memnet']
ALPHA_LEVELS_STR = ['alpha_-4', 'alpha_-2', 'alpha_0', 'alpha_2', 'alpha_4']
ALPHA_LEVELS_NUM = np.array([-4, -2, 0, 2, 4])


def setup_plotting_style():
    """Sets the global matplotlib and seaborn plotting styles."""
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
    """Loads all model scores and human data into memory."""
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
                    # Handle missing file gracefully if needed
                    model_data[net][model][sub] = None

    # --- Load human data ---
    try:
        human_df = pd.read_csv(HUMAN_DATA_PATH)
        # Map conditions to numeric alpha levels for easier grouping
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
    """Performs min-max normalization on a numpy array."""
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
    """
    if human_data is None:
        print("Skipping mean score plots due to missing human data.")
        return

    for net in NETWORKS:
        # --- Prepare human data for the current network ---
        if net == 'emonet':
            conditions = ['valence-4', 'valence-2', 'alpha0', 'valence+2', 'valence+4']
            human_col = 'ValenceRating'
        else:
            conditions = ['mem-4', 'mem-2', 'alpha0', 'mem+2', 'mem+4']
            human_col = 'MemorabilityRating'

        human_net_df = human_data[human_data['Condition'].isin(conditions)]
        human_means = human_net_df.groupby('Alpha')[human_col].mean().reindex(ALPHA_LEVELS_NUM)
        human_norm = normalize_scores(human_means)
        df_human_plot = pd.DataFrame({
            'Alpha': ALPHA_LEVELS_NUM,
            'NormalizedScore': human_norm,
            'Model': 'Human (mean)'
        })

        # --- Plot 1: Per-Subject Comparison ---
        fig_sub, axs_sub = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axs_sub = axs_sub.flatten()

        for idx, sub in enumerate(SUBJECTS):
            ax = axs_sub[idx]
            # Plot human data
            # Around line 185
            ax.plot(df_human_plot['Alpha'], df_human_plot['NormalizedScore'], marker='o', label='Human (mean)' if idx == 0 else None)

            # Plot model data
            for model in MODELS:
                if model_data[net][model][sub] is None: continue
                means = [np.mean(model_data[net][model][sub][alpha]) for alpha in ALPHA_LEVELS_STR]
                norm_means = normalize_scores(means)
                ax.plot(ALPHA_LEVELS_NUM, norm_means, marker='o', label=model if idx == 0 else None)
            
            ax.set_title(f"Subject {sub:02d}")
            ax.set_xticks(ALPHA_LEVELS_NUM)
        
        fig_sub.supxlabel("Alpha Level", fontweight='bold')
        fig_sub.supylabel("Normalized Mean Score", fontweight='bold')
        fig_sub.suptitle(f"{net.capitalize()} Network: Model vs. Human Scores by Subject", fontsize=20)
        fig_sub.legend(*axs_sub[0].get_legend_handles_labels(), title="Model", loc='center right')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        output_filename = OUTPUT_DIR / f"scores_{net}_by_subject.png"
        plt.savefig(output_filename, dpi=300)
        plt.close(fig_sub)
        print(f"📈 Saved plot: {output_filename}")

        # --- Plot 2: Averaged Across Subjects ---
        fig_avg, ax_avg = plt.subplots(figsize=(10, 7))
        
        # Plot human data
        ax_avg.plot(df_human_plot['Alpha'], df_human_plot['NormalizedScore'], marker='o', label='Human (mean)')

        
        # Calculate and plot averaged model data
        for model in MODELS:
            subj_norms = []
            for sub in SUBJECTS:
                if model_data[net][model][sub] is None: continue
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

        output_filename = OUTPUT_DIR / f"scores_{net}_averaged.png"
        plt.savefig(output_filename, dpi=300)
        plt.close(fig_avg)
        print(f"📈 Saved plot: {output_filename}")


def plot_slope_histograms(model_data):
    """
    Generates histograms of response slopes for each model.
    1. By Subject: A 2x2 grid of histograms for each subject.
    2. Pooled: A single histogram with data pooled across subjects.
    """
    for net in NETWORKS:
        # --- Data structures for plots ---
        fig_sub, axs_sub = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axs_sub = axs_sub.flatten()
        pooled_slopes = {model: [] for model in MODELS}

        # --- Plot 1: Per-Subject Histograms ---
        for idx, sub in enumerate(SUBJECTS):
            ax = axs_sub[idx]
            subject_slopes = {}
            for model in MODELS:
                if model_data[net][model][sub] is None: continue
                
                # Stack scores into [n_images x n_alphas] matrix
                scores = np.vstack([model_data[net][model][sub][alpha] for alpha in ALPHA_LEVELS_STR]).T
                
                # Calculate slope for each image
                slopes = [np.polyfit(ALPHA_LEVELS_NUM, img_scores, 1)[0] for img_scores in scores]
                subject_slopes[model] = slopes
                pooled_slopes[model].extend(slopes)
            
            # Determine shared bins for this subject's plot
            all_subj_slopes = np.concatenate(list(subject_slopes.values()))
            bins = np.linspace(all_subj_slopes.min(), all_subj_slopes.max(), 30)

            # Plot histograms on the subplot
            for model, slopes in subject_slopes.items():
                 ax.hist(slopes, bins=bins, alpha=0.6, label=model, edgecolor='black')
            
            ax.set_title(f"Subject {sub:02d}")
            ax.legend()

        fig_sub.supxlabel("Response Slope (Δ Score per α-unit)", fontweight='bold')
        fig_sub.supylabel("Number of Images", fontweight='bold')
        fig_sub.suptitle(f"{net.capitalize()} Network: Response Slope Distribution by Subject", fontsize=20)
        plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])

        output_filename = OUTPUT_DIR / f"slopes_{net}_by_subject.png"
        plt.savefig(output_filename, dpi=300)
        plt.close(fig_sub)
        print(f"📊 Saved plot: {output_filename}")
        
        # --- Plot 2: Pooled Histograms ---
        fig_pool, ax_pool = plt.subplots(figsize=(10, 7))

        # Determine shared bins for pooled data
        all_pooled_slopes = np.concatenate(list(pooled_slopes.values()))
        bins = np.linspace(all_pooled_slopes.min(), all_pooled_slopes.max(), 40)
        
        # Plot pooled histograms
        for model, slopes in pooled_slopes.items():
            if slopes:
                ax_pool.hist(slopes, bins=bins, alpha=0.6, label=model, edgecolor='black')

        ax_pool.set_title(f"{net.capitalize()} Network: Pooled Response Slope Distribution", fontsize=20)
        ax_pool.set_xlabel("Response Slope (Δ Score per α-unit)")
        ax_pool.set_ylabel("Number of Images")
        ax_pool.legend(title="Model")
        plt.tight_layout()

        output_filename = OUTPUT_DIR / f"slopes_{net}_pooled.png"
        plt.savefig(output_filename, dpi=300)
        plt.close(fig_pool)
        print(f"📊 Saved plot: {output_filename}")




def main():
    """Main execution function."""
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
    
    print("\n--- Script finished successfully ---")



if __name__ == "__main__":
    main()