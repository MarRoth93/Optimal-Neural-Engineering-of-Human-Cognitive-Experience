#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate exactly these figures:

1) scores_averaged_side_by_side.png
2) roc_vs_human_side_by_side.png
3) correlations_alpha0_side_by_side.png
4) new_pixcorr_col.png

Paths and core logic are preserved from your provided scripts.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path

# =============================================================================
# --- Configuration (unchanged paths) ---
# =============================================================================
BASE_DIR = Path("/home/rothermm/brain-diffuser")
OUTPUT_DIR = BASE_DIR / "results" / "graphs"
MODEL_SCORE_DIR = BASE_DIR / "results" / "assessor_scores"
HUMAN_DATA_PATH = BASE_DIR / "data" / "human_data" / "human_df_detrended.csv"

# For PixCorr figure
PIXCORR_ROOT = BASE_DIR / "results" / "metrics" / "pixcorr"

SUBJECTS = [1, 2, 5, 7]
MODELS = ['vdvae', 'versatile']
NETWORKS = ['emonet', 'memnet']
ALPHA_LEVELS_STR = ['alpha_-4', 'alpha_-2', 'alpha_0', 'alpha_2', 'alpha_4']
ALPHA_LEVELS_NUM = np.array([-4, -2, 0, 2, 4])

# Consistent, colorblind-safe palette & role colors
PALETTE = sns.color_palette("colorblind")
COLOR_MAP = {
    "Human Rating": PALETTE[0],
    "VDVAE":        PALETTE[1],
    "Versatile Diffusion": PALETTE[2],
    "EmoNet":       PALETTE[0],
    "MemNet":       PALETTE[2],
}

# Task colors/labels for new_pixcorr_col.png (from the second script)
TASK_COLOR = {
    "emonet": PALETTE[0],  # Valence
    "memnet": PALETTE[2],  # Memorability
}
TASK_LABEL = {"emonet": "Valence", "memnet": "Memorability"}
TASK_ORDER = ("emonet", "memnet")

# =============================================================================
# --- Helpers (unchanged behavior) ---
# =============================================================================
def setup_plotting_style():
    sns.set_style("whitegrid", {"grid.color": "#e0e0e0",
                                "grid.linestyle": "-",
                                "grid.linewidth": 1.0})
    sns.set_context("notebook", font_scale=1.2)
    sns.set_palette(PALETTE)

    mpl.rcParams.update({
        "figure.figsize": (20, 14),
        "axes.facecolor": "#FFFFFF",
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,

        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",

        "xtick.labelsize": 16,
        "ytick.labelsize": 16,

        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "legend.frameon": True,

        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "lines.markeredgewidth": 0,

        "errorbar.capsize": 0,
    })
    print("🎨 Plotting style configured (colorblind palette, +2pt labels, lighter grid).")


def load_data():
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


def _bold_ticks(ax):
    ax.tick_params(axis='both', labelsize=16)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def normalize_scores(scores):
    s_arr = np.array(scores, dtype=float)
    min_val, max_val = np.nanmin(s_arr), np.nanmax(s_arr)
    if max_val == min_val:
        return np.zeros_like(s_arr)
    return (s_arr - min_val) / (max_val - min_val)


def pearson_safe(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = min(len(a), len(b))
    if n < 2:
        return np.nan
    a = a[:n]; b = b[:n]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def _subject_alpha_means(d_dict):
    """Return np.array of means per alpha ([-4,-2,0,2,4]) in ALPHA_LEVELS_STR order."""
    return np.array([np.nanmean(d_dict[a]) if (d_dict and a in d_dict) else np.nan
                     for a in ALPHA_LEVELS_STR], dtype=float)

# -------- PixCorr lookup (from per-subject summary.csv) --------
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
    pix_map[network][model][subject][alpha_num] -> mean_r
    Uses INTRA comparisons: 'intra:{model}:{network}:alpha_±X_vs_alpha_0'
    """
    import re
    pat = re.compile(r"^intra:(.*?):(.*?):alpha_([+-]?\d+)_vs_alpha_0$")
    pix_map = {net: {m: {} for m in MODELS} for net in NETWORKS}

    for subj in SUBJECTS:
        df = _load_pixcorr_summary_for_subject(subj)
        if df.empty or "comparison" not in df or "mean_r" not in df:
            continue

        for comp, mean_r in zip(df["comparison"].values, df["mean_r"].values):
            m = pat.match(str(comp))
            if not m:
                continue
            model, network, a = m.groups()
            model_key = "versatile" if model == "versatile_diffusion" else model
            try:
                a_num = int(a.replace("+", ""))  # "+2" -> 2
            except Exception:
                continue
            pix_map.setdefault(network, {}).setdefault(model_key, {}) \
                   .setdefault(subj, {})[a_num] = float(mean_r)
    return pix_map


def _get_pixcorr(pix_map, net: str, model: str, subj: int, alpha_num: int) -> float:
    if alpha_num == 0:
        return 1.0
    try:
        return pix_map[net][model].get(subj, {}).get(alpha_num, np.nan)
    except KeyError:
        return np.nan

# =============================================================================
# --- Figure 1: scores_averaged_side_by_side.png ---
# =============================================================================
def plot_scores_averaged_side_by_side(model_data, human_data):
    if human_data is None:
        print("Skipping averaged mean score plot due to missing human data.")
        return

    avg_store = {
        'emonet': {'human_x': None, 'human_y': None, 'models': {}},
        'memnet': {'human_x': None, 'human_y': None, 'models': {}},
    }

    for net in NETWORKS:
        # Human column and conditions for each network
        if net == 'emonet':
            conditions = ['valence-4', 'valence-2', 'alpha0', 'valence+2', 'valence+4']
            human_col = 'ValenceRating'
        else:
            conditions = ['mem-4', 'mem-2', 'alpha0', 'mem+2', 'mem+4']
            human_col = 'MemorabilityRating'

        human_net_df = human_data[human_data['Condition'].isin(conditions)]
        human_means = human_net_df.groupby('Alpha')[human_col].mean().reindex(ALPHA_LEVELS_NUM)
        human_norm = normalize_scores(human_means.to_numpy(dtype=float))

        human_x = ALPHA_LEVELS_NUM.copy()
        human_y = human_norm.copy()
        # For MemNet panel, display human without ±4 (unchanged behavior)
        if net == 'memnet':
            keep = ~np.isin(human_x, [-4, 4])
            human_x = human_x[keep]
            human_y = human_y[keep]

        avg_store[net]['human_x'] = human_x
        avg_store[net]['human_y'] = human_y

        # Model curves averaged across subjects (per-subject normalized)
        for model in MODELS:
            subj_norms = []
            for sub in SUBJECTS:
                d = model_data[net][model].get(sub)
                if not d:
                    continue
                means = [np.nanmean(d[a]) for a in ALPHA_LEVELS_STR]
                subj_norms.append(normalize_scores(means))
            avg_store[net]['models'][model] = (
                np.nanmean(np.vstack(subj_norms), axis=0) if subj_norms else None
            )

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left: EmoNet
    ax = axs[0]
    ax.plot(avg_store['emonet']['human_x'], avg_store['emonet']['human_y'],
            marker='o', label='Human Rating', color=COLOR_MAP["Human Rating"])
    if avg_store['emonet']['models']['vdvae'] is not None:
        ax.plot(ALPHA_LEVELS_NUM, avg_store['emonet']['models']['vdvae'],
                marker='o', label='VDVAE', color=COLOR_MAP["VDVAE"])
    if avg_store['emonet']['models']['versatile'] is not None:
        ax.plot(ALPHA_LEVELS_NUM, avg_store['emonet']['models']['versatile'],
                marker='o', label='Versatile Diffusion', color=COLOR_MAP["Versatile Diffusion"])
    ax.set_title("Valence")
    ax.set_xlabel("Alpha Level", fontweight="bold")
    ax.set_ylabel("Normalized Mean Score", fontweight="bold")
    ax.set_xticks(ALPHA_LEVELS_NUM)
    ax.set_ylim(0, 1.05)

    # Right: MemNet
    ax = axs[1]
    ax.plot(avg_store['memnet']['human_x'], avg_store['memnet']['human_y'],
            marker='o', label='Human Rating', color=COLOR_MAP["Human Rating"])
    if avg_store['memnet']['models']['vdvae'] is not None:
        ax.plot(ALPHA_LEVELS_NUM, avg_store['memnet']['models']['vdvae'],
                marker='o', label='VDVAE', color=COLOR_MAP["VDVAE"])
    if avg_store['memnet']['models']['versatile'] is not None:
        ax.plot(ALPHA_LEVELS_NUM, avg_store['memnet']['models']['versatile'],
                marker='o', label='Versatile Diffusion', color=COLOR_MAP["Versatile Diffusion"])
    ax.set_title("Memorability")
    ax.set_xlabel("Alpha Level", fontweight="bold")
    ax.set_xticks(ALPHA_LEVELS_NUM)
    ax.set_ylim(0, 1.05)

    _bold_ticks(axs[0])
    _bold_ticks(axs[1])

    # Shared legend (ordered)
    handles, labels = axs[0].get_legend_handles_labels()
    order = ['Human Rating', 'VDVAE', 'Versatile Diffusion']
    uniq = {l: h for h, l in zip(handles, labels)}
    handles_final = [uniq[l] for l in order if l in uniq]
    labels_final = [l for l in order if l in uniq]

    fig.legend(handles_final, labels_final, title="Source",
               loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3)

    fig.suptitle("Mean Scores Averaged Across Subjects", fontsize=22, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "scores_averaged_side_by_side.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved side-by-side averaged plot: {out_path}")


# =============================================================================
# --- Figure 2: roc_vs_human_side_by_side.png ---
# =============================================================================
def plot_rate_of_change_vs_human(human_df, prob_per_bin: bool = False):
    if human_df is None:
        print("Skipping ROC vs human plot due to missing human data.")
        return

    alpha_map = {
        'valence-4': -4, 'valence-2': -2, 'alpha0': 0, 'valence+2': 2, 'valence+4': 4,
        'mem-4': -4, 'mem-2': -2, 'mem+2': 2, 'mem+4': 4
    }
    human_df = human_df.copy()
    human_df['alpha'] = human_df['Condition'].map(alpha_map)

    # Collect human ROC values with their alphas
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

    # Model ROC arrays for a network
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

    rates_v_emo, rates_vs_emo = get_model_rates('emonet')
    human_pool_emo = human_val

    rates_v_mem, rates_vs_mem = get_model_rates('memnet')
    if human_mem_alpha.size:
        keep = (human_mem_alpha != -4) & (human_mem_alpha != 4)
        human_pool_mem = human_mem[keep]
    else:
        human_pool_mem = human_mem

    if human_pool_mem.size == 0:
        print("⚠️ No human samples for MemNet after filtering (±4 removed). Plotting models only.")

    # Global bins across both panels for comparability
    global_combined = np.concatenate([
        rates_v_emo, rates_vs_emo, human_pool_emo,
        rates_v_mem, rates_vs_mem,
        human_pool_mem if human_pool_mem.size else np.array([])
    ])
    bins = np.histogram_bin_edges(global_combined, bins=60)

    # Create side-by-side figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Common bar styling per recommendations
    edgec = "#444444"
    alpha_fill = 0.4  # less transparent (more opaque)

    # Helper to draw hist + KDE
    def draw_panel(ax, model_v, model_vs, human_pool, title):
        if prob_per_bin:
            ax.hist(model_v,  bins=bins, weights=weights_for(model_v),  alpha=alpha_fill,
                    label="VDVAE", edgecolor=edgec, color=COLOR_MAP["VDVAE"])
            ax.hist(model_vs, bins=bins, weights=weights_for(model_vs), alpha=alpha_fill,
                    label="Versatile Diffusion", edgecolor=edgec, color=COLOR_MAP["Versatile Diffusion"])
            if human_pool.size:
                ax.hist(human_pool, bins=bins, weights=weights_for(human_pool), alpha=alpha_fill,
                        label="Human Rating", edgecolor=edgec, color=COLOR_MAP["Human Rating"])
            ax.set_ylabel("Probability per bin (sums to 1)")
        else:
            ax.hist(model_v,  bins=bins, density=True, alpha=alpha_fill,
                    label="VDVAE", edgecolor=edgec, color=COLOR_MAP["VDVAE"])
            ax.hist(model_vs, bins=bins, density=True, alpha=alpha_fill,
                    label="Versatile Diffusion", edgecolor=edgec, color=COLOR_MAP["Versatile Diffusion"])
            if human_pool.size:
                ax.hist(human_pool, bins=bins, density=True, alpha=alpha_fill,
                        label="Human Rating", edgecolor=edgec, color=COLOR_MAP["Human Rating"])
            ax.set_ylabel("Probability density")

        # KDE overlays for clarity
        try:
            sns.kdeplot(model_v,  ax=ax, bw_adjust=0.9, clip=(0, 3),
                        linewidth=2.5, fill=False, label=None, color=COLOR_MAP["VDVAE"])
            sns.kdeplot(model_vs, ax=ax, bw_adjust=0.9, clip=(0, 3),
                        linewidth=2.5, fill=False, label=None, color=COLOR_MAP["Versatile Diffusion"])
            if human_pool.size:
                sns.kdeplot(human_pool, ax=ax, bw_adjust=0.9, clip=(0, 3),
                            linewidth=2.5, fill=False, label=None, color=COLOR_MAP["Human Rating"])
        except Exception as e:
            print(f"ℹ️ KDE overlay skipped for {title}: {e}")

        ax.set_title(title)
        ax.set_xlabel("Rate of Change (Δ/α)", fontweight="bold")
        ax.set_xlim(0, 3)
        _bold_ticks(ax)

    # Left: EmoNet
    draw_panel(axs[0], rates_v_emo, rates_vs_emo, human_pool_emo, "Valence")
    # Right: MemNet
    draw_panel(axs[1], rates_v_mem, rates_vs_mem, human_pool_mem, "Memorability")

    # Shared legend (order & placement)
    handles, labels = axs[0].get_legend_handles_labels()
    order = ['Human Rating', 'VDVAE', 'Versatile Diffusion']
    uniq = {l: h for h, l in zip(handles, labels)}
    handles_final = [uniq[l] for l in order if l in uniq]
    labels_final = [l for l in order if l in uniq]

    fig.suptitle("Rate-of-Change Distributions", fontsize=22, fontweight="bold")
    fig.legend(handles_final, labels_final, title="Source",
               loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "roc_vs_human_side_by_side.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved side-by-side ROC vs human plot: {out_path}")


# =============================================================================
# --- Figure 3: correlations_alpha0_side_by_side.png ---
# =============================================================================
def plot_alpha0_correlations(model_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("i2i", "i2i_key"),
        ("alpha_-4", "alpha_-4"),
        ("alpha_-2", "alpha_-2"),
        ("alpha_2",  "alpha_2"),
        ("alpha_4",  "alpha_4"),
    ]
    xtick_labels = ["i2i", "α−4", "α−2", "α+2", "α+4"]

    i2i_key_for_model = {
        "vdvae": "vdvae_image_to_image",
        "versatile": "clip_image_to_image"
    }

    def compute_means_sems_for_model(model: str):
        stats = {net: {lbl: [] for (lbl, _) in comparisons} for net in NETWORKS}
        for net in NETWORKS:
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

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    panel_defs = [("vdvae", "VDVAE"), ("versatile", "Versatile Diffusion")]

    x = np.arange(len(comparisons))
    width = 0.45
    offset = 0.23

    error_kw = dict(elinewidth=2.2, capthick=2.2)
    capsize = 6

    shared_handles, shared_labels = None, None

    for ax, (model_key, model_title) in zip(axs, panel_defs):
        means, sems = compute_means_sems_for_model(model_key)

        em_col = ax.bar(x - offset, means["emonet"], width,
                        yerr=sems["emonet"], error_kw=error_kw, capsize=capsize,
                        label="Valence", color=COLOR_MAP["EmoNet"], edgecolor="#333333")
        mm_col = ax.bar(x + offset, means["memnet"], width,
                        yerr=sems["memnet"], error_kw=error_kw, capsize=capsize,
                        label="Memorability", color=COLOR_MAP["MemNet"], edgecolor="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel(r"Pearson $r$", fontweight="bold")
        ax.set_title(model_title)
        ax.set_ylim(-0.2, 1.0)
        _bold_ticks(ax)

        # Bold, 1-decimal annotations
        ymin, ymax = ax.get_ylim()
        pad = 0.06 * (ymax - ymin)

        for net, bars in zip(["emonet", "memnet"], [em_col, mm_col]):
            for idx, b in enumerate(bars):
                r_val = means[net][idx]
                label = "NA" if np.isnan(r_val) else f"{r_val:.1f}"
                height = 0 if np.isnan(b.get_height()) else b.get_height()
                err = 0 if np.isnan(sems[net][idx]) else sems[net][idx]
                ax.text(
                    b.get_x() + b.get_width()/2,
                    height + err + pad,
                    label,
                    ha="center", va="bottom", fontsize=14, fontweight="bold"
                )

        if shared_handles is None:
            shared_handles, shared_labels = ax.get_legend_handles_labels()

    fig.suptitle("Correlation vs α=0 (Averaged Across Subjects)", fontsize=22, fontweight="bold")

    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, title="Assessor",
                   loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = OUTPUT_DIR / "correlations_alpha0_side_by_side.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📊 Saved: {out_path}")


# =============================================================================
# --- Figure 4: new_pixcorr_col.png (relative /(1 − PixCorr), α≠0; 2x1: VDVAE, Versatile)
# =============================================================================
def new_pixcorr(model_data):
    """
    2x1 panels: top=VDVAE, bottom=Versatile Diffusion.
    Task-colored lines (Valence/EmoNet & Memorability/MemNet).
    y = (score(α)/score(0)) / (1 − PixCorr(α, α=0)), α≠0 only (linear y-axis).
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
                with np.errstate(divide="ignore", invalid="ignore"):
                    penalized = rel / denom
                penalized[~np.isfinite(penalized)] = np.nan
                subj_curves.append(penalized[ALPHA_LEVELS_NUM != 0])
            series[model_key][net_key] = (np.nanmean(np.vstack(subj_curves), axis=0)
                                          if subj_curves else None)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    panels = [("vdvae", "VDVAE", axs[0]), ("versatile", "Versatile Diffusion", axs[1])]

    for model_key, model_name, ax in panels:
        for net_key in TASK_ORDER:
            y = series[model_key].get(net_key)
            if y is None:
                continue
            ax.plot(alpha_no0, y, marker="o", linewidth=2.5,
                    label=TASK_LABEL[net_key], color=TASK_COLOR[net_key])

        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Alpha Level", fontweight="bold")
        ax.set_xticks(alpha_no0)
        _bold_ticks(ax)
        handles, labels = ax.get_legend_handles_labels()
        order_map = {TASK_LABEL[k]: i for i, k in enumerate(TASK_ORDER)}
        idx = np.argsort([order_map.get(l, 999) for l in labels])
        ax.legend([handles[i] for i in idx], [labels[i] for i in idx], title="Task")

    axs[0].set_ylabel("Relative Mean Score / (1 − PixCorr)", fontweight="bold")
    axs[1].set_ylabel("Relative Mean Score / (1 − PixCorr)", fontweight="bold")

    fig.suptitle("Relative Mean Scores (vs α=0) Penalized by (1 − PixCorr)", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUTPUT_DIR / "new_pixcorr_col.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved: {out_path}")


# =============================================================================
# --- Main ---
# =============================================================================
def main():
    print("--- Starting Plot Script ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_plotting_style()
    model_data, human_data = load_data()

    print("\n--- Generating scores_averaged_side_by_side.png ---")
    plot_scores_averaged_side_by_side(model_data, human_data)

    print("\n--- Generating roc_vs_human_side_by_side.png ---")
    plot_rate_of_change_vs_human(human_data)

    print("\n--- Generating correlations_alpha0_side_by_side.png ---")
    plot_alpha0_correlations(model_data)

    print("\n--- Generating new_pixcorr_col.png ---")
    new_pixcorr(model_data)

    print("\n--- Script finished successfully ---")


if __name__ == "__main__":
    main()
