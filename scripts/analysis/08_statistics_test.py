#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full statistics pipeline for assessor scores and human ratings, now with SSIM analysis.

This version loads pickles with exact filenames like:
  /home/rothermm/brain-diffuser/results/assessor_scores/subjXX/{net}_{model}_subXX.pkl
for net in {emonet, memnet} and model in {vdvae, versatile} — matching your plotting script.

SSIM inputs are read from your SSIM script outputs:
  results/metrics/ssim/{vdvae|versatile_diffusion}/summary_{emonet|memnet}.csv
Each summary_* CSV contains per-subject per-alpha SSIM means (α ∈ {-4,-2,0,2,4} vs α=0).

Outputs (under /home/rothermm/brain-diffuser/results/statistics/):
  - descriptives_by_alpha.csv
  - slope_one_sample.csv
  - slope_model_vs_model.csv
  - model_vs_human_curve_correlations.csv   (includes n, 95% CI (Fisher z), permutation p)
  - alpha_linear_trend_tests.csv
  - mixed_effects_alpha.csv                 (mixed model: score_norm ~ Alpha + (1|subject))
  - ssim_summary_combined.csv               (weighted subject-avg SSIM per model/network/alpha)
  - ssim_delta_join.csv                     (joins SSIM (D=1-SSIM) with Δ scores by α)
  - ssim_vs_delta_correlations.csv          (correlations D ↔ Δ scores, model & human)
  - report_statistics_<timestamp>.md        (roll-up Markdown, now includes SSIM section)
"""

import glob
import pickle
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence convergence + random effects singularity chatter
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", message="Random effects covariance is singular")

# Optional dependency for DataFrame.to_markdown()
try:
    import tabulate  # noqa: F401
    _HAS_TABULATE = True
except Exception:
    _HAS_TABULATE = False

# =============================================================================
# --- Configuration ---
# =============================================================================
BASE_DIR = Path("/home/rothermm/brain-diffuser")
OUTPUT_DIR = BASE_DIR / "results" / "graphs"            # (not used here, kept for consistency)
MODEL_SCORE_DIR = BASE_DIR / "results" / "assessor_scores"
HUMAN_DATA_PATH = BASE_DIR / "data" / "human_data" / "human_df_detrended.csv"

SUBJECTS = [1, 2, 5, 7]
NETWORKS = ['emonet', 'memnet']
MODELS   = ['vdvae', 'versatile']        # <- only these two, like your plotting script
ALPHA_LEVELS_STR = ['alpha_-4', 'alpha_-2', 'alpha_0', 'alpha_2', 'alpha_4']
ALPHA_LEVELS_NUM = np.array([-4, -2, 0, 2, 4], dtype=float)

STAT_DIR = BASE_DIR / "results" / "statistics"

# SSIM locations produced by your SSIM script
SSIM_ROOT = BASE_DIR / "results" / "metrics" / "ssim"
# Map our "versatile" model name to the SSIM folder "versatile_diffusion"
SSIM_MODEL_MAP = {"vdvae": "vdvae", "versatile": "versatile_diffusion"}

# =============================================================================
# --- Utilities ---
# =============================================================================
def df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Render DataFrame to markdown if possible; otherwise fall back to plain text."""
    if df is None:
        return ""
    try:
        if _HAS_TABULATE and hasattr(df, "to_markdown"):
            return df.to_markdown(index=index)
    except Exception:
        pass
    return df.to_string(index=index)

def _safe_to_csv(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        print(f"⚠️  Skipping write: no rows for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Wrote {path} ({len(df)} rows)")

def normalize_scores(scores):
    s_arr = np.asarray(scores, dtype=float)
    s_arr = s_arr[np.isfinite(s_arr)]
    if s_arr.size == 0:
        return np.array([], dtype=float)
    lo, hi = np.nanmin(s_arr), np.nanmax(s_arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(s_arr)
    return (s_arr - lo) / (hi - lo)

def pearson_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return np.nan
    a = a[:n]; b = b[:n]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def fisher_z_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)
    r_clip = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3)
    z_lo = z + scipy.stats.norm.ppf(alpha/2) * se
    z_hi = z + scipy.stats.norm.ppf(1 - alpha/2) * se
    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))

def perm_test_corr(a, b, iters=10000, rng=None):
    rng = np.random.default_rng(rng)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    r_obs = pearson_safe(a, b)
    if not np.isfinite(r_obs):
        return (np.nan, np.nan)
    cnt = 0
    for _ in range(iters):
        perm_b = rng.permutation(b)
        r_perm = pearson_safe(a, perm_b)
        if np.abs(r_perm) >= np.abs(r_obs):
            cnt += 1
    p = (cnt + 1) / (iters + 1)
    return (r_obs, float(p))

def _cohen_d_one_sample(sample, mu=0.0):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float((x.mean() - mu) / x.std(ddof=1))

def _cohen_d_independent(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    if sa2 == 0 and sb2 == 0:
        return np.nan
    sp = np.sqrt((sa2 + sb2) / 2.0)
    if sp == 0 or not np.isfinite(sp):  # degenerate
        return np.nan
    d = (a.mean() - b.mean()) / sp
    J = 1 - (3 / (4*(a.size + b.size) - 9))
    return float(d * J)

def _describe_array(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, q1=np.nan,
                    median=np.nan, q3=np.nan, max=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        min=float(np.min(x)),
        q1=float(np.quantile(x, 0.25)),
        median=float(np.median(x)),
        q3=float(np.quantile(x, 0.75)),
        max=float(np.max(x)),
    )

# =============================================================================
# --- Data Loading ---
# =============================================================================
def load_data():
    """
    Loads model assessor data and human ratings using exact filenames:
      subjXX/{net}_{model}_subXX.pkl
    """
    model_data = {net: {m: {} for m in MODELS} for net in NETWORKS}

    for sub in SUBJECTS:
        for net in NETWORKS:
            for model in MODELS:
                p = MODEL_SCORE_DIR / f"subj{sub:02d}" / f"{net}_{model}_sub{sub:02d}.pkl"
                if not p.exists():
                    print(f"❌ Could not find: {p}")
                    model_data[net][model][sub] = None
                    continue
                try:
                    with open(p, "rb") as f:
                        model_data[net][model][sub] = pickle.load(f)
                except Exception as e:
                    print(f"❌ Failed to load {p}: {e}")
                    model_data[net][model][sub] = None

    # Human ratings
    try:
        human_df = pd.read_csv(HUMAN_DATA_PATH)
        alpha_map = {
            'valence-4': -4, 'valence-2': -2, 'alpha0': 0, 'valence+2': 2, 'valence+4': 4,
            'mem-4': -4, 'mem-2': -2, 'alpha0': 0, 'mem+2': 2, 'mem+4': 4
        }
        human_df['Alpha'] = human_df['Condition'].map(alpha_map)
        if 'Subject' not in human_df.columns:
            human_df['Subject'] = 'S'
        print("✅ Loaded human ratings.")
    except FileNotFoundError:
        print(f"❌ Human data not found at {HUMAN_DATA_PATH}")
        human_df = None

    print("✅ Loaded model assessor data.")
    return model_data, human_df

# =============================================================================
# --- Core Computations (existing) ---
# =============================================================================
def _alpha_descriptives(model_data, human_df):
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
                vals = df[df['Alpha'] == alpha][col].to_numpy()
                desc = _describe_array(vals)
                rows.append(dict(kind="human", network=net, model="human", subject="all",
                                 alpha=int(alpha), **desc))
    # Models (pooled across subjects by averaging descriptive moments; n sums)
    for net in NETWORKS:
        for model in MODELS:
            agg = []
            for sub in SUBJECTS:
                d = model_data[net][model].get(sub)
                if not d:
                    continue
                for a_str, a_num in zip(ALPHA_LEVELS_STR, ALPHA_LEVELS_NUM):
                    vals = np.asarray(d.get(a_str, []), dtype=float)
                    desc = _describe_array(vals)
                    agg.append((net, model, sub, int(a_num), desc))
            if agg:
                df_tmp = pd.DataFrame([dict(alpha=a, **desc) for _, _, _, a, desc in agg])
                pooled = df_tmp.groupby('alpha', as_index=False).agg(
                    n=('n', 'sum'),
                    mean=('mean', 'mean'),
                    std=('std', 'mean'),
                    min=('min', 'mean'),
                    q1=('q1', 'mean'),
                    median=('median', 'mean'),
                    q3=('q3', 'mean'),
                    max=('max', 'mean'),
                )
                for _, r in pooled.iterrows():
                    rows.append(dict(kind="model", network=net, model=model, subject="pooled",
                                     alpha=int(r['alpha']),
                                     n=int(r['n']), mean=float(r['mean']), std=float(r['std']),
                                     min=float(r['min']), q1=float(r['q1']), median=float(r['median']),
                                     q3=float(r['q3']), max=float(r['max'])))
    return pd.DataFrame(rows)

def _collect_image_slopes_for(model_data, net, model):
    slopes_all = []
    for sub in SUBJECTS:
        d = model_data[net][model].get(sub)
        if not d:
            continue
        try:
            M = np.vstack([np.asarray(d[a], dtype=float) for a in ALPHA_LEVELS_STR]).T  # n_img x 5
        except Exception:
            continue
        for row in M:
            if np.all(np.isfinite(row)):
                slope = np.polyfit(ALPHA_LEVELS_NUM, row, 1)[0]
                slopes_all.append(float(slope))
    return np.asarray(slopes_all, dtype=float) if slopes_all else np.array([], dtype=float)

def _slope_tests(model_data):
    rows_1samp, rows_cmp = [], []
    for net in NETWORKS:
        per_model = {m: _collect_image_slopes_for(model_data, net, m) for m in MODELS}
        # one-sample vs 0
        for m, arr in per_model.items():
            arr = arr[np.isfinite(arr)]
            if arr.size >= 2:
                t, p = scipy.stats.ttest_1samp(arr, popmean=0.0)
                d = _cohen_d_one_sample(arr, 0.0)
                sw_n = min(arr.size, 5000)
                try:
                    sw_p = float(scipy.stats.shapiro(arr[:sw_n]).pvalue)
                except Exception:
                    sw_p = np.nan
                rows_1samp.append(dict(network=net, model=m, n=int(arr.size),
                                       mean=float(arr.mean()), std=float(arr.std(ddof=1)),
                                       t=float(t), p=float(p), cohen_d=float(d),
                                       shapiro_p=sw_p))
            else:
                rows_1samp.append(dict(network=net, model=m, n=int(arr.size),
                                       mean=np.nan, std=np.nan, t=np.nan, p=np.nan,
                                       cohen_d=np.nan, shapiro_p=np.nan))
        # Welch comparison between models (vdvae vs versatile)
        a = per_model.get('vdvae', np.array([])); b = per_model.get('versatile', np.array([]))
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if a.size >= 2 and b.size >= 2:
            t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
            d = _cohen_d_independent(a, b)
        else:
            t = p = d = np.nan
        rows_cmp.append(dict(network=net, model_A='vdvae', model_B='versatile',
                             n_A=int(a.size), n_B=int(b.size),
                             t=float(t) if np.isfinite(t) else np.nan,
                             p=float(p) if np.isfinite(p) else np.nan,
                             cohen_d=float(d) if np.isfinite(d) else np.nan))
    return pd.DataFrame(rows_1samp), pd.DataFrame(rows_cmp)

def _model_curve_across_subjects(model_data, net, model, reducer="mean"):
    subj_norms = []
    for sub in SUBJECTS:
        d = model_data[net][model].get(sub)
        if not d:
            continue
        try:
            vals = [float(np.mean(np.asarray(d[a], dtype=float))) for a in ALPHA_LEVELS_STR]
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

def _curve_correlations(model_data, human_df, do_perm=True, iters=10000):
    rows = []
    for net in NETWORKS:
        human = _human_curve(human_df, net)
        for model in MODELS:
            curve = _model_curve_across_subjects(model_data, net, model, reducer="mean")
            if curve is None or human is None:
                r = np.nan; n = 0; r_lo = r_hi = np.nan; p_perm = np.nan
            else:
                a = np.asarray(curve, dtype=float)
                b = np.asarray(human, dtype=float)
                n = int(min(len(a), len(b)))
                r = pearson_safe(a, b)
                r_lo, r_hi = fisher_z_ci(r, n) if np.isfinite(r) else (np.nan, np.nan)
                p_perm = perm_test_corr(a, b, iters=iters)[1] if (do_perm and n >= 5) else np.nan
            rows.append(dict(network=net, model=model, n=n,
                             pearson_r=float(r) if np.isfinite(r) else np.nan,
                             ci_low=float(r_lo), ci_high=float(r_hi),
                             p_perm=float(p_perm)))
        # Extra: memnet with ±4 removed
        if net == 'memnet' and human is not None:
            keep = ~np.isin(ALPHA_LEVELS_NUM, [-4, 4])
            for model in MODELS:
                curve = _model_curve_across_subjects(model_data, net, model, reducer="mean")
                if curve is None:
                    r2 = np.nan; n2 = 0; r2_lo = r2_hi = np.nan; p2 = np.nan
                else:
                    a = np.asarray(curve, dtype=float)[keep]
                    b = np.asarray(human, dtype=float)[keep]
                    n2 = int(min(len(a), len(b)))
                    r2 = pearson_safe(a, b)
                    r2_lo, r2_hi = fisher_z_ci(r2, n2) if np.isfinite(r2) else (np.nan, np.nan)
                    p2 = perm_test_corr(a, b, iters=iters)[1] if (do_perm and n2 >= 3) else np.nan
                rows.append(dict(network=net, model=model + " (±4 removed)", n=n2,
                                 pearson_r=float(r2) if np.isfinite(r2) else np.nan,
                                 ci_low=float(r2_lo), ci_high=float(r2_hi),
                                 p_perm=float(p2)))
    return pd.DataFrame(rows)

def _alpha_linear_trend_tests(model_data):
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
                mask = np.isfinite(ALPHA_LEVELS_NUM) & np.isfinite(y)
                if mask.sum() >= 2:
                    res = scipy.stats.linregress(ALPHA_LEVELS_NUM[mask], y[mask])
                    rows.append(dict(network=net, model=model, subject=f"{sub:02d}",
                                     slope=float(res.slope), intercept=float(res.intercept),
                                     r=float(res.rvalue), p=float(res.pvalue), stderr=float(res.stderr)))
                    subj_slopes.append(float(res.slope))
            subj_slopes = np.asarray(subj_slopes, dtype=float)
            if subj_slopes.size >= 2:
                t, p = scipy.stats.ttest_1samp(subj_slopes, 0.0)
                rows.append(dict(network=net, model=model, subject="MEAN",
                                 slope=float(subj_slopes.mean()), intercept=np.nan,
                                 r=np.nan, p=float(p),
                                 stderr=float(subj_slopes.std(ddof=1)/np.sqrt(subj_slopes.size))))
            else:
                rows.append(dict(network=net, model=model, subject="MEAN",
                                 slope=np.nan, intercept=np.nan, r=np.nan, p=np.nan, stderr=np.nan))
    return pd.DataFrame(rows)

def _mixed_effects_alpha(model_data, human_df):
    """
    Mixed effects on normalized scores:
        score_norm ~ Alpha + (1 | subject)
    """
    rows = []
    for net in NETWORKS:
        model_list = MODELS + (['human'] if human_df is not None else [])
        for model in model_list:
            records = []
            if model == 'human':
                if net == 'emonet':
                    conds = ['valence-4','valence-2','alpha0','valence+2','valence+4']; col = 'ValenceRating'
                else:
                    conds = ['mem-4','mem-2','alpha0','mem+2','mem+4']; col = 'MemorabilityRating'
                df = human_df[human_df['Condition'].isin(conds)][['Subject','Alpha',col]].copy()
                df = df.rename(columns={'Subject':'subject', col:'score'})
                df['image'] = -1
                records = df.to_dict('records')
            else:
                for sub in SUBJECTS:
                    d = model_data[net][model].get(sub)
                    if not d:
                        continue
                    try:
                        M = np.vstack([np.asarray(d[a], dtype=float) for a in ALPHA_LEVELS_STR]).T
                        for i in range(M.shape[0]):
                            for a_idx, a_num in enumerate(ALPHA_LEVELS_NUM):
                                val = M[i, a_idx]
                                if np.isfinite(val):
                                    records.append({'subject': f'{sub:02d}', 'image': i, 'Alpha': float(a_num), 'score': float(val)})
                    except Exception:
                        for a_str, a_num in zip(ALPHA_LEVELS_STR, ALPHA_LEVELS_NUM):
                            arr = np.asarray(d.get(a_str, []), dtype=float)
                            if arr.size:
                                m = float(np.nanmean(arr))
                                records.append({'subject': f'{sub:02d}', 'image': -1, 'Alpha': float(a_num), 'score': m})
            if not records:
                continue
            df_long = pd.DataFrame(records)

            # normalize within subject
            def _minmax(x):
                if np.nanmax(x) > np.nanmin(x):
                    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
                return np.zeros_like(x)

            df_long['score_norm'] = df_long.groupby('subject')['score'].transform(_minmax)

            # enforce numeric types
            for col in ['Alpha','score','score_norm']:
                df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
            df_long = df_long.dropna(subset=['Alpha', 'score_norm', 'subject'])

            # NEW: ensure there is enough variation/groups to fit the model
            if df_long['subject'].nunique() < 2 or df_long['Alpha'].nunique() < 2:
                rows.append(dict(network=net, model=model, n=int(len(df_long)),
                                F=np.nan, df1=np.nan, df2=np.nan, p=np.nan,
                                error='Insufficient groups or Alpha variation'))
                continue

            # (optional but helpful): center Alpha to aid convergence
            df_long['Alpha'] = df_long['Alpha'] - df_long['Alpha'].mean()

            try:
                md = smf.mixedlm("score_norm ~ Alpha", df_long, groups=df_long["subject"])
                mdf = md.fit(reml=False, method='lbfgs', maxiter=200)
                wald = mdf.t_test([0, 1])  # Alpha coefficient
                tval = float(wald.tvalue)
                F = tval * tval
                rows.append(dict(network=net, model=model,
                                 n=int(len(df_long)), F=float(F),
                                 df1=1, df2=int(mdf.df_resid), p=float(wald.pvalue)))
            except Exception as e:
                rows.append(dict(network=net, model=model, n=int(len(df_long)),
                                 F=np.nan, df1=np.nan, df2=np.nan, p=np.nan, error=str(e)))
    return pd.DataFrame(rows)

# =============================================================================
# --- SSIM ANALYSIS (NEW) ---
# =============================================================================
def _alpha_key_to_num(alpha_key: str) -> int:
    # accepts "alpha -4", "alpha 2", etc.
    try:
        return int(alpha_key.replace("alpha", "").strip())
    except Exception:
        return np.nan

def _load_ssim_summaries() -> pd.DataFrame:
    """
    Load SSIM summaries from results/metrics/ssim/{model}/summary_{network}.csv
    Returns a unified DataFrame with columns:
      model, network, subject, alpha (int), n, ssim_mean, ssim_std
    """
    rows = []
    for m in MODELS:
        ssim_model_name = SSIM_MODEL_MAP.get(m, m)
        for net in NETWORKS:
            csv_path = SSIM_ROOT / ssim_model_name / f"summary_{net}.csv"
            if not csv_path.exists():
                print(f"⚠️ SSIM summary missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            # expected columns: model, network, subject, alpha (e.g., "alpha -4"), n, ssim_mean, ssim_std
            df['alpha_num'] = df['alpha'].apply(_alpha_key_to_num)
            df['model_std'] = m  # keep our canonical model name
            keep = df[['model_std','network','subject','alpha_num','n','ssim_mean','ssim_std']].copy()
            keep = keep.rename(columns={'model_std':'model','alpha_num':'alpha'})
            rows.append(keep)
    if not rows:
        return pd.DataFrame(columns=['model','network','subject','alpha','n','ssim_mean','ssim_std'])
    out = pd.concat(rows, ignore_index=True)
    # keep only α in {-4,-2,2,4}
    out = out[out['alpha'].isin([-4,-2,2,4])]
    # subject string normalize (e.g., "subj01" -> "01")
    def _sub_norm(s):
        s = str(s)
        if s.lower().startswith("subj") and len(s) >= 6 and s[4:6].isdigit():
            return s[4:6]
        return s
    out['subject'] = out['subject'].apply(_sub_norm)
    return out

def _weighted_mean(group, val_col, w_col):
    v = group[val_col].to_numpy(dtype=float)
    w = group[w_col].to_numpy(dtype=float)
    if np.all(~np.isfinite(v)) or np.all(w <= 0):
        return np.nan
    v = np.nan_to_num(v, nan=np.nanmean(v))
    w = np.where(np.isfinite(w) & (w>0), w, 0.0)
    if w.sum() == 0:
        return np.nan
    return float(np.sum(v*w) / np.sum(w))

def _combine_ssim(df_ssim: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted average (by n) SSIM across subjects, per (model, network, alpha).
    Returns columns: model, network, alpha, n_total, ssim_mean_w, ssim_sd_across_subjects
    """
    if df_ssim.empty:
        return df_ssim
    grp = df_ssim.groupby(['model','network','alpha'], as_index=False)
    agg_mean = grp.apply(lambda g: pd.Series({
        'n_total': int(np.nansum(g['n'].to_numpy())),
        'ssim_mean_w': _weighted_mean(g, 'ssim_mean', 'n'),
        'ssim_sd_across_subjects': float(np.nanstd(g['ssim_mean'].to_numpy(), ddof=1)) if len(g)>1 else 0.0
    }))
    return agg_mean

def _compute_model_delta_table(model_data) -> pd.DataFrame:
    """
    Build per-subject Δ score (mean over images) relative to α=0, then aggregate across subjects.
    Returns rows: network, model, subject, alpha, delta
    """
    rows = []
    for net in NETWORKS:
        for model in MODELS:
            for sub in SUBJECTS:
                d = model_data[net][model].get(sub)
                if not d:
                    continue
                # per-alpha means
                means = {}
                for a_str, a_num in zip(ALPHA_LEVELS_STR, ALPHA_LEVELS_NUM):
                    arr = np.asarray(d.get(a_str, []), dtype=float)
                    means[int(a_num)] = float(np.nanmean(arr)) if arr.size else np.nan
                if not np.isfinite(means.get(0, np.nan)):
                    continue
                base = means[0]
                for a in [-4,-2,2,4]:
                    if np.isfinite(means.get(a, np.nan)):
                        rows.append(dict(network=net, model=model, subject=f"{sub:02d}",
                                         alpha=int(a), delta=float(means[a]-base)))
    return pd.DataFrame(rows)

def _compute_human_delta_table(human_df) -> pd.DataFrame:
    """
    Per-network Δ human rating vs α=0 (means across trials), per 'Subject' in human_df.
    Returns rows: network, model='human', subject, alpha, delta
    """
    if human_df is None or human_df.empty:
        return pd.DataFrame(columns=['network','model','subject','alpha','delta'])
    out_rows = []
    for net in NETWORKS:
        if net == 'emonet':
            conds = ['valence-4','valence-2','alpha0','valence+2','valence+4']; col = 'ValenceRating'
        else:
            conds = ['mem-4','mem-2','alpha0','mem+2','mem+4']; col = 'MemorabilityRating'
        df = human_df[human_df['Condition'].isin(conds)].copy()
        # per human subject means
        base = (df[df['Alpha']==0].groupby('Subject')[col].mean()).rename('base')
        for a in [-4,-2,2,4]:
            means_a = (df[df['Alpha']==a].groupby('Subject')[col].mean()).rename('mean_a')
            j = pd.concat([base, means_a], axis=1).dropna()
            for subj, row in j.iterrows():
                out_rows.append(dict(network=net, model='human', subject=str(subj), alpha=int(a),
                                     delta=float(row['mean_a']-row['base'])))
    return pd.DataFrame(out_rows)

def _join_ssim_and_deltas(df_ssim_w: pd.DataFrame,
                          df_delta_model: pd.DataFrame,
                          df_delta_human: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a row per (network, model, alpha) with weighted SSIM mean and subject-avg Δ.
    For model deltas: average across subjects.
    For human deltas: average across human subjects (kept as separate 'target=human').
    Returns columns:
      network, model, alpha, ssim_mean_w, distortion, delta_model_mean, delta_human_mean
    """
    if df_ssim_w.empty:
        return pd.DataFrame()
    # Model deltas aggregated across subjects
    dm = (df_delta_model.groupby(['network','model','alpha'])
                      .agg(delta_model_mean=('delta','mean'),
                           delta_model_sd=('delta','std'),
                           n_subjects=('delta','size'))
                      .reset_index())
    # Human deltas aggregated across human raters
    dh = (df_delta_human.groupby(['network','alpha'])
                      .agg(delta_human_mean=('delta','mean'),
                           delta_human_sd=('delta','std'),
                           n_hum=('delta','size'))
                      .reset_index())
    # Merge SSIM with deltas
    merged = df_ssim_w.merge(dm, on=['network','model','alpha'], how='left')
    merged = merged.merge(dh, on=['network','alpha'], how='left')
    merged['distortion'] = 1.0 - merged['ssim_mean_w']
    # reorder
    cols = ['network','model','alpha','n_total','ssim_mean_w','ssim_sd_across_subjects',
            'distortion','delta_model_mean','delta_model_sd','n_subjects',
            'delta_human_mean','delta_human_sd','n_hum']
    return merged[cols].sort_values(['network','model','alpha'])

def _correlate_distortion_vs_delta(df_join: pd.DataFrame) -> pd.DataFrame:
    """
    For each (network, model), compute correlation across α ∈ {-4,-2,2,4}:
      distortion ↔ delta_model_mean  (target='model')
      distortion ↔ delta_human_mean  (target='human')
    Returns rows: network, model, target, n, pearson_r, pearson_p, spearman_rho, spearman_p
    """
    rows = []
    if df_join.empty:
        return pd.DataFrame(columns=['network','model','target','n','pearson_r','pearson_p','spearman_rho','spearman_p'])
    for net in NETWORKS:
        for model in MODELS:
            sub = df_join[(df_join['network']==net) & (df_join['model']==model)]
            if sub.empty:
                continue
            # model-based Δ
            x = sub['distortion'].to_numpy(dtype=float)
            y = sub['delta_model_mean'].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                pr, pp = scipy.stats.pearsonr(x[mask], y[mask])
                sr, sp = scipy.stats.spearmanr(x[mask], y[mask])
                rows.append(dict(network=net, model=model, target='model',
                                 n=int(mask.sum()), pearson_r=float(pr), pearson_p=float(pp),
                                 spearman_rho=float(sr), spearman_p=float(sp)))
            else:
                rows.append(dict(network=net, model=model, target='model',
                                 n=int(mask.sum()), pearson_r=np.nan, pearson_p=np.nan,
                                 spearman_rho=np.nan, spearman_p=np.nan))
            # human-based Δ
            y2 = sub['delta_human_mean'].to_numpy(dtype=float)
            mask2 = np.isfinite(x) & np.isfinite(y2)
            if mask2.sum() >= 3:
                pr2, pp2 = scipy.stats.pearsonr(x[mask2], y2[mask2])
                sr2, sp2 = scipy.stats.spearmanr(x[mask2], y2[mask2])
                rows.append(dict(network=net, model=model, target='human',
                                 n=int(mask2.sum()), pearson_r=float(pr2), pearson_p=float(pp2),
                                 spearman_rho=float(sr2), spearman_p=float(sp2)))
            else:
                rows.append(dict(network=net, model=model, target='human',
                                 n=int(mask2.sum()), pearson_r=np.nan, pearson_p=np.nan,
                                 spearman_rho=np.nan, spearman_p=np.nan))
    return pd.DataFrame(rows)

# =============================================================================
# --- Report Generation ---
# =============================================================================
def generate_statistics_report(model_data, human_data):
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1/8: Descriptives by alpha …")
    df_desc = _alpha_descriptives(model_data, human_data)
    csv_desc = STAT_DIR / "descriptives_by_alpha.csv"
    _safe_to_csv(df_desc, csv_desc)

    print("Step 2/8: Per-image slope tests …")
    df_slope_1samp, df_slope_cmp = _slope_tests(model_data)
    csv_1s   = STAT_DIR / "slope_one_sample.csv"
    csv_cmp  = STAT_DIR / "slope_model_vs_model.csv"
    _safe_to_csv(df_slope_1samp, csv_1s)
    _safe_to_csv(df_slope_cmp, csv_cmp)

    print("Step 3/8: Curve correlations with human …")
    df_corr = _curve_correlations(model_data, human_data, do_perm=True, iters=10000)
    csv_corr = STAT_DIR / "model_vs_human_curve_correlations.csv"
    _safe_to_csv(df_corr, csv_corr)

    print("Step 4/8: Linear trend tests …")
    df_trend = _alpha_linear_trend_tests(model_data)
    csv_trnd = STAT_DIR / "alpha_linear_trend_tests.csv"
    _safe_to_csv(df_trend, csv_trnd)

    print("Step 5/8: Mixed effects for alpha modulation …")
    df_mixed = _mixed_effects_alpha(model_data, human_data)
    csv_mixed = STAT_DIR / "mixed_effects_alpha.csv"
    _safe_to_csv(df_mixed, csv_mixed)

    print("Step 6/8: Load SSIM summaries …")
    df_ssim = _load_ssim_summaries()
    if df_ssim.empty:
        print("⚠️ No SSIM summaries found — skipping SSIM analysis steps.")
        df_ssim_w = pd.DataFrame()
        df_delta_model = pd.DataFrame()
        df_delta_human = pd.DataFrame()
        df_join = pd.DataFrame()
        df_corr_ssim = pd.DataFrame()
    else:
        print("Step 7/8: Aggregate SSIM and compute Δ scores …")
        df_ssim_w = _combine_ssim(df_ssim)
        df_delta_model = _compute_model_delta_table(model_data)
        df_delta_human = _compute_human_delta_table(human_data)
        df_join = _join_ssim_and_deltas(df_ssim_w, df_delta_model, df_delta_human)
        df_corr_ssim = _correlate_distortion_vs_delta(df_join)

    # Write SSIM CSVs
    csv_ssim_sum = STAT_DIR / "ssim_summary_combined.csv"
    csv_ssim_join = STAT_DIR / "ssim_delta_join.csv"
    csv_ssim_corr = STAT_DIR / "ssim_vs_delta_correlations.csv"
    _safe_to_csv(df_ssim_w, csv_ssim_sum)
    _safe_to_csv(df_join, csv_ssim_join)
    _safe_to_csv(df_corr_ssim, csv_ssim_corr)

    print("Step 8/8: Compose Markdown roll-up …")
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
    lines.append(f"- Alpha levels: {list(map(int, ALPHA_LEVELS_NUM))}")
    lines.append("")

    # Descriptives (models pooled across subjects)
    lines.append("## Descriptive Statistics by Alpha (Models, pooled across subjects)")
    df_desc_models = df_desc[(df_desc["kind"]=="model") & (df_desc["subject"]=="pooled")].copy()
    df_desc_models = df_desc_models.sort_values(["network","model","alpha"])
    lines.append(df_to_md(df_desc_models[["network","model","alpha","n","mean","std","min","q1","median","q3","max"]], index=False))
    lines.append("")

    if not df_desc[df_desc["kind"]=="human"].empty:
        lines.append("## Human Ratings: Descriptive Statistics by Alpha")
        df_desc_h = df_desc[df_desc["kind"]=="human"].sort_values(["network","alpha"]).copy()
        lines.append(df_to_md(df_desc_h[["kind","network","model","subject","alpha","n","mean","std","min","q1","median","q3","max"]], index=False))
        lines.append("")

    lines.append("## Per-Image Slope Tests (One-sample vs 0)")
    lines.append(df_to_md(df_slope_1samp, index=False))
    lines.append("")

    lines.append("## Per-Image Slope Comparison: VDVAE vs Versatile (Welch t-test)")
    lines.append(df_to_md(df_slope_cmp, index=False))
    lines.append("")

    lines.append("## Correlation Between Averaged Model Curves and Human Curves")
    lines.append(df_to_md(df_corr, index=False))
    lines.append("")

    lines.append("## Linear Trend of Mean Score vs Alpha (Per Subject)")
    lines.append(df_to_md(df_trend, index=False))
    lines.append("")

    lines.append("## Mixed-Effects Tests for Alpha Modulation (score_norm ~ Alpha + (1|subject))")
    lines.append(df_to_md(df_mixed, index=False))
    lines.append("")

    # --- SSIM Section ---
    lines.append("## SSIM vs Δ Cognitive Scores (α compared to α=0)")
    if df_ssim_w.empty:
        lines.append("_No SSIM summaries found at:_")
        lines.append(f"- `{SSIM_ROOT}`")
        lines.append("")
    else:
        lines.append("### Weighted SSIM Means (aggregated across subjects)")
        lines.append(df_to_md(df_ssim_w[['model','network','alpha','n_total','ssim_mean_w','ssim_sd_across_subjects']], index=False))
        lines.append("")
        if not df_join.empty:
            lines.append("### Distortion–Δ Score Join (per model/network)")
            lines.append(df_to_md(df_join[['network','model','alpha','ssim_mean_w','distortion','delta_model_mean','delta_human_mean']], index=False))
            lines.append("")
        if not df_corr_ssim.empty:
            lines.append("### Distortion–Δ Score Correlations")
            lines.append(df_to_md(df_corr_ssim, index=False))
            lines.append("")

    lines.append("## Files")
    lines.append(f"- Descriptives CSV: `{csv_desc}`")
    lines.append(f"- Slope (one-sample) CSV: `{csv_1s}`")
    lines.append(f"- Slope (model-vs-model) CSV: `{csv_cmp}`")
    lines.append(f"- Model–Human curve correlations CSV: `{csv_corr}`")
    lines.append(f"- Linear trend tests CSV: `{csv_trnd}`")
    lines.append(f"- Mixed-effects CSV: `{csv_mixed}`")
    lines.append(f"- SSIM summary (weighted): `{csv_ssim_sum}`")
    lines.append(f"- SSIM–Δ join: `{csv_ssim_join}`")
    lines.append(f"- SSIM–Δ correlations: `{csv_ssim_corr}`")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📝 Statistics report written to: {md_path}")

# =============================================================================
# --- Main ---
# =============================================================================
def main():
    print("--- Starting Statistical Analysis ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    model_data, human_data = load_data()

    print("\n--- Generating Markdown & CSV statistics ---")
    generate_statistics_report(model_data, human_data)

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
