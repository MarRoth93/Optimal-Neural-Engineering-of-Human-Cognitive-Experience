#!/usr/bin/env python3
"""
Analyze and visualize the effects of different theta variants on assessor scores.

This script:
1. Loads assessor scores for all subjects, variants, and alpha values
2. Computes statistics (mean, std, effect sizes) across subjects
3. Performs statistical tests to compare variants
4. Generates publication-quality visualizations

Output:
- Statistical summaries (CSV files)
- Comparison plots (PNG/PDF)
- Effect size analyses
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_rel, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.3)

def load_all_scores(base_dir, subjects, assessor):
    """
    Load scores for all subjects for a given assessor.
    
    Returns:
        dict: {variant: {alpha: [scores across all subjects]}}
    """
    all_data = {}
    
    for sub in subjects:
        pkl_path = base_dir / f"subj{sub:02d}" / f"{assessor}_scores.pkl"
        if not pkl_path.exists():
            print(f"⚠️  Warning: File not found: {pkl_path}")
            continue
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Aggregate across subjects
        for variant, alpha_dict in data.items():
            if variant not in all_data:
                all_data[variant] = {}
            for alpha, scores in alpha_dict.items():
                if alpha not in all_data[variant]:
                    all_data[variant][alpha] = []
                all_data[variant][alpha].extend(scores)
    
    return all_data

def compute_effect_size(baseline_scores, manipulated_scores):
    """
    Compute Cohen's d effect size.
    """
    mean_diff = np.mean(manipulated_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt((np.var(baseline_scores) + np.var(manipulated_scores)) / 2)
    if pooled_std == 0:
        return 0
    return mean_diff / pooled_std

def create_summary_dataframe(data, assessor):
    """
    Create a summary dataframe with mean, std, and counts for each condition.
    """
    rows = []
    for variant, alpha_dict in data.items():
        for alpha, scores in alpha_dict.items():
            rows.append({
                'assessor': assessor,
                'variant': variant,
                'alpha': alpha,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_images': len(scores),
                'sem': np.std(scores) / np.sqrt(len(scores))
            })
    return pd.DataFrame(rows)

def compute_effect_sizes(data):
    """
    Compute effect sizes relative to baseline (alpha_0) for each variant.
    """
    rows = []
    for variant, alpha_dict in data.items():
        if 'alpha_0' not in alpha_dict:
            continue
        baseline = alpha_dict['alpha_0']
        
        for alpha, scores in alpha_dict.items():
            if alpha == 'alpha_0':
                effect_size = 0
            else:
                effect_size = compute_effect_size(baseline, scores)
            
            rows.append({
                'variant': variant,
                'alpha': alpha,
                'effect_size': effect_size
            })
    return pd.DataFrame(rows)

def perform_statistical_tests(data):
    """
    Perform statistical tests comparing variants at each alpha level.
    """
    results = []
    
    # Get list of alphas (excluding baseline for some tests)
    alphas = sorted(set(alpha for alpha_dict in data.values() for alpha in alpha_dict.keys()))
    
    for alpha in alphas:
        # Collect scores for each variant at this alpha
        variant_scores = {}
        for variant, alpha_dict in data.items():
            if alpha in alpha_dict:
                variant_scores[variant] = alpha_dict[alpha]
        
        if len(variant_scores) < 2:
            continue
        
        # Friedman test (non-parametric repeated measures)
        # We need to match images across variants (assuming same order)
        min_len = min(len(scores) for scores in variant_scores.values())
        truncated_scores = [scores[:min_len] for scores in variant_scores.values()]
        
        if min_len > 0:
            stat, p_value = friedmanchisquare(*truncated_scores)
            results.append({
                'alpha': alpha,
                'test': 'Friedman',
                'statistic': stat,
                'p_value': p_value,
                'n_variants': len(variant_scores),
                'n_samples': min_len
            })
    
    return pd.DataFrame(results)

def plot_score_trajectories(data, assessor, output_dir):
    """
    Plot mean scores across alpha values for each variant.
    """
    # Prepare data for plotting
    plot_data = []
    for variant, alpha_dict in data.items():
        for alpha, scores in alpha_dict.items():
            # Extract numeric alpha value
            alpha_val = float(alpha.replace('alpha_', ''))
            plot_data.append({
                'variant': variant,
                'alpha': alpha_val,
                'mean_score': np.mean(scores),
                'sem': np.std(scores) / np.sqrt(len(scores))
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors and line styles for variants
    variant_order = ['original', 'balanced', 'semantic_heavy', 'semantic_only', 
                     'structural_heavy', 'structural_only']
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_order)))
    
    for i, variant in enumerate(variant_order):
        if variant not in df['variant'].values:
            continue
        variant_df = df[df['variant'] == variant].sort_values('alpha')
        ax.plot(variant_df['alpha'], variant_df['mean_score'], 
                marker='o', linewidth=2.5, markersize=8,
                label=variant.replace('_', ' ').title(),
                color=colors[i], alpha=0.8)
        
        # Add error bars
        ax.fill_between(variant_df['alpha'], 
                        variant_df['mean_score'] - variant_df['sem'],
                        variant_df['mean_score'] + variant_df['sem'],
                        alpha=0.2, color=colors[i])
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline (α=0)')
    ax.set_xlabel('Alpha (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{assessor.title()} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Effect of Theta Variants on {assessor.title()} Scores\n(Averaged across all subjects)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{assessor}_score_trajectories.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{assessor}_score_trajectories.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {assessor}_score_trajectories.png/pdf")

def plot_effect_sizes(effect_df, assessor, output_dir):
    """
    Plot effect sizes (Cohen's d) for each variant across alpha values.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    variant_order = ['original', 'balanced', 'semantic_heavy', 'semantic_only', 
                     'structural_heavy', 'structural_only']
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_order)))
    
    for i, variant in enumerate(variant_order):
        if variant not in effect_df['variant'].values:
            continue
        variant_df = effect_df[effect_df['variant'] == variant].copy()
        variant_df['alpha_numeric'] = variant_df['alpha'].str.replace('alpha_', '').astype(float)
        variant_df = variant_df.sort_values('alpha_numeric')
        
        ax.plot(variant_df['alpha_numeric'], variant_df['effect_size'],
                marker='s', linewidth=2.5, markersize=8,
                label=variant.replace('_', ' ').title(),
                color=colors[i], alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add reference lines for small, medium, large effect sizes
    ax.axhline(y=0.2, color='green', linestyle=':', linewidth=1, alpha=0.3, label='Small effect (d=0.2)')
    ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.3, label='Medium effect (d=0.5)')
    ax.axhline(y=0.8, color='red', linestyle=':', linewidth=1, alpha=0.3, label='Large effect (d=0.8)')
    
    ax.set_xlabel('Alpha (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
    ax.set_title(f'Effect Sizes Relative to Baseline (α=0) - {assessor.title()}\n(Averaged across all subjects)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{assessor}_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{assessor}_effect_sizes.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {assessor}_effect_sizes.png/pdf")

def plot_variant_comparison_heatmap(data, assessor, output_dir):
    """
    Create a heatmap showing mean scores for each variant-alpha combination.
    """
    # Prepare data
    rows = []
    for variant, alpha_dict in data.items():
        for alpha, scores in alpha_dict.items():
            alpha_val = float(alpha.replace('alpha_', ''))
            rows.append({
                'variant': variant,
                'alpha': alpha_val,
                'mean_score': np.mean(scores)
            })
    
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='variant', columns='alpha', values='mean_score')
    
    # Reorder variants
    variant_order = ['structural_only', 'structural_heavy', 'balanced', 
                     'semantic_heavy', 'semantic_only', 'original']
    pivot = pivot.reindex([v for v in variant_order if v in pivot.index])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=pivot.loc['original'].mean(),
                cbar_kws={'label': f'{assessor.title()} Score'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Alpha (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Theta Variant', fontsize=14, fontweight='bold')
    ax.set_title(f'Mean {assessor.title()} Scores by Variant and Alpha\n(Averaged across all subjects)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Format y-axis labels
    ax.set_yticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_yticklabels()], 
                       rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{assessor}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{assessor}_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {assessor}_heatmap.png/pdf")

def plot_extreme_alpha_comparison(data, assessor, output_dir):
    """
    Bar plot comparing variants at extreme alpha values.
    """
    extreme_alphas = ['alpha_-1.5', 'alpha_0', 'alpha_1.5']
    
    plot_data = []
    for variant, alpha_dict in data.items():
        for alpha in extreme_alphas:
            if alpha in alpha_dict:
                scores = alpha_dict[alpha]
                plot_data.append({
                    'variant': variant.replace('_', ' ').title(),
                    'alpha': alpha.replace('alpha_', 'α='),
                    'mean_score': np.mean(scores),
                    'sem': np.std(scores) / np.sqrt(len(scores))
                })
    
    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create grouped bar plot
    x = np.arange(len(df['variant'].unique()))
    width = 0.25
    alphas_list = df['alpha'].unique()
    
    for i, alpha in enumerate(alphas_list):
        alpha_df = df[df['alpha'] == alpha]
        offset = (i - len(alphas_list)/2 + 0.5) * width
        bars = ax.bar(x + offset, alpha_df['mean_score'], width, 
                      label=alpha, alpha=0.8)
        
        # Add error bars
        ax.errorbar(x + offset, alpha_df['mean_score'], 
                   yerr=alpha_df['sem'], fmt='none', 
                   color='black', capsize=3, alpha=0.5)
    
    ax.set_xlabel('Theta Variant', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{assessor.title()} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Comparison of Variants at Extreme Alpha Values - {assessor.title()}\n(Averaged across all subjects)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['variant'].unique(), rotation=45, ha='right', fontsize=11)
    ax.legend(title='Alpha', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{assessor}_extreme_alpha_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{assessor}_extreme_alpha_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {assessor}_extreme_alpha_comparison.png/pdf")

def rank_variants_by_effect(effect_df, assessor, output_dir):
    """
    Rank variants by their maximum absolute effect size.
    """
    # Calculate max absolute effect for each variant
    max_effects = effect_df.groupby('variant')['effect_size'].apply(
        lambda x: x.abs().max()
    ).sort_values(ascending=False)
    
    # Create ranking table
    ranking = pd.DataFrame({
        'Rank': range(1, len(max_effects) + 1),
        'Variant': [v.replace('_', ' ').title() for v in max_effects.index],
        'Max Absolute Effect Size': max_effects.values
    })
    
    # Save to CSV
    ranking.to_csv(output_dir / f'{assessor}_variant_ranking.csv', index=False)
    print(f"  ✓ Saved: {assessor}_variant_ranking.csv")
    
    return ranking

def main():
    print("="*70)
    print("Hybrid Theta Variant Analysis")
    print("="*70)
    
    # Paths
    BASE_DIR = Path("/home/rothermm/brain-diffuser")
    SCORES_DIR = BASE_DIR / "results" / "assessor_scores" / "hybrid_theta"
    OUTPUT_DIR = BASE_DIR / "results" / "statistics" / "hybrid_theta_analysis"
    FIG_DIR = BASE_DIR / "figures" / "hybrid_theta_analysis"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Subjects
    SUBJECTS = [1, 2, 5, 7]
    ASSESSORS = ['emonet', 'memnet']
    
    print(f"\nConfiguration:")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Assessors: {ASSESSORS}")
    print(f"  Input dir: {SCORES_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Figures dir: {FIG_DIR}")
    print()
    
    # Process each assessor
    for assessor in ASSESSORS:
        print(f"\n{'='*70}")
        print(f"Processing: {assessor.upper()}")
        print(f"{'='*70}")
        
        # Load all scores
        print(f"\n1. Loading scores for all subjects...")
        data = load_all_scores(SCORES_DIR, SUBJECTS, assessor)
        
        if not data:
            print(f"  ⚠️  No data found for {assessor}")
            continue
        
        n_variants = len(data)
        n_alphas = len(next(iter(data.values())))
        print(f"  ✓ Loaded {n_variants} variants × {n_alphas} alpha values")
        
        # Create summary statistics
        print(f"\n2. Computing summary statistics...")
        summary_df = create_summary_dataframe(data, assessor)
        summary_df.to_csv(OUTPUT_DIR / f'{assessor}_summary_statistics.csv', index=False)
        print(f"  ✓ Saved: {assessor}_summary_statistics.csv")
        
        # Compute effect sizes
        print(f"\n3. Computing effect sizes...")
        effect_df = compute_effect_sizes(data)
        effect_df.to_csv(OUTPUT_DIR / f'{assessor}_effect_sizes.csv', index=False)
        print(f"  ✓ Saved: {assessor}_effect_sizes.csv")
        
        # Rank variants
        print(f"\n4. Ranking variants by effect magnitude...")
        ranking = rank_variants_by_effect(effect_df, assessor, OUTPUT_DIR)
        print(f"\n  Variant Ranking (by max absolute effect):")
        for _, row in ranking.iterrows():
            print(f"    {row['Rank']}. {row['Variant']}: d={row['Max Absolute Effect Size']:.3f}")
        
        # Statistical tests
        print(f"\n5. Performing statistical tests...")
        stats_df = perform_statistical_tests(data)
        stats_df.to_csv(OUTPUT_DIR / f'{assessor}_statistical_tests.csv', index=False)
        print(f"  ✓ Saved: {assessor}_statistical_tests.csv")
        print(f"  Significant differences at p<0.05: {(stats_df['p_value'] < 0.05).sum()}/{len(stats_df)}")
        
        # Generate visualizations
        print(f"\n6. Generating visualizations...")
        plot_score_trajectories(data, assessor, FIG_DIR)
        plot_effect_sizes(effect_df, assessor, FIG_DIR)
        plot_variant_comparison_heatmap(data, assessor, FIG_DIR)
        plot_extreme_alpha_comparison(data, assessor, FIG_DIR)
        
        print(f"\n✓ Completed analysis for {assessor}")
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to:")
    print(f"  Statistics: {OUTPUT_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"\nGenerated files per assessor:")
    print(f"  - summary_statistics.csv (means, stds, SEMs)")
    print(f"  - effect_sizes.csv (Cohen's d values)")
    print(f"  - variant_ranking.csv (variants ranked by effect)")
    print(f"  - statistical_tests.csv (Friedman test results)")
    print(f"  - score_trajectories.png/pdf")
    print(f"  - effect_sizes.png/pdf")
    print(f"  - heatmap.png/pdf")
    print(f"  - extreme_alpha_comparison.png/pdf")
    print()

if __name__ == "__main__":
    main()
