#!/bin/bash
#
# Analyze Hybrid Theta Variant Effects
#
# This script performs comprehensive statistical analysis and visualization of
# the effects of different theta variants on assessor scores.
#
# Analysis includes:
#   - Summary statistics (means, standard deviations, SEMs)
#   - Effect size calculations (Cohen's d)
#   - Statistical significance testing (Friedman test)
#   - Variant ranking by effect magnitude
#
# Visualizations generated:
#   1. Score trajectories: Mean scores across alpha values for each variant
#   2. Effect sizes: Cohen's d values relative to baseline (α=0)
#   3. Heatmap: Mean scores for all variant-alpha combinations
#   4. Extreme comparison: Bar plots at α=-1.5, 0, and 1.5
#
# Data is averaged across all subjects (1, 2, 5, 7)
#
# Input: results/assessor_scores/hybrid_theta/subj{XX}/{assessor}_scores.pkl
# Output:
#   - Statistics: results/statistics/hybrid_theta_analysis/
#   - Figures: figures/hybrid_theta_analysis/
#
# Usage: sbatch 06_analyze_hybrid_theta_effects.sh
#

#SBATCH --job-name=06_theta_analysis
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# --- ENVIRONMENT ---
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda env at: $(which python)"

# --- JOB LOGIC ---
echo ""
echo "==== Analyzing hybrid theta variant effects ===="
echo "Subjects: 1, 2, 5, 7 (averaged)"
echo "Assessors: Emonet, MemNet"
echo "Variants: 6 (original, semantic_heavy, semantic_only, balanced, structural_heavy, structural_only)"
echo "Alpha values: 7 (-1.5 to 1.5)"
echo ""

python -u /home/rothermm/brain-diffuser/scripts/analysis/analyze_hybrid_theta_effects.py \
    2>&1 | tee logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "!! ERROR: Analysis failed with exit code $exit_code"
    echo "==== Job finished at $(date) ===="
    exit $exit_code
fi

echo ""
echo "==== Job finished successfully at $(date) ===="
echo ""
echo "Output files generated:"
echo "  Statistics (per assessor):"
echo "    - summary_statistics.csv"
echo "    - effect_sizes.csv"
echo "    - variant_ranking.csv"
echo "    - statistical_tests.csv"
echo ""
echo "  Figures (per assessor):"
echo "    - score_trajectories.png/pdf"
echo "    - effect_sizes.png/pdf"
echo "    - heatmap.png/pdf"
echo "    - extreme_alpha_comparison.png/pdf"
echo ""
echo "Locations:"
echo "  - results/statistics/hybrid_theta_analysis/"
echo "  - figures/hybrid_theta_analysis/"
echo ""

exit 0
