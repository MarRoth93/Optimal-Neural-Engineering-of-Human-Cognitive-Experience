#!/bin/bash
#
# Hybrid Theta Variants - Full Dataset Reconstruction
#
# This script reconstructs ALL test images using ALL theta variants.
# Unlike the comparison script (03_*), this processes the full dataset rather than
# just a single image for visualization.
#
# Variants (6 total):
#   - original (1.0, 1.0, 1.0): Baseline, no layer-specific weighting
#   - semantic_heavy (1.0, 0.3, 0.05): Strong emphasis on LOW layers (semantic)
#   - semantic_only (1.0, 0.2, 0.0): Only LOW layers active (semantic)
#   - balanced (1.0, 0.6, 0.2): Moderate contribution from all layers
#   - structural_heavy (0.3, 0.6, 1.0): Strong emphasis on HIGH layers (structural)
#   - structural_only (0.0, 0.2, 1.0): Only HIGH layers active (structural)
#
# Alpha range: [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
#   - Negative values: Decrease attribute (valence/memorability)
#   - Zero: Baseline (no manipulation)
#   - Positive values: Increase attribute
#
# Output: results/hybrid_theta_reconstructions/subj{XX}/{assessor}/{variant}/alpha_{alpha}/
#         Contains all test images (~982 per subject) at each alpha value
#
# Usage: sbatch 04_semantic_heavy_reconstruct_all.sh
#

#SBATCH --job-name=04_all_variants
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
subjects=(1 2 5 7)
overall_status=0

# Process all subjects sequentially
# Each subject: 2 assessors × 6 variants × 7 alphas × ~982 images = ~82,488 images
# Total across all subjects: ~329,952 images

for sub in "${subjects[@]}"; do
    echo "==== Reconstructing all images for subject $sub at $(date) ===="
    echo "Assessors: emonet, memnet (2 total)"
    echo "Variants: original, semantic_heavy, semantic_only, balanced, structural_heavy, structural_only (6 total)"
    echo "Alpha values: -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5 (7 total)"
    echo "Expected output: ~82,488 images per subject"

    python -u /home/rothermm/brain-diffuser/scripts/analysis/reconstruct_semantic_heavy_all.py \
        --sub "$sub" \
        --bs 30 \
        2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Reconstruction for subject $sub failed"
        overall_status=1
    else
        echo "✓ Successfully generated all images for subject $sub"
    fi

    echo "==== Finished subject $sub at $(date) ===="
    echo ""
done

echo "==== Job finished at $(date) with overall status: $overall_status ===="
echo "Total images generated: ~329,952 (across 4 subjects, 6 variants)"
exit $overall_status
