#!/bin/bash
#
# Semantic Heavy Theta - Full Dataset Reconstruction
#
# This script reconstructs ALL test images using the semantic_heavy theta variant.
# Unlike the comparison script (03_*), this processes the full dataset rather than
# just a single image for visualization.
#
# Variant: semantic_heavy (1.0, 0.3, 0.05)
#   - Strong emphasis on LOW layers (semantic/abstract features)
#   - Reduced MID layer contribution (0.3)
#   - Minimal HIGH layer contribution (0.05)
#   - Best balance for semantic manipulation with structure preservation
#
# Alpha range: [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
#   - Negative values: Decrease attribute (valence/memorability)
#   - Zero: Baseline (no manipulation)
#   - Positive values: Increase attribute
#
# Output: results/hybrid_theta_reconstructions/subj{XX}/{assessor}/semantic_heavy/alpha_{alpha}/
#         Contains all test images (~982 per subject) at each alpha value
#
# Usage: sbatch 04_semantic_heavy_reconstruct_all.sh
#

#SBATCH --job-name=04_semantic_heavy_all
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=04:00:00
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
# Each subject: 2 assessors × 7 alphas × ~982 images = ~13,748 images
# Total across all subjects: ~54,992 images

for sub in "${subjects[@]}"; do
    echo "==== Reconstructing all images for subject $sub at $(date) ===="
    echo "Variant: semantic_heavy (1.0, 0.3, 0.05)"
    echo "Assessors: emonet, memnet"
    echo "Alpha values: -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5"
    echo "Expected output: ~13,748 images per subject"

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
echo "Total images generated: ~54,992 (across 4 subjects)"
exit $overall_status
