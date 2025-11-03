#!/bin/bash
#
# Hybrid Theta Variants Reconstruction
#
# This script runs image reconstruction using the new hybrid theta variants:
#   - original: Full unweighted theta (1.0, 1.0, 1.0)
#   - semantic_heavy: Strong LOW layer emphasis (1.0, 0.3, 0.05)
#   - semantic_only: Only LOW layers active (1.0, 0.2, 0.0)
#   - balanced: Moderate all layers (1.0, 0.6, 0.2)
#
# Reconstructs image index 2 for each subject with alpha values [0, 50]
# to compare semantic manipulation strength across different weighting schemes.
#
# Output: results/hybrid_theta_reconstructions/subj{XX}/{assessor}/{variant}/alpha_{alpha}/
#         + comparison visualization plots
#
# Usage: sbatch 03_vdvae_reconstruct_hierarchical_theta.sh
#

#SBATCH --job-name=03_hybrid_theta_reconstruct
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=02:00:00
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

# Note: Script now only reconstructs image index 2 (hardcoded in Python script)
# This allows comparison of all 4 theta variants × 2 alpha values × 2 assessors
# Total: 4 variants × 2 alphas × 2 assessors × 4 subjects = 64 images + comparison plots

for sub in "${subjects[@]}"; do
    echo "==== Reconstructing subject $sub at $(date) ===="
    echo "Generating image 2 with all hybrid theta variants (original, semantic_heavy, semantic_only, balanced)"
    echo "Assessors: emonet, memnet | Alpha values: 0, 50"

    python -u /home/rothermm/brain-diffuser/scripts/analysis/reconstructions_from_hierachical_theta.py \
        --sub "$sub" \
        --bs 30 \
        2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Reconstruction for subject $sub failed"
        overall_status=1
    else
        echo "✓ Generated comparison visualization for subject $sub"
    fi

    echo "==== Finished subject $sub at $(date) ===="
done

echo "==== Job finished at $(date) with overall status: $overall_status ===="
exit $overall_status
