#!/bin/bash
#
# Versatile Diffusion Enhancement for Hierarchical Theta Reconstructions
#
# This script applies Versatile Diffusion to hierarchical theta-manipulated images
# to enhance quality while preserving structural similarity. The key is using
# LOW diffusion strength (0.2-0.4) to maintain structure while allowing semantic
# improvements from CLIP guidance.
#
# Structure Preservation:
#   - Diffusion strength controls how much the image can change
#   - Lower strength (0.2-0.4) = preserves structure, enhances quality
#   - Higher strength (>0.5) = allows structural changes (not desired here)
#
# Output: results/versatile_diffusion_hierarchical/subj{XX}/{assessor}/{group}/alpha_{alpha}/
#
# Usage: sbatch 04_versatile_diffusion_hierarchical.sh
#

#SBATCH --job-name=04_versatile_diffusion_hierarchical
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

# --- CONFIGURATION ---
subjects=(1 2 5 7)
overall_status=0

# CRITICAL PARAMETER: Diffusion strength for structure preservation
# 0.2 = Maximum structure preservation (subtle quality improvement)
# 0.3 = RECOMMENDED - Good balance between preservation and enhancement
# 0.4 = More enhancement, some structure changes possible
# 0.5+ = Significant changes (NOT recommended for preserving theta effects)
DIFF_STR=0.3

# Mixing ratio between vision and text conditioning
# 0.5 = Balanced (recommended)
# <0.5 = More text influence
# >0.5 = More vision influence
MIX_STR=0.5

# Which layer groups to process
# Options: all, low, mid, high
# For most analyses, "all" is sufficient
LAYER_GROUPS="all"

# Which assessors to process
ASSESSORS="emonet memnet"

# Which alpha values to process
ALPHAS="-50 0 50"

echo "======================================"
echo "VERSATILE DIFFUSION CONFIGURATION"
echo "======================================"
echo "Diffusion strength: $DIFF_STR (structure preservation)"
echo "Mixing ratio: $MIX_STR"
echo "Layer groups: $LAYER_GROUPS"
echo "Assessors: $ASSESSORS"
echo "Alphas: $ALPHAS"
echo "======================================"
echo ""

# --- JOB LOGIC ---
for sub in "${subjects[@]}"; do
    echo "======================================"
    echo "Processing Subject $sub at $(date)"
    echo "======================================"
    echo ""

    python -u /home/rothermm/brain-diffuser/scripts/analysis/versatile_diffusion_hierachical.py \
        --sub "$sub" \
        --diff_str $DIFF_STR \
        --mix_str $MIX_STR \
        --assessors $ASSESSORS \
        --alphas $ALPHAS \
        --layer_groups $LAYER_GROUPS \
        2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Versatile Diffusion processing for subject $sub failed"
        overall_status=1
    fi

    echo ""
    echo "======================================"
    echo "Finished subject $sub at $(date)"
    echo "======================================"
    echo ""
done

echo ""
echo "======================================"
echo "Job finished at $(date)"
echo "======================================"
echo "Overall status: $overall_status"
echo ""
echo "Output location:"
echo "  /home/rothermm/brain-diffuser/results/versatile_diffusion_hierarchical/"
echo ""
echo "Structure preservation notes:"
echo "  - Diffusion strength: $DIFF_STR"
echo "  - Lower = more structure preserved"
echo "  - Higher = more quality enhancement but risks structure changes"
echo "  - Your theta manipulations should remain visible in the output"
echo "======================================"

exit $overall_status
