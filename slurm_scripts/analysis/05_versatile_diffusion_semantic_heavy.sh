#!/bin/bash
#
# Versatile Diffusion - Semantic Heavy Refinement
#
# This script applies Versatile Diffusion to semantic_heavy theta reconstructions
# to achieve high-quality photorealistic images while preserving:
#   1. Semantic manipulations from theta (valence/memorability changes)
#   2. Structural similarity to original reconstructions
#
# Input: results/hybrid_theta_reconstructions/subj{XX}/{assessor}/semantic_heavy/
# Output: results/versatile_diffusion_semantic_heavy/subj{XX}/{assessor}/semantic_heavy/
#
# Key Parameters for Structure Preservation:
#   - diff_str: 0.3 (30% diffusion strength) - Lower preserves more structure
#   - mix_str: 0.5 (balanced vision/text conditioning)
#   - ddim_steps: 50 (quality vs speed tradeoff)
#   - scale: 7.5 (unconditional guidance scale)
#
# The low diffusion strength (0.3) is crucial: it allows CLIP guidance to enhance
# quality and semantics without destroying the structural properties encoded in
# the VDVAE latent space manipulations.
#
# Usage: sbatch 05_versatile_diffusion_semantic_heavy.sh
#

#SBATCH --job-name=05_vd_semantic_heavy
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=06:00:00
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
# Each subject: 2 assessors × 7 alphas × ~982 images = ~13,748 refined images
# Total across all subjects: ~54,992 images

for sub in "${subjects[@]}"; do
    echo "==== Processing Versatile Diffusion for subject $sub at $(date) ===="
    echo "Input: semantic_heavy theta reconstructions"
    echo "Variant: semantic_heavy (1.0, 0.3, 0.05)"
    echo "Assessors: emonet, memnet"
    echo "Alpha values: -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5"
    echo "Diffusion strength: 0.3 (structure preservation mode)"
    echo "Expected output: ~13,748 refined images per subject"

    python -u /home/rothermm/brain-diffuser/scripts/analysis/versatile_diffusion_semantic_heavy.py \
        --sub "$sub" \
        --diff_str 0.3 \
        --mix_str 0.5 \
        --ddim_steps 50 \
        --scale 7.5 \
        --ddim_eta 0.0 \
        --assessors emonet memnet \
        --alphas -1.5 -1 -0.5 0 0.5 1 1.5 \
        2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Versatile Diffusion refinement for subject $sub failed"
        overall_status=1
    else
        echo "✓ Successfully refined all images for subject $sub"
        echo "✓ Generated comparison visualization"
    fi

    echo "==== Finished subject $sub at $(date) ===="
    echo ""
done

echo "==== Job finished at $(date) with overall status: $overall_status ===="
echo "Total refined images: ~54,992 (across 4 subjects)"
echo ""
echo "Output structure:"
echo "  /results/versatile_diffusion_semantic_heavy/subj{01,02,05,07}/"
echo "    ├── emonet/semantic_heavy/alpha_{-1.5...1.5}/"
echo "    └── memnet/semantic_heavy/alpha_{-1.5...1.5}/"
echo ""
echo "Settings used:"
echo "  - Diffusion strength: 0.3 (structure preservation)"
echo "  - Mixing ratio: 0.5 (balanced vision/text)"
echo "  - DDIM steps: 50"
echo "  - Guidance scale: 7.5"
exit $overall_status
