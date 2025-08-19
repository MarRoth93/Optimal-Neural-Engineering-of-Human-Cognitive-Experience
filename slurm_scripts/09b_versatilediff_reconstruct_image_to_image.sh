#!/bin/bash
#SBATCH --job-name=vd_original_latents
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/slurm_scripts/analysis/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/slurm_scripts/analysis/logs/%x_%j.err
#SBATCH --chdir=/home/rothermm/brain-diffuser


set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="

module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"

SUB=01   # embeddings live under subj paths; change if needed
I2I_DIR="/home/rothermm/brain-diffuser/results/vdvae/image_to_image"
CLIP_VISION="/home/rothermm/brain-diffuser/data/extracted_features/subj${SUB}/nsd_clipvision_test.npy"
CLIP_TEXT="/home/rothermm/brain-diffuser/data/extracted_features/subj${SUB}/nsd_cliptext_test.npy"

python -u /home/rothermm/brain-diffuser/scripts/versatilediffusion_reconstruct_image_to_image.py \
  -sub ${SUB} \
  --i2i-dir "${I2I_DIR}" \
  --clip-vision-npy "${CLIP_VISION}" \
  --clip-text-npy "${CLIP_TEXT}" \
  --ddim-steps 50 \
  --strength 0.50 \
  --mixing 0.20 \
  --scale 20.0 \
  --seed 0 \
  2>&1 | tee "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

status=${PIPESTATUS[0]}
echo "==== Job finished with status ${status} at $(date) ===="
exit ${status}
