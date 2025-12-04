#!/bin/bash

#SBATCH --job-name=03_assessor_scores_original_sub01
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# Ensure logs dir exists for tee output below
mkdir -p /home/rothermm/brain-diffuser/logs

# --- ENVIRONMENT ---
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# --- JOB LOGIC ---
sub=1
script="/home/rothermm/brain-diffuser/scripts/analysis/09_assessor_scores_original_images.py"
log="/home/rothermm/brain-diffuser/logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log"

echo "==== Computing EmoNet/MemNet on ORIGINAL test images for subject $sub at $(date) ===="
python -u "$script" --subject "$sub" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

if [ $status -ne 0 ]; then
  echo "!! ERROR: original assessor scoring failed for subject $sub (exit $status)"
else
  echo "==== Finished subject $sub at $(date) with status 0 ===="
fi

echo "==== Job finished at $(date) with overall status: $status ===="
exit $status
