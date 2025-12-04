#!/bin/bash
#SBATCH --job-name=vdvae_roundtrip_universal_test
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda: $(which python)"

# >>>>>>>>>>>> EDIT THIS: universal test images file <<<<<<<<<<<<
IMAGES_NPY="/home/rothermm/brain-diffuser/data/processed_data/subj01/nsd_test_stim_sub1.npy"

BS=64
OUTSIZE=512

mkdir -p logs

python -u /home/rothermm/brain-diffuser/scripts/04b_vdvae_image_to_image.py \
  --images-npy "${IMAGES_NPY}" \
  --bs "${BS}" \
  --resize "${OUTSIZE}" \
  --save-latents --save-flattened \
  2>&1 | tee "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

status=${PIPESTATUS[0]}
echo "==== Job finished at $(date) with status ${status} ===="
exit ${status}
