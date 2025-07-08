#!/bin/bash

#SBATCH --job-name=versdiff_recon_thetas_memnet
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/slurm_scripts/logs/%x_sub%a_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/slurm_scripts/logs/%x_sub%a_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# --- VALIDATION ---
if [ -z "$1" ]; then
    echo "!! ERROR: No subject number provided." >&2
    echo "   Usage: sbatch $0 <subject_number>" >&2
    echo "   where <subject_number> must be one of: 1 2 5 7" >&2
    exit 1
fi

SUBJECT_ID="$1"
VALID_SUBJECTS="1 2 5 7"

if [[ ! " $VALID_SUBJECTS " =~ (^|[[:space:]])"$SUBJECT_ID"($|[[:space:]]) ]]; then
    echo "!! ERROR: Invalid subject number '${SUBJECT_ID}'." >&2
    echo "   Must be one of: ${VALID_SUBJECTS}" >&2
    exit 1
fi

echo "Processing Subject ID: ${SUBJECT_ID}"

# --- ENVIRONMENT ---
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# --- JOB LOGIC ---
echo "==== Starting Versatile Diffusion for subject ${SUBJECT_ID} at $(date) ===="

python -u /home/rothermm/brain-diffuser/scripts/analysis/vd_recon_thetas_memnet.py \
    --sub "${SUBJECT_ID}" \
    2>&1 | tee /home/rothermm/brain-diffuser/slurm_scripts/logs/versdiff_sub${SUBJECT_ID}_${SLURM_JOB_ID}.log

SCRIPT_STATUS=${PIPESTATUS[0]}

if [ "${SCRIPT_STATUS}" -ne 0 ]; then
    echo "!! ERROR: Script failed for subject ${SUBJECT_ID} with exit code ${SCRIPT_STATUS}" >&2
else
    echo "Script completed successfully for subject ${SUBJECT_ID}"
fi

echo "==== Finished subject ${SUBJECT_ID} at $(date) ===="
echo "==== Job finished at $(date) with final status: ${SCRIPT_STATUS} ===="

exit ${SCRIPT_STATUS}
