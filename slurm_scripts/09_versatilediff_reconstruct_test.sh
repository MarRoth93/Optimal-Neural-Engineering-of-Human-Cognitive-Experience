#!/bin/bash

#SBATCH --job-name=08_versatilediff_reconstruct
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_sub%a_%j.out
#SBATCH --error=logs/%x_sub%a_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# --- VALIDATION ---
# Check if a subject ID was provided as an argument.
# In a sbatch script, command-line arguments start from $1.
if [ -z "$1" ]; then
    echo "!! ERROR: No subject ID provided." >&2
    echo "   Usage: sbatch $0 <subject_id>" >&2
    exit 1
fi

# Assign the first argument to a variable for clarity
SUBJECT_ID=$1
echo "Processing Subject ID: ${SUBJECT_ID}"

# --- ENVIRONMENT ---
# Load Conda properly
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# --- JOB LOGIC ---
echo "==== Starting subject ${SUBJECT_ID} at $(date) ===="

# Execute the python script for the specified subject
# The script's final exit code will be determined by the python command's success or failure
python -u /home/rothermm/brain-diffuser/scripts/versatilediffusion_reconstruct_images_original-test.py --sub "${SUBJECT_ID}" \
2>&1 | tee logs/${SLURM_JOB_NAME}_sub${SUBJECT_ID}_${SLURM_JOB_ID}.log

# Capture the exit status of the python script (from the left side of the pipe)
script_status=${PIPESTATUS[0]}

if [ "${script_status}" -ne 0 ]; then
    echo "!! ERROR: Versatile Diffusion reconstruction for subject ${SUBJECT_ID} failed with status ${script_status}"
fi

echo "==== Finished subject ${SUBJECT_ID} at $(date) ===="

echo "==== Job finished at $(date) with final status: ${script_status} ===="
# Exit with the python script's status. Slurm will interpret any non-zero value as a job failure.
exit $script_status
