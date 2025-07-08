#!/bin/bash

#SBATCH --job-name=05_cliptext_regression
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# --- ENVIRONMENT ---
# Load Conda properly
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# --- JOB LOGIC ---
# List of subjects to process
subjects=(5 7)
# Initialize an overall status to track failures
overall_status=0

for sub in "${subjects[@]}"; do
    echo "==== Starting subject $sub at $(date) ===="
    
    # Using relative path for the tee log file
    python -u /home/rothermm/brain-diffuser/scripts/cliptext_regression.py --sub "$sub" \
    2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    # Check the exit status of the python script (the left side of the pipe)
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: CLIP text regression for subject $sub failed"
        # Set the overall status to 1 to indicate failure
        overall_status=1
    fi
    echo "==== Finished subject $sub at $(date) ===="
done

echo "==== Job finished at $(date) with overall status: $overall_status ===="
# Exit with the overall status. Slurm will see this as a failure if it's non-zero.
exit $overall_status