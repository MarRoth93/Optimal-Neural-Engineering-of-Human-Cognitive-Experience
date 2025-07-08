#!/bin/bash

#SBATCH --job-name=01_compute_theta
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
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
echo "Activated Conda environment: $(which python)"

# --- JOB LOGIC ---
subjects=(1 2 5 7)
overall_status=0

for sub in "${subjects[@]}"; do
    echo "==== Starting subject $sub at $(date) ===="

    python -u /home/rothermm/brain-diffuser/scripts/analysis/compute_theta.py --sub "$sub" \
    2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Theta computation for subject $sub failed"
        overall_status=1
    fi

    echo "==== Finished subject $sub at $(date) ===="
done

echo "==== Job finished at $(date) with overall status: $overall_status ===="
exit $overall_status
