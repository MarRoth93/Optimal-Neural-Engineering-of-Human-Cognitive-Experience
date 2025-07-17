#!/bin/bash

#SBATCH --job-name=graphs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # reserve 4 CPU cores for this task
#SBATCH --mem=16G                  # total memory per node
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
overall_status=0

echo "==== Running graphs.py at $(date) ===="
python -u /home/rothermm/brain-diffuser/scripts/analysis/graphs.py \
    2>&1 | tee logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "!! ERROR: graphs.py failed"
    overall_status=1
fi

echo "==== Finished graphs.py at $(date) ===="
echo "==== Job finished at $(date) with status: $overall_status ===="
exit $overall_status
