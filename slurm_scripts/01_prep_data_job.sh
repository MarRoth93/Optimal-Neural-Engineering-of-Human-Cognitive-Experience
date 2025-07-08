#!/bin/bash
#SBATCH --job-name=prep_data_job
#SBATCH --ntasks=1
#SBATCH --output=logs/prep_data_%j.out
#SBATCH --error=logs/prep_data_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/brain-diffuser/data/

# Debugging output
echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Load Conda properly
module purge
module load miniconda
echo "Loaded miniconda"

source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# Run script with passed arguments
echo "Starting prepare_nsddata.py with args: $@"
python -u prepare_nsddata.py "$@" | tee logs/prep_data_${SLURM_JOB_ID}.debug.log

echo "==== Job finished at $(date) ===="

