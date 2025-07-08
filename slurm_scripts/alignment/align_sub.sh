#!/bin/bash
#SBATCH --job-name=align_sub_job
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/scripts/logs/align_sub_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/scripts/logs/align_sub_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1         # remove or set to gpu:0 if no GPU needed
#SBATCH --mem=128G
#SBATCH --partition=normal


# --- Check for subject argument ---------------------------------------------
if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <subject_number>"
  echo "e.g.: sbatch $0 2"
  exit 1
fi
sub=$1

# --- Load your conda environment ---------------------------------------------
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

# --- Run alignment -----------------------------------------------------------
echo "==== Running alignment for subject ${sub} at $(date) ===="
python -u /home/rothermm/brain-diffuser/scripts/alignment/align_sub.py --sub "${sub}" \
    2>&1 | tee /home/rothermm/brain-diffuser/scripts/logs/align_sub_sub${sub}_${SLURM_JOB_ID}.log
echo "==== Finished subject ${sub} at $(date) ===="

echo "==== Job finished at $(date) ===="
