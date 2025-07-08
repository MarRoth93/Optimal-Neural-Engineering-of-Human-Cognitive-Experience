#!/bin/bash
#SBATCH --job-name=versdiff_recon
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/scripts/logs/versdiff_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/scripts/logs/versdiff_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal

echo "==== Job started on $(hostname) at $(date) ===="

# Expect the subject number as the first argument
if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <subject_number>"
  echo "  where <subject_number> must be one of: 1 2 5 7"
  exit 1
fi

sub="$1"
if [[ ! "1 2 5 7" =~ (^|[[:space:]])"$sub"($|[[:space:]]) ]]; then
  echo "Error: Subject number must be one of 1, 2, 5, or 7."
  exit 1
fi

echo "==== Running Versatile Diffusion for subject $sub at $(date) ===="

# Load Conda environment
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

python -u /home/rothermm/brain-diffuser/scripts/analysis/versatile_diffusion_reconstruct_images_thetas.py \
       --sub "$sub" \
    2>&1 | tee /home/rothermm/brain-diffuser/scripts/logs/versdiff_sub${sub}_${SLURM_JOB_ID}.log

echo "==== Finished subject $sub at $(date) ===="
echo "==== Job finished at $(date) ===="
