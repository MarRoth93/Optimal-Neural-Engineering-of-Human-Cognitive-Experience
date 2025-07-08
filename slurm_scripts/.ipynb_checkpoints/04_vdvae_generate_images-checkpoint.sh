#!/bin/bash
#SBATCH --job-name=vdvae_reconstruct
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/scripts/logs/vdvae_reconstruct_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/scripts/logs/vdvae_reconstruct_%j.err
#SBATCH --time=08:00:00  # Adjust depending on number of subjects and time per subject
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal

echo "==== Job started on $(hostname) at $(date) ===="

# Load Conda properly
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# List of subjects to process
subjects=(1 2 5 7)

for sub in "${subjects[@]}"; do
    echo "==== Starting subject $sub at $(date) ===="
    python -u /home/rothermm/brain-diffuser/scripts/vdvae_reconstruct_images.py --sub "$sub" \
    2>&1 | tee /home/rothermm/brain-diffuser/scripts/logs/vdvae_reconstruct_sub${sub}_${SLURM_JOB_ID}.log
    echo "==== Finished subject $sub at $(date) ===="
done

echo "==== Job finished at $(date) ===="

