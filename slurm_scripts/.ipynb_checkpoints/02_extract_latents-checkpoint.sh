#!/bin/bash
#SBATCH --job-name=extract_latents_job
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/scripts/logs/extract_latents_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/scripts/logs/extract_latents_%j.err
#SBATCH --time=08:00:00  # Adjust time for all subjects
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal
#SBATCH --mail-user=marco.rothermel@uni-marburg.de
#SBATCH --mail-type=END

# Load Conda
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

# Subject list
subjects=(1 2 5 7)

# Loop over each subject
for sub in "${subjects[@]}"; do
    echo "==== Running for subject $sub at $(date) ===="
    python -u /home/rothermm/brain-diffuser/scripts/vdvae_extract_features.py --sub "$sub" --bs 30 \
    2>&1 | tee /home/rothermm/brain-diffuser/scripts/logs/extract_features_sub${sub}_${SLURM_JOB_ID}.log
    echo "==== Finished subject $sub at $(date) ===="
done

echo "==== Job finished at $(date) ===="

