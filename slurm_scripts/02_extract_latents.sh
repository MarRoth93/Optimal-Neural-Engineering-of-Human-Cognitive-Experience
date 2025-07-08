#!/bin/bash
#SBATCH --job-name=01_extract_latents
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=normal


# Load Conda
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

# Subject list
subjects=(1 2 5 7)
overall_status=0

for sub in "${subjects[@]}"; do
    echo "==== Running for subject $sub at $(date) ===="
    python -u /home/rothermm/brain-diffuser/scripts/vdvae_extract_features.py --sub "$sub" --bs 30 \
    2>&1 | tee logs/${SLURM_JOB_NAME}_sub${sub}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Subject $sub failed"
        overall_status=1
    fi
    echo "==== Finished subject $sub at $(date) ===="
done

echo "==== Job finished at $(date) ===="
exit $overall_status
