#!/bin/bash

#SBATCH --job-name=03_visualize_layer_theta
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=04:00:00
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
echo "Activated Conda env at: $(which python)"

# --- JOB LOGIC ---
assessors=("emonet" "memnet")
overall_status=0

for assessor in "${assessors[@]}"; do
    echo "==== Processing ${assessor} at $(date) ===="

    python -u /home/rothermm/brain-diffuser/scripts/analysis/visualize_layer_specific_theta.py \
        --assessor "$assessor" \
        --alpha 50.0 \
        --n_images 3 \
        --output_dir /home/rothermm/brain-diffuser/results/hierachical_theta \
        2>&1 | tee logs/${SLURM_JOB_NAME}_${assessor}_${SLURM_JOB_ID}.log

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "!! ERROR: Visualization for ${assessor} failed"
        overall_status=1
    fi

    echo "==== Finished ${assessor} at $(date) ===="
done

echo "==== Job finished at $(date) with overall status: $overall_status ===="
exit $overall_status
