#!/bin/bash

#SBATCH --job-name=ssim_all_models
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# --- PATHS & DEFAULTS ---
PY_FILE="/home/rothermm/brain-diffuser/scripts/analysis/ssim.py"
BASE_DIR="/home/rothermm/brain-diffuser/results"
OUT_DIR="${BASE_DIR}/metrics/ssim"
MODELS=("vdvae" "versatile_diffusion")
NETWORKS=("emonet" "memnet")
IMG_EXT=(".png" ".jpg" ".jpeg")
DEVICE="auto"   # auto | cpu | cuda

# Allow simple overrides via environment variables if desired
# e.g., sbatch --export=ALL,DEVICE=cuda this_script.sh
MODELS=(${MODELS_OVERRIDE:-${MODELS[@]}})
NETWORKS=(${NETWORKS_OVERRIDE:-${NETWORKS[@]}})
IMG_EXT=(${IMG_EXT_OVERRIDE:-${IMG_EXT[@]}})
BASE_DIR="${BASE_OVERRIDE:-$BASE_DIR}"
OUT_DIR="${OUT_OVERRIDE:-$OUT_DIR}"
DEVICE="${DEVICE:-auto}"

echo "Python file: ${PY_FILE}"
echo "Base dir:    ${BASE_DIR}"
echo "Out dir:     ${OUT_DIR}"
echo "Models:      ${MODELS[@]}"
echo "Networks:    ${NETWORKS[@]}"
echo "Img ext:     ${IMG_EXT[@]}"
echo "Device:      ${DEVICE}"

# --- VALIDATION ---
if [ ! -f "${PY_FILE}" ]; then
    echo "!! ERROR: Python script not found: ${PY_FILE}" >&2
    exit 1
fi

# --- ENVIRONMENT ---
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# Optional: ensure deps (quietly). Comment out if offline.
python - <<'PYCHK' || { echo "[deps] Installing deps…"; pip install -q --upgrade piq pytorch-msssim Pillow torch torchvision; }
import importlib
for m in ("piq","PIL","torch","torchvision"):
    importlib.import_module(m)
PYCHK

# --- JOB LOGIC ---
echo "==== Starting SSIM computation at $(date) ===="

# Build CLI arrays
MODEL_ARGS=()
for m in "${MODELS[@]}";   do MODEL_ARGS+=( "$m" ); done

NET_ARGS=()
for n in "${NETWORKS[@]}"; do NET_ARGS+=( "$n" ); done

EXT_ARGS=()
for e in "${IMG_EXT[@]}";  do EXT_ARGS+=( "$e" ); done

set -o pipefail
python -u "${PY_FILE}" \
  --base "${BASE_DIR}" \
  --models "${MODEL_ARGS[@]}" \
  --networks "${NET_ARGS[@]}" \
  --img-ext "${EXT_ARGS[@]}" \
  --device "${DEVICE}" \
  --out "${OUT_DIR}" \
  2>&1 | tee "/home/rothermm/brain-diffuser/slurm_scripts/logs/ssim_all_${SLURM_JOB_ID}.log"
SCRIPT_STATUS=${PIPESTATUS[0]}

if [ "${SCRIPT_STATUS}" -ne 0 ]; then
    echo "!! ERROR: SSIM job failed with exit code ${SCRIPT_STATUS}" >&2
else
    echo "SSIM job completed successfully."
fi

echo "==== Finished at $(date) with final status: ${SCRIPT_STATUS} ===="
exit ${SCRIPT_STATUS}
