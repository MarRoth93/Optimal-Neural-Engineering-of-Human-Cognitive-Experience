#!/bin/bash

#SBATCH --job-name=pixcorr_all_models
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --partition=normal

# --- SETUP ---
echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"

# --- PATHS & DEFAULTS ---
PY_FILE="/home/rothermm/brain-diffuser/scripts/analysis/pixel_corr.py"
BASE_RESULTS="/home/rothermm/brain-diffuser/results"
BASE_DATA="/home/rothermm/brain-diffuser/data"
OUT_DIR="${BASE_RESULTS}/metrics/pixcorr"
MODELS=("vdvae" "versatile_diffusion")
NETWORKS=("emonet" "memnet")
SUBJECTS=(1 2 5 7)
IMG_EXT=(".png" ".jpg" ".jpeg")
N=982

# Allow simple overrides via environment variables if desired, e.g.:
# sbatch --export=ALL,MODELS_OVERRIDE="vdvae",SUBJECTS_OVERRIDE="1 2" run_pixcorr_all_models.sh
MODELS=(${MODELS_OVERRIDE:-${MODELS[@]}})
NETWORKS=(${NETWORKS_OVERRIDE:-${NETWORKS[@]}})
SUBJECTS=(${SUBJECTS_OVERRIDE:-${SUBJECTS[@]}})
IMG_EXT=(${IMG_EXT_OVERRIDE:-${IMG_EXT[@]}})
BASE_RESULTS="${BASE_RESULTS_OVERRIDE:-$BASE_RESULTS}"
BASE_DATA="${BASE_DATA_OVERRIDE:-$BASE_DATA}"
OUT_DIR="${OUT_OVERRIDE:-$OUT_DIR}"
N="${N_OVERRIDE:-$N}"

echo "Python file: ${PY_FILE}"
echo "Base results: ${BASE_RESULTS}"
echo "Base data:    ${BASE_DATA}"
echo "Out dir:      ${OUT_DIR}"
echo "Models:       ${MODELS[@]}"
echo "Networks:     ${NETWORKS[@]}"
echo "Subjects:     ${SUBJECTS[@]}"
echo "Img ext:      ${IMG_EXT[@]}"
echo "N images:     ${N}"

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

# Optional: quick dependency probe; install if missing (quiet)
python - <<'PYCHK' || { echo "[deps] Installing deps…"; pip install -q --upgrade Pillow numpy; }
import importlib
for m in ("PIL","numpy"):
    importlib.import_module(m)
PYCHK

# --- JOB LOGIC ---
echo "==== Starting PixCorr computation at $(date) ===="

# Build CLI arrays
MODEL_ARGS=()
for m in "${MODELS[@]}";   do MODEL_ARGS+=( "$m" ); done

NET_ARGS=()
for n in "${NETWORKS[@]}"; do NET_ARGS+=( "$n" ); done

SUBJ_ARGS=()
for s in "${SUBJECTS[@]}"; do SUBJ_ARGS+=( "$s" ); done

EXT_ARGS=()
for e in "${IMG_EXT[@]}";  do EXT_ARGS+=( "$e" ); done

set -o pipefail
python -u "${PY_FILE}" \
  --base-results "${BASE_RESULTS}" \
  --base-data "${BASE_DATA}" \
  --subjects "${SUBJ_ARGS[@]}" \
  --models "${MODEL_ARGS[@]}" \
  --networks "${NET_ARGS[@]}" \
  --img-ext "${EXT_ARGS[@]}" \
  --n "${N}" \
  --out-dir "${OUT_DIR}" \
  2>&1 | tee "/home/rothermm/brain-diffuser/logs/pixcorr_all_${SLURM_JOB_ID}.log"
SCRIPT_STATUS=${PIPESTATUS[0]}

if [ "${SCRIPT_STATUS}" -ne 0 ]; then
  echo "!! ERROR: PixCorr job failed with exit code ${SCRIPT_STATUS}" >&2
else
  echo "PixCorr job completed successfully."
fi

echo "==== Finished at $(date) with final status: ${SCRIPT_STATUS} ===="
exit ${SCRIPT_STATUS}
