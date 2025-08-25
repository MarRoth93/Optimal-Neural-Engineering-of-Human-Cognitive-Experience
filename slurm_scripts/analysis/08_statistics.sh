#!/bin/bash
#
# SBATCH DIRECTIVES
#SBATCH --job-name=stats_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/logs/%x_%j.err

# --- PATHS ---
PY_FILE="/home/rothermm/brain-diffuser/scripts/analysis/statistics_test.py"
LOG_DIR="/home/rothermm/brain-diffuser/slurm_scripts/logs"
mkdir -p "/home/rothermm/brain-diffuser/logs" "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working Directory: $(pwd)"
echo "Python file: ${PY_FILE}"
echo "Log dir:     ${LOG_DIR}"

# --- ENVIRONMENT ---
set -euo pipefail

module purge
module load miniconda

# Initialize conda for non-interactive shells
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not on PATH after 'module load miniconda'." >&2
  exit 1
fi

# Activate env
conda activate brain-diffuser || { echo "ERROR: failed to activate env 'brain-diffuser'." >&2; exit 1; }
echo "Activated Conda environment: $(which python)"
python -V

# Helpful diagnostics (won't fail the job if missing)
python - <<'PY' || true
import sys
print("Runtime Python:", sys.executable)
try:
    import pandas as pd
    print("pandas:", pd.__version__, "has to_markdown:", hasattr(pd.DataFrame({"a":[1]}), "to_markdown"))
except Exception as e:
    print("pandas import failed:", e)
try:
    import tabulate
    print("tabulate: OK")
except Exception as e:
    print("tabulate NOT available:", e)
PY

# --- RUN ---
echo "==== Running statistics.py at $(date) ===="
set -o pipefail
python -u "${PY_FILE}" 2>&1 | tee "${LOG_DIR}/stats_full_${SLURM_JOB_ID}.log"
SCRIPT_STATUS=${PIPESTATUS[0]}

if [ "${SCRIPT_STATUS}" -ne 0 ]; then
    echo "!! ERROR: Statistics job failed with exit code ${SCRIPT_STATUS}" >&2
else
    echo "Statistics job completed successfully."
fi

echo "==== Finished at $(date) with final status: ${SCRIPT_STATUS} ===="
exit ${SCRIPT_STATUS}
