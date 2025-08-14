#!/bin/bash
#SBATCH --job-name=make_mean_epi_subj01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.err

set -euo pipefail
trap 'echo "ERROR at line $LINENO (exit $?)" >&2' ERR

echo "==== Job started on $(hostname) at $(date) ===="
LOGDIR=/home/rothermm/brain-diffuser/freesurfer/logs
mkdir -p "$LOGDIR"

# --- Modules (order matters on your cluster) ---
module purge
module load gnu9/9.4.0
module load R-src/4.2.0
module load freesurfer/8.0.0-beta
module load fsl/6.0.7.16
module load miniconda

# --- Conda activation (robust) ---
if [[ -n "${CONDA_ROOT:-}" && -f "$CONDA_ROOT/bin/activate" ]]; then
  source "$CONDA_ROOT/bin/activate"
else
  CONDA_BASE=$(conda info --base)
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate brain-diffuser
echo "Using Python: $(which python) ($(python -V))"

# --- Env and license ---
export ROOT=/home/rothermm/brain-diffuser
export SUBJECT=subj01
export SUBJECTS_DIR=$ROOT/freesurfer
export FS_LICENSE=$ROOT/freesurfer/license.txt
export MPLBACKEND=Agg
[[ -r "$FS_LICENSE" ]] || { echo "No readable FS_LICENSE at $FS_LICENSE"; exit 2; }

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OUT_FILE="$ROOT/data/nsddata/ppdata/${SUBJECT}/func1pt8mm/mean_epi_from_betas.nii.gz"
if [[ -f "$OUT_FILE" ]]; then
  echo "Mean EPI already exists: $OUT_FILE (skipping)"
  exit 0
fi

# ---- Inline Python with full logging ----
python - << 'PY' 2>&1 | tee "/home/rothermm/brain-diffuser/freesurfer/logs/make_mean_epi_subj01_${SLURM_JOB_ID}.log"
import os, sys
from pathlib import Path
import numpy as np
import nibabel as nib
ROOT = os.environ.get("ROOT", "/home/rothermm/brain-diffuser")
SUBJECT = os.environ.get("SUBJECT", "subj01")
betas_dir = Path(f"{ROOT}/data/nsddata_betas/ppdata/{SUBJECT}/func1pt8mm/betas_fithrf_GLMdenoise_RR")
out_dir   = Path(f"{ROOT}/data/nsddata/ppdata/{SUBJECT}/func1pt8mm")
out_dir.mkdir(parents=True, exist_ok=True)
out_file  = out_dir / "mean_epi_from_betas.nii.gz"

print(f"[INFO] ROOT={ROOT}")
print(f"[INFO] SUBJECT={SUBJECT}")
print(f"[INFO] betas_dir={betas_dir}")
print(f"[INFO] out_file={out_file}")

if not betas_dir.exists():
    print(f"[ERROR] Betas directory does not exist: {betas_dir}", file=sys.stderr)
    sys.exit(2)

sess_files = sorted(betas_dir.glob("betas_session*.nii.gz"))
print(f"[INFO] Found {len(sess_files)} session files")
if not sess_files:
    print(f"[ERROR] No betas_session*.nii.gz files in {betas_dir}", file=sys.stderr)
    sys.exit(2)

sum_img = None
ref_img = None

for i, f in enumerate(sess_files, 1):
    try:
        img = nib.load(str(f))
        # (X,Y,Z,750) — compute per-session mean along time
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim != 4 or data.shape[-1] < 1:
            print(f"[WARN] Unexpected shape {data.shape} in {f.name}; skipping", file=sys.stderr)
            continue
        m = data.mean(axis=3)  # (X,Y,Z)
        if sum_img is None:
            sum_img = m
            ref_img = img
        else:
            sum_img += m
        print(f"[INFO] Session {i:02d}/{len(sess_files)}: {f.name} mean ok, shape {m.shape}")
        del data, m, img
    except Exception as e:
        print(f"[ERROR] Failed on {f.name}: {e}", file=sys.stderr)
        sys.exit(3)

if sum_img is None or ref_img is None:
    print("[ERROR] No valid sessions processed; cannot write output", file=sys.stderr)
    sys.exit(3)

grand_mean = (sum_img / float(len(sess_files))).astype(np.float32)
out_img = nib.Nifti1Image(grand_mean, ref_img.affine, ref_img.header)
nib.save(out_img, str(out_file))
print(f"[OK] Wrote {out_file}")
PY

# Fail if Python failed
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "!! Python failed; see log in $LOGDIR/make_mean_epi_subj01_${SLURM_JOB_ID}.log"
  exit 1
fi

# Final existence check
[[ -f "$OUT_FILE" ]] || { echo "!! mean EPI not created: $OUT_FILE"; exit 1; }

echo "==== Job finished at $(date) ===="
