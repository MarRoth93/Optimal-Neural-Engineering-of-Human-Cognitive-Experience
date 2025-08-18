#!/bin/bash
#SBATCH --job-name=reconall_subj01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.err

set -euo pipefail
trap 'echo "ERROR at line $LINENO (exit $?)" >&2' ERR
echo "==== recon-all started on $(hostname) at $(date) ===="

# Modules (order matters on your cluster)
module purge
module load gnu9/9.4.0
module load R-src/4.2.0
module load freesurfer/8.0.0-beta

# Env / license
export ROOT=/home/rothermm/brain-diffuser
export SUBJECTS_DIR=$ROOT/freesurfer
export SUBJECT=subj01
export FS_LICENSE=$ROOT/freesurfer/license.txt
[[ -r "$FS_LICENSE" ]] || { echo "No readable FS_LICENSE at $FS_LICENSE"; exit 2; }

mkdir -p "$SUBJECTS_DIR" "$ROOT/freesurfer/logs"
echo "SUBJECTS_DIR=$SUBJECTS_DIR"

# 1) Find or set the T1w
# Common NSD guess:
T1_GUESS="$ROOT/data/nsddata/ppdata/$SUBJECT/anat/T1w.nii.gz"
T1W="${T1W_OVERRIDE:-}"

if [[ -z "${T1W}" ]]; then
  if [[ -f "$T1_GUESS" ]]; then
    T1W="$T1_GUESS"
  else
    # search for a likely T1 file
    T1W=$(find "$ROOT/data" -type f \( -iname '*subj01*T1*.nii*' -o -iname '*T1w*.nii*' \) 2>/dev/null | head -n 1 || true)
  fi
fi

if [[ -z "${T1W}" || ! -f "$T1W" ]]; then
  echo "!! Could not find T1w. Set T1W_OVERRIDE to the path and re-run."
  echo "   Example: export T1W_OVERRIDE=/path/to/subj01_T1w.nii.gz ; sbatch $0"
  exit 1
fi

echo "Using T1W: $T1W"

# 2) Run recon-all (parallel CPU)
recon-all -i "$T1W" -s "$SUBJECT" -sd "$SUBJECTS_DIR"

# If this subject was already imported, continue with -all. Otherwise this initializes; run -all:
recon-all -s "$SUBJECT" -sd "$SUBJECTS_DIR" -all -parallel -openmp ${SLURM_CPUS_PER_TASK:-8}

# 3) Verify expected outputs
[[ -f "$SUBJECTS_DIR/$SUBJECT/mri/brainmask.mgz" ]] || { echo "!! brainmask missing"; exit 1; }
[[ -f "$SUBJECTS_DIR/$SUBJECT/surf/lh.white" ]] || { echo "!! lh.white missing"; exit 1; }
[[ -f "$SUBJECTS_DIR/$SUBJECT/surf/rh.white" ]] || { echo "!! rh.white missing"; exit 1; }

echo "✅ recon-all complete for $SUBJECT at $(date)"
