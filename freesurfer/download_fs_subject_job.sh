#!/bin/bash
#SBATCH --job-name=dl_fs_subj01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.err

set -euo pipefail
trap 'echo "ERROR at line $LINENO (exit $?)" >&2' ERR

echo "==== Job started on $(hostname) at $(date) ===="
LOGDIR=/home/rothermm/brain-diffuser/freesurfer/logs
mkdir -p "$LOGDIR"

# Modules: we only need awscli + a Python to run the script
module purge
module load awscli 2>/dev/null || true
module load miniconda

# Conda activation (robust)
if [[ -n "${CONDA_ROOT:-}" && -f "$CONDA_ROOT/bin/activate" ]]; then
  source "$CONDA_ROOT/bin/activate"
else
  CONDA_BASE=$(conda info --base)
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate brain-diffuser
echo "Using Python: $(which python) ($(python -V))"
aws --version || { echo "!! aws CLI not available. Try: module load awscli"; exit 2; }

export ROOT=/home/rothermm/brain-diffuser
export SUBJECT=subj01

# Run the downloader, tee logs
python -u "$ROOT/freesurfer/download_fs_subject.py" \
  --subject "$SUBJECT" \
  --root "$ROOT" \
  --also-fsaverage \
  --mode safe \
  2>&1 | tee "$LOGDIR/download_${SUBJECT}_${SLURM_JOB_ID}.log"



if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "!! Download failed; see $LOGDIR/download_${SUBJECT}_${SLURM_JOB_ID}.log"
  exit 1
fi

# Double-check key files exist
for f in mri/brainmask.mgz surf/lh.white surf/lh.pial surf/rh.white surf/rh.pial; do
  test -f "$ROOT/freesurfer/$SUBJECT/$f" || { echo "!! Missing $f"; exit 1; }
done

echo "✅ FreeSurfer subject ready at $ROOT/freesurfer/$SUBJECT"
echo "==== Job finished at $(date) ===="
