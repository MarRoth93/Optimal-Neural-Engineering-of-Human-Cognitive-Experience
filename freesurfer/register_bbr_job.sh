#!/bin/bash
#SBATCH --job-name=register_subj01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.err

set -euo pipefail
trap 'echo "ERROR at line $LINENO (exit $?)" >&2' ERR

module purge
module load gnu9/9.4.0
module load R-src/4.2.0
module load freesurfer/8.0.0-beta
module load fsl/6.0.7.16
module load miniconda

# Conda activation
if [[ -n "${CONDA_ROOT:-}" && -f "$CONDA_ROOT/bin/activate" ]]; then
  source "$CONDA_ROOT/bin/activate"
else
  CONDA_BASE=$(conda info --base)
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate brain-diffuser

export ROOT=/home/rothermm/brain-diffuser
export SUBJECTS_DIR=$ROOT/freesurfer
export SUBJECT=subj01
export FS_LICENSE=$ROOT/freesurfer/license.txt
export MPLBACKEND=Agg

MEAN_EPI="$ROOT/data/nsddata/ppdata/${SUBJECT}/func1pt8mm/mean_epi_from_betas.nii.gz"
REGFILE="$ROOT/results/nifti/${SUBJECT}_register.dat"

[[ -f "$MEAN_EPI" ]] || { echo "!! Missing mean EPI: $MEAN_EPI"; exit 1; }
[[ -f "$SUBJECTS_DIR/$SUBJECT/mri/brainmask.mgz" ]] || { echo "!! No brainmask at $SUBJECTS_DIR/$SUBJECT/mri/brainmask.mgz"; exit 1; }

python -u "$ROOT/freesurfer/register_bbr.py" \
  --subject "$SUBJECT" \
  --root "$ROOT" \
  --subjects_dir "$SUBJECTS_DIR" \
  --mean-epi "$MEAN_EPI" \
  --out "$REGFILE" \
  --init fsl \
  --force \
  2>&1 | tee "/home/rothermm/brain-diffuser/freesurfer/logs/register_${SUBJECT}_${SLURM_JOB_ID}.log"

[[ -s "$REGFILE" ]] && echo "✅ register.dat ready: $REGFILE" || { echo "!! register.dat missing"; exit 1; }
