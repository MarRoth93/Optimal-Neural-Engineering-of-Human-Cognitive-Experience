#!/bin/bash
#SBATCH --job-name=vol2surf_flatmaps_subj01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.out
#SBATCH --error=/home/rothermm/brain-diffuser/freesurfer/logs/%x_%j.err

set -euo pipefail
trap 'echo "ERROR at line $LINENO (exit $?)" >&2' ERR

# --- Modules (order matters on your cluster) ---
module purge
module load gnu9/9.4.0
module load R-src/4.2.0
module load freesurfer/8.0.0-beta
module load fsl/6.0.7.16
module load miniconda

# --- Conda activation ---
if [[ -n "${CONDA_ROOT:-}" && -f "$CONDA_ROOT/bin/activate" ]]; then
  source "$CONDA_ROOT/bin/activate"
else
  CONDA_BASE=$(conda info --base)
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate brain-diffuser
echo "Using Python: $(which python) ($(python -V))"

# --- Env ---
export ROOT=/home/rothermm/brain-diffuser
export SUBJECT=subj01
export SUBJECTS_DIR="$ROOT/freesurfer"
export FS_LICENSE="$ROOT/freesurfer/license.txt"
export MPLBACKEND=Agg
[[ -r "$FS_LICENSE" ]] || { echo "No readable FS_LICENSE at $FS_LICENSE"; exit 2; }

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# --- Force PyCortex to use project-local filestore & isolated config ---
export CORTEX_DB="$ROOT/results/pycortex_db"
export XDG_CONFIG_HOME="$ROOT/results/.pycortex_cfg"
mkdir -p "$CORTEX_DB" "$XDG_CONFIG_HOME/pycortex"

# Write a minimal options.cfg so legacy PyCortex honors our filestore
if [[ ! -f "$XDG_CONFIG_HOME/pycortex/options.cfg" ]]; then
  cat > "$XDG_CONFIG_HOME/pycortex/options.cfg" <<EOF
[basic]
filestore = $CORTEX_DB
EOF
fi

# Paths
IN_DIR="$ROOT/data/brain_mapping/$SUBJECT"
REGFILE="$ROOT/results/nifti/${SUBJECT}_register.dat"

# Sanity checks
need=(
  "$IN_DIR/recon_fmri_alpha-4_idx755_memnet.nii.gz"
  "$IN_DIR/recon_fmri_alpha+4_idx755_memnet.nii.gz"
  "$REGFILE"
  "$SUBJECTS_DIR/$SUBJECT/mri/brainmask.mgz"
)
for f in "${need[@]}"; do [[ -e "$f" ]] || { echo "!! Missing: $f"; exit 1; }; done

# Add this section before the python call in your SLURM script

# --- PyCortex subject cleanup (prevent interactive prompts) ---
echo "Checking PyCortex subject state..."
PYCORTEX_SUBJ_DIR="$CORTEX_DB/$SUBJECT"

if [[ -d "$PYCORTEX_SUBJ_DIR" ]]; then
    echo "Found existing PyCortex subject directory: $PYCORTEX_SUBJ_DIR"
    
    # Check if it's complete (has essential files)
    if [[ -f "$PYCORTEX_SUBJ_DIR/transforms.hdf" ]] && \
       [[ -d "$PYCORTEX_SUBJ_DIR/surf" ]] && \
       [[ -f "$PYCORTEX_SUBJ_DIR/surf/lh.fiducial" ]] && \
       [[ -f "$PYCORTEX_SUBJ_DIR/surf/rh.fiducial" ]]; then
        echo "PyCortex subject appears complete; will skip import"
    else
        echo "PyCortex subject appears incomplete; removing for clean import"
        rm -rf "$PYCORTEX_SUBJ_DIR"
    fi
fi

# Ensure non-interactive environment
export DEBIAN_FRONTEND=noninteractive
export CORTEX_NONINTERACTIVE=1

# Run
python -u "$ROOT/freesurfer/vol2surf_and_flatmaps.py" \
  --subject "$SUBJECT" \
  --root "$ROOT" \
  --subjects_dir "$SUBJECTS_DIR" \
  --register "$REGFILE" \
  --symmetric \
  2>&1 | tee "/home/rothermm/brain-diffuser/freesurfer/logs/vol2surf_flatmaps_${SUBJECT}_${SLURM_JOB_ID}.log"

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "!! vol2surf/flatmaps failed; see freesurfer/logs/vol2surf_flatmaps_${SUBJECT}_${SLURM_JOB_ID}.log"
  exit 1
fi

echo "✅ Surf files in:  $ROOT/results/surf_native/$SUBJECT"
echo "✅ Flatmaps in:    $ROOT/results/flatmaps/$SUBJECT"
echo "✅ Filestore at:   $CORTEX_DB"
