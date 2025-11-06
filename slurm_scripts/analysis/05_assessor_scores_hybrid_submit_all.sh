#!/bin/bash
#
# Submit assessor scoring jobs for all subjects
#
# This is a convenience script to submit separate jobs for each subject
# in parallel. Each subject will run independently.
#
# Usage: bash 05_assessor_scores_submit_all.sh
#

echo "=========================================="
echo "Submitting assessor scoring jobs for all subjects"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/05_assessor_scores_hybrid_thetas.sh"

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

SUBJECTS=(1 2 5 7)
JOB_IDS=()

for sub in "${SUBJECTS[@]}"; do
    echo "Submitting job for subject $sub..."
    JOB_ID=$(sbatch "$SLURM_SCRIPT" "$sub" | awk '{print $NF}')
    
    if [ -n "$JOB_ID" ]; then
        echo "  ✓ Subject $sub submitted with Job ID: $JOB_ID"
        JOB_IDS+=("$JOB_ID")
    else
        echo "  ✗ Failed to submit job for subject $sub"
    fi
done

echo ""
echo "=========================================="
echo "Submission complete!"
echo "=========================================="
echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[@]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  logs/05_hybrid_scores_sub*"
echo ""
