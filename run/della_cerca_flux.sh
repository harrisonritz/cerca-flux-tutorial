#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH -J cerca-flux
#SBATCH --output logs/%A_cerca-flux_%a.txt
#SBATCH --array=0,7-15,18-39,41-44


# ============================================================================
# SLURM array job: the cerca_flux motor-response pipeline [DELLA CLUSTER]
# ============================================================================
#
# Array index 0       -> the Oxford reference participant
# Array index 7..44   -> that Princeton subject number (as in della_mne_batch.sh)
#
# Run it from the repository root so logs/ exists:
#
#   mkdir -p logs
#   JOB=$(sbatch --parsable run/della_cerca_flux.sh)
#   sbatch --dependency=afterany:$JOB run/della_cerca_flux_compare.sh
#
# Try a preprocessing option without editing any file (the variant name keeps its
# outputs in a separate derivatives folder, and the same options must be passed
# to the compare job):
#
#   sbatch --export=ALL,VARIANT=hfc3,EXTRA="--set hfc.order=3" run/della_cerca_flux.sh
#
# Stages (the `motor` preset): qc, hfc, annotate, ica, epochs, motor, forward,
# motor_source, report.  Every stage caches its output, so a failed task can simply
# be resubmitted:  sbatch --array=24 run/della_cerca_flux.sh
# ============================================================================


# ==== EDITABLE DEFAULTS ====

VARIANT="${VARIANT:-default}"   # names this set of analysis choices
EXTRA="${EXTRA:-}"              # extra cerca-flux arguments, e.g. "--set hfc.order=3"

# Paths: this repository sits next to data/, mne-opm/ and TSX_OPM/ in the TSX folder
BASE_PATH="/projects/NDAW/harrison_ritz"   # "/projects/NDAW/harrison_ritz" OR "/scratch/gpfs/NDAW/harrison_ritz"
export TSX_DIR="$BASE_PATH/TSX"
REPO_DIR="$TSX_DIR/cerca-flux-tutorial"

# ==== END EDITABLE DEFAULTS ====


cd "$REPO_DIR" || exit 1

export MPLBACKEND=agg
export MNE_BROWSER_BACKEND=agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    SITE="Oxford"
    CONFIG="configs/motor/oxford.yaml"
    SELECT=()
else
    SITE="Princeton"
    CONFIG="configs/motor/princeton.yaml"
    SELECT=(--subjects "$(printf '%03d' "$SLURM_ARRAY_TASK_ID")")
fi

echo "============================================"
echo "cerca_flux motor response"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Site:          $SITE ${SELECT[*]}"
echo "Config:        $CONFIG"
echo "Variant:       $VARIANT"
echo "Extra args:    $EXTRA"
echo "TSX_DIR:       $TSX_DIR"
echo "Node:          $SLURM_NODELIST"
echo "Start time:    $(date)"
echo "============================================"

# random onset delay, to spread the load on the shared file system
DELAY=$((RANDOM % 60))
echo "Starting after a random delay of $DELAY seconds."
sleep $DELAY

# shellcheck disable=SC2086   # $EXTRA is deliberately word-split
uv run cerca-flux run \
    --config "$CONFIG" \
    "${SELECT[@]}" \
    --preset motor \
    --variant "$VARIANT" \
    $EXTRA

EXIT_STATUS=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "============================================"

exit $EXIT_STATUS
