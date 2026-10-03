#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH -J cerca-flux-compare
#SBATCH --output logs/%A_cerca-flux-compare.txt


# ============================================================================
# Overlay the Oxford participant (thick black) on every Princeton participant
# (thin coloured lines).  Submit after the array has finished; use `afterany`, not
# `afterok`, so a participant whose job failed does not stop the plots being made
# for the others -- the figure and the metrics table list who is missing.
#
#   sbatch --dependency=afterany:<array-jobid> run/della_cerca_flux_compare.sh
#   sbatch --export=ALL,VARIANT=hfc3,EXTRA="--set hfc.order=3" \
#          --dependency=afterany:<array-jobid> run/della_cerca_flux_compare.sh
#
# Output: outputs/motor_response/<variant>/ in the repository.
# ============================================================================

VARIANT="${VARIANT:-default}"
EXTRA="${EXTRA:-}"

BASE_PATH="/projects/NDAW/harrison_ritz"
export TSX_DIR="$BASE_PATH/TSX"
cd "$TSX_DIR/cerca-flux-tutorial" || exit 1

export MPLBACKEND=agg

# shellcheck disable=SC2086
uv run cerca-flux compare \
    --reference configs/motor/oxford.yaml \
    --cohort configs/motor/princeton.yaml \
    --variant "$VARIANT" \
    $EXTRA

EXIT_STATUS=$?
echo "Exit status: $EXIT_STATUS"
exit $EXIT_STATUS
