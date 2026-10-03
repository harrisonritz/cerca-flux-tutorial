#!/bin/bash
# =============================================================================
# Oxford vs Princeton motor response, run in the background
# =============================================================================
# Runs the cerca_flux pipeline over the Oxford participant and then every Princeton
# participant, with IDENTICAL preprocessing (configs/motor/preprocessing.yaml), and finally
# draws the overlay (Oxford thick black, Princeton thin coloured lines).  It detaches from
# your terminal, so you can close it or log out; check on it whenever you like.
#
#   run/run_motor_compare.sh start  [options]    check the setup, then run in the background
#   run/run_motor_compare.sh status [--variant]  is it running? how far has it got?
#   run/run_motor_compare.sh log    [-f]         show the log (-f follows it)
#   run/run_motor_compare.sh stop                stop it (and everything it started)
#   run/run_motor_compare.sh run    [options]    the same as `start`, in the foreground
#
# Options (start / run):
#   --variant NAME      name this set of options; its results get their own folder (default: default)
#   --set KEY=VALUE     change a setting for BOTH sites, repeatable:   --set hfc.order=3
#   --jobs N            Princeton recordings processed at once (default 2; memory-hungry)
#   --subjects "007 008" only these Princeton subjects (default: the full sample in princeton.yaml)
#   --data DIR          the data folder, if it is not $TSX_DIR/data  (sets TSX_DATA)
#   --tsx DIR           the TSX folder, if this repository is not inside it (sets TSX_DIR)
#   --out DIR           where the figures and metrics table go (default: outputs/motor_response/<variant>)
#   --overwrite         recompute stages whose results already exist
#   --require-all       refuse to start unless every participant passes the pre-flight check
#   --no-compare        process only; do not draw the figures
#
# Changing preprocessing (see `uv run cerca-flux settings --config configs/motor/princeton.yaml`):
#   run/run_motor_compare.sh start --variant hfc3 --set hfc.order=3
#   run/run_motor_compare.sh start --variant no-ecg-keep-flat --set ica.detect_ecg=true --set qc.flag_flat=true
# Results of different variants never mix, and `compare` refuses results whose settings differ
# between the two sites or that predate a settings change.
#
# Results:  outputs/motor_response/<variant>/        (figures + metrics table)
# Logs:     logs/motor_compare/<variant>/            (latest.log, run.pid, steps.txt)
# =============================================================================

# NB: written for bash 3.2 (macOS): no `set -u` (it aborts on empty arrays), no mapfile.

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERCA_FLUX="${CERCA_FLUX:-uv run cerca-flux}"     # override for testing: CERCA_FLUX="python -m cerca_flux.cli"
read -r -a CF <<< "$CERCA_FLUX"

usage() { awk 'NR>1 && /^# =+$/ {n++; if (n==3) exit} NR>1 {sub(/^# ?/, ""); print}' "$SELF"; }

# ---- arguments ---------------------------------------------------------------------

COMMAND="${1:-}"; [ $# -gt 0 ] && shift
VARIANT="default"; JOBS=2; SUBJECTS=""; OUT=""; OVERWRITE=0; REQUIRE_ALL=0; COMPARE=1; FOLLOW=0
SETS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --variant)     VARIANT="${2:?--variant needs a name}"; shift 2 ;;
        --set)         SETS+=(--set "${2:?--set needs KEY=VALUE}"); shift 2 ;;
        --jobs)        JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --subjects)    SUBJECTS="${2:?--subjects needs a list}"; shift 2 ;;
        --data)        export TSX_DATA="${2:?--data needs a directory}"; shift 2 ;;
        --tsx)         export TSX_DIR="${2:?--tsx needs a directory}"; shift 2 ;;
        --out)         OUT="${2:?--out needs a directory}"; shift 2 ;;
        --overwrite)   OVERWRITE=1; shift ;;
        --require-all) REQUIRE_ALL=1; shift ;;
        --no-compare)  COMPARE=0; shift ;;
        -f|--follow)   FOLLOW=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

case "$VARIANT" in *[!A-Za-z0-9.-]*|"") echo "--variant may only contain letters, digits, '.' and '-'" >&2; exit 2 ;; esac
case "$JOBS" in *[!0-9]*|"") echo "--jobs must be a number" >&2; exit 2 ;; esac

# This repository sits next to data/, mne-opm/ and TSX_OPM/ in the TSX folder.
: "${TSX_DIR:=$(cd "$REPO_DIR/.." && pwd)}"
export TSX_DIR

RUN_DIR="${MOTOR_COMPARE_LOGS:-$REPO_DIR/logs/motor_compare}/$VARIANT"
PID_FILE="$RUN_DIR/run.pid"
STEPS_FILE="$RUN_DIR/steps.txt"
LATEST_LOG="$RUN_DIR/latest.log"

# ---- helpers -----------------------------------------------------------------------

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

kill_tree() {   # kill a process and all its descendants (portable: no setsid / process groups)
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
    kill -"${2:-TERM}" "$pid" 2>/dev/null
}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

step_status() { echo "$(stamp)  $*" >> "$STEPS_FILE"; echo "[$(stamp)] $*"; }

run_step() {   # run a command so that the stop trap can interrupt it (bash defers traps otherwise)
    "$@" &
    CHILD=$!
    wait "$CHILD"
    local rc=$?
    CHILD=""
    return $rc
}

kill_children() { local c; for c in $(pgrep -P $$ 2>/dev/null); do kill_tree "$c" TERM; done; }

# ---- the work ----------------------------------------------------------------------

do_run() {
    cd "$REPO_DIR" || exit 1
    mkdir -p "$RUN_DIR"
    export MPLBACKEND=Agg MNE_BROWSER_BACKEND=agg PYVISTA_OFF_SCREEN=true
    : "${OMP_NUM_THREADS:=4}"; export OMP_NUM_THREADS

    # stop cleanly when asked, taking every child process with us
    trap 'step_status "STOPPED"; kill_children; exit 143' TERM INT
    # keep a laptop awake while this runs (macOS only; harmless elsewhere)
    if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w $$ & fi

    local -a common; common=(--variant "$VARIANT" "${SETS[@]}")
    local -a extra=()
    [ "$OVERWRITE" = 1 ] && extra+=(--overwrite)
    local -a pick=()
    # shellcheck disable=SC2206
    [ -n "$SUBJECTS" ] && pick=(--subjects $SUBJECTS)

    : > "$STEPS_FILE"
    echo "variant: $VARIANT | TSX_DIR: $TSX_DIR | TSX_DATA: ${TSX_DATA:-<TSX_DIR>/data} | jobs: $JOBS"
    echo "overrides: ${SETS[*]:-none}"
    echo "settings that differ from the Cerca defaults:"
    "${CF[@]}" settings --config configs/motor/princeton.yaml --changed "${common[@]}" | sed 's/^/    /'

    local failed=0
    step_status "oxford: started"
    if run_step "${CF[@]}" run --config configs/motor/oxford.yaml --preset motor "${common[@]}" "${extra[@]}"; then
        step_status "oxford: done"
    else
        step_status "oxford: FAILED (exit $?)"; failed=1
    fi

    step_status "princeton: started"
    if run_step "${CF[@]}" run --config configs/motor/princeton.yaml --preset motor --n-jobs "$JOBS" \
            "${pick[@]}" "${common[@]}" "${extra[@]}"; then
        step_status "princeton: done"
    else
        # Some participants failing is expected and reported; the others are still compared.
        step_status "princeton: finished with failures (exit $?) - see the metrics table"
    fi

    if [ "$COMPARE" = 1 ]; then
        step_status "compare: started"
        if run_step "${CF[@]}" compare --reference configs/motor/oxford.yaml --cohort configs/motor/princeton.yaml \
                "${common[@]}" ${OUT:+--out "$OUT"}; then
            step_status "compare: done -> ${OUT:-outputs/motor_response/$VARIANT/}"
        else
            step_status "compare: FAILED (exit $?)"; failed=1
        fi
    fi
    step_status "FINISHED$([ "$failed" = 1 ] && echo ' with errors')"
    return "$failed"
}

do_preflight() {
    cd "$REPO_DIR" || exit 1
    local -a common; common=(--variant "$VARIANT" "${SETS[@]}")
    echo "Checking the setup (TSX_DIR=$TSX_DIR${TSX_DATA:+, TSX_DATA=$TSX_DATA}) ..."
    if ! "${CF[@]}" check --config configs/motor/oxford.yaml "${common[@]}"; then
        echo; echo "The Oxford reference is not ready, so there is nothing to compare against. Not starting." >&2
        return 1
    fi
    local -a pick=()
    # shellcheck disable=SC2206
    [ -n "$SUBJECTS" ] && pick=(--subjects $SUBJECTS)
    if ! "${CF[@]}" check --config configs/motor/princeton.yaml "${pick[@]}" "${common[@]}"; then
        if [ "$REQUIRE_ALL" = 1 ]; then
            echo; echo "Not starting (--require-all). Those participants will fail if you continue." >&2
            return 1
        fi
        echo; echo "Some Princeton participants are not ready; they will be skipped and listed as missing."
    fi
}

# ---- commands ----------------------------------------------------------------------

case "$COMMAND" in
    run)
        do_run
        ;;

    start)
        if is_running; then
            echo "A run for variant '$VARIANT' is already running (pid $(cat "$PID_FILE")). Use: $0 status --variant $VARIANT"
            exit 1
        fi
        do_preflight || exit 1
        mkdir -p "$RUN_DIR"
        LOG="$RUN_DIR/run_$(date '+%Y%m%d_%H%M%S').log"
        ln -sf "$(basename "$LOG")" "$LATEST_LOG"
        # Re-invoke ourselves as the foreground worker, detached from this terminal.
        nohup bash "$SELF" run --variant "$VARIANT" --jobs "$JOBS" \
            ${SUBJECTS:+--subjects "$SUBJECTS"} ${OUT:+--out "$OUT"} "${SETS[@]}" \
            $([ "$OVERWRITE" = 1 ] && echo --overwrite) \
            $([ "$COMPARE" = 0 ] && echo --no-compare) \
            > "$LOG" 2>&1 < /dev/null &
        echo $! > "$PID_FILE"
        disown 2>/dev/null || true
        echo
        echo "Started in the background (pid $(cat "$PID_FILE"))."
        echo "  watch it:  $0 status --variant $VARIANT"
        echo "  the log:   $0 log -f --variant $VARIANT     ($LOG)"
        echo "  stop it:   $0 stop --variant $VARIANT"
        echo "  results:   ${OUT:-outputs/motor_response/$VARIANT/}"
        ;;

    status)
        if is_running; then
            echo "variant '$VARIANT': RUNNING (pid $(cat "$PID_FILE"))"
        elif [ -f "$STEPS_FILE" ]; then
            echo "variant '$VARIANT': not running"
        else
            echo "variant '$VARIANT': no run recorded in $RUN_DIR"; exit 1
        fi
        [ -f "$STEPS_FILE" ] && { echo "steps:"; sed 's/^/  /' "$STEPS_FILE"; }
        if [ -f "$LATEST_LOG" ]; then echo "last lines of the log:"; tail -n 6 "$LATEST_LOG" | cut -c1-200 | sed 's/^/  /'; fi
        ;;

    log)
        [ -f "$LATEST_LOG" ] || { echo "no log for variant '$VARIANT' ($RUN_DIR)"; exit 1; }
        if [ "$FOLLOW" = 1 ]; then tail -n 40 -f "$LATEST_LOG"; else cat "$LATEST_LOG"; fi
        ;;

    stop)
        if ! is_running; then echo "variant '$VARIANT': not running"; exit 0; fi
        PID="$(cat "$PID_FILE")"
        echo "Stopping variant '$VARIANT' (pid $PID) ..."
        kill -TERM "$PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        if kill -0 "$PID" 2>/dev/null; then echo "still alive; forcing it"; kill_tree "$PID" KILL; fi
        rm -f "$PID_FILE"
        echo "Stopped. Finished steps are cached, so starting again resumes where it left off."
        ;;

    ""|-h|--help|help) usage ;;

    *) echo "unknown command: $COMMAND (start | status | log | stop | run)" >&2; exit 2 ;;
esac
