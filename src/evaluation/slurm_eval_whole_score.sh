#!/bin/bash
# =============================================================================
# Slurm Parallel Whole-Score MV2H Evaluation — two-phase pipeline
# =============================================================================
#
# Phase 1 (render): one job. Every reference and prediction score is exported
#                   to MIDI and converted; the export cache makes a repeat run
#                   nearly free, and it is the memory-heavy half.
# Phase 2 (eval):   slurm array, depends on Phase 1 (afterok). Each task scores
#                   its range of recordings; the last task merges the shards.
#
# Usage (from project root):
#   MAPPING=... ARMS="ours=a/pairs.jsonl ours_seed91=b/pairs.jsonl" OUTPUT_DIR=... \
#     bash src/evaluation/slurm_eval_whole_score.sh
#
#   # Run it here instead of submitting
#   MAPPING=... ARMS=... OUTPUT_DIR=... bash src/evaluation/slurm_eval_whole_score.sh --local
#
#   # Override per-job size (default 60)
#   RECORDINGS_PER_JOB=30 MAPPING=... bash src/evaluation/slurm_eval_whole_score.sh
#
# Each arm set must use its own OUTPUT_DIR.

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

# A submit shell may have an unrelated Python environment active. Clear its
# markers so Poetry resolves this project's environment on every worker.
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV POETRY_ACTIVE

MAPPING="${MAPPING:-}"
ARMS="${ARMS:-}"
MV2H_BIN="${MV2H_BIN:-external/MV2H/bin}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

RECORDINGS_PER_JOB="${RECORDINGS_PER_JOB:-60}"
RECORDING_TIMEOUT="${RECORDING_TIMEOUT:-}"

arm_arguments=()
for spec in $ARMS; do
    arm_arguments+=(--arm "$spec")
done

# ─────────────────────────────────────────────────────────────────────
# Worker: Phase 1 — export and convert every score once
# ─────────────────────────────────────────────────────────────────────

if [ "$1" = "--render-worker" ]; then
    echo "[render] exporting the whole inventory"
    poetry run python -m src.evaluation.eval_whole_score \
        --phase render \
        --mapping "$MAPPING" \
        "${arm_arguments[@]}" \
        --mv2h-bin "$MV2H_BIN" \
        --output-dir "$OUTPUT_DIR" \
        --workers "${RENDER_WORKERS:-12}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# Worker: Phase 2 — score one range of recordings, last task merges
# ─────────────────────────────────────────────────────────────────────

if [ "$1" = "--eval-worker" ]; then
    TOTAL_UNITS=$2
    N_JOBS=${SLURM_ARRAY_TASK_COUNT}
    TASK_ID=${SLURM_ARRAY_TASK_ID}

    SHARD_SIZE=$(( (TOTAL_UNITS + N_JOBS - 1) / N_JOBS ))
    SHARD_START=$(( TASK_ID * SHARD_SIZE ))
    SHARD_END=$(( SHARD_START + SHARD_SIZE ))
    if [ "$SHARD_END" -gt "$TOTAL_UNITS" ]; then
      SHARD_END=$TOTAL_UNITS
    fi

    echo "[eval $TASK_ID/$N_JOBS] units [$SHARD_START, $SHARD_END) / $TOTAL_UNITS"
    timeout_argument=()
    if [ -n "$RECORDING_TIMEOUT" ]; then
        timeout_argument=(--timeout "$RECORDING_TIMEOUT")
    fi

    if [ "$SHARD_START" -lt "$SHARD_END" ]; then
        poetry run python -m src.evaluation.eval_whole_score \
            --phase eval \
            --mapping "$MAPPING" \
            "${arm_arguments[@]}" \
            --mv2h-bin "$MV2H_BIN" \
            --output-dir "$OUTPUT_DIR" \
            --chunk-start "$SHARD_START" \
            --chunk-end "$SHARD_END" \
            --workers "${WORKERS:-1}" \
            "${timeout_argument[@]}"
    fi

    SHARD_JSON="${OUTPUT_DIR}/results_shard_$(printf '%06d' $SHARD_START)_$(printf '%06d' $SHARD_END).json"
    if [ "$SHARD_START" -lt "$SHARD_END" ] && [ ! -f "$SHARD_JSON" ]; then
        echo "ERROR: shard result missing: $SHARD_JSON"; exit 1
    fi

    # Completion markers must outlive workers that have not counted them yet.
    DONE_DIR="${OUTPUT_DIR}/.done_${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$DONE_DIR"
    touch "$DONE_DIR/$TASK_ID"

    DONE_COUNT=$(ls "$DONE_DIR" | wc -l)
    if [ "$DONE_COUNT" -eq "$N_JOBS" ]; then
        MERGE_LOCK="${OUTPUT_DIR}/.merge_${SLURM_ARRAY_JOB_ID}.lock"
        if mkdir "$MERGE_LOCK" 2>/dev/null; then
            echo "[eval $TASK_ID] all shards complete, merging"
            if poetry run python -m src.evaluation.eval_whole_score \
                --phase merge \
                --mapping "$MAPPING" \
                "${arm_arguments[@]}" \
                --mv2h-bin "$MV2H_BIN" \
                --output-dir "$OUTPUT_DIR" \
                "${timeout_argument[@]}"; then
                touch "$MERGE_LOCK/complete"
            else
                rmdir "$MERGE_LOCK"
                exit 1
            fi
        else
            echo "[eval $TASK_ID] all shards complete; another task owns the merge"
        fi
    else
        echo "[eval $TASK_ID] waiting for other shards ($DONE_COUNT / $N_JOBS done)"
    fi
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# Submit mode (default): count recordings, submit Phase 1 + Phase 2
# ─────────────────────────────────────────────────────────────────────

for name in MAPPING ARMS OUTPUT_DIR; do
    if [ -z "${!name}" ]; then
        echo "ERROR: $name is required"
        echo "Usage: MAPPING=... ARMS=\"name=pairs.jsonl ...\" OUTPUT_DIR=... $0 [--local]"
        exit 1
    fi
done
mkdir -p "$OUTPUT_DIR"

REQUIRED_PATHS=("$MAPPING" "$MV2H_BIN")
for spec in $ARMS; do
    REQUIRED_PATHS+=("${spec#*=}")
done
for p in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: $p does not exist"; exit 1
    fi
done

if [ "${1:-}" = "--local" ]; then
    timeout_argument=()
    if [ -n "$RECORDING_TIMEOUT" ]; then
        timeout_argument=(--timeout "$RECORDING_TIMEOUT")
    fi
    poetry run python -m src.evaluation.eval_whole_score \
        --phase all \
        --mapping "$MAPPING" \
        "${arm_arguments[@]}" \
        --mv2h-bin "$MV2H_BIN" \
        --output-dir "$OUTPUT_DIR" \
        --workers "${WORKERS:-12}" \
        "${timeout_argument[@]}"
    exit 0
fi

N_ARMS=$(printf '%s\n' $ARMS | wc -l)
N_RECORDINGS=$(wc -l < "$MAPPING")
TOTAL_UNITS=$(( N_RECORDINGS * N_ARMS ))
N_JOBS=$(( (TOTAL_UNITS + RECORDINGS_PER_JOB - 1) / RECORDINGS_PER_JOB ))
MAX_IDX=$(( N_JOBS - 1 ))
ARRAY_SPEC="0-${MAX_IDX}${MAX_CONCURRENT:+%${MAX_CONCURRENT}}"

echo "Mapping:         $MAPPING ($N_RECORDINGS recordings)"
echo "Arms:            $ARMS"
echo "Output dir:      $OUTPUT_DIR"
echo "Units:           $TOTAL_UNITS (recordings x arms)"
echo "Units per job:   $RECORDINGS_PER_JOB"
echo "Array size:      $N_JOBS (0-$MAX_IDX)"
echo ""

LOG_DIR="logs/eval_whole_score_$(basename "$OUTPUT_DIR")"
mkdir -p "$LOG_DIR"

export MAPPING ARMS MV2H_BIN OUTPUT_DIR RECORDINGS_PER_JOB RECORDING_TIMEOUT \
       RENDER_WORKERS PROJECT_ROOT

# Phase 1: export and convert every score
PHASE1_ID=$(sbatch --parsable \
    --job-name="eval_whole_render" \
    --partition=compute \
    --cpus-per-task="${RENDER_WORKERS:-12}" \
    --mem=32G \
    --time=8:00:00 \
    --output="${LOG_DIR}/render_%j.out" \
    --error="${LOG_DIR}/render_%j.err" \
    --export=ALL \
    "$0" --render-worker)

echo "Phase 1 (render) submitted: job $PHASE1_ID"

# Phase 2: single-path MV2H, depends on Phase 1
PHASE2_ID=$(sbatch --parsable \
    --job-name="eval_whole_mv2h" \
    --partition=compute \
    --array="$ARRAY_SPEC" \
    --cpus-per-task=1 \
    --mem=8G \
    --time=4:00:00 \
    --dependency="afterok:${PHASE1_ID}" \
    --output="${LOG_DIR}/eval_%A_%a.out" \
    --error="${LOG_DIR}/eval_%A_%a.err" \
    --export=ALL \
    "$0" --eval-worker "$TOTAL_UNITS")

echo "Phase 2 (eval)   submitted: job $PHASE2_ID (depends on $PHASE1_ID)"
echo ""
echo "Watch with: squeue -u $USER"
echo "Logs:       $LOG_DIR/"
echo "Final CSV:  ${OUTPUT_DIR}/eval_whole_score.csv"
