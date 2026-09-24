#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV POETRY_ACTIVE

PAIRS="${PAIRS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MUSICDIFF_ROOT="${MUSICDIFF_ROOT:-external/efficient-musicdiff}"
PAIRS_PER_JOB="${PAIRS_PER_JOB:-5}"

MODE="${1:-}"
if [ "$MODE" = "--worker" ]; then
    TOTAL_PAIRS="$2"
    N_JOBS="${SLURM_ARRAY_TASK_COUNT}"
    TASK_ID="${SLURM_ARRAY_TASK_ID}"
    SHARD_SIZE=$(( (TOTAL_PAIRS + N_JOBS - 1) / N_JOBS ))
    SHARD_START=$(( TASK_ID * SHARD_SIZE ))
    SHARD_END=$(( SHARD_START + SHARD_SIZE ))
    if [ "$SHARD_END" -gt "$TOTAL_PAIRS" ]; then
        SHARD_END="$TOTAL_PAIRS"
    fi

    poetry run python -m src.evaluation.omr_ned_shard \
        --pairs "$PAIRS" \
        --output-dir "$OUTPUT_DIR" \
        --musicdiff-root "$MUSICDIFF_ROOT" \
        --start "$SHARD_START" \
        --end "$SHARD_END"

    DONE_DIR="$OUTPUT_DIR/.done_${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$DONE_DIR"
    touch "$DONE_DIR/$TASK_ID"
    DONE_COUNT=$(find "$DONE_DIR" -maxdepth 1 -type f | wc -l)
    if [ "$DONE_COUNT" -eq "$N_JOBS" ]; then
        MERGE_LOCK="$OUTPUT_DIR/.merge_${SLURM_ARRAY_JOB_ID}.lock"
        if mkdir "$MERGE_LOCK" 2>/dev/null; then
            poetry run python -m src.evaluation.omr_ned_shard \
                --pairs "$PAIRS" \
                --output-dir "$OUTPUT_DIR" \
                --musicdiff-root "$MUSICDIFF_ROOT" \
                --merge
            touch "$MERGE_LOCK/complete"
        fi
    fi
    exit 0
fi

if [ "$MODE" = "--local-shard" ]; then
    SHARD_START="$2"
    SHARD_END="$3"
    if [ "$SHARD_START" -ge "$SHARD_END" ]; then
        exit 0
    fi
    SHARD_OUT="$OUTPUT_DIR/$(printf 'shard_%06d_%06d.jsonl' "$SHARD_START" "$SHARD_END")"
    # A finished shard is kept so an interrupted local run only redoes the missing ranges.
    if [ -f "$SHARD_OUT" ]; then
        exit 0
    fi
    poetry run python -m src.evaluation.omr_ned_shard \
        --pairs "$PAIRS" \
        --output-dir "$OUTPUT_DIR" \
        --musicdiff-root "$MUSICDIFF_ROOT" \
        --start "$SHARD_START" \
        --end "$SHARD_END"
    exit 0
fi

if [ -n "$MODE" ] && [ "$MODE" != "--local" ]; then
    echo "Usage: PAIRS=... OUTPUT_DIR=... bash src/evaluation/slurm_omr_ned.sh [--local]" >&2
    exit 2
fi
if [ -z "$PAIRS" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: PAIRS and OUTPUT_DIR are required" >&2
    exit 2
fi
for path in "$PAIRS" "$MUSICDIFF_ROOT"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing $path" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR"
TOTAL_PAIRS=$(awk 'NF { count += 1 } END { print count + 0 }' "$PAIRS")
if [ "$TOTAL_PAIRS" -le 0 ]; then
    echo "ERROR: pair inventory is empty" >&2
    exit 2
fi
N_JOBS=$(( (TOTAL_PAIRS + PAIRS_PER_JOB - 1) / PAIRS_PER_JOB ))
MAX_IDX=$(( N_JOBS - 1 ))
LOG_DIR="logs/omr_ned_$(basename "$OUTPUT_DIR")"
mkdir -p "$LOG_DIR"

export PAIRS OUTPUT_DIR MUSICDIFF_ROOT PROJECT_ROOT

# Local mode runs the same shards through xargs instead of a slurm array and
# merges once every shard has returned; shard files and the summary land in the
# same places as the slurm run.
if [ "$MODE" = "--local" ]; then
    WORKERS="${WORKERS:-16}"
    SHARD_SIZE=$(( (TOTAL_PAIRS + N_JOBS - 1) / N_JOBS ))
    for (( task = 0; task < N_JOBS; task++ )); do
        SHARD_START=$(( task * SHARD_SIZE ))
        SHARD_END=$(( SHARD_START + SHARD_SIZE ))
        if [ "$SHARD_END" -gt "$TOTAL_PAIRS" ]; then
            SHARD_END="$TOTAL_PAIRS"
        fi
        echo "$SHARD_START $SHARD_END"
    done | xargs -P "$WORKERS" -n 2 bash "$0" --local-shard
    # Merge writes results.jsonl and then fails closed when any pair did not
    # score; the caller's failure-policy summary needs those results, so only a
    # merge that stopped before writing them (inventory mismatch) aborts here.
    # Stale merge outputs are removed first so they cannot stand in for this merge.
    rm -f "$OUTPUT_DIR/results.jsonl" "$OUTPUT_DIR/summary.json"
    MERGE_ERR="$OUTPUT_DIR/.merge_local.err"
    set +e
    poetry run python -m src.evaluation.omr_ned_shard \
        --pairs "$PAIRS" \
        --output-dir "$OUTPUT_DIR" \
        --musicdiff-root "$MUSICDIFF_ROOT" \
        --merge 2> "$MERGE_ERR"
    MERGE_STATUS=$?
    set -e
    cat "$MERGE_ERR" >&2
    if [ "$MERGE_STATUS" -ne 0 ]; then
        if [ -f "$OUTPUT_DIR/results.jsonl" ] && grep -q "failed closed" "$MERGE_ERR"; then
            echo "OMR-NED local merge: some pairs did not score; results.jsonl kept for the failure-policy summary"
        else
            exit "$MERGE_STATUS"
        fi
    fi
    echo "OMR-NED local run finished: $TOTAL_PAIRS pairs, $N_JOBS shards, $WORKERS workers"
    echo "Final summary: $OUTPUT_DIR/summary.json"
    exit 0
fi

JOB_ID=$(sbatch --parsable \
    --job-name="omr_ned" \
    --partition=compute \
    --array="0-${MAX_IDX}" \
    --cpus-per-task=1 \
    --mem=4G \
    --time=4:00:00 \
    --output="$LOG_DIR/eval_%A_%a.out" \
    --error="$LOG_DIR/eval_%A_%a.err" \
    --export=ALL \
    "$0" --worker "$TOTAL_PAIRS")

echo "OMR-NED array submitted: $JOB_ID ($TOTAL_PAIRS pairs, $N_JOBS jobs)"
echo "Final summary: $OUTPUT_DIR/summary.json"
