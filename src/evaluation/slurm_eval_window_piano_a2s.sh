#!/bin/bash
# =============================================================================
# Slurm MV2H evaluation against Piano-A2S reference MIDI (Table 1) — two-phase pipeline
# =============================================================================
#
# Phase 1 (render): slurm array, parallel per-chunk kern→MIDI render. The
#                   reference root is read-only and shared by every shard.
# Phase 2 (eval):   slurm array, depends on Phase 1 (afterok). Each task
#                   reads pre-rendered chunk MIDIs and runs MV2H on its range.
#                   Last finishing task merges shard CSVs.
#
# Both phases shard the grounding manifest by row range. Per-chunk kern→MIDI
# bypasses music21's full-piece duration-type failures.
#
# Usage (from project root):
#   PRED_DIR=... GROUNDING=... REFERENCE_ROOT=... OUTPUT_DIR=... \
#     bash src/evaluation/slurm_eval_window_piano_a2s.sh
#
#   # Override per-job size (default 100, matching slurm_eval_syn.sh)
#   CHUNKS_PER_JOB=200 PRED_DIR=... bash src/evaluation/slurm_eval_window_piano_a2s.sh
#
# Each model must use its own OUTPUT_DIR.

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

# A submit shell may have an unrelated Python environment active. Clear its
# markers so Poetry resolves this project's environment on every worker.
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV POETRY_ACTIVE

PRED_DIR="${PRED_DIR:-}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set the Piano-A2S results root that holds the reference MIDI}"
GROUNDING="${GROUNDING:-$REFERENCE_ROOT/grounding.jsonl}"
MV2H_BIN="${MV2H_BIN:-external/MV2H/bin}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

CHUNKS_PER_JOB="${CHUNKS_PER_JOB:-100}"
CHUNK_TIMEOUT="${CHUNK_TIMEOUT:-300}"

# ─────────────────────────────────────────────────────────────────────
# Worker: Phase 1 — per-chunk kern of our system→prediction MIDI
# ─────────────────────────────────────────────────────────────────────

if [ "$1" = "--render-worker" ]; then
    TOTAL_CHUNKS=$2
    N_JOBS=${SLURM_ARRAY_TASK_COUNT}
    TASK_ID=${SLURM_ARRAY_TASK_ID}

    SHARD_SIZE=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))
    SHARD_START=$(( TASK_ID * SHARD_SIZE ))
    SHARD_END=$(( SHARD_START + SHARD_SIZE ))
    if [ "$SHARD_END" -gt "$TOTAL_CHUNKS" ]; then
      SHARD_END=$TOTAL_CHUNKS
    fi

    echo "[render $TASK_ID/$N_JOBS] chunks [$SHARD_START, $SHARD_END) / $TOTAL_CHUNKS"

    poetry run python -m src.evaluation.eval_window_piano_a2s \
        --phase render \
        --pred-dir "$PRED_DIR" \
        --grounding "$GROUNDING" \
        --reference-root "$REFERENCE_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --chunk-start "$SHARD_START" \
        --chunk-end "$SHARD_END" \
        --workers 1

    echo "[render $TASK_ID] done"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# Worker: Phase 2 — MV2H eval on one shard
# ─────────────────────────────────────────────────────────────────────

if [ "$1" = "--eval-worker" ]; then
    TOTAL_CHUNKS=$2
    N_JOBS=${SLURM_ARRAY_TASK_COUNT}
    TASK_ID=${SLURM_ARRAY_TASK_ID}

    SHARD_SIZE=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))
    SHARD_START=$(( TASK_ID * SHARD_SIZE ))
    SHARD_END=$(( SHARD_START + SHARD_SIZE ))
    if [ "$SHARD_END" -gt "$TOTAL_CHUNKS" ]; then
      SHARD_END=$TOTAL_CHUNKS
    fi

    echo "[eval $TASK_ID/$N_JOBS] chunks [$SHARD_START, $SHARD_END) / $TOTAL_CHUNKS"

    poetry run python -m src.evaluation.eval_window_piano_a2s \
        --phase eval \
        --pred-dir "$PRED_DIR" \
        --grounding "$GROUNDING" \
        --reference-root "$REFERENCE_ROOT" \
        --mv2h-bin "$MV2H_BIN" \
        --output-dir "$OUTPUT_DIR" \
        --chunk-start "$SHARD_START" \
        --chunk-end "$SHARD_END" \
        --workers 1 \
        --chunk-timeout "$CHUNK_TIMEOUT"

    SHARD_CSV="${OUTPUT_DIR}/eval_asap_shard_$(printf '%06d' $SHARD_START)_$(printf '%06d' $SHARD_END).csv"
    echo "[eval $TASK_ID] done: $SHARD_CSV"

    # Completion markers must outlive workers that have not counted them yet.
    DONE_DIR="${OUTPUT_DIR}/.done_${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$DONE_DIR"
    touch "$DONE_DIR/$TASK_ID"

    DONE_COUNT=$(ls "$DONE_DIR" | wc -l)
    if [ "$DONE_COUNT" -eq "$N_JOBS" ]; then
        MERGE_LOCK="${OUTPUT_DIR}/.merge_${SLURM_ARRAY_JOB_ID}.lock"
        if mkdir "$MERGE_LOCK" 2>/dev/null; then
            echo "All shards complete. Merging..."
            if poetry run python -m src.evaluation.eval_window_piano_a2s \
                --phase merge \
                --pred-dir "$PRED_DIR" \
                --reference-root "$REFERENCE_ROOT" \
                --chunk-timeout "$CHUNK_TIMEOUT" \
                --grounding "$GROUNDING" \
                --output-dir "$OUTPUT_DIR"; then
                touch "$MERGE_LOCK/complete"
            else
                rmdir "$MERGE_LOCK"
                exit 1
            fi
        else
            echo "All shards complete; another task owns the merge."
        fi
    else
        echo "Waiting for other shards ($DONE_COUNT / $N_JOBS done)"
    fi
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# Submit mode (default): count chunks, submit Phase 1 + Phase 2 arrays
# ─────────────────────────────────────────────────────────────────────

for name in PRED_DIR GROUNDING REFERENCE_ROOT OUTPUT_DIR; do
    if [ -z "${!name}" ]; then
        echo "ERROR: $name is required"; exit 1
    fi
done
mkdir -p "$OUTPUT_DIR"

REQUIRED_PATHS=("$PRED_DIR" "$MV2H_BIN" "$GROUNDING" "$REFERENCE_ROOT")
for p in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: $p does not exist"; exit 1
    fi
done

if [ "${1:-}" = "--local" ]; then
    poetry run python -m src.evaluation.eval_window_piano_a2s \
        --phase all \
        --pred-dir "$PRED_DIR" \
        --grounding "$GROUNDING" \
        --reference-root "$REFERENCE_ROOT" \
        --mv2h-bin "$MV2H_BIN" \
        --output-dir "$OUTPUT_DIR" \
        --workers "${WORKERS:-16}" \
        --chunk-timeout "$CHUNK_TIMEOUT"
    exit 0
fi

TOTAL_CHUNKS=$(wc -l < "$GROUNDING")
N_JOBS=$(( (TOTAL_CHUNKS + CHUNKS_PER_JOB - 1) / CHUNKS_PER_JOB ))
MAX_IDX=$(( N_JOBS - 1 ))
ARRAY_SPEC="0-${MAX_IDX}${MAX_CONCURRENT:+%${MAX_CONCURRENT}}"
EVAL_TIME="4:00:00"
if [ "$CHUNK_TIMEOUT" -gt 10 ]; then
    EVAL_TIME=$(( (CHUNKS_PER_JOB * (CHUNK_TIMEOUT + 2) + 59) / 60 + 30 ))
fi

echo "Pred dir:        $PRED_DIR"
echo "Grounding:       $GROUNDING ($TOTAL_CHUNKS chunks)"
echo "Reference root:  $REFERENCE_ROOT"
echo "Output dir:      $OUTPUT_DIR"
echo "Chunks per job:  $CHUNKS_PER_JOB"
echo "Array size:      $N_JOBS (0-$MAX_IDX)"
echo ""

LOG_DIR="logs/eval_asap_$(basename "$(dirname "$PRED_DIR")")_$(basename "$PRED_DIR")"
mkdir -p "$LOG_DIR"

export PRED_DIR GROUNDING REFERENCE_ROOT \
       MV2H_BIN OUTPUT_DIR CHUNKS_PER_JOB CHUNK_TIMEOUT PROJECT_ROOT

# Phase 1: render the kern of our system→prediction MIDI
PHASE1_ID=$(sbatch --parsable \
    --job-name="eval_asap_render" \
    --partition=compute \
    --array="$ARRAY_SPEC" \
    --cpus-per-task=1 \
    --mem=4G \
    --time=4:00:00 \
    --output="${LOG_DIR}/render_%A_%a.out" \
    --error="${LOG_DIR}/render_%A_%a.err" \
    --export=ALL \
    "$0" --render-worker "$TOTAL_CHUNKS")

echo "Phase 1 (render) submitted: job $PHASE1_ID"

# Phase 2: MV2H eval, depends on Phase 1
PHASE2_ID=$(sbatch --parsable \
    --job-name="eval_asap_mv2h" \
    --partition=compute \
    --array="$ARRAY_SPEC" \
    --cpus-per-task=1 \
    --mem=4G \
    --time="$EVAL_TIME" \
    --dependency="afterok:${PHASE1_ID}" \
    --output="${LOG_DIR}/eval_%A_%a.out" \
    --error="${LOG_DIR}/eval_%A_%a.err" \
    --export=ALL \
    "$0" --eval-worker "$TOTAL_CHUNKS")

echo "Phase 2 (eval)   submitted: job $PHASE2_ID (depends on $PHASE1_ID)"
echo ""
echo "Watch with: squeue -u $USER"
echo "Logs:       $LOG_DIR/"
echo "Final CSV:  ${OUTPUT_DIR}/eval_asap.csv"
