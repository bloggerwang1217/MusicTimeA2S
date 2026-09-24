#!/bin/bash
# =============================================================================
# MusicTime-A2S Bar-Mode Inference Script
# =============================================================================
#
# Downbeat-guided inference producing direct five-bar kern chunks.
#
# Usage:
#   # Default: GPU 0, the full arm's seed-42 checkpoint and its config, full test manifest
#   ./src/a2s/piano/run_inference.sh
#
#   # Custom GPU
#   GPU=1 ./src/a2s/piano/run_inference.sh
#
#   # Custom checkpoint and output (avoid clobbering an existing test_kern_pred)
#   CHECKPOINT=checkpoints/<arm>/best.pt \
#   OUTPUT_DIR=data/experiments/syn/test_kern_pred_a0prime_4dec \
#   ./src/a2s/piano/run_inference.sh
#
#   # Smoke test on a few pieces
#   MAX_SAMPLES=1 ./src/a2s/piano/run_inference.sh
#

set -e

# =============================================================================
# Configuration
# =============================================================================

GPU="${GPU:-0}"

CHECKPOINT="${CHECKPOINT:-checkpoints/full_seed42.pt}"
CONFIG="${CONFIG:-configs/piano_2gpu.yaml}"

MANIFEST_DIR="${MANIFEST_DIR:-data/experiments/syn}"
MANIFEST="${MANIFEST:-$MANIFEST_DIR/test_manifest.json}"
METADATA="${METADATA:-}"
GROUNDING="${GROUNDING:-}"
if [ -z "$METADATA" ] && [ -f "$MANIFEST_DIR/augmentation_metadata.json" ]; then
  METADATA="$MANIFEST_DIR/augmentation_metadata.json"
fi

# NOTE: defaults to overwriting any existing files with the same
# <piece_id>.<chunk_idx>.krn name in this directory. Set OUTPUT_DIR to a
# fresh path when comparing against a previous checkpoint's predictions.
OUTPUT_DIR="${OUTPUT_DIR:-$MANIFEST_DIR/test_kern_pred}"

N_BARS="${N_BARS:-5}"          # max bars per chunk
NUM_BEAMS="${NUM_BEAMS:-1}"    # 1 = greedy
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-1}"
MAX_LEN="${MAX_LEN:-2048}"     # max decode length per chunk
START_IDX="${START_IDX:-0}"    # manifest offset (resume)
MAX_SAMPLES="${MAX_SAMPLES:-}" # cap pieces processed (unset = all)
RESUME="${RESUME:-1}"          # skip complete five-bar inventories
COORDINATE_INTERVENTION="${COORDINATE_INTERVENTION:-none}"  # oracle | other-work | zero

LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/inference_bar_${TIMESTAMP}.log"

# =============================================================================
# Print configuration
# =============================================================================

echo "=============================================="
echo "MusicTime-A2S Bar-Mode Inference"
echo "=============================================="
echo "GPU: $GPU"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Manifest: $MANIFEST"
echo "Coordinates: ${METADATA:-embedded in manifest}"
if [ -n "$GROUNDING" ]; then
  echo "Five-bar grounding: $GROUNDING"
fi
echo "Kern output: $OUTPUT_DIR"
echo "n_bars/chunk: $N_BARS  num_beams: $NUM_BEAMS  max_len: $MAX_LEN"
echo "Coordinate intervention: $COORDINATE_INTERVENTION"
echo "Decode batch: $INFERENCE_BATCH_SIZE  encoder batch: $ENCODER_BATCH_SIZE"
echo "Resume: $RESUME"
if [ -n "$MAX_SAMPLES" ]; then
  echo "max_samples: $MAX_SAMPLES"
fi
echo "Log file: $LOG_FILE"
echo "=============================================="
echo ""

EXTRA_ARGS=()
if [ -n "$METADATA" ]; then
  EXTRA_ARGS+=(--metadata "$METADATA")
fi
if [ -n "$GROUNDING" ]; then
  EXTRA_ARGS+=(--grounding "$GROUNDING")
fi
if [ "$RESUME" != "0" ]; then
  EXTRA_ARGS+=(--resume)
fi
if [ -n "$MAX_SAMPLES" ]; then
  EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [ "$COORDINATE_INTERVENTION" != "none" ]; then
  EXTRA_ARGS+=(--coordinate-intervention "$COORDINATE_INTERVENTION")
fi
if [ -n "${BEATTHIS_DIR:-}" ]; then
  EXTRA_ARGS+=(--beatthis-dir "$BEATTHIS_DIR")
fi

# =============================================================================
# Run inference (with logging)
# =============================================================================

{
  CUDA_VISIBLE_DEVICES=$GPU poetry run python -m src.a2s.piano.inference \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-bars "$N_BARS" \
    --num-beams "$NUM_BEAMS" \
    --max-len "$MAX_LEN" \
    --inference-batch-size "$INFERENCE_BATCH_SIZE" \
    --encoder-batch-size "$ENCODER_BATCH_SIZE" \
    --start-idx "$START_IDX" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}"

  echo ""
  echo "=============================================="
  echo "Inference complete!"
  echo "Five-bar kern predictions: $OUTPUT_DIR"
  echo "Log saved to: $LOG_FILE"
  echo "=============================================="
} 2>&1 | tee "$LOG_FILE"
