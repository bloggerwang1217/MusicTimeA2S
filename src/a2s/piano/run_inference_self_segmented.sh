#!/bin/bash
# =============================================================================
# MusicTime-A2S Self-Segmented Whole-Piece Inference
# =============================================================================
#
# Scope origins come from the model's own downbeat phase; the decoder and
# reconstructor are the bar-mode ones. One whole-piece kern per performance.
#
# Usage:
#   ./src/a2s/piano/run_inference_self_segmented.sh
#   GPU=1 MAX_SAMPLES=2 ./src/a2s/piano/run_inference_self_segmented.sh
#   MANIFEST_DIR=data/experiments/syn \
#   MANIFEST=data/experiments/syn/valid_manifest.json \
#   ID_REGEX='_v0(~|$)' \
#   OUTPUT_DIR=data/experiments/syn/valid_kern_pred_self_segmented \
#   ./src/a2s/piano/run_inference_self_segmented.sh
#

set -eo pipefail

GPU="${GPU:-0}"

CHECKPOINT="${CHECKPOINT:-checkpoints/full_seed42.pt}"
CONFIG="${CONFIG:-configs/piano_2gpu.yaml}"

MANIFEST_DIR="${MANIFEST_DIR:-data/experiments/asap102}"
MANIFEST="${MANIFEST:-$MANIFEST_DIR/test_manifest.json}"
# Cross-scope meter/key switch penalties (nats); the default output name carries them.
METER_SWITCH_PENALTY="${METER_SWITCH_PENALTY:-8.3}"
KEY_SWITCH_PENALTY="${KEY_SWITCH_PENALTY:-6.3}"
OUTPUT_DIR="${OUTPUT_DIR:-$MANIFEST_DIR/test_kern_pred_self_segmented_forward400_meter${METER_SWITCH_PENALTY}_key${KEY_SWITCH_PENALTY}}"

BATCH_SIZE="${BATCH_SIZE:-4}"   # phase windows per forward pass
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-1}"
MAX_LEN="${MAX_LEN:-2048}"      # max decode length per scope
START_IDX="${START_IDX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
ID_REGEX="${ID_REGEX:-}"
RESUME="${RESUME:-1}"
PIECE_SCOPE="${PIECE_SCOPE:-five_bar}"   # five_bar | cursor

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/inference_self_segmented_${TIMESTAMP}.log"

echo "=============================================="
echo "MusicTime-A2S Self-Segmented Inference"
echo "=============================================="
echo "GPU: $GPU"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Manifest: $MANIFEST"
echo "Kern output: $OUTPUT_DIR"
echo "Decode batch: $INFERENCE_BATCH_SIZE  encoder batch: $ENCODER_BATCH_SIZE"
echo "batch_size: $BATCH_SIZE  max_len: $MAX_LEN  resume: $RESUME"
if [ -n "$MAX_SAMPLES" ]; then echo "max_samples: $MAX_SAMPLES"; fi
if [ -n "$ID_REGEX" ]; then echo "id_regex: $ID_REGEX"; fi
if [ -n "${COORDINATE_INTERVENTION:-}" ]; then echo "coordinate_intervention: $COORDINATE_INTERVENTION"; fi
if [ -n "${BAR_BOUNDARIES:-}" ]; then echo "bar_boundaries: $BAR_BOUNDARIES"; fi
echo "piece_scope: $PIECE_SCOPE"
echo "key_switch_penalty: $KEY_SWITCH_PENALTY  meter_switch_penalty: ${METER_SWITCH_PENALTY:-none}"
echo "Log file: $LOG_FILE"
echo "=============================================="
echo ""

EXTRA_ARGS=()
if [ "${DRY_RUN:-0}" = "1" ]; then EXTRA_ARGS+=(--dry-run); fi
if [ "$RESUME" != "0" ]; then EXTRA_ARGS+=(--resume); fi
if [ -n "$MAX_SAMPLES" ]; then EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES"); fi
if [ -n "$ID_REGEX" ]; then EXTRA_ARGS+=(--id-regex "$ID_REGEX"); fi
if [ -n "${COORDINATE_INTERVENTION:-}" ] && [ "$COORDINATE_INTERVENTION" != "none" ]; then EXTRA_ARGS+=(--coordinate-intervention "$COORDINATE_INTERVENTION"); fi
if [ -n "${BAR_BOUNDARIES:-}" ] && [ "$BAR_BOUNDARIES" != "predicted" ]; then EXTRA_ARGS+=(--bar-boundaries "$BAR_BOUNDARIES"); fi
if [ -n "${BEATTHIS_DIR:-}" ]; then EXTRA_ARGS+=(--beatthis-dir "$BEATTHIS_DIR"); fi
if [ -n "${METER_SWITCH_PENALTY:-}" ]; then EXTRA_ARGS+=(--meter-switch-penalty "$METER_SWITCH_PENALTY"); fi
if [ -n "${KEY_SWITCH_PENALTY:-}" ]; then EXTRA_ARGS+=(--key-switch-penalty "$KEY_SWITCH_PENALTY"); fi
EXTRA_ARGS+=(--piece-scope "$PIECE_SCOPE")


{
  CUDA_VISIBLE_DEVICES=$GPU poetry run python -m src.a2s.piano.self_segmented \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --max-len "$MAX_LEN" \
    --inference-batch-size "$INFERENCE_BATCH_SIZE" \
    --encoder-batch-size "$ENCODER_BATCH_SIZE" \
    --start-idx "$START_IDX" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}"

  echo ""
  echo "=============================================="
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "Dry run complete; no model was loaded."
  else
    echo "Self-segmented inference complete!"
    echo "Whole-piece kern predictions: $OUTPUT_DIR"
  fi
  echo "Log saved to: $LOG_FILE"
  echo "=============================================="
} 2>&1 | tee "$LOG_FILE"
