#!/bin/bash
# =============================================================================
# MusicTime-A2S Single-GPU Training Script
# =============================================================================
#
# All hyperparameters come from the config YAML — this script only handles
# GPU selection and optional resume/wandb overrides.
#
# Usage (from project root):
#   bash src/a2s/piano/train_single_gpu.sh configs/piano_2gpu.yaml
#
#   # Custom GPU
#   GPU=2 bash src/a2s/piano/train_single_gpu.sh configs/piano_2gpu.yaml
#
#   # Resume from checkpoint
#   RESUME=checkpoints/<arm>/best.pt \
#     bash src/a2s/piano/train_single_gpu.sh configs/piano_2gpu.yaml
#
#   # Disable wandb (default project: musictime-a2s)
#   WANDB=false bash src/a2s/piano/train_single_gpu.sh configs/piano_2gpu.yaml

set -euo pipefail

# =============================================================================
# Environment / GPU selection (the only things that belong in shell)
# =============================================================================

CONFIG="${1:-${CONFIG:-}}"
if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG is required. Pass as first arg or CONFIG=... env var." >&2
    echo "  e.g.: bash src/a2s/piano/train_single_gpu.sh configs/piano_2gpu.yaml" >&2
    exit 1
fi

GPU="${GPU:-0}"
RESUME="${RESUME:-}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-musictime-a2s}"

# =============================================================================
# Print summary
# =============================================================================

echo "============================================="
echo "MusicTime-A2S Single-GPU Training"
echo "============================================="
echo "Config:      $CONFIG  (single source of truth)"
echo "GPU:         $GPU"
if [ -n "$RESUME" ]; then
    echo "Resume:      $RESUME"
fi
echo "Wandb:       $WANDB"
echo "============================================="
echo ""

# =============================================================================
# Build and run command
# =============================================================================

CMD="poetry run python -m src.a2s.piano.train"
CMD="$CMD --config $CONFIG"
CMD="$CMD --num-workers 0"

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb --wandb-project $WANDB_PROJECT"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "Running:"
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo ""

# PCI order so GPU indices match nvidia-smi; CUDA's default fastest-first
# order can move a card between runs.
PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $CMD
