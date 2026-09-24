#!/bin/bash
# =============================================================================
# MusicTime-A2S DDP Training Script
# =============================================================================
#
# All hyperparameters come from the config YAML — this script only handles
# GPU selection and optional resume/wandb overrides.
#
# Usage (from project root):
#   # Default: GPU 1 and 2
#   bash src/a2s/piano/train_ddp.sh
#
#   # Custom GPUs
#   GPUS=0,1 bash src/a2s/piano/train_ddp.sh
#
#   # Resume from checkpoint
#   RESUME=checkpoints/<arm>/best.pt bash src/a2s/piano/train_ddp.sh
#
#   # Enable wandb with custom project
#   WANDB=true WANDB_PROJECT=my-project bash src/a2s/piano/train_ddp.sh
#
#   # Overfit one batch through the normal training entry point
#   SANITY_CHECK=true WANDB=false bash src/a2s/piano/train_ddp.sh

set -e

# =============================================================================
# Environment / GPU selection (the only things that belong in shell)
# =============================================================================

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG="${1:-${CONFIG:-configs/piano_2gpu.yaml}}"
GPUS="${GPUS:-1,2}"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME="${RESUME:-}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-musictime-a2s}"
SANITY_CHECK="${SANITY_CHECK:-false}"

# =============================================================================
# Print summary
# =============================================================================

echo "=============================================="
echo "MusicTime-A2S DDP Training"
echo "=============================================="
echo "Config:      $CONFIG  (single source of truth)"
echo "GPUs:        $GPUS  ($NUM_GPUS GPUs)"
echo "Master port: $MASTER_PORT"
if [ -n "$RESUME" ]; then
    echo "Resume:      $RESUME"
fi
echo "Wandb:       $WANDB"
echo "Sanity:      $SANITY_CHECK"
echo "=============================================="
echo ""

# =============================================================================
# Check GPU memory
# =============================================================================

echo "Checking GPU memory..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv --id=$GPUS
echo ""

# =============================================================================
# Build and run command
# =============================================================================

CMD="poetry run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m src.a2s.piano.train"
CMD="$CMD --config $CONFIG"

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb"
    CMD="$CMD --wandb-project $WANDB_PROJECT"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

if [ "$SANITY_CHECK" = "true" ]; then
    CMD="$CMD --sanity-check"
fi

echo "Running:"
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS $CMD"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS $CMD
