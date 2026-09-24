#!/usr/bin/env bash
set -euo pipefail

DEST="external/virtuosoNet/pretrained_weights/han_measnote_gru"
BASE_URL="https://huggingface.co/dasaem/virtuosonet/resolve/main"

mkdir -p "$DEST"

echo "Downloading VirtuosoNet weights (RenCon 2025, HAN+GRU)..."
wget -q --show-progress -O "$DEST/checkpoint_best.pt"    "$BASE_URL/checkpoint_best.pt"
wget -q --show-progress -O "$DEST/han_measnote_gru.yml"  "$BASE_URL/han_measnote_gru.yml"

echo "Done. Weights saved to $DEST/"
