#!/bin/bash
# Run on rank 0. Starts an OpenAI-compatible server on all 4 Macs (tensor parallel).
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$VENV/bin/activate"
mlx.launch --backend jaccl --hostfile "$HOSTFILE" \
  --env MLX_METAL_FAST_SYNCH=1 --env K31_INDEXER_FP8="$K31_INDEXER_FP8" --env K31_PRESPLIT_DIR="$DST" -- \
  "$VENV/bin/python" -m mlx_lm.server --model "$DST" --trust-remote-code \
  --host 0.0.0.0 --port "$PORT"
