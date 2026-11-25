#!/bin/bash
# Run on rank 0. Tensor-parallel check on the TINY model over the real cluster.
# Needs the tiny model converted at the same path on every Mac (TINY below).
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$VENV/bin/activate"
TINY="${TINY:-$BASE/tiny_mlx}"
S="$K31_PKG/scripts"
K31_INDEXER_FP8=0 python "$S/tp_check.py" --model "$TINY" --ref /tmp/k31_ref.npy --fp32 --topk 8 --tokens 64
for h in $HOSTS; do scp -q /tmp/k31_ref.npy "$h:/tmp/k31_ref.npy" || true; done
mlx.launch --backend jaccl --hostfile "$HOSTFILE" --env K31_INDEXER_FP8=0 -- \
  "$VENV/bin/python" "$S/tp_check.py" --model "$TINY" --ref /tmp/k31_ref.npy --fp32 --topk 8 --tokens 64
