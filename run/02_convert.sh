#!/bin/bash
# Run on rank 0 after the download. Repacks the weights for MLX (no requantization).
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$VENV/bin/activate"
if [ -e "$FULL" ]; then echo "$FULL already exists; remove it first"; exit 1; fi
caffeinate -i python -m mlx_lm convert --hf-path "$SRC" --mlx-path "$FULL" --trust-remote-code
python "$K31_PKG/scripts/verify_full.py" --src "$SRC" --mlx "$FULL"
du -sh "$FULL"
