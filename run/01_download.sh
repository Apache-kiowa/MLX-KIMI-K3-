#!/bin/bash
# Run on the Mac that converts (rank 0), inside tmux: about 40 h at 10 MB/s.
# Interrupted? Run again: it resumes.
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$VENV/bin/activate"
mkdir -p "$SRC"
df -h "$SRC"
echo "Needs 1.56 TB here plus 1.56 TB for $DST"
caffeinate -i "$VENV/bin/hf" download moonshotai/Kimi-K3.1-Pro --local-dir "$SRC"
python "$K31_PKG/scripts/inspect_ckpt.py" "$SRC" | head -40
