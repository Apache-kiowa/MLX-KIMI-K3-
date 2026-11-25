#!/bin/bash
# Run on EVERY Mac. Creates the venv, installs MLX and mlx-lm, adds kimi_k31.py.
set -euo pipefail
source "$(dirname "$0")/config.sh"

PY=/opt/homebrew/bin/python3.12
if [ ! -x "$PY" ]; then
  echo "Python 3.12 not found; installing with Homebrew"; /opt/homebrew/bin/brew install python@3.12
fi
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip
python -m pip install -U "mlx>=0.32.4" "huggingface_hub[hf_xet]" numpy safetensors tiktoken

if [ ! -d "$MLX_LM_DIR/.git" ]; then
  git clone https://github.com/ml-explore/mlx-lm "$MLX_LM_DIR"
fi
git -C "$MLX_LM_DIR" fetch -q origin
git -C "$MLX_LM_DIR" checkout -q "$MLX_LM_COMMIT"
python -m pip install -e "$MLX_LM_DIR"
cp "$K31_PKG/kimi_k31.py" "$MLX_LM_DIR/mlx_lm/models/kimi_k31.py"

python - <<'PY'
import mlx.core as mx, mlx_lm, mlx_lm.models.kimi_k31 as k
print("mlx", mx.__version__, "| mlx-lm", mlx_lm.__version__, "| kimi_k31 DSA:", hasattr(k, "KimiK31Indexer"))
PY
