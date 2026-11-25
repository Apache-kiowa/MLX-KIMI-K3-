# Edit these, then every script in run/ reads them.
# Hosts as reachable over SSH from mac1 (aliases from ~/.ssh/config or IPs).
# Order = rank: the first is rank 0 (the Mac you run the scripts on).
HOSTS="mac1 mac2 mac3 mac4"

# Shared paths, identical on every Mac whatever the user name.
# /Users/Shared is writable by every user, so no sudo is needed.
BASE="/Users/Shared/k31"
K31_PKG="$BASE/kimi-k31-mlx"           # this archive
VENV="$BASE/env"                       # Python 3.12 venv
MLX_LM_DIR="$BASE/mlx-lm"              # mlx-lm from git
MLX_LM_COMMIT="c4c9c76"                # tested commit; newer should work too

# Only on mac1 (rank 0):
SRC="$HOME/models/Kimi-K3.1-Pro"       # original weights from Hugging Face (1.56 TB)
FULL="$HOME/models/Kimi-K3.1-Pro-mlx"  # converted full MLX model (1.56 TB)

# On every Mac: this Mac's part only (~400 GB), cut by scripts/presplit.py
TP=4
DST="$BASE/Kimi-K3.1-Pro-tp$TP"
HOSTFILE="$BASE/k31-hosts.json"

PORT=8080
# 0 = full precision as NeMo (default, reproducible); 1 = FP8 indexer scoring as vLLM serving kernels
K31_INDEXER_FP8=0
