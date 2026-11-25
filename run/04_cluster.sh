#!/bin/bash
# Run on rank 0 once the Thunderbolt 5 mesh is cabled and RDMA is enabled on every Mac.
# Writes the hostfile for the jaccl backend and checks every Mac has the same software.
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$VENV/bin/activate"
hosts=$(echo $HOSTS | tr ' ' ',')
mlx.distributed_config --backend jaccl --over thunderbolt --hosts "$hosts" --output-hostfile "$HOSTFILE"
cat "$HOSTFILE"
for h in $HOSTS; do
  echo "== $h"
  ssh "$h" "'$VENV/bin/python' -c 'import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__)'; \
            shasum '$MLX_LM_DIR/mlx_lm/models/kimi_k31.py'; sysctl iogpu.wired_limit_mb; ls '$DST' | grep -c safetensors; du -sh '$DST'"
done
echo "Every Mac needs: sudo sysctl iogpu.wired_limit_mb=480000 (after each reboot)"
