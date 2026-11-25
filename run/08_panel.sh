#!/bin/bash
# Run on mac1. Cluster panel: status of every Mac, start/stop, settings, logs, chat.
#   bash 08_panel.sh                  -> http://localhost:7070
#   bash 08_panel.sh --host 0.0.0.0   -> also from other devices on your network (no password!)
#   K31_PANEL_DEMO=1 bash 08_panel.sh -> made-up data, to look around
set -euo pipefail
source "$(dirname "$0")/config.sh"
exec "$VENV/bin/python" "$K31_PKG/panel/k31_panel.py" "$@"
