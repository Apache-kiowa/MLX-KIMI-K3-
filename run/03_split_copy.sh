#!/bin/bash
# Run on mac1, one Mac at a time:   bash 03_split_copy.sh mac4
# Cuts that Mac's part (~400 GB) out of the full model, copies it and this
# archive over, checks the sizes, then deletes the temporary copy on mac1.
# For mac1 itself (rank 0) the part is written straight into $DST.
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$VENV/bin/activate"
h="${1:?usage: 03_split_copy.sh <host from HOSTS>}"
rank=-1; i=0
for x in $HOSTS; do [ "$x" = "$h" ] && rank=$i; i=$((i+1)); done
[ $rank -ge 0 ] || { echo "$h is not in HOSTS=\"$HOSTS\""; exit 1; }
first=$(echo $HOSTS | awk '{print $1}')
echo "== $h is rank $rank of $TP"

free_gb() { df -g "$1" | awk 'NR==2{print $4}'; }
if [ "$h" = "$first" ]; then
  out="$DST"
else
  out="$HOME/k31-part-r$rank"
  ssh "$h" "mkdir -p '$BASE'"
  room=$(ssh "$h" "df -g '$BASE' | awk 'NR==2{print \$4}'")
  echo "$h free: ${room} GB"
  [ "$room" -ge 430 ] || { echo "Not enough space on $h (needs ~430 GB)"; exit 1; }
fi
mkdir -p "$(dirname "$out")"
room=$(free_gb "$(dirname "$out")")
echo "mac1 free: ${room} GB"
[ "$room" -ge 430 ] || { echo "Not enough space on mac1 for the part (needs ~430 GB)"; exit 1; }

caffeinate -i python "$K31_PKG/scripts/presplit.py" --model "$FULL" --out "$out" --size $TP --rank $rank

if [ "$h" != "$first" ]; then
  rsync -a "$K31_PKG/" "$h:$K31_PKG/"
  caffeinate -i rsync -a --partial --progress "$out/" "$h:$DST/"
  local_sz=$(cd "$out" && ls -l tp$TP-r$rank-*.safetensors | awk '{s+=$5} END{print s}')
  remote_sz=$(ssh "$h" "cd '$DST' && ls -l tp$TP-r$rank-*.safetensors | awk '{s+=\$5} END{print s}'")
  if [ "$local_sz" = "$remote_sz" ]; then
    echo "[ok] $h has rank $rank: $remote_sz bytes"; rm -rf "$out"
  else
    echo "[FAIL] size differs: mac1 $local_sz, $h $remote_sz. Run this again (rsync resumes)."; exit 1
  fi
fi
