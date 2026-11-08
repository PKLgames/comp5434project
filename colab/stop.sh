#!/usr/bin/env bash
# Stop the Colab runtime — this is what actually ends compute-unit billing.
# A closed terminal/SSH connection only drops the tunnel; the VM keeps running.
#
#   ./colab/stop.sh            # stop session "work"
#   ./colab/stop.sh train      # stop session "train"
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SESSION=${1:-$SESSION}
echo "[colab] stopping session '$SESSION'"
"$COLAB_BIN" stop -s "$SESSION"
echo "[colab] done; verify with: colab sessions && colab usage"
