#!/usr/bin/env bash
# Interactive shell on the runtime, in the project directory.
#
#   ./colab/shell.sh            # session "work"
#   ./colab/shell.sh train
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SESSION=${1:-$SESSION}
echo "[colab] opening a shell on '$SESSION' (exit with Ctrl-D; the VM keeps running)"
exec "$COLAB_BIN" ssh -s "$SESSION" ${COLAB_SSH_ARGS:-} -- \
  "cd '$REMOTE_ROOT' 2>/dev/null; exec bash -l"
