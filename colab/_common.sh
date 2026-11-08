#!/usr/bin/env bash
# Shared helpers for the Colab helper scripts.  Sourced, never executed.
#
#   COLAB_SESSION  session name              (default: work)
#   COLAB_BIN      colab CLI executable      (default: colab)
#   REMOTE_ROOT    project dir on the runtime(default: /content/comp5434project)
#   COLAB_SSH_ARGS extra flags for `colab ssh`, e.g. "-i ~/.ssh/mykey"
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=${COLAB_SESSION:-work}
# The CLI keeps its state (sessions.json, keyring token, log) under $HOME, and the
# authenticated store lives in the colab-cli checkout, not in the real $HOME.
# COLAB_HOME can override; everything else derives from it.
COLAB_HOME_DEFAULT=/home/pklgames/coding/colab-cli/home
if [ -d "${COLAB_HOME:-$COLAB_HOME_DEFAULT}" ]; then
  export HOME=${COLAB_HOME:-$COLAB_HOME_DEFAULT}
fi
export PYTHON_KEYRING_BACKEND=${PYTHON_KEYRING_BACKEND:-keyrings.alt.file.PlaintextKeyring}
COLAB_BIN=${COLAB_BIN:-/home/pklgames/coding/colab-cli/bin/colab}
REMOTE_ROOT=${REMOTE_ROOT:-/content/comp5434project}

colab_ssh() {
  # Runs one command on the existing runtime.  `colab ssh -- CMD` never creates
  # a session: if it is missing the CLI exits non-zero instead of silently
  # burning compute units on a new VM.
  "$COLAB_BIN" ssh -s "$SESSION" ${COLAB_SSH_ARGS:-} -- "$@"
}
