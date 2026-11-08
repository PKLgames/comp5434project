#!/usr/bin/env bash
# Push code + validation cache + checkpoint to an existing Colab runtime.
#
#   ./colab/push.sh              # session "work", minimal set for validation
#   ./colab/push.sh work --full  # also push the training cache (~2.4 GB, slow)
#   REBUILD=1 ./colab/push.sh    # rebuild the cached bundle first
#
# The bundle is cached in colab/.cache/ so a second push of a fresh runtime does
# not pay the tar time again.  Transfer runs as one binary stream over
# `colab ssh -- CMD`, i.e. inside the same WebSocket tunnel the CLI uses; no
# Drive mount, no Contents-API base64 round-trip.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) SESSION="$arg" ;;
  esac
done

if [ "$FULL" = "1" ]; then
  BUNDLE="$ROOT/colab/.cache/bundle-full.tgz"
  PATHS=(task2_model tests artifacts runs/full3/best.pt runs/full_tmlp/best.pt)
else
  BUNDLE="$ROOT/colab/.cache/bundle-valid.tgz"
  PATHS=(task2_model tests
         artifacts/meta.json artifacts/vocab_title.npy artifacts/vocab_comment.npy
         artifacts/valid runs/full3/best.pt)
fi

if [ ! -f "$BUNDLE" ] || [ "${REBUILD:-0}" = "1" ]; then
  echo "[colab] packing ${PATHS[*]}"
  mkdir -p "$(dirname "$BUNDLE")"
  tar -C "$ROOT" -czf "$BUNDLE" "${PATHS[@]}"
fi
echo "[colab] bundle: $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"

echo "[colab] preparing $REMOTE_ROOT on session '$SESSION'"
colab_ssh "mkdir -p '$REMOTE_ROOT'"

echo "[colab] streaming bundle (one shot; keep the terminal open)"
cat "$BUNDLE" | colab_ssh "tar xzf - -C '$REMOTE_ROOT'"

echo "[colab] verifying remote tree"
colab_ssh "cd '$REMOTE_ROOT' && \
  printf 'code    : '; ls task2_model/*.py | wc -l; \
  printf 'valid   : '; ls artifacts/valid/*.npz | wc -l; \
  printf 'ckpt    : '; du -h runs/full3/best.pt | cut -f1; \
  printf 'python  : '; python -V; \
  python -c 'import torch,numpy,pyarrow;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())'"

echo "[colab] done — next: ./colab/run_valid.sh"
