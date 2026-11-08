#!/usr/bin/env bash
# Run the full validation split on the Colab runtime and bring the metrics back.
#
#   ./colab/run_valid.sh                 # session "work", checkpoint runs/full3
#   ./colab/run_valid.sh work
#   EXTRA_ARGS="--max-shards 1" ./colab/run_valid.sh   # quick smoke
#   FETCH_PREDS=1 ./colab/run_valid.sh  # also pull the 12 MB val_preds.npz
#
# The prediction file itself stays on the runtime (it is 12 MB); only the
# 200-byte metrics JSON crosses the tunnel unless FETCH_PREDS=1.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SESSION=${1:-$SESSION}
OUT_REMOTE="runs/colab_valid"
LOCAL_DIR="$ROOT/$OUT_REMOTE"

echo "[colab] scoring the validation split on '$SESSION' (this is the compute-unit part)"
colab_ssh "cd '$REMOTE_ROOT' && mkdir -p '$OUT_REMOTE' && \
  python task2_model/predict.py \
    --data artifacts --split valid \
    --checkpoint runs/full3/best.pt \
    --out '$OUT_REMOTE/val_preds.npz' ${EXTRA_ARGS:-}"

mkdir -p "$LOCAL_DIR"
colab_ssh "cat '$REMOTE_ROOT/$OUT_REMOTE/val_preds_metrics.json'" > "$LOCAL_DIR/val_preds_metrics.json"

echo
echo "[colab] metrics (saved to $OUT_REMOTE/val_preds_metrics.json):"
cat "$LOCAL_DIR/val_preds_metrics.json"
echo
echo "[colab] published reference on torch 2.9.1/CPU: RMSE 1.1573, MAE 0.8646"

if [ "${FETCH_PREDS:-0}" = "1" ]; then
  echo "[colab] fetching val_preds.npz"
  "$COLAB_BIN" download -s "$SESSION" "$REMOTE_ROOT/$OUT_REMOTE/val_preds.npz" \
    "$LOCAL_DIR/val_preds.npz"
  echo "[colab] compare with the published run:"
  echo "  $ROOT/.venv/bin/python task2_model/compare.py \\"
  echo "    --base runs/full3/val_preds.npz --candidate $OUT_REMOTE/val_preds.npz \\"
  echo "    --out $OUT_REMOTE/comparison_vs_published.json"
fi

echo "[colab] remember to free the GPU: ./colab/stop.sh"
