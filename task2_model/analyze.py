#!/usr/bin/env python
"""Post-hoc analysis of validation predictions.

Usage:
    python analyze.py --preds runs/full/val_preds.npz --out runs/full/analysis.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with np.load(args.preds) as npz:
        pred = npz["pred"].astype(np.float64)
        y = npz["y"].astype(np.float64)

    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    per_rating = {}
    for rating in range(1, 6):
        mask = y == rating
        if mask.any():
            per_rating[str(rating)] = {
                "count": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))),
                "bias": float(np.mean(pred[mask] - y[mask])),
                "pred_mean": float(pred[mask].mean()),
            }

    # Linear recalibration a*pred+b (diagnostic only; a≈1 means the model is
    # already well calibrated for squared error).
    design = np.stack([pred, np.ones_like(pred)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    calibrated = np.clip(design @ coef, 1.0, 5.0)
    calibration_rmse = float(np.sqrt(np.mean((calibrated - y) ** 2)))

    result = {
        "rows": int(y.size),
        "rmse": rmse,
        "mae": mae,
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "label_mean": float(y.mean()),
        "label_std": float(y.std()),
        "per_rating": per_rating,
        "calibration": {"a": float(coef[0]), "b": float(coef[1]),
                        "rmse": calibration_rmse},
        "pred_quantiles": np.quantile(pred, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist(),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
