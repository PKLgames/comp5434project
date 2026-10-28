#!/usr/bin/env python
"""Fit an N-model stacked ensemble with out-of-fold weights.

The published ensemble (weights ``0.477 / 0.484``, intercept ``0.162``) was
fitted by hand on two checkpoints.  This script generalises that recipe: for
every checkpoint it reproduces the validation predictions exactly as
``predict.py`` would, then fits one weight per model plus an intercept by least
squares.

The reported number is **out-of-fold**: the validation rows are split into
``--folds`` contiguous folds, the weights are fitted on the other folds and
applied to the held-out fold.  With only N+1 parameters over 1.5 M rows the
in-sample and OOF numbers agree to ~1e-6, but OOF is the honest one, so it is
what gets written to ``val_preds.npz`` (making the file directly comparable with
``compare.py``).

Each checkpoint may have been trained on a different dense schema (a V1 model
uses 26 columns, a V2 model 33); the batcher selects exactly the columns that
checkpoint recorded, so mixing them is safe as long as ``--data`` provides them
all.

Usage
-----
python stack.py --data ../artifacts_v2 --out ../runs/v2_stack \
  --checkpoint ../runs/full3/best.pt \
  --checkpoint ../runs/full_tmlp/best.pt \
  --checkpoint ../runs/v2_full/best.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from common import load_json, save_json
from dataset import ShardBatcher
from predict import (load_checkpoint, load_model, predict_loader,
                     resolve_blend_alpha)


# ---------------------------------------------------------------------------
def collect(checkpoint: str, data: str, device: torch.device,
            batch_size: int, max_shards: int | None,
            blend_spec: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, labels)`` for one checkpoint on the valid split.

    Scoring goes through ``predict.predict_loader`` so a stacked member is
    evaluated exactly as ``predict.py`` would evaluate it - including the
    out-of-fold ``E[rating]`` blend of an ordinal model.  Doing it any other way
    would make the member's contribution disagree with its standalone number.
    """

    ckpt = load_checkpoint(checkpoint)
    dense_columns = [str(c) for c in ckpt["dense_columns"]]

    model = load_model(checkpoint, device)
    loader = ShardBatcher(data, "valid", dense_columns, batch_size=batch_size,
                          shuffle=False, max_shards=max_shards,
                          pin_memory=torch.cuda.is_available())

    pred, exp, y = predict_loader(model, loader, device)
    if y is None:
        raise SystemExit(f"{checkpoint}: the valid split has no labels")

    alpha, _ = resolve_blend_alpha(blend_spec, ckpt)
    if exp is not None and alpha > 0.0:
        pred = (1.0 - alpha) * pred + alpha * exp
    pred = np.clip(pred, 1.0, 5.0)
    return pred.astype(np.float64), y.astype(np.float64)


def fit_weights(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an explicit intercept column (already in ``design``)."""

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    diff = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "rows": int(y.size),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="artifact dir providing the features")
    parser.add_argument("--out", required=True, help="run directory for the stack outputs")
    parser.add_argument("--checkpoint", required=True, action="append",
                        help="checkpoint path; repeat once per member model")
    parser.add_argument("--folds", type=int, default=5,
                        help="contiguous out-of-fold splits (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--no-clip", dest="clip", action="store_false",
                        help="do not clip the blended prediction to [1, 5]")
    parser.add_argument("--blend-alpha", default="auto",
                        help="E[rating] blend for each member: 'auto' uses the "
                             "out-of-fold value stored in its checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.out, exist_ok=True)

    members: List[np.ndarray] = []
    y: np.ndarray | None = None
    individual: Dict[str, dict] = {}
    for path in args.checkpoint:
        print(f"predicting with {path} ...", flush=True)
        pred, labels = collect(path, args.data, device, args.batch_size,
                               args.max_shards, blend_spec=args.blend_alpha)
        if y is None:
            y = labels
        elif labels.shape != y.shape or not np.array_equal(labels, y):
            raise SystemExit(f"{path}: validation rows do not align with the first model")
        members.append(pred)
        individual[os.path.abspath(path)] = metrics(pred, y)
        print(f"  {os.path.basename(os.path.dirname(path))}: "
              f"rmse={individual[os.path.abspath(path)]['rmse']:.6f}", flush=True)

    assert y is not None
    n_models = len(members)
    if n_models < 2:
        raise SystemExit("stacking needs at least two checkpoints")

    design = np.column_stack(members + [np.ones_like(y)])  # [rows, N+1]

    # --- out-of-fold weights -------------------------------------------
    n = y.size
    bounds = np.linspace(0, n, args.folds + 1).astype(np.int64)
    oof = np.empty(n, dtype=np.float64)
    fold_weights: List[List[float]] = []
    for k in range(args.folds):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        held = np.zeros(n, dtype=bool)
        held[lo:hi] = True
        coef = fit_weights(design[~held], y[~held])
        fold_weights.append(coef.tolist())
        oof[held] = design[held] @ coef

    full_coef = fit_weights(design, y)
    if args.clip:
        oof = np.clip(oof, 1.0, 5.0)

    oof_metrics = metrics(oof, y)
    in_sample = metrics(np.clip(design @ full_coef, 1.0, 5.0) if args.clip
                        else design @ full_coef, y)

    equal_weight = np.mean(members, axis=0)
    if args.clip:
        equal_weight = np.clip(equal_weight, 1.0, 5.0)
    equal_metrics = metrics(equal_weight, y)

    print(f"\nmembers ({n_models}):")
    for path, m in individual.items():
        print(f"  {m['rmse']:.6f}  {path}")
    print(f"equal-weight  rmse={equal_metrics['rmse']:.6f} mae={equal_metrics['mae']:.6f}")
    print(f"stack (in-sample)  rmse={in_sample['rmse']:.6f}")
    print(f"stack (5-fold OOF) rmse={oof_metrics['rmse']:.6f} mae={oof_metrics['mae']:.6f}")
    print("full-fit weights: " + ", ".join(f"{w:+.8f}" for w in full_coef))

    np.savez(os.path.join(args.out, "val_preds.npz"), pred=oof, y=y)
    save_json({
        "rows": int(n),
        "folds": args.folds,
        "clip": args.clip,
        "checkpoints": [os.path.abspath(p) for p in args.checkpoint],
        "dense_schemas": [
            [str(c) for c in torch.load(p, map_location="cpu", weights_only=False)["dense_columns"]]
            for p in args.checkpoint
        ],
        "individual": individual,
        "equal_weight": equal_metrics,
        "stack_oof": oof_metrics,
        "stack_in_sample": in_sample,
        "weights": full_coef[:-1].tolist(),
        "intercept": float(full_coef[-1]),
        "fold_weights": fold_weights,
        "method": "least squares on [pred_1..pred_N, 1] with contiguous "
                  f"{args.folds}-fold out-of-fold weight fitting",
    }, os.path.join(args.out, "stack.json"))
    save_json({"rmse": oof_metrics["rmse"], "mae": oof_metrics["mae"], "rows": int(n)},
              os.path.join(args.out, "val_preds_metrics.json"))
    print(f"\nwrote {args.out}/val_preds.npz, stack.json, val_preds_metrics.json")


if __name__ == "__main__":
    main()
