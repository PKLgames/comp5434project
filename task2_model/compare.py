#!/usr/bin/env python
"""Paired comparison of two validation prediction files.

This is the objective arbiter for the V2 plan: it answers "is the candidate
actually better than the baseline, or is the difference noise?"  Both inputs are
``.npz`` files written by ``predict.py`` / ``stack.py`` containing ``pred`` and
``y`` for the *same* validation rows in the *same* order, so every statistic here
is paired.

Outputs RMSE / MAE deltas, per-rating deltas and a paired bootstrap confidence
interval on the deltas.  A candidate is only adopted when the interval excludes
0 (see ``--decision-threshold``).

Usage
-----
python compare.py --base ../runs/ensemble_stack/val_preds.npz \
                  --candidate ../runs/v2_stack/val_preds.npz \
                  --out ../runs/v2_stack/comparison.json
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np

from common import save_json


# ---------------------------------------------------------------------------
def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def load_preds(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        for key in ("pred", "y"):
            if key not in npz.files:
                raise SystemExit(f"{path} has no {key!r} array (keys: {npz.files})")
        pred = npz["pred"].astype(np.float64)
        y = npz["y"].astype(np.float64)
    if pred.shape != y.shape:
        raise SystemExit(f"{path}: pred {pred.shape} and y {y.shape} disagree")
    return pred, y


def per_rating(base: np.ndarray, cand: np.ndarray, y: np.ndarray) -> Dict[str, dict]:
    """Per-rating RMSE for both models plus the delta."""

    buckets = np.clip(np.rint(y), 1, 5).astype(np.int64)
    out: Dict[str, dict] = {}
    for rating in range(1, 6):
        mask = buckets == rating
        rows = int(mask.sum())
        if not rows:
            continue
        b, c = rmse(base[mask], y[mask]), rmse(cand[mask], y[mask])
        out[str(rating)] = {
            "rows": rows,
            "base_rmse": b,
            "candidate_rmse": c,
            "delta_rmse": c - b,
        }
    return out


def paired_bootstrap(
    base: np.ndarray,
    cand: np.ndarray,
    y: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, dict]:
    """Percentile bootstrap CI of ``RMSE(candidate) - RMSE(base)``.

    Rows are resampled *with replacement* and both models are scored on exactly
    the same resampled rows, so the correlation between the two prediction
    vectors is preserved (an unpaired CI would be far too wide).
    """

    n = y.size
    base_sq = (base - y) ** 2
    cand_sq = (cand - y) ** 2
    base_abs = np.abs(base - y)
    cand_abs = np.abs(cand - y)

    rng = np.random.default_rng(seed)
    d_rmse = np.empty(reps, dtype=np.float64)
    d_mae = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        d_rmse[i] = (np.sqrt(cand_sq[idx].mean()) - np.sqrt(base_sq[idx].mean()))
        d_mae[i] = (cand_abs[idx].mean() - base_abs[idx].mean())

    def summarise(samples: np.ndarray, point: float) -> dict:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {
            "point": point,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "std_error": float(samples.std(ddof=1)),
            "reps": int(reps),
        }

    return {
        "delta_rmse": summarise(d_rmse, 0.0),
        "delta_mae": summarise(d_mae, 0.0),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="baseline val_preds.npz")
    parser.add_argument("--candidate", required=True, help="candidate val_preds.npz")
    parser.add_argument("--out", required=True, help="output comparison.json")
    parser.add_argument("--reps", type=int, default=5000,
                        help="paired bootstrap resamples (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=-0.001,
                        help="adopt the candidate only if delta_rmse <= this value "
                             "and the CI excludes 0 (default: -0.001)")
    args = parser.parse_args()

    base, y_base = load_preds(args.base)
    cand, y_cand = load_preds(args.candidate)
    if base.shape != cand.shape:
        raise SystemExit(f"row count mismatch: base {base.size:,} vs candidate {cand.size:,}")
    if not np.array_equal(y_base, y_cand):
        raise SystemExit("the two files do not describe the same validation labels "
                         "(or do not have the same row order)")

    base_rmse, cand_rmse = rmse(base, y_base), rmse(cand, y_cand)
    base_mae, cand_mae = mae(base, y_base), mae(cand, y_cand)
    d_rmse, d_mae = cand_rmse - base_rmse, cand_mae - base_mae

    print(f"rows         {y_base.size:,}")
    print(f"base RMSE    {base_rmse:.6f}   MAE {base_mae:.6f}")
    print(f"cand RMSE    {cand_rmse:.6f}   MAE {cand_mae:.6f}")
    print(f"delta        {d_rmse:+.6f}      {d_mae:+.6f}")
    print(f"bootstrapping {args.reps:,} paired resamples ...", flush=True)

    boot = paired_bootstrap(base, cand, y_base, args.reps, args.seed)
    # fill in the point estimates (the bootstrap gets its own mean)
    boot["delta_rmse"]["point"] = d_rmse
    boot["delta_mae"]["point"] = d_mae

    ci = boot["delta_rmse"]
    adopt = bool(d_rmse <= args.decision_threshold and ci["excludes_zero"])
    print(f"delta_rmse 95% CI [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}] "
          f"(excludes 0: {ci['excludes_zero']})")
    print(f"delta_mae  95% CI [{boot['delta_mae']['ci95_low']:+.6f}, "
          f"{boot['delta_mae']['ci95_high']:+.6f}]")
    print(f"decision: {'ADOPT candidate' if adopt else 'KEEP baseline'} "
          f"(threshold {args.decision_threshold:+.4f} on delta_rmse)")

    report = {
        "base": os.path.abspath(args.base),
        "candidate": os.path.abspath(args.candidate),
        "rows": int(y_base.size),
        "base_metrics": {"rmse": base_rmse, "mae": base_mae},
        "candidate_metrics": {"rmse": cand_rmse, "mae": cand_mae},
        "delta": {"rmse": d_rmse, "mae": d_mae},
        "bootstrap": boot,
        "per_rating": per_rating(base, cand, y_base),
        "decision": {
            "threshold": args.decision_threshold,
            "adopt": adopt,
            "rule": "delta_rmse <= threshold AND the paired bootstrap 95% CI "
                    "on delta_rmse excludes 0",
        },
    }
    save_json(report, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
