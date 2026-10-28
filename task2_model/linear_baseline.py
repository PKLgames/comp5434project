#!/usr/bin/env python
"""Linear TF-IDF + numeric baseline (streaming SGD).

This model is intentionally simple.  It serves three purposes:

1. a sanity check that the feature pipeline is informative,
2. a lower-bound reference for the neural model, and
3. a diverse member for the ensemble (its errors differ from the MLP's).

The text columns are already L2-normalised TF-IDF, so no further scaling is
needed.  Dense columns are standardised with the training statistics stored in
``meta.json``.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDRegressor

from common import DENSE_COLS, load_json, save_json, select_dense_columns
from dataset import ShardBatcher


def shard_matrix(loader: ShardBatcher, shard: dict, dense_index: np.ndarray) -> csr_matrix:
    path = os.path.join(loader.shard_dir, shard["file"])
    with np.load(path) as npz:
        t = csr_matrix((npz["t_data"], npz["t_indices"], npz["t_indptr"]),
                       shape=(npz["dense"].shape[0], loader.vocab_title))
        c = csr_matrix((npz["c_data"], npz["c_indices"], npz["c_indptr"]),
                       shape=(npz["dense"].shape[0], loader.vocab_comment))
        dense = npz["dense"][:, dense_index]
        y = npz["y"].astype(np.float32) if "y" in npz.files else None
    dense = (dense - loader.dense_mean) / loader.dense_std
    dense = np.clip(dense, -5.0, 5.0).astype(np.float32)
    return hstack([t, c, csr_matrix(dense)], format="csr"), y


def run_epoch(model: SGDRegressor, loader: ShardBatcher, epochs: int,
              rng: np.random.RandomState, dense_index: np.ndarray):
    for epoch in range(epochs):
        order = rng.permutation(len(loader.shards))
        for idx in order:
            X, y = shard_matrix(loader, loader.shards[idx], dense_index)
            perm = rng.permutation(X.shape[0])
            model.partial_fit(X[perm], y[perm])
        print(f"  linear epoch {epoch + 1}/{epochs} done", flush=True)


def predict(model, loader: ShardBatcher, dense_index: np.ndarray) -> np.ndarray:
    preds = []
    for shard in loader.shards:
        X, _ = shard_matrix(loader, shard, dense_index)
        preds.append(model.predict(X).astype(np.float32))
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1e-6, help="L2 penalty")
    parser.add_argument("--eta0", type=float, default=0.02)
    parser.add_argument("--lr-mode", default="invscaling",
                        choices=["constant", "invscaling", "adaptive"])
    parser.add_argument("--no-average", dest="average", action="store_false",
                        help="disable averaged SGD weights")
    parser.add_argument("--dense-mode", default="all",
                        choices=["all", "numeric", "entity", "none"])
    parser.add_argument("--limit-train-shards", type=int, default=None)
    parser.add_argument("--limit-valid-shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_json(vars(args), os.path.join(args.out, "args.json"))
    meta = load_json(os.path.join(args.data, "meta.json"))
    dense_columns = select_dense_columns(args.dense_mode)
    dense_index = np.array([DENSE_COLS.index(c) for c in dense_columns], dtype=np.int64)

    train_loader = ShardBatcher(args.data, "train", dense_columns, batch_size=1,
                                shuffle=False, max_shards=args.limit_train_shards)
    valid_loader = ShardBatcher(args.data, "valid", dense_columns, batch_size=1,
                                shuffle=False, max_shards=args.limit_valid_shards)

    model = SGDRegressor(
        loss="squared_error", penalty="l2", alpha=args.alpha,
        learning_rate=args.lr_mode, eta0=args.eta0, power_t=0.25,
        fit_intercept=True, average=args.average, random_state=args.seed,
    )
    # SGDRegressor cannot infer the feature count on the first partial_fit, so
    # warm it up with one shard and then train normally.
    first_X, first_y = shard_matrix(train_loader, train_loader.shards[0], dense_index)
    model.partial_fit(first_X, first_y)

    started = time.time()
    rng = np.random.RandomState(args.seed)
    run_epoch(model, train_loader, args.epochs, rng, dense_index)
    train_seconds = time.time() - started

    val_pred = predict(model, valid_loader, dense_index)
    y_parts = []
    for shard in valid_loader.shards:
        with np.load(os.path.join(valid_loader.shard_dir, shard["file"])) as npz:
            y_parts.append(npz["y"])
    y_val = np.concatenate(y_parts).astype(np.float32)
    rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    mae = float(np.mean(np.abs(val_pred - y_val)))

    np.savez(os.path.join(args.out, "val_preds.npz"), pred=val_pred, y=y_val)
    save_json({"rmse": rmse, "mae": mae, "rows": int(y_val.size),
               "train_seconds": round(train_seconds, 2)},
              os.path.join(args.out, "metrics.json"))
    print(f"linear baseline: val RMSE={rmse:.4f} MAE={mae:.4f} "
          f"({train_seconds:.1f}s train)")


if __name__ == "__main__":
    main()
