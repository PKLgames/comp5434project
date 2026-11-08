"""Full-data linear model (streaming SGD) — the stable, verifiable deliverable.

The torch model was abandoned: with 2^17 hashed features and Adam on a 33.5M-row
embedding table it needs far more epochs than the time budget allows to approach the
closed-form solution that Ridge reaches on a 690k-row subsample (1.2099). Streaming
SGD over all 7,492,528 training rows reaches a comparable solution in minutes and is
fully reproducible on CPU.

Usage:
  python -m task2_model.audit.train_sgd --train /tmp/abl2/train_full \
      --valid /tmp/abl/valid --epochs 3 --out /tmp/abl2/sgd_full.npz
  python -m task2_model.audit.train_sgd --predict /tmp/abl2/test/all \
      --ckpt /tmp/abl2/sgd_full.npz --submission comp5434project/submission.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDRegressor


def shard_files(d, pattern="shard_*.npz"):
    fs = sorted(glob.glob(os.path.join(d, pattern)))
    if not fs:
        raise SystemExit(f"no shards under {d}")
    return fs


def rows(f, batch_rows, with_y=True):
    d = np.load(f)
    c_ptr, c_idx, c_dat = d["c_indptr"], d["c_indices"], d["c_data"]
    t_ptr, t_idx, t_dat = d["t_indptr"], d["t_indices"], d["t_data"]
    dense, y, ids = d["dense"], d["y"] if with_y else None, d["ids"]
    n = len(c_ptr) - 1
    for r0 in range(0, n, batch_rows):
        r1 = min(n, r0 + batch_rows)
        m = r1 - r0
        c = sp.csr_matrix(
            (c_dat[c_ptr[r0]:c_ptr[r1]], c_idx[c_ptr[r0]:c_ptr[r1]],
             c_ptr[r0:r1 + 1] - c_ptr[r0]),
            shape=(m, 2 ** 17), dtype=np.float32,
        )
        t = sp.csr_matrix(
            (t_dat[t_ptr[r0]:t_ptr[r1]], t_idx[t_ptr[r0]:t_ptr[r1]],
             t_ptr[r0:r1 + 1] - t_ptr[r0]),
            shape=(m, 2 ** 15), dtype=np.float32,
        )
        X = sp.hstack([c, t], format="csr")
        yy = None if y is None else y[r0:r1].astype(np.float32)
        yield X, yy, dense[r0:r1], ids[r0:r1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--valid")
    ap.add_argument("--predict")
    ap.add_argument("--ckpt")
    ap.add_argument("--submission")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-rows", type=int, default=50000)
    ap.add_argument("--alpha", type=float, default=1e-5)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.predict:
        ck = np.load(a.ckpt)
        coef, b0 = ck["coef"], float(ck["intercept"])
        rows_out = []
        for f in shard_files(a.predict):
            for X, _, _, ids in rows(f, a.batch_rows, with_y=False):
                p = np.clip(X @ coef + b0, 1, 5)
                rows_out.append(np.column_stack([ids, p]))
        sub = np.concatenate(rows_out)
        sub = sub[np.argsort(sub[:, 0])]
        with open(a.submission, "w") as fh:
            fh.write("id,rating\n")
            for i, p in sub:
                fh.write(f"{int(i)},{p:.4f}\n")
        print(f"[predict] {len(sub)} rows -> {a.submission}")
        return

    model = SGDRegressor(
        loss="squared_error", penalty="l2", alpha=a.alpha, learning_rate="invscaling",
        eta0=0.02, power_t=0.25, max_iter=1, tol=None, shuffle=False, random_state=0,
        average=False,
    )
    best = 9.9
    for ep in range(1, a.epochs + 1):
        tot, cnt = 0.0, 0
        for f in shard_files(a.train):
            for X, y, _, _ in rows(f, a.batch_rows):
                model.partial_fit(X, y)
                p = np.clip(X @ model.coef_ + model.intercept_[0], 1, 5)
                tot += float(((p - y) ** 2).sum())
                cnt += len(y)
        msg = f"[ep{ep}] train rmse {math.sqrt(tot / cnt):.4f}"
        if a.valid:
            err, n = 0.0, 0
            for f in shard_files(a.valid):
                for X, y, _, _ in rows(f, a.batch_rows):
                    p = np.clip(X @ model.coef_ + model.intercept_[0], 1, 5)
                    err += float(((p - y) ** 2).sum())
                    n += len(y)
            v = math.sqrt(err / n)
            msg += f" | val rmse {v:.4f}"
            if v < best:
                best = v
                np.savez(a.out, coef=model.coef_, intercept=model.intercept_[0], val=v)
                msg += " *saved*"
        print(msg, flush=True)
    print(f"[done] best val {best:.4f}")


if __name__ == "__main__":
    main()
