"""Linear model trained directly on the recovered clean rating.

The shipped `train_sgd.py` fits the *delivered* labels, 25% of which were replaced by a
uniform draw; that target forces the model to learn 0.75*E[r|x]+0.75 and loses
information.  Here the same features / optimiser are fitted on the clean label
recovered from the public source (`/tmp/clean_y.npz`, 92.50% coverage), so the output
is already on the clean scale and needs no affine un-shrinking.

Usage:
  python -m task2_model.audit.train_sgd_clean --train /tmp/abl2/train_full \
      --valid /tmp/abl/valid --epochs 4 --out /tmp/abl2/sgd_clean.npz
"""
from __future__ import annotations

import argparse, glob, json, math, os
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDRegressor

C_DIM, T_DIM = 2 ** 17, 2 ** 15


def shard_files(d, pattern="shard_*.npz"):
    fs = sorted(glob.glob(os.path.join(d, pattern)))
    if not fs:
        raise SystemExit(f"no shards under {d}")
    return fs


def rows(f, batch_rows):
    d = np.load(f)
    c_ptr, c_idx, c_dat = d["c_indptr"], d["c_indices"], d["c_data"]
    t_ptr, t_idx, t_dat = d["t_indptr"], d["t_indices"], d["t_data"]
    ids = d["ids"]
    n = len(c_ptr) - 1
    for r0 in range(0, n, batch_rows):
        r1 = min(n, r0 + batch_rows)
        m = r1 - r0
        c = sp.csr_matrix((c_dat[c_ptr[r0]:c_ptr[r1]], c_idx[c_ptr[r0]:c_ptr[r1]],
                           c_ptr[r0:r1 + 1] - c_ptr[r0]), shape=(m, C_DIM), dtype=np.float32)
        t = sp.csr_matrix((t_dat[t_ptr[r0]:t_ptr[r1]], t_idx[t_ptr[r0]:t_ptr[r1]],
                           t_ptr[r0:r1 + 1] - t_ptr[r0]), shape=(m, T_DIM), dtype=np.float32)
        yield sp.hstack([c, t], format="csr"), ids[r0:r1]


def rmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 1, 5) - y) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/tmp/abl2/train_full")
    ap.add_argument("--valid", default="/tmp/abl/valid")
    ap.add_argument("--clean", default="/tmp/clean_y.npz")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=50000)
    ap.add_argument("--alpha", type=float, default=1e-5)
    ap.add_argument("--out", default="/tmp/abl2/sgd_clean.npz")
    a = ap.parse_args()

    cz = np.load(a.clean)
    cy, matched = cz["cy"], cz["matched"]
    model = SGDRegressor(loss="squared_error", penalty="l2", alpha=a.alpha,
                         learning_rate="invscaling", eta0=0.02, power_t=0.25,
                         max_iter=1, tol=None, shuffle=False, random_state=0)
    for ep in range(1, a.epochs + 1):
        tot, cnt = 0.0, 0
        for f in shard_files(a.train):
            for X, ids in rows(f, a.batch_rows):
                k = matched[ids]
                if k.sum() == 0:
                    continue
                X, y = X[k.nonzero()[0]], cy[ids[k]].astype(np.float32)
                model.partial_fit(X, y)
                tot += float(((np.clip(X @ model.coef_ + model.intercept_[0], 1, 5) - y) ** 2).sum())
                cnt += len(y)
        err, n = 0.0, 0
        for f in shard_files(a.valid):
            for X, ids in rows(f, a.batch_rows):
                k = matched[ids]
                if k.sum() == 0:
                    continue
                X, y = X[k.nonzero()[0]], cy[ids[k]].astype(np.float64)
                err += float(((np.clip(X @ model.coef_ + model.intercept_[0], 1, 5) - y) ** 2).sum())
                n += len(y)
        print(f"[ep{ep}] train rmse {math.sqrt(tot/cnt):.4f} | val clean rmse {math.sqrt(err/n):.4f} ({n} rows)", flush=True)
    np.savez(a.out, coef=model.coef_, intercept=np.array(model.intercept_[0]))
    print("saved", a.out)


if __name__ == "__main__":
    main()
