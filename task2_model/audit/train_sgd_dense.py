"""Delivered-label linear model + the 9 per-review numeric features.

Same protocol as `train_sgd.py` (which is text-only) but appends the standardised
dense block to the sparse matrix, so we can measure whether votes / year / length
carry anything beyond the text.  Evaluate afterwards with `eval_clean.py`.
"""
from __future__ import annotations
import argparse, glob, math, os
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import SGDRegressor

C_DIM, T_DIM = 2 ** 17, 2 ** 15


def sf(d):
    return sorted(glob.glob(os.path.join(d, "shard_*.npz")))


def rows(f, batch_rows, mu=None, sd=None):
    d = np.load(f)
    c_ptr, c_idx, c_dat = d["c_indptr"], d["c_indices"], d["c_data"]
    t_ptr, t_idx, t_dat = d["t_indptr"], d["t_indices"], d["t_data"]
    n = len(c_ptr) - 1
    for r0 in range(0, n, batch_rows):
        r1 = min(n, r0 + batch_rows); m = r1 - r0
        c = sp.csr_matrix((c_dat[c_ptr[r0]:c_ptr[r1]], c_idx[c_ptr[r0]:c_ptr[r1]],
                           c_ptr[r0:r1+1]-c_ptr[r0]), shape=(m, C_DIM), dtype=np.float32)
        t = sp.csr_matrix((t_dat[t_ptr[r0]:t_ptr[r1]], t_idx[t_ptr[r0]:t_ptr[r1]],
                           t_ptr[r0:r1+1]-t_ptr[r0]), shape=(m, T_DIM), dtype=np.float32)
        dn = d["dense"][r0:r1].astype(np.float64)
        if mu is not None:
            dn = (dn - mu) / sd
        yield sp.hstack([c, t, sp.csr_matrix(dn.astype(np.float32))], format="csr"), d["y"][r0:r1].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/tmp/abl2/train_full")
    ap.add_argument("--valid", default="/tmp/abl/valid")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=50000)
    ap.add_argument("--out", default="/tmp/abl2/sgd_dense.npz")
    a = ap.parse_args()
    z = np.load(sf(a.train)[0])
    sample = z["dense"].astype(np.float64)
    mu = sample.mean(0); sd = np.maximum(sample.std(0), 0.05)
    model = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-5,
                         learning_rate="invscaling", eta0=0.02, power_t=0.25,
                         max_iter=1, tol=None, shuffle=False, random_state=0)
    for ep in range(1, a.epochs + 1):
        tot, cnt = 0.0, 0
        for f in sf(a.train):
            for X, y in rows(f, a.batch_rows, mu, sd):
                model.partial_fit(X, y)
                tot += float(((np.clip(X @ model.coef_ + model.intercept_[0], 1, 5) - y) ** 2).sum()); cnt += len(y)
        err, n = 0.0, 0
        for f in sf(a.valid):
            for X, y in rows(f, a.batch_rows, mu, sd):
                err += float(((np.clip(X @ model.coef_ + model.intercept_[0], 1, 5) - y) ** 2).sum()); n += len(y)
        print(f"[ep{ep}] train {math.sqrt(tot/cnt):.4f} | val(deliv) {math.sqrt(err/n):.4f}", flush=True)
    np.savez(a.out, coef=model.coef_, intercept=np.array(model.intercept_[0]), mu=mu, sd=sd)
    print("saved", a.out)


if __name__ == "__main__":
    main()
