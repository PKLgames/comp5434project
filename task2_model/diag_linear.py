"""Cheap diagnostic: how much signal is in the existing cached features?

Fits Ridge (sparse TF-IDF + dense) on a subsample of the cache and reports
validation RMSE for text-only / dense-only / combined, plus per-feature
correlations.  Nothing is written except to stdout.
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge

DATA = "artifacts"
N_TRAIN_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N_VAL_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def load(split, n_shards):
    man = json.load(open(f"{DATA}/{split}/manifest.json"))
    rows_t, rows_c, rows_d, ys = [], [], [], []
    for s in man["shards"][:n_shards]:
        d = np.load(f"{DATA}/{split}/{s['file']}")
        n = d["y"].shape[0]
        rows_t.append(sp.csr_matrix((d["t_data"], d["t_indices"], d["t_indptr"]), shape=(n, 55731)))
        rows_c.append(sp.csr_matrix((d["c_data"], d["c_indices"], d["c_indptr"]), shape=(n, 182382)))
        rows_d.append(d["dense"])
        ys.append(d["y"].astype(np.float32))
    Xt = sp.vstack(rows_t).tocsr()
    Xc = sp.vstack(rows_c).tocsr()
    Xd = np.vstack(rows_d)
    y = np.concatenate(ys)
    return Xt, Xc, Xd, y


def dense_matrix(Xd):
    # log1p + standardise the heavy-tailed count columns, keep the rest as is
    Z = Xd.copy()
    Z[:, 4] = np.log1p(np.clip(Xd[:, 4], 0, None))
    Z[:, 12] = np.log1p(np.clip(Xd[:, 12], 0, None))
    Z[:, 15] = np.log1p(np.clip(Xd[:, 15], 0, None))
    Z[:, 18] = np.log1p(np.clip(Xd[:, 18], 0, None))
    mu = Z.mean(0)
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    return ((Z - mu) / sd).astype(np.float32), mu, sd


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.clip(pred, 1, 5) - y) ** 2)))


def main():
    t0 = time.time()
    Xt_tr, Xc_tr, Xd_tr, y_tr = load("train", N_TRAIN_SHARDS)
    Xt_va, Xc_va, Xd_va, y_va = load("valid", N_VAL_SHARDS)
    print(f"train rows={len(y_tr):,} valid rows={len(y_va):,} ({time.time()-t0:.0f}s)", flush=True)

    Z_tr, mu, sd = dense_matrix(Xd_tr)
    Z_va = ((dense_matrix(Xd_va)[0] * sd) + mu)  # only to reuse shape; replaced below
    Z_va, _, _ = dense_matrix(Xd_va)

    names = json.load(open(f"{DATA}/meta.json"))["dense_columns"]
    print("baseline mean RMSE:", round(rmse(np.full_like(y_va, y_tr.mean()), y_va), 4), flush=True)

    # per-feature correlation with the target
    yy = (y_tr - y_tr.mean()) / y_tr.std()
    corr = ((Z_tr - Z_tr.mean(0)) * yy[:, None]).mean(0)
    order = np.argsort(-np.abs(corr))
    print("\ndense correlations with rating:")
    for i in order:
        print(f"  {names[i]:32s} {corr[i]:+.3f}", flush=True)

    for tag, (A_tr, A_va) in {
        "text_only": (sp.hstack([Xc_tr, Xt_tr]).tocsr(), sp.hstack([Xc_va, Xt_va]).tocsr()),
        "dense_only": (sp.csr_matrix(Z_tr), sp.csr_matrix(Z_va)),
        "text+dense": (sp.hstack([Xc_tr, Xt_tr, sp.csr_matrix(Z_tr)]).tocsr(),
                       sp.hstack([Xc_va, Xt_va, sp.csr_matrix(Z_va)]).tocsr()),
    }.items():
        for alpha in (1.0, 10.0):
            t1 = time.time()
            m = Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=200)
            m.fit(A_tr, y_tr)
            p = m.predict(A_va)
            print(f"{tag:12s} alpha={alpha:<5} val RMSE={rmse(p, y_va):.4f}  ({time.time()-t1:.0f}s)", flush=True)
        del A_tr, A_va


if __name__ == "__main__":
    main()
