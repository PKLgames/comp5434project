"""A/B/C/D of 'clean the training set' tricks, on the delivered (noisy) labels.

Arms
  A plain      : ridge on y
  B trimmed    : ridge, drop 10% largest-residual rows, refit
  C em_soft    : 2 EM-style iterations, target t = 0.8*y + 0.2*m(x)
  D affine     : ridge, then a,b fitted on a training slice (delivered RMSE)
  E hard_relabel: y := round(m(x)) where |y - m(x)| > 2, refit

All arms see exactly the same features and the same validation rows.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, hstack as sph, vstack


def load(split_dir):
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    parts = {"C": [], "T": [], "dense": [], "y": [], "ids": []}
    for f in files:
        d = np.load(os.path.join(split_dir, f))
        n = d["y"].shape[0]
        C = csr_matrix((d["c_data"], d["c_indices"], d["c_indptr"]), shape=(n, 2 ** 17))
        T = csr_matrix((d["t_data"], d["t_indices"], d["t_indptr"]), shape=(n, 2 ** 15))
        parts["C"].append(C)
        parts["T"].append(T)
        parts["dense"].append(d["dense"])
        parts["y"].append(d["y"])
        parts["ids"].append(d["ids"])
    C = vstack(parts["C"]).tocsr()
    T = vstack(parts["T"]).tocsr()
    dense = np.concatenate(parts["dense"])
    y = np.concatenate(parts["y"]).astype(np.float64)
    ids = np.concatenate(parts["ids"])
    return C, T, dense, y, ids


def rmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 1, 5) - y) ** 2)))


def main():
    from sklearn.linear_model import Ridge

    pool_dir, valid_dir = "/tmp/abl/pool", "/tmp/abl/valid"
    Cp, Tp, Dp, yp, _ = load(pool_dir)
    Cv, Tv, Dv, yv, _ = load(valid_dir)
    print(f"[data] pool {Cp.shape[0]:,} valid {Cv.shape[0]:,}", flush=True)

    mu, sd = Dp.mean(0), Dp.std(0) + 1e-6
    Dp = (Dp - mu) / sd
    Dv = (Dv - mu) / sd

    # calibration slice: last 10% of pool, never used for the reported metric
    n_cal = len(yp) // 10
    tr = slice(0, len(yp) - n_cal)
    cal = slice(len(yp) - n_cal, len(yp))

    def design(C, T, D, rows=None):
        X = sph([C, T, csr_matrix(D)], format="csr")
        return X[rows] if rows is not None else X

    Xp = design(Cp, Tp, Dp)
    Xv = design(Cv, Tv, Dv)
    Xc = Xp[cal]
    yc = yp[cal]
    Xtr = Xp[tr]
    ytr = yp[tr]
    print(f"[data] design {Xtr.shape} nnz/row {Xtr.nnz/Xtr.shape[0]:.1f}", flush=True)

    def fit(X, y, alpha=1.0):
        return Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=400).fit(X, y)

    res = {}
    # ---- A plain
    for a in (0.3, 1.0, 3.0, 10.0):
        m = fit(Xtr, ytr, a)
        r = rmse(m.predict(Xv), yv)
        res[f"A_plain_a{a}"] = r
        print(f"[A] alpha {a:5.1f} delivered {r:.4f}", flush=True)
    best_a = min((v, k) for k, v in res.items() if k.startswith("A_"))[1].split("_a")[-1]
    best_a = float(best_a)
    print(f"[A] best alpha {best_a}", flush=True)

    # ---- B trimmed
    for q in (0.95, 0.90, 0.80):
        m = fit(Xtr, ytr, best_a)
        p = m.predict(Xtr)
        keep = np.abs(ytr - p) <= np.quantile(np.abs(ytr - p), q)
        m2 = fit(Xtr[keep], ytr[keep], best_a)
        r = rmse(m2.predict(Xv), yv)
        res[f"B_trim_q{int(q*100)}"] = r
        print(f"[B] trim q{int(q*100)} keep {keep.mean():.2f} delivered {r:.4f}", flush=True)

    # ---- C em_soft
    m = fit(Xtr, ytr, best_a)
    for it in (1, 2):
        t = 0.8 * ytr + 0.2 * m.predict(Xtr)
        m = fit(Xtr, t, best_a)
        r = rmse(m.predict(Xv), yv)
        res[f"C_em_soft_it{it}"] = r
        print(f"[C] em_soft iter {it} delivered {r:.4f}", flush=True)

    # ---- D affine calibration on the held-out slice
    m = fit(Xtr, ytr, best_a)
    pc_ = m.predict(Xc)
    ab = np.polyfit(pc_, yc, 1)
    pv = ab[0] * m.predict(Xv) + ab[1]
    res["D_affine"] = rmse(pv, yv)
    res["D_ab"] = [float(ab[0]), float(ab[1])]
    print(f"[D] affine a,b={ab[0]:.4f},{ab[1]:.4f} delivered {res['D_affine']:.4f}", flush=True)

    # ---- E hard relabel
    m = fit(Xtr, ytr, best_a)
    p = m.predict(Xtr)
    y2 = ytr.copy()
    hot = np.abs(ytr - p) > 2.0
    y2[hot] = np.clip(np.round(p[hot]), 1, 5)
    m2 = fit(Xtr, y2, best_a)
    res["E_hard_relabel"] = rmse(m2.predict(Xv), yv)
    res["E_hot_frac"] = float(hot.mean())
    print(f"[E] relabelled {hot.mean():.2%} delivered {res['E_hard_relabel']:.4f}", flush=True)

    json.dump(res, open("/tmp/abl/ablation.json", "w"), indent=2)
    print("BEST", min((v, k) for k, v in res.items() if isinstance(v, float)), flush=True)


if __name__ == "__main__":
    main()
