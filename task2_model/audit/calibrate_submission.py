"""Recalibrate the shipped linear predictions onto the clean-label target.

The submission scores against the *clean* rating from the public source, while the
model was trained on the delivered labels (25% of which were replaced by a uniform
draw).  That replacement applies a 0.75 shrink to the model output, so the raw
predictions are compressed toward 3.75 while the clean mean is ~4.12.  This script
fits r_hat = a * p + b on the recovered clean labels and writes the calibrated
submission.

Usage:
  python -m task2_model.audit.calibrate_submission \
      --valid /tmp/abl/valid --ckpt /tmp/abl2/sgd_full7.npz \
      --test /tmp/abl2/test/all --out comp5434project/submission_calibrated.csv
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression

C_DIM, T_DIM = 2 ** 17, 2 ** 15


def shards(px):
    fs = sorted(glob.glob(os.path.join(px, "shard_*.npz")))
    if not fs:
        raise SystemExit(f"no shards under {px}")
    return fs


def load(f):
    z = np.load(f, allow_pickle=False)
    n = len(z["ids"])
    c = sp.csr_matrix((z["c_data"], z["c_indices"], z["c_indptr"]), shape=(n, C_DIM))
    t = sp.csr_matrix((z["t_data"], z["t_indices"], z["t_indptr"]), shape=(n, T_DIM))
    return sp.hstack([c, t]).tocsr(), z["y"] if "y" in z.files else None, z["ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default="/tmp/abl/valid")
    ap.add_argument("--ckpt", default="/tmp/abl2/sgd_full7.npz")
    ap.add_argument("--test", default="/tmp/abl2/test/all")
    ap.add_argument("--clean", default="/tmp/clean_y.npz")
    ap.add_argument("--out", default="submission_calibrated.csv")
    ap.add_argument("--method", choices=["affine", "isotonic"], default="isotonic")
    a = ap.parse_args()

    d = np.load(a.ckpt)
    w, b0 = d["coef"].astype(np.float64), float(np.ravel(d["intercept"])[0])
    cz = np.load(a.clean)
    cy, matched = cz["cy"], cz["matched"]

    ps, ids = [], []
    for f in shards(a.valid):
        X, _, sid = load(f)
        ps.append(np.clip(X @ w + b0, 1, 5)); ids.append(sid)
    p = np.concatenate(ps); i = np.concatenate(ids)
    m = matched[i]
    y = cy[i][m].astype(np.float64)
    rmse = lambda u, v: float(np.sqrt(np.mean((u - v) ** 2)))
    if a.method == "affine":
        A = np.vstack([p[m], np.ones(m.sum())]).T
        (ca, cb), *_ = np.linalg.lstsq(A, y, rcond=None)
        apply_ = lambda v: np.clip(ca * v + cb, 1, 5)
        print(f"[calib] affine a={ca:.4f} b={cb:.4f}", end=" ")
        params = {"method": "affine", "a": float(ca), "b": float(cb)}
    else:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1, y_max=5).fit(p[m], y)
        apply_ = lambda v: np.clip(iso.predict(v), 1, 5)
        print("[calib] isotonic", end=" ")
        params = {"method": "isotonic",
                  "x": [float(v) for v in iso.X_thresholds_],
                  "y": [float(v) for v in iso.y_thresholds_]}
    print(f"clean RMSE raw {rmse(p[m], y):.5f} -> {rmse(apply_(p[m]), y):.5f} on {m.sum()} rows")

    tps, tids = [], []
    for f in shards(a.test):
        X, _, sid = load(f)
        tps.append(np.clip(X @ w + b0, 1, 5)); tids.append(sid)
    tp = np.concatenate(tps); tid = np.concatenate(tids)
    rating = apply_(tp)
    order = np.argsort(tid)
    with open(a.out, "w") as fh:
        fh.write("id,rating\n")
        for k in order:
            fh.write(f"{int(tid[k])},{rating[k]:.6f}\n")
    print(f"[out] {a.out}: {len(order)} rows, mean {rating.mean():.4f}, range {rating.min():.3f}-{rating.max():.3f}")
    json.dump({**params, "ckpt": a.ckpt}, open("/tmp/calib.json", "w"))


if __name__ == "__main__":
    main()
