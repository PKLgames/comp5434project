"""Evaluate a linear checkpoint on the clean-label target.

Usage:
  python audit/eval_clean.py --valid /tmp/abl/valid --ckpt /tmp/abl2/sgd_full7.npz
Reports RMSE against (a) delivered labels, (b) recovered clean labels, and
(c) the best affine recalibration fitted on half the validation rows and
measured on the other half.
"""
import argparse, glob, os, numpy as np, scipy.sparse as sp

def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    return d

def shard_xy(path, c_dim, t_dim, want_ids=True):
    z = np.load(path, allow_pickle=False)
    n = len(z["y"]) if "y" in z.files else len(z["ids"])
    c = sp.csr_matrix((z["c_data"], z["c_indices"], z["c_indptr"]), shape=(n, c_dim))
    t = sp.csr_matrix((z["t_data"], z["t_indices"], z["t_indptr"]), shape=(n, t_dim))
    return sp.hstack([c, t]).tocsr(), (z["y"] if "y" in z.files else None), (z["ids"] if want_ids else None)

def predict(X, d):
    w = d["coef"].astype(np.float64)
    b = float(np.ravel(d["intercept"])[0])
    return X @ w + b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default="/tmp/abl/valid")
    ap.add_argument("--ckpt", default="/tmp/abl2/sgd_full7.npz")
    ap.add_argument("--clean", default="/tmp/clean_y.npz")
    ap.add_argument("--c-dim", type=int, default=2 ** 17)
    ap.add_argument("--t-dim", type=int, default=2 ** 15)
    a = ap.parse_args()
    d = load_ckpt(a.ckpt)
    assert a.c_dim + a.t_dim == len(d["coef"]), (a.c_dim, a.t_dim, len(d["coef"]))
    cz = np.load(a.clean)
    cy, matched, ag = cz["cy"], cz["matched"], cz["ag"]
    ys, ps, ids = [], [], []
    for f in sorted(glob.glob(os.path.join(a.valid, "shard_*.npz"))):
        X, y, sid = shard_xy(f, a.c_dim, a.t_dim)
        ps.append(predict(X, d)); ys.append(y); ids.append(sid)
    y = np.concatenate(ys).astype(np.float64); p = np.concatenate(ps); i = np.concatenate(ids)
    yc = cy[i].astype(np.float64); m = matched[i]
    p = np.clip(p, 1, 5)
    rmse = lambda a_, b_: float(np.sqrt(np.mean((a_ - b_) ** 2)))
    print("rows            %d  (matched %.2f%%)" % (len(i), 100 * m.mean()))
    print("RMSE delivered  %.5f" % rmse(p, y))
    print("RMSE clean(all) %.5f" % rmse(p, yc))
    print("RMSE clean(mch) %.5f" % rmse(p[m], yc[m]))
    # affine recalibration, fitted on even index, scored on odd
    idx = np.arange(len(i)); fit = (idx % 2 == 0) & m; ev = (idx % 2 == 1) & m
    A = np.vstack([p[fit], np.ones(fit.sum())]).T
    coef, *_ = np.linalg.lstsq(A, yc[fit], rcond=None)
    cal = np.clip(coef[0] * p[ev] + coef[1], 1, 5)
    print("calib a=%.4f b=%.4f -> RMSE clean(mch,heldout) %.5f  (raw %.5f)"
          % (coef[0], coef[1], rmse(cal, yc[ev]), rmse(p[ev], yc[ev])))
    print("pred mean %.4f  clean mean %.4f  delivered mean %.4f" % (p.mean(), yc[m].mean(), y[m].mean()))

if __name__ == "__main__":
    main()
