"""Equal-sample comparison: bag-of-words vs sentence embeddings vs both.

Train rows are restricted to id % 20 == 0 (the rows encoded in /tmp/emb), validation is
the usual id % 40 == 0 set. Every arm is scored against the recovered clean rating,
raw and after isotonic calibration fitted on half of the validation rows.
"""
from __future__ import annotations

import glob
import numpy as np
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

C_DIM, T_DIM = 2 ** 17, 2 ** 15


def bow_for_ids(want):
    """Rows of the cached sparse features whose id is in `want` (np.int64 set)."""
    order = np.sort(want)
    blocks, keep = [], []
    for f in sorted(glob.glob("/tmp/abl2/train_full/shard_*.npz")):
        z = np.load(f)
        ids = z["ids"]
        sel = np.isin(ids, order)  # shard is sorted by id, so this is cheap
        if not sel.any():
            continue
        idx = np.nonzero(sel)[0]
        n = len(ids)
        c_ptr, c_idx, c_dat = z["c_indptr"], z["c_indices"], z["c_data"]
        t_ptr, t_idx, t_dat = z["t_indptr"], z["t_indices"], z["t_data"]
        c = sp.csr_matrix((c_dat[c_ptr[0]:c_ptr[n]], c_idx[c_ptr[0]:c_ptr[n]],
                           c_ptr[:n + 1]), shape=(n, C_DIM))[idx]
        t = sp.csr_matrix((t_dat[t_ptr[0]:t_ptr[n]], t_idx[t_ptr[0]:t_ptr[n]],
                           t_ptr[:n + 1]), shape=(n, T_DIM))[idx]
        blocks.append(sp.hstack([c, t]).tocsr())
        keep.append(ids[idx])
    return sp.vstack(blocks).tocsr(), np.concatenate(keep)


def valid_bow():
    blocks, ids = [], []
    for f in sorted(glob.glob("/tmp/abl/valid/shard_*.npz")):
        z = np.load(f)
        n = len(z["ids"])
        c = sp.csr_matrix((z["c_data"], z["c_indices"], z["c_indptr"]), shape=(n, C_DIM))
        t = sp.csr_matrix((z["t_data"], z["t_indices"], z["t_indptr"]), shape=(n, T_DIM))
        blocks.append(sp.hstack([c, t]).tocsr())
        ids.append(z["ids"])
    return sp.vstack(blocks).tocsr(), np.concatenate(ids)


def score(name, p, y, fit_mask, ev_mask):
    r = lambda u, v: float(np.sqrt(np.mean((u - v) ** 2)))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1, y_max=5).fit(p[fit_mask], y[fit_mask])
    q = np.clip(iso.predict(p[ev_mask]), 1, 5)
    print("%-22s raw %.4f  iso %.4f" % (name, r(np.clip(p[ev_mask], 1, 5), y[ev_mask]), r(q, y[ev_mask])))


def main():
    cz = np.load("/tmp/clean_y.npz")
    cy, matched = cz["cy"], cz["matched"]

    tr_emb = np.load("/tmp/emb/train_emb.npy").astype(np.float32)
    tr_ids = np.load("/tmp/emb/train_ids.npy")
    va_emb = np.load("/tmp/emb/valid_emb.npy").astype(np.float32)
    va_ids = np.load("/tmp/emb/valid_ids.npy")
    print("emb shapes", tr_emb.shape, va_emb.shape)

    keep = matched[tr_ids]
    Xe_tr, ytr, ids_tr = tr_emb[keep], cy[tr_ids[keep]].astype(np.float64), tr_ids[keep]
    print("train rows with clean label: %d" % len(ytr))

    Xb_tr, b_ids = bow_for_ids(ids_tr)
    assert np.array_equal(b_ids, ids_tr), "id order mismatch"

    Xb_va, v_ids = valid_bow()
    pos = {int(v): k for k, v in enumerate(v_ids)}
    m_va = matched[v_ids] & (cy[v_ids] > 0)
    keep_va = np.nonzero(m_va)[0]
    order_emb = np.array([pos[int(i)] for i in va_ids if matched[i] and cy[i] > 0])
    Xe_va = va_emb[[k for k, i in enumerate(va_ids) if matched[i] and cy[i] > 0]]
    y = cy[v_ids[keep_va]].astype(np.float64)
    idx = np.arange(len(y))
    fit, ev = idx % 2 == 0, idx % 2 == 1

    for name, Xtr, Xva in (("bow", Xb_tr, Xb_va[keep_va]),
                           ("emb", Xe_tr, Xe_va),
                           ("bow+emb", sp.hstack([Xb_tr, sp.csr_matrix(Xe_tr * 0.1)]).tocsr(),
                            sp.hstack([Xb_va[keep_va], sp.csr_matrix(Xe_va * 0.1)]).tocsr())):
        mdl = Ridge(alpha=1.0, solver="lsqr", tol=1e-3, max_iter=300).fit(Xtr, ytr)
        p = mdl.predict(Xva)
        score(name, p, y, fit, ev)


if __name__ == "__main__":
    main()
