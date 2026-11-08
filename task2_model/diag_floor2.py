"""Stratified + full-feature noise floor.

1) Text floor reweighted to the population distribution of comment length
   (duplicate texts are short/generic, so the raw estimate is pessimistic).
2) Full-feature floor: rows sharing text AND the 26 dense values cannot be
   separated by ANY model built on the delivered feature file -> its
   within-group variance is the hard ceiling of the current feature medium.
"""
import glob
import numpy as np

P1, P2, P3, P4 = np.uint64(1000003), np.uint64(7919), np.uint64(104729), np.uint64(31337)


def row_fp(idx, val, indptr):
    n = indptr.size - 1
    counts = np.diff(indptr).astype(np.int64)
    starts = indptr[:-1].astype(np.int64)
    nz = counts > 0
    s1 = np.zeros(n, np.int64); s2 = np.zeros(n, np.int64)
    if nz.any():
        s1[nz] = np.add.reduceat(idx.astype(np.int64), starts[nz])
        s2[nz] = np.add.reduceat(np.round(val * 1e6).astype(np.int64), starts[nz])
    return counts, s1, s2


def floor_from_groups(inv, n_groups, y, min_size=2):
    counts = np.bincount(inv, minlength=n_groups)
    sel = counts >= min_size
    gsum = np.bincount(inv, weights=y, minlength=n_groups)
    gsum2 = np.bincount(inv, weights=y * y, minlength=n_groups)
    ss = (gsum2[sel] - gsum[sel] ** 2 / counts[sel]).sum()
    dof = (counts[sel] - 1).sum()
    return ss / dof, counts, dof


def main():
    keys, dense, ys = [], [], []
    for p in sorted(glob.glob("artifacts/train/shard_*.npz")):
        d = np.load(p)
        c, s1, s2 = row_fp(d["c_indices"], d["c_data"], d["c_indptr"])
        keys.append(np.stack([c, s1, s2], 1))
        dense.append(d["dense"].astype(np.float32))
        ys.append(d["y"])
        del d
    keys = np.concatenate(keys); dense = np.concatenate(dense); y = np.concatenate(ys).astype(np.float64)
    N = y.size
    tot_var = y.var()
    print(f"rows {N:,}  total sd {np.sqrt(tot_var):.4f}")

    k = keys[:, 0].astype(np.uint64) * P1 + keys[:, 1].astype(np.uint64) * P2 + keys[:, 2].astype(np.uint64) * P3
    uk, inv, counts = np.unique(k, return_inverse=True, return_counts=True)

    # ---- 1. text floor stratified by comment nnz -------------------------
    var_all, _, _ = floor_from_groups(inv, uk.size, y)
    print(f"\n[text floor] pooled            : {np.sqrt(var_all):.4f}")
    nnz = keys[:, 0]
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10**9)]
    rows = []
    for lo, hi in buckets:
        m = (nnz >= lo) & (nnz <= hi)
        if not m.any():
            continue
        cnt = np.bincount(inv[m], minlength=uk.size).astype(np.float64)
        gs = np.bincount(inv[m], weights=y[m], minlength=uk.size)
        gs2 = np.bincount(inv[m], weights=y[m] * y[m], minlength=uk.size)
        c2 = cnt >= 2
        if not c2.any():
            continue
        dof = (cnt[c2] - 1).sum()
        v = (gs2[c2] - gs[c2] ** 2 / cnt[c2]).sum() / dof
        rows.append((lo, hi, m.mean(), np.sqrt(v), cnt[c2].sum() / m.sum()))
    print("  bucket      rowshare  nndup share  within-sd")
    for lo, hi, share, sd, dup_share in rows:
        print(f"  nnz {lo:>3}-{hi:<4} {share:8.2%} {dup_share:10.2%} {sd:11.4f}")
    wsum = sum(r[2] for r in rows)
    floor_strat = np.sqrt(sum(r[2] * r[3] ** 2 for r in rows) / wsum)
    print(f"  -> population-weighted text floor: {floor_strat:.4f}")

    # ---- 2. full-feature floor -------------------------------------------
    dh = np.zeros(N, np.uint64)
    for j in range(dense.shape[1]):
        q = np.round(dense[:, j].astype(np.float64) * 100).astype(np.int64).astype(np.uint64)
        dh = dh * P4 + q
    struct = np.zeros(N, dtype=[("a", np.uint64), ("b", np.uint64)])
    struct["a"] = k; struct["b"] = dh
    uf, invf, countsf = np.unique(struct.view(np.void), return_inverse=True, return_counts=True)
    vf, _, dof = floor_from_groups(invf, uf.size, y)
    print(f"\n[feature floor] text+dense groups: {int((countsf>=2).sum()):,} with >=2 rows, "
          f"{int(countsf[countsf>=2].sum()):,} rows")
    print(f"[feature floor] within-group sd  : {np.sqrt(vf):.4f}")

    # ---- 3. text + time/votes only (the non-text info a model can use) ----
    dh2 = np.zeros(N, np.uint64)
    for j in range(dense.shape[1]):
        q = np.round(dense[:, j].astype(np.float64) * 100).astype(np.int64).astype(np.uint64)
        dh2 = dh2 * P4 + q * np.uint64(j + 1)
    st2 = np.zeros(N, dtype=[("a", np.uint64), ("b", np.uint64)])
    st2["a"] = k; st2["b"] = dh2
    u2, inv2, cnt2 = np.unique(st2.view(np.void), return_inverse=True, return_counts=True)
    v2, _, _ = floor_from_groups(inv2, u2.size, y)
    print(f"[feature floor] alt hash        : {np.sqrt(v2):.4f}  rows in dup groups "
          f"{int(cnt2[cnt2>=2].sum()):,}")
    print(f"\npublished NN val RMSE 1.1573 | linear text 1.2140 | constant {np.sqrt(tot_var):.4f}")


if __name__ == "__main__":
    main()
