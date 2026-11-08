"""Empirical noise floor of the rating given the review text.

Rows whose title+comment produce the *identical* hashed-TF-IDF vector have the
same text (hash collisions aside).  If those rows carry different ratings, no
model that sees only the text can predict them.  The within-group variance of
duplicate-text groups is therefore a direct estimate of E[Var(y | text)], and
sqrt of it is a lower bound on the achievable RMSE for any text-only model.
"""
import glob
import numpy as np

P1 = np.uint64(1000003)
P2 = np.uint64(7919)
P3 = np.uint64(104729)
P4 = np.uint64(31337)


def row_fp(indices, values, indptr):
    n = indptr.size - 1
    counts = np.diff(indptr).astype(np.int64)
    starts = indptr[:-1].astype(np.int64)
    nz = counts > 0
    s_idx = np.zeros(n, dtype=np.int64)
    s_val = np.zeros(n, dtype=np.int64)
    if nz.any():
        s_idx[nz] = np.add.reduceat(indices.astype(np.int64), starts[nz])
        s_val[nz] = np.add.reduceat(np.round(values * 1e6).astype(np.int64), starts[nz])
    return counts, s_idx, s_val


def main():
    keys, ys = [], []
    for p in sorted(glob.glob("artifacts/train/shard_*.npz")):
        d = np.load(p)
        c, s1, s2 = row_fp(d["c_indices"], d["c_data"], d["c_indptr"])
        keys.append(np.stack([c, s1, s2], axis=1))
        ys.append(d["y"])
        del d
    keys = np.concatenate(keys)
    y = np.concatenate(ys).astype(np.float64)
    print(f"rows {keys.shape[0]:,}")

    # 128-bit-ish composite key: hash the comment fingerprint
    k = (keys[:, 0].astype(np.uint64) * P1
         + keys[:, 1].astype(np.uint64) * P2
         + keys[:, 2].astype(np.uint64) * P3)
    uk, inv, counts = np.unique(k, return_inverse=True, return_counts=True)
    print(f"unique comment-text fingerprints: {uk.size:,}  ({uk.size/keys.shape[0]:.2%} of rows)")

    # groups: sum y and sum y^2 per fingerprint
    gsum = np.bincount(inv, weights=y, minlength=uk.size)
    gsum2 = np.bincount(inv, weights=y * y, minlength=uk.size)
    g = counts.astype(np.float64)
    dup = counts >= 2
    print(f"duplicate-text rows: {counts[dup].sum():,} in {int(dup.sum()):,} groups "
          f"({counts[dup].sum()/keys.shape[0]:.2%} of rows)")

    ss_within = (gsum2[dup] - gsum[dup] ** 2 / g[dup]).sum()
    dof = (g[dup] - 1).sum()
    var_within = ss_within / dof
    print(f"\nwithin-duplicate-text variance = {var_within:.4f}  (sd = {np.sqrt(var_within):.4f})")
    print(f"total variance of y          = {y.var():.4f}  (sd = {y.std():.4f})")
    print(f"=> share of variance explainable by text alone at most: {1 - var_within/y.var():.2%}")
    print(f"=> RMSE floor for a perfect text-only model: >= {np.sqrt(var_within):.4f}")

    # weight by group size so the estimate reflects the distribution of inference rows
    w = (g[dup] - 1) / dof
    var_w = (w * (gsum2[dup] - gsum[dup] ** 2 / g[dup]) / (g[dup] - 1)).sum()
    print(f"size-weighted floor: {np.sqrt(var_w):.4f}")

    # what does the ambiguous "short generic text" class look like?
    order = np.argsort(-counts)[:15]
    print("\nlargest duplicate-text groups (nnz, group size, mean rating, sd):")
    for i in order:
        m = y[inv == i]
        print(f"  nnz={keys[np.argmax(inv == i), 0]:>3}  n={counts[i]:>6}  "
              f"mean={m.mean():.2f}  sd={m.std():.2f}")

    # rating distribution inside all duplicate groups vs overall
    print("\noverall rating shares:", np.round(np.bincount(y.astype(int), minlength=6)[1:] / y.size, 3))
    m_dup = y[dup[inv]]
    print("duplicate-group rating shares:", np.round(np.bincount(m_dup.astype(int), minlength=6)[1:] / m_dup.size, 3))


if __name__ == "__main__":
    main()
