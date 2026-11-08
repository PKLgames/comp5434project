"""Exact-match test: are the 10k test rows literally inside the training corpus?

Both splits carry IDF-weighted hashed TF-IDF vectors from the same Part-A
pipeline, so a row present in both files has an identical comment index list and
value list.  We fingerprint each row by (nnz, sum(indices), round(sum(values)*1e6)),
sort the 6.1M visible train fingerprints by a 64-bit hash, look the 10k test
fingerprints up, verify element-wise, and report the ratings the matches carry
in train.
"""
import glob
import numpy as np
import pyarrow.parquet as pq

P1 = np.uint64(1000003)
P2 = np.uint64(7919)
P3 = np.uint64(104729)


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


def hash3(c, s1, s2):
    return (c.astype(np.uint64) * P1
            + s1.astype(np.uint64) * P2
            + s2.astype(np.uint64) * P3)


def read_list_col(comp, name):
    arr = comp.field(name)
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    offsets = np.asarray(arr.offsets)
    vals = arr.values.to_numpy(zero_copy_only=False)
    return vals, offsets


def read_test_fp(pattern="test_features_with_entity/part-*.parquet"):
    cs, s1s, s2s = [], [], []
    for p in sorted(glob.glob(pattern)):
        tab = pq.read_table(p, columns=["comment_tfidf"])
        comp = tab["comment_tfidf"].combine_chunks()
        idx, offs = read_list_col(comp, "indices")
        val, _ = read_list_col(comp, "values")
        c, s1, s2 = row_fp(idx, val, offs)
        cs.append(c); s1s.append(s1); s2s.append(s2)
        print(f"  {p.split('/')[-1][:24]}... rows={c.size}")
        del tab, comp, idx, val
    return np.concatenate(cs), np.concatenate(s1s), np.concatenate(s2s)


def main():
    print("reading test tfidf...")
    tc, t1, t2 = read_test_fp()
    print(f"test rows: {tc.size}  mean nnz {tc.mean():.1f}")

    keys, ys = [], []
    for p in sorted(glob.glob("artifacts/train/shard_*.npz")):
        d = np.load(p)
        c, s1, s2 = row_fp(d["c_indices"], d["c_data"], d["c_indptr"])
        keys.append(np.stack([c, s1, s2], axis=1))
        ys.append(d["y"])
        del d
    keys = np.concatenate(keys)
    y = np.concatenate(ys)
    print(f"train rows fingerprinted: {keys.shape[0]:,}  y mean {y.mean():.4f}")

    K = hash3(keys[:, 0], keys[:, 1], keys[:, 2])
    order = np.argsort(K, kind="stable")
    K = K[order]
    keys = keys[order]

    qk = hash3(tc, t1, t2)
    pos = np.clip(np.searchsorted(K, qk), 0, K.size - 1)
    hit = (K[pos] == qk) & (keys[pos, 0] == tc) & (keys[pos, 1] == t1) & (keys[pos, 2] == t2)

    n = tc.size
    print(f"\nexact-fingerprint matches in visible train: {int(hit.sum()):,} / {n} = {hit.mean():.2%}")
    if hit.any():
        m = y[order[pos[hit]]]
        print("ratings of matched train rows:", np.bincount(m, minlength=6)[1:], " mean", float(m.mean()))
    print("empty-comment test rows:", int((tc == 0).sum()))


if __name__ == "__main__":
    main()
