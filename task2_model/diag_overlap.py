"""Does the 10k test set sit inside the 7.68M training rows?

Both splits carry the same hashed TF-IDF vectors.  If a test row *is* a train
row, its comment/title index+value arrays are byte-identical, so a fingerprint
match rate near 100% means the test set is a subset of train; a rate near the
duplicate-text background rate (~0.3%) means the splits are disjoint.
"""
import glob
import numpy as np
import pyarrow.parquet as pq

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def fingerprints(indices, values, indptr):
    n = indptr.size - 1
    counts = np.diff(indptr).astype(np.uint64)
    starts = indptr[:-1].astype(np.int64)
    s1 = np.zeros(n, dtype=np.float64)
    s2 = np.zeros(n, dtype=np.float64)
    nz = counts > 0
    if nz.any():
        s1[nz] = np.add.reduceat(indices.astype(np.float64), starts[nz])
        s2[nz] = np.add.reduceat(values.astype(np.float64), starts[nz])
    k1 = s1.astype(np.uint64)
    k2 = (np.round(s2 * 1e6).astype(np.int64).astype(np.uint64) << np.uint64(8)) | counts
    h = (k1 * np.uint64(1000003) + k2 * np.uint64(7919) + counts * np.uint64(104729)) & MASK
    return h


def shard_fp(path):
    d = np.load(path)
    return fingerprints(d["c_indices"], d["c_data"], d["c_indptr"])


test_fp = np.concatenate([
    shard_fp(f) for f in sorted(glob.glob("artifacts/valid/*.npz"))
][:0]) if False else None

# test set fingerprints come from the parquet file (they are not in artifacts/)
import pyarrow as pa
test_hashes = []
for f in sorted(glob.glob("test_features_with_entity/part-*.parquet")):
    t = pq.read_table(f, columns=["comment_tfidf", "title_tfidf"])["comment_tfidf"]
    idx = t.combine_chunks().field("indices").to_numpy(zero_copy_only=False)
    val = t.combine_chunks().field("values").to_numpy(zero_copy_only=False)
    offsets = np.asarray(t.combine_chunks().offsets)
    idx = np.concatenate([np.asarray(x, dtype=np.int32) for x in idx]) if idx.dtype == object else idx
    val = np.concatenate([np.asarray(x, dtype=np.float64) for x in val]) if val.dtype == object else val
    test_hashes.append(fingerprints(idx, val, offsets))

test_h = np.concatenate(test_hashes)
print("test rows:", test_h.size)

train_hashes = []
n_rows = 0
for path in sorted(glob.glob("artifacts/train/shard_*.npz")):
    train_hashes.append(shard_fp(path))
    n_rows += train_hashes[-1].size
train_h = np.concatenate(train_hashes)
print("train rows:", train_h.size)

train_sorted = np.sort(train_h)
pos = np.searchsorted(train_sorted, test_h)
pos = np.clip(pos, 0, train_sorted.size - 1)
hit = train_sorted[pos] == test_h
print(f"test rows with an identical train comment-tfidf fingerprint: {hit.sum():,} / {test_h.size} = {hit.mean():.2%}")
np.save("/tmp/test_comment_fp.npy", test_h)
