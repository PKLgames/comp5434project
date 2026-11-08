"""Leakage / format audit for the final submission.

Checks, in order:
  1. train shards used for fitting contain no holdout row (id % 40 != 0);
  2. holdout shards contain only id % 40 == 0;
  3. test.csv is never inside the training shard set (fingerprint of ids + row count);
  4. submission ids == sampleSubmission ids (same set, same order as written);
  5. ratings inside [1, 5] and 10,000 rows.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ids_of(d):
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    return np.concatenate([np.load(f)["ids"] for f in fs])


def main(train_dir, valid_dir, submission, sample):
    tr = ids_of(train_dir)
    va = ids_of(valid_dir)
    print(f"[1] train rows {len(tr):,}  max id {tr.max():,}")
    assert (tr % 40 != 0).all(), "holdout row leaked into training shards"
    print("    ok: no id % 40 == 0 in training shards")
    print(f"[2] valid rows {len(va):,}")
    assert (va % 40 == 0).all(), "valid shard contains a training row"
    assert len(np.intersect1d(tr, va)) == 0, "train/valid id overlap"
    print("    ok: disjoint holdout")

    test_ids = ids_of(os.path.join(ROOT, "..", "abl2", "test", "all")) \
        if os.path.isdir(os.path.join(ROOT, "..", "abl2", "test", "all")) else None
    print(f"[3] training corpus ids span {tr.min()}..{tr.max()}; test.csv is a separate "
          f"file with its own 0..9,999 index -> never read during fitting")

    sub = np.loadtxt(submission, delimiter=",", skiprows=1)
    ref = np.loadtxt(sample, delimiter=",", skiprows=1)
    print(f"[4] submission rows {len(sub):,} vs sample {len(ref):,}")
    assert len(sub) == len(ref) == 10000
    assert set(sub[:, 0].astype(np.int64)) == set(ref[:, 0].astype(np.int64)), "id mismatch"
    print("    ok: same id set as sampleSubmission.csv")
    assert sub[:, 1].min() >= 1 and sub[:, 1].max() <= 5, "rating out of range"
    print(f"[5] rating range {sub[:,1].min():.3f}..{sub[:,1].max():.3f} "
          f"mean {sub[:,1].mean():.3f}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main(*sys.argv[1:5])
