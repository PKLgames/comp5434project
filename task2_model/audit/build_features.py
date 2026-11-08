"""Stream dataset2_metadata CSVs into sharded sparse feature caches.

Text  : hashed word 1-2 grams, l2-normalised (stateless -> no fit, no leakage).
Dense : review-level numerics only (NO identity features by design, see §7.2).

    python -m task2_model.audit.build_features --splits pool,valid \
        --out-dir /tmp/abl --hash-comment 17 --hash-title 15 --shard-rows 200000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLS = ["id", "title", "comment", "time", "votes", "purchased", "rating"]
DENSE_COLS = [
    "votes_log", "votes_missing", "purchased", "review_year", "review_month",
    "log_comment_char", "log_title_char", "comment_missing", "title_missing",
]


def split_fn(splits: str):
    """Return {name: predicate} over int64 id arrays."""
    table = {}
    for name in splits.split(","):
        name = name.strip()
        if name == "pool":
            table[name] = lambda i: (i % 8 == 0) & (i % 40 != 0)
        elif name == "valid":
            table[name] = lambda i: i % 40 == 0
        elif name == "train_full":          # everything except the holdout
            table[name] = lambda i: i % 40 != 0
        elif name == "all":
            table[name] = lambda i: np.ones(len(i), dtype=bool)
        else:
            raise SystemExit(f"unknown split {name}")
    return table


class ShardWriter:
    def __init__(self, out_dir: str, shard_rows: int):
        self.out_dir = out_dir
        self.shard_rows = shard_rows
        os.makedirs(out_dir, exist_ok=True)
        self.buf = {k: [] for k in
                    ["c_indptr", "c_indices", "c_data", "t_indptr", "t_indices",
                     "t_data", "dense", "y", "ids"]}
        self.n_rows = 0
        self.n_shards = 0

    def _emit(self):
        if not self.buf["ids"]:
            return
        out = {}
        dense = np.concatenate(self.buf["dense"], axis=0)
        y = np.concatenate(self.buf["y"], axis=0)
        ids = np.concatenate(self.buf["ids"], axis=0)
        rows = ids.shape[0]
        for pref, ind_key, idx_key, dat_key in (("c", "c_indptr", "c_indices", "c_data"),
                                                ("t", "t_indptr", "t_indices", "t_data")):
            mats = self.buf[ind_key]
            total = 0
            indptr_parts = []
            for m in mats:
                indptr_parts.append((m[:-1] + total).astype(np.int32))
                total += m[-1]
            indptr = np.concatenate(indptr_parts + [np.array([total], np.int32)])
            out[pref + "_indptr"] = indptr
            out[pref + "_indices"] = np.concatenate(
                [a for a in self.buf[idx_key]]).astype(np.int32)
            out[pref + "_data"] = np.concatenate(
                [a for a in self.buf[dat_key]]).astype(np.float32)
        out["dense"] = dense.astype(np.float32)
        out["y"] = y.astype(np.int8)
        out["ids"] = ids.astype(np.int64)
        path = os.path.join(self.out_dir, f"shard_{self.n_shards:05d}.npz")
        np.savez(path, **out)
        self.n_shards += 1
        self.n_rows += rows
        self.buf = {k: [] for k in self.buf}

    def add(self, tables: dict):
        for k, v in tables.items():
            self.buf[k].append(v)
        cur = sum(a.shape[0] for a in self.buf["ids"])
        if cur >= self.shard_rows:
            self._emit()

    def close(self):
        self._emit()
        return self.n_rows, self.n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "Dataset2_metadata", "train.csv"))
    ap.add_argument("--splits", default="pool,valid")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hash-comment", type=int, default=17)
    ap.add_argument("--hash-title", type=int, default=15)
    ap.add_argument("--shard-rows", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    hv_c = HashingVectorizer(n_features=2 ** args.hash_comment, ngram_range=(1, 2),
                             alternate_sign=False, norm="l2", lowercase=True)
    hv_t = HashingVectorizer(n_features=2 ** args.hash_title, ngram_range=(1, 2),
                             alternate_sign=False, norm="l2", lowercase=True)
    preds = split_fn(args.splits)
    writers = {name: ShardWriter(os.path.join(args.out_dir, name), args.shard_rows)
               for name in preds}

    header = open(args.csv, "r", encoding="utf-8", errors="ignore").readline().rstrip("\n").split(",")
    has_label = "rating" in header
    cols = list(COLS) if has_label else [c for c in COLS if c != "rating"]
    print(f"[cfg] {args.csv}: has_label={has_label}", flush=True)
    reader = pcsv.open_csv(
        args.csv,
        read_options=pcsv.ReadOptions(block_size=64 << 20),
        parse_options=pcsv.ParseOptions(newlines_in_values=True),
        convert_options=pcsv.ConvertOptions(include_columns=cols),
    )
    t0 = time.time()
    seen = 0
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        ids = batch.column("id").to_numpy(zero_copy_only=False).astype(np.int64)
        seen += ids.shape[0]
        for name, fn in preds.items():
            mask = fn(ids)
            if not mask.any():
                continue
            sub = batch.filter(pa.array(mask))
            sid = sub.column("id").to_numpy(zero_copy_only=False).astype(np.int64)
            comments = sub.column("comment").to_pylist()
            titles = sub.column("title").to_pylist()
            comments = [c if c is not None else "" for c in comments]
            titles = [t if t is not None else "" for t in titles]
            C = hv_c.transform(comments).astype(np.float32).tocsr()
            T = hv_t.transform(titles).astype(np.float32).tocsr()
            ts = sub.column("time").to_numpy(zero_copy_only=False).astype(np.float64)
            votes = sub.column("votes").to_numpy(zero_copy_only=False).astype(np.float32)
            pur = sub.column("purchased").to_numpy(zero_copy_only=False).astype(np.float32)
            if has_label:
                rating = sub.column("rating").to_numpy(zero_copy_only=False).astype(np.int8)
            else:
                rating = np.full(sub.num_rows, -1, np.int8)
            dt = (ts / 1000.0).astype(np.int64).astype("datetime64[s]")
            yr = (dt.astype("datetime64[Y]").astype(np.int64) + 1970).astype(np.float32)
            month = (dt.astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.float32)
            lc = np.array([len(c) for c in comments], np.float32)
            lt = np.array([len(t) for t in titles], np.float32)
            dense = np.column_stack([
                np.log1p(np.maximum(votes, 0)), (votes < 0).astype(np.float32), pur,
                yr, month, np.log1p(lc), np.log1p(lt),
                (lc == 0).astype(np.float32), (lt == 0).astype(np.float32),
            ])
            writers[name].add({
                "c_indptr": C.indptr.astype(np.int64), "c_indices": C.indices,
                "c_data": C.data, "t_indptr": T.indptr.astype(np.int64),
                "t_indices": T.indices, "t_data": T.data,
                "dense": dense, "y": rating, "ids": sid,
            })
        if args.limit and seen >= args.limit:
            break
        if seen % 2_000_000 < 65_000:
            print(f"[{time.time()-t0:6.0f}s] rows {seen:,}", flush=True)

    for name, w in writers.items():
        n, s = w.close()
        print(f"[done] {name}: {n:,} rows, {s} shards -> {w.out_dir}", flush=True)


if __name__ == "__main__":
    main()
