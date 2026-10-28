#!/usr/bin/env python
"""Convert the Part-A Spark parquet shards into compact CSR shards.

Why this step exists
--------------------
The handover files store two Spark ``SparseVector`` columns with a hashed
dimension of 65,536 (title) and 262,144 (comment).  In practice only the
indices that occurred in the training set matter, so we compact both hashed
spaces into contiguous vocabularies (typically ~13k title / ~35k comment
features).  That shrinks the embedding tables used by the neural model by
roughly 6x and removes the need to materialise a 327,680-column matrix.

Output layout
-------------
``<out>/meta.json``            feature names, vocab sizes, dense statistics
``<out>/vocab_title.npy``      sorted original hashed indices (train only)
``<out>/vocab_comment.npy``
``<out>/<split>/shard_XXXXX.npz``
    t_indptr, t_indices, t_data   title CSR (remapped indices, float32)
    c_indptr, c_indices, c_data   comment CSR
    dense                          float32 [rows, 26]
    y                              int8 rating (omitted for test)
``<out>/<split>/manifest.json``    shard list + row counts

Usage
-----
# Build train/valid caches (run once, ~2-3 minutes for the full data)
python preprocess.py --train-dir ../Task2数据交接/output/train_features_with_entity \
                     --valid-dir ../Task2数据交接/output/validation_features_with_entity \
                     --out ../artifacts

# Later, when Part A delivers the test features, reuse the saved vocabulary:
python preprocess.py --test-dir <test_features_dir> --reuse ../artifacts --out ../artifacts
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from common import (
    DENSE_COLS,
    FEATURE_GROUPS,
    TARGET_COL,
    build_dense_matrix,
    load_json,
    save_json,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def list_parquet(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet part files under {folder}")
    return files


def read_sparse_column(table, column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(indptr, indices, data)`` for a Spark SparseVector column.

    Spark serialises a sparse vector as a struct of ``type/size/indices/values``
    where ``indices`` and ``values`` are Arrow list columns.  Flattening the
    lists lets us rebuild the CSR arrays without a Python loop over rows.
    """

    struct = table.column(column).combine_chunks()
    indices_list = struct.field("indices")
    values_list = struct.field("values")

    indptr = indices_list.offsets.to_numpy().astype(np.int64, copy=False)
    indices = indices_list.values.to_numpy().astype(np.int32, copy=False)
    data = values_list.values.to_numpy().astype(np.float32, copy=False)

    if indptr.size != table.num_rows + 1 or indptr[-1] != indices.size:
        raise ValueError(f"malformed sparse column {column}")
    return indptr, indices, data


def remap_to_vocab(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    vocab: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map original hashed indices to contiguous ids.

    Indices absent from the training vocabulary (possible for validation and
    test rows) are mapped to slot 0 with weight 0.  A zero ``per_sample_weight``
    makes ``EmbeddingBag`` ignore the token, which is equivalent to dropping it.
    """

    if indices.size == 0:
        return indptr, indices, data

    pos = np.searchsorted(vocab, indices)
    clipped = np.minimum(pos, vocab.size - 1)
    valid = vocab[clipped] == indices
    new_indices = clipped.astype(np.int32, copy=False)
    new_data = np.where(valid, data, np.float32(0.0)).astype(np.float32, copy=False)
    return indptr, new_indices, new_data


def scan_vocabulary(files: List[str], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Collect the union of hashed indices used by the training shards."""

    chunks: Dict[str, List[np.ndarray]] = {"title_tfidf": [], "comment_tfidf": []}
    started = time.time()
    for i, path in enumerate(files):
        table = pq.read_table(path, columns=["title_tfidf", "comment_tfidf"])
        for column in chunks:
            _, indices, _ = read_sparse_column(table, column)
            chunks[column].append(np.unique(indices))
        if verbose and (i + 1) % 16 == 0:
            print(f"  vocab scan {i + 1}/{len(files)} shards "
                  f"({time.time() - started:.1f}s)", flush=True)

    vocab = {}
    for column, parts in chunks.items():
        merged = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int32)
        vocab[column] = merged.astype(np.int32)
        if verbose:
            print(f"  {column}: {merged.size} distinct hashed indices", flush=True)
    return vocab


# ---------------------------------------------------------------------------
# Shard conversion
# ---------------------------------------------------------------------------
def convert_shard(
    path: str,
    out_path: str,
    vocab: Dict[str, np.ndarray],
    with_target: bool,
) -> int:
    """Convert one parquet part file into one ``.npz`` cache shard."""

    # Reading every column is simpler than maintaining two column lists and
    # still costs <0.5 s per 100k-row shard.
    table = pq.read_table(path)
    n_rows = table.num_rows

    t_indptr, t_indices, t_data = read_sparse_column(table, "title_tfidf")
    c_indptr, c_indices, c_data = read_sparse_column(table, "comment_tfidf")

    t_indptr, t_indices, t_data = remap_to_vocab(
        t_indptr, t_indices, t_data, vocab["title_tfidf"])
    c_indptr, c_indices, c_data = remap_to_vocab(
        c_indptr, c_indices, c_data, vocab["comment_tfidf"])

    dense = build_dense_matrix(table)

    payload = dict(
        t_indptr=t_indptr.astype(np.int32, copy=False),
        t_indices=t_indices,
        t_data=t_data,
        c_indptr=c_indptr.astype(np.int32, copy=False),
        c_indices=c_indices,
        c_data=c_data,
        dense=dense,
    )
    if with_target:
        payload["y"] = table.column(TARGET_COL).combine_chunks().to_numpy().astype(np.int8)

    np.savez(out_path, **payload)
    return n_rows


def process_split(
    files: List[str],
    out_dir: str,
    vocab: Dict[str, np.ndarray],
    split_name: str,
    with_target: bool,
    dense_stats: Optional[dict] = None,
) -> Tuple[dict, Optional[dict]]:
    """Convert all shards of one split and optionally accumulate dense stats."""

    os.makedirs(out_dir, exist_ok=True)
    n_cols = len(DENSE_COLS)
    total = np.zeros(n_cols, np.float64)
    total_sq = np.zeros(n_cols, np.float64)
    n_rows = 0
    manifest = {"split": split_name, "rows": 0, "shards": []}
    started = time.time()

    for i, path in enumerate(files):
        shard_path = os.path.join(out_dir, f"shard_{i:05d}.npz")
        rows = convert_shard(path, shard_path, vocab, with_target)
        manifest["shards"].append({"file": os.path.basename(shard_path), "rows": rows})
        manifest["rows"] += rows

        if dense_stats is None and with_target:
            arr = np.load(shard_path)["dense"]
            total += arr.sum(axis=0, dtype=np.float64)
            total_sq += np.square(arr, dtype=np.float64).sum(axis=0)
            n_rows += rows

        if (i + 1) % 16 == 0 or i + 1 == len(files):
            print(f"  {split_name}: {i + 1}/{len(files)} shards, "
                  f"{manifest['rows']:,} rows, {time.time() - started:.1f}s",
                  flush=True)

    save_json(manifest, os.path.join(out_dir, "manifest.json"))

    if dense_stats is None and with_target and n_rows:
        mean = total / n_rows
        var = np.maximum(total_sq / n_rows - mean * mean, 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-6, 1.0, std)
        dense_stats = {
            "rows": n_rows,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return manifest, dense_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", help="folder with train part-*.parquet")
    parser.add_argument("--valid-dir", help="folder with validation part-*.parquet")
    parser.add_argument("--test-dir", help="folder with test part-*.parquet")
    parser.add_argument("--out", required=True, help="artifact/cache directory")
    parser.add_argument("--reuse", help="existing artifact dir whose vocab/meta to reuse")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="debug: only convert the first N train shards")
    parser.add_argument("--limit-valid", type=int, default=None,
                        help="debug: only convert the first N validation shards")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------
    # Test-only mode: reuse an existing vocabulary and statistics.
    # ------------------------------------------------------------------
    if args.test_dir:
        reuse = os.path.abspath(args.reuse or out)
        meta = load_json(os.path.join(reuse, "meta.json"))
        vocab = {
            "title_tfidf": np.load(os.path.join(reuse, "vocab_title.npy")),
            "comment_tfidf": np.load(os.path.join(reuse, "vocab_comment.npy")),
        }
        files = list_parquet(args.test_dir)
        print(f"converting {len(files)} test shards with the saved vocabulary")
        manifest, _ = process_split(
            files, os.path.join(out, "test"), vocab, "test", with_target=False)
        meta.setdefault("splits", {})["test"] = {
            "rows": manifest["rows"],
            "shards": manifest["shards"],
        }
        save_json(meta, os.path.join(out, "meta.json"))
        print(f"done: {manifest['rows']:,} test rows")
        return

    if not args.train_dir or not args.valid_dir:
        parser.error("--train-dir and --valid-dir are required unless --test-dir is used")

    train_files = list_parquet(args.train_dir)
    valid_files = list_parquet(args.valid_dir)
    if args.limit_train:
        train_files = train_files[: args.limit_train]
    if args.limit_valid:
        valid_files = valid_files[: args.limit_valid]

    print(f"train shards: {len(train_files)}, validation shards: {len(valid_files)}")

    print("[1/3] scanning training vocabulary ...")
    vocab = scan_vocabulary(train_files)
    np.save(os.path.join(out, "vocab_title.npy"), vocab["title_tfidf"])
    np.save(os.path.join(out, "vocab_comment.npy"), vocab["comment_tfidf"])

    print("[2/3] converting training shards ...")
    train_manifest, dense_stats = process_split(
        train_files, os.path.join(out, "train"), vocab, "train", with_target=True)

    print("[3/3] converting validation shards ...")
    valid_manifest, _ = process_split(
        valid_files, os.path.join(out, "valid"), vocab, "valid",
        with_target=True, dense_stats=dense_stats)

    meta = {
        "version": 1,
        "source": {"train": os.path.abspath(args.train_dir),
                   "valid": os.path.abspath(args.valid_dir)},
        "dense_columns": DENSE_COLS,
        "feature_groups": FEATURE_GROUPS,
        "target": TARGET_COL,
        "vocab": {k: int(v.size) for k, v in vocab.items()},
        "splits": {"train": train_manifest, "valid": valid_manifest},
        "dense_stats": dense_stats,
    }

    # Rating distribution is useful for sanity checks and for the constant
    # baseline (predict the training mean).
    ratings = np.zeros(1, dtype=np.int64)
    y_all = []
    for item in train_manifest["shards"]:
        y = np.load(os.path.join(out, "train", item["file"]))["y"]
        y_all.append(y)
    y_all = np.concatenate(y_all)
    counts = np.bincount(y_all.astype(np.int64), minlength=6)[1:6]
    meta["train_rating_distribution"] = counts.tolist()
    meta["train_rating_mean"] = float(y_all.mean())
    save_json(meta, os.path.join(out, "meta.json"))

    print(f"done. train={train_manifest['rows']:,} valid={valid_manifest['rows']:,}")
    print(f"vocab title={meta['vocab']['title_tfidf']:,} "
          f"comment={meta['vocab']['comment_tfidf']:,}")


if __name__ == "__main__":
    main()
