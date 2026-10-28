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
python preprocess.py --train-dir ../Task2_Token_Dataset/tokenized/train_features_with_entity \
                     --valid-dir ../Task2_Token_Dataset/tokenized/validation_features_with_entity \
                     --out ../artifacts

# Build the V2 overlay on top of an existing (read-only) V1 cache.  This writes
# only meta.json + stats.json (a few KB); the 2.3 GB of shards is reused as is.
python preprocess.py --feature-set v2 --reuse ../artifacts --out ../artifacts_v2

# Later, when Part A delivers the test features, reuse the saved vocabulary:
python preprocess.py --test-dir <test_features_dir> --reuse ../artifacts --out ../artifacts
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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
    dense_schema,
    load_json,
    save_json,
)
from features_v2 import build_matrix


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
# V2 overlay
# ---------------------------------------------------------------------------
def base_fingerprint(base: str) -> str:
    """Cheap, stable fingerprint of the underlying V1 cache.

    Hashing the 2.3 GB of shards on every build would be wasteful and the
    overlay is reproducible anyway, so this digests the artifacts that pin the
    data down: the V1 meta (column list + row counts + dense statistics), both
    manifests (shard names and per-shard row counts) and the two vocabularies.
    If Part A ever delivers genuinely different features, this changes.
    """

    digest = hashlib.sha256()
    parts = [
        os.path.join(base, "meta.json"),
        os.path.join(base, "train", "manifest.json"),
        os.path.join(base, "valid", "manifest.json"),
        os.path.join(base, "vocab_title.npy"),
        os.path.join(base, "vocab_comment.npy"),
    ]
    for path in parts:
        digest.update(os.path.basename(path).encode("utf-8"))
        if not os.path.exists(path):
            digest.update(b"<missing>")
            continue
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_v2_overlay(args: argparse.Namespace) -> None:
    """Write ``meta.json`` + ``stats.json`` for a V2 overlay over a V1 cache."""

    if not args.reuse:
        raise SystemExit(
            "--feature-set v2 requires --reuse <v1 artifact dir>: the overlay "
            "reads the existing shards and never writes them")
    base = os.path.abspath(args.reuse)
    out = os.path.abspath(args.out)
    if os.path.normpath(base) == os.path.normpath(out):
        raise SystemExit(
            "--out must differ from --reuse: V2 never modifies the V1 cache")

    base_meta = load_json(os.path.join(base, "meta.json"))
    if list(base_meta["dense_columns"]) != list(DENSE_COLS):
        raise SystemExit(
            f"{base}/meta.json does not describe a V1 (26-column) cache; "
            "point --reuse at the original artifacts directory")

    interactions = bool(args.interactions)
    columns = dense_schema("v2", interactions)
    v2_columns = columns[len(DENSE_COLS):]
    shape_cols = [c for c in v2_columns if c in FEATURE_GROUPS["text_shape"]]
    inter_cols = [c for c in v2_columns if c in FEATURE_GROUPS["interactions"]]
    print(f"[v2] base={base}")
    print(f"[v2] out={out}  columns={len(columns)} "
          f"(26 v1 + {len(v2_columns)} v2, interactions={interactions})")

    t_map = np.load(os.path.join(base, "vocab_title.npy"))
    c_map = np.load(os.path.join(base, "vocab_comment.npy"))

    shards = list(base_meta["splits"]["train"]["shards"])
    if args.limit_train:
        shards = shards[: args.limit_train]

    n_cols = len(columns)
    total = np.zeros(n_cols, dtype=np.float64)
    total_sq = np.zeros(n_cols, dtype=np.float64)
    n_rows = 0
    started = time.time()
    print(f"[v2] fitting mean/std on {len(shards)} training shards ...", flush=True)

    for i, shard in enumerate(shards):
        with np.load(os.path.join(base, "train", shard["file"])) as npz:
            dense = npz["dense"]
            t_indptr, t_indices, t_data = npz["t_indptr"], npz["t_indices"], npz["t_data"]
            c_indptr, c_indices, c_data = npz["c_indptr"], npz["c_indices"], npz["c_data"]
        raw = build_matrix(dense, t_indptr, t_indices, t_data,
                           c_indptr, c_indices, c_data,
                           t_map, c_map, interactions)
        if raw.shape != (dense.shape[0], n_cols):
            raise SystemExit(f"unexpected V2 block shape {raw.shape} for {shard['file']}")

        # Identical accumulation to the V1 path, so the first 26 statistics come
        # out bit-for-bit equal to the ones already stored in the V1 meta.
        total += raw.sum(axis=0, dtype=np.float64)
        total_sq += np.square(raw, dtype=np.float64).sum(axis=0)
        n_rows += raw.shape[0]

        if (i + 1) % 8 == 0 or i + 1 == len(shards):
            print(f"  {i + 1}/{len(shards)} shards, {n_rows:,} rows, "
                  f"{time.time() - started:.1f}s", flush=True)

    if not n_rows:
        raise SystemExit("no training rows; nothing to fit")

    mean = total / n_rows
    var = np.maximum(total_sq / n_rows - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)

    # The first 26 statistics must reproduce the V1 ones exactly; a mismatch
    # would mean the overlay standardises V1 columns differently from the model
    # that produced the published numbers.
    base_mean = np.asarray(base_meta["dense_stats"]["mean"], dtype=np.float64)
    base_std = np.asarray(base_meta["dense_stats"]["std"], dtype=np.float64)
    drift_mean = float(np.max(np.abs(mean[: len(DENSE_COLS)] - base_mean))) if base_mean.size else 0.0
    drift_std = float(np.max(np.abs(std[: len(DENSE_COLS)] - base_std))) if base_std.size else 0.0
    print(f"[v2] v1-statistic drift: max|dmean|={drift_mean:.3e} "
          f"max|dstd|={drift_std:.3e}")

    stats = {"rows": int(n_rows), "mean": mean.tolist(), "std": std.tolist()}

    fingerprint = base_fingerprint(base)
    rel_base = os.path.relpath(base, out)

    meta = {
        "version": 2,
        "feature_set": "v2",
        "interactions": interactions,
        # Relative when possible so the overlay survives moving the repository.
        "base_dir": rel_base,
        "dense_columns": columns,
        "base_dense_columns": list(DENSE_COLS),
        "feature_groups": FEATURE_GROUPS,
        "target": TARGET_COL,
        "vocab": base_meta["vocab"],
        "splits": base_meta["splits"],
        "dense_stats": stats,
        "train_rating_distribution": base_meta.get("train_rating_distribution", []),
        "train_rating_mean": base_meta.get("train_rating_mean", 0.0),
        "source": {
            "base": base,
            "base_source": base_meta.get("source", {}),
            "note": "V2 overlay: shards are read from base_dir, only the schema "
                    "and standardisation statistics are new",
        },
        "v2": {
            "feature_set": "v2",
            "interactions": interactions,
            "columns": shape_cols,
            "text_shape_columns": shape_cols,
            "interaction_columns": inter_cols,
            "fit_split": "train",
            "base_fingerprint": fingerprint,
            "v1_stat_drift": {"max_abs_dmean": drift_mean, "max_abs_dstd": drift_std},
        },
    }

    os.makedirs(out, exist_ok=True)
    save_json(stats, os.path.join(out, "stats.json"))
    save_json(meta, os.path.join(out, "meta.json"))

    print("[v2] fitted statistics:")
    for name, mu, sd in zip(columns[len(DENSE_COLS):], mean[len(DENSE_COLS):], std[len(DENSE_COLS):]):
        print(f"    {name:<24} mean={mu:10.4f} std={sd:10.4f}")
    print(f"[v2] wrote {out}/meta.json and stats.json "
          f"({os.path.getsize(os.path.join(out, 'meta.json')) / 1024:.1f} KB)")


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
    parser.add_argument("--feature-set", default="v1", choices=["v1", "v2"],
                        help="v1 (default) writes the 26-column cache; v2 writes a "
                             "lightweight overlay over an existing --reuse cache")
    parser.add_argument("--interactions", dest="interactions", action="store_true",
                        default=True,
                        help="v2: also fit the 4 interaction columns (default: on, so "
                             "one overlay serves both the core and the +interactions "
                             "ablation; a run selects the subset it wants)")
    parser.add_argument("--no-interactions", dest="interactions", action="store_false",
                        help="v2: fit only the 7 text-shape columns")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------
    # V2 overlay: no parquet is read and no shard is written.
    # ------------------------------------------------------------------
    if args.feature_set == "v2":
        build_v2_overlay(args)
        return

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
