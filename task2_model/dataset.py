"""Shard-based batching for the compact CSR caches.

The full training set has 6.1M rows but only ~160M non-zeros, so one shard
(~100k rows, ~2.5M non-zeros) fits comfortably in memory.  We therefore load
one shard at a time, optionally permute its rows, and emit contiguous
mini-batches.  This keeps RAM usage flat and avoids any dependency on
``torch.utils.data.DataLoader`` worker processes.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix

from common import DENSE_COLS, dense_schema, load_json
from features_v2 import build_matrix


class ShardBatcher:
    """Iterate mini-batches of one split stored as ``.npz`` CSR shards.

    A V2 cache is an *overlay*: ``<artifacts>/meta.json`` describes the extended
    schema and points at a read-only V1 cache through ``base_dir``.  The shards
    stay V1 (26 dense columns); the V2 columns are recomputed from the CSR arrays
    of each shard on the fly, so no second 2.3 GB cache is ever written.
    """

    def __init__(
        self,
        artifacts: str,
        split: str,
        dense_columns: List[str],
        batch_size: int = 4096,
        shuffle: bool = False,
        seed: int = 42,
        max_shards: Optional[int] = None,
        max_rows: Optional[int] = None,
        pin_memory: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory

        meta = load_json(os.path.join(artifacts, "meta.json"))
        split_info = meta["splits"][split]
        self.vocab_title = int(meta["vocab"]["title_tfidf"])
        self.vocab_comment = int(meta["vocab"]["comment_tfidf"])

        # --- V1 / V2 overlay resolution --------------------------------
        # ``meta["dense_columns"]`` is authoritative (V1 caches already contain
        # exactly the same list as ``common.DENSE_COLS``), which lets an overlay
        # describe itself instead of relying on an import.
        self.schema = [str(name) for name in meta["dense_columns"]]
        v2_info = meta.get("v2") or {}
        self.feature_set = str(meta.get("feature_set", v2_info.get("feature_set", "v1")))
        # Whether the cache was built *with* the interaction block.  This is a
        # property of the cache (its statistics cover those columns), not of the
        # current run: a run may select a subset of the schema, e.g. the 33
        # text-shape columns out of a 37-column overlay.
        self.schema_interactions = bool(
            meta.get("interactions", v2_info.get("interactions", False)))
        expected = dense_schema(self.feature_set, self.schema_interactions)
        if self.schema != expected:
            raise ValueError(
                f"meta.json declares {len(self.schema)} dense columns but "
                f"feature_set={self.feature_set!r} "
                f"(interactions={self.schema_interactions}) has {len(expected)}; "
                f"rebuild the cache with preprocess.py"
            )

        base_dir = meta.get("base_dir")
        if base_dir:
            self.base_dir = (base_dir if os.path.isabs(base_dir)
                             else os.path.normpath(os.path.join(artifacts, base_dir)))
        else:
            self.base_dir = artifacts
        self.is_v2 = self.feature_set == "v2"
        # Wide of the raw ``dense`` array stored in the underlying shards.
        self.base_width = len(meta.get("base_dense_columns") or DENSE_COLS)

        if self.is_v2:
            self.t_map = np.load(os.path.join(self.base_dir, "vocab_title.npy"))
            self.c_map = np.load(os.path.join(self.base_dir, "vocab_comment.npy"))
        else:
            self.t_map = self.c_map = None

        shards = split_info["shards"]
        if max_shards:
            shards = shards[:max_shards]
        self.shards = shards
        self.shard_dir = os.path.join(self.base_dir, split)

        total = sum(s["rows"] for s in shards)
        if max_rows:
            total = min(total, max_rows)
        self.num_rows = total

        # Dense column selection and train-fitted standardisation parameters.
        # The statistics are fitted on the *whole* schema, so the index array is
        # resolved against it and must agree in length (guards the classic bug of
        # standardising V2 columns with V1 statistics).
        self.dense_index_names = [str(c) for c in dense_columns]
        missing = [c for c in self.dense_index_names if c not in self.schema]
        if missing:
            raise ValueError(
                f"{artifacts}/meta.json does not provide the dense column(s) "
                f"{missing}; build a V2 cache with preprocess.py --feature-set v2")
        self.dense_index = np.array([self.schema.index(c) for c in self.dense_index_names],
                                    dtype=np.int64)
        stats = meta["dense_stats"]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.size != len(self.schema) or std.size != len(self.schema):
            raise ValueError(
                f"dense_stats has {mean.size} mean / {std.size} std values but the "
                f"schema has {len(self.schema)} columns; rebuild the cache"
            )
        self.dense_mean = mean[self.dense_index] if self.dense_index.size else mean[:0]
        self.dense_std = std[self.dense_index] if self.dense_index.size else std[:0]
        self.has_dense = bool(self.dense_index.size)

    def __len__(self) -> int:
        return (self.num_rows + self.batch_size - 1) // self.batch_size

    def steps_per_epoch(self) -> int:
        """Exact number of batches emitted by :meth:`batches`."""

        total = 0
        remaining = self.num_rows
        for shard in self.shards:
            if remaining <= 0:
                break
            rows = min(shard["rows"], remaining)
            total += (rows + self.batch_size - 1) // self.batch_size
            remaining -= rows
        return total

    # ------------------------------------------------------------------
    def _load_shard(self, shard: dict, rng: np.random.RandomState):
        path = os.path.join(self.shard_dir, shard["file"])
        with np.load(path) as npz:
            t_indptr = npz["t_indptr"]
            t_indices = npz["t_indices"]
            t_data = npz["t_data"]
            c_indptr = npz["c_indptr"]
            c_indices = npz["c_indices"]
            c_data = npz["c_data"]
            dense = npz["dense"]
            y = npz["y"] if "y" in npz.files else None

        n = dense.shape[0]
        if self.shuffle:
            perm = rng.permutation(n)
            t_csr = csr_matrix((t_data, t_indices, t_indptr),
                               shape=(n, self.vocab_title))
            c_csr = csr_matrix((c_data, c_indices, c_indptr),
                               shape=(n, self.vocab_comment))
            t_csr = t_csr[perm]
            c_csr = c_csr[perm]
            t_indptr, t_indices, t_data = t_csr.indptr, t_csr.indices, t_csr.data
            c_indptr, c_indices, c_data = c_csr.indptr, c_csr.indices, c_csr.data
            dense = dense[perm]
            if y is not None:
                y = y[perm]

        if self.has_dense:
            if self.is_v2:
                # Zero-copy overlay: recompute the V2 columns from this shard's
                # CSR arrays and append them *before* standardisation, so one
                # mean/std vector covers the whole schema.
                dense = build_matrix(
                    dense, t_indptr, t_indices, t_data,
                    c_indptr, c_indices, c_data,
                    self.t_map, self.c_map, self.schema_interactions)
                if dense.shape[1] != len(self.schema):
                    raise ValueError(
                        f"V2 overlay produced {dense.shape[1]} dense columns but "
                        f"meta.json declares {len(self.schema)}")
            elif dense.shape[1] != self.base_width:
                raise ValueError(
                    f"shard holds {dense.shape[1]} dense columns, expected "
                    f"{self.base_width}")
            dense = dense[:, self.dense_index]
            dense = (dense - self.dense_mean) / self.dense_std
            dense = np.clip(dense, -5.0, 5.0).astype(np.float32, copy=False)
        else:
            dense = np.zeros((n, 0), dtype=np.float32)
        return (n, t_indptr, t_indices, t_data,
                c_indptr, c_indices, c_data, dense, y)

    def batches(self, epoch: int = 0) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield batches.  Set ``epoch`` to vary the shuffling order."""

        rng = np.random.RandomState(self.seed + epoch)
        order = np.arange(len(self.shards))
        if self.shuffle:
            rng.shuffle(order)

        emitted = 0
        for shard_idx in order:
            shard = self.shards[shard_idx]
            (n, t_indptr, t_indices, t_data,
             c_indptr, c_indices, c_data, dense, y) = self._load_shard(shard, rng)

            start = 0
            while start < n and emitted < self.num_rows:
                end = min(start + self.batch_size, n,
                          start + (self.num_rows - emitted))
                yield self._make_batch(
                    t_indptr, t_indices, t_data,
                    c_indptr, c_indices, c_data,
                    dense, y, start, end)
                emitted += end - start
                start = end
            if emitted >= self.num_rows:
                return

    # ------------------------------------------------------------------
    def _make_batch(self, t_indptr, t_indices, t_data,
                    c_indptr, c_indices, c_data,
                    dense, y, start: int, end: int) -> Dict[str, torch.Tensor]:
        """Slice contiguous rows [start, end) without materialising a matrix."""

        batch = {}
        t_lo, t_hi = int(t_indptr[start]), int(t_indptr[end])
        t_off = t_indptr[start:end + 1].astype(np.int64)
        t_off -= t_off[0]
        batch["t_idx"] = torch.from_numpy(t_indices[t_lo:t_hi].astype(np.int64))
        batch["t_off"] = torch.from_numpy(t_off)
        batch["t_val"] = torch.from_numpy(t_data[t_lo:t_hi])

        c_lo, c_hi = int(c_indptr[start]), int(c_indptr[end])
        c_off = c_indptr[start:end + 1].astype(np.int64)
        c_off -= c_off[0]
        batch["c_idx"] = torch.from_numpy(c_indices[c_lo:c_hi].astype(np.int64))
        batch["c_off"] = torch.from_numpy(c_off)
        batch["c_val"] = torch.from_numpy(c_data[c_lo:c_hi])

        batch["dense"] = torch.from_numpy(dense[start:end])
        if y is not None:
            batch["y"] = torch.from_numpy(y[start:end].astype(np.float32))
        if self.pin_memory and torch.cuda.is_available():
            for key, value in batch.items():
                batch[key] = value.pin_memory()
        return batch
