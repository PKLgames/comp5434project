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

from common import DENSE_COLS, load_json


class ShardBatcher:
    """Iterate mini-batches of one split stored as ``.npz`` CSR shards."""

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

        shards = split_info["shards"]
        if max_shards:
            shards = shards[:max_shards]
        self.shards = shards
        self.shard_dir = os.path.join(artifacts, split)

        total = sum(s["rows"] for s in shards)
        if max_rows:
            total = min(total, max_rows)
        self.num_rows = total

        # Dense column selection and train-fitted standardisation parameters.
        self.dense_index = np.array([DENSE_COLS.index(c) for c in dense_columns], dtype=np.int64)
        stats = meta["dense_stats"]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
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
