"""End-to-end smoke test of the V2 overlay.

Builds a one-shard overlay from the real V1 cache, checks the schema it produces,
runs a single optimiser step and asserts the loss is finite with
``dense.shape[1] == 33`` (the 26 V1 columns plus the 7 text-shape columns).
The heavier CLI paths are exercised with row limits so the whole file runs in
well under a minute.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "task2_model"
sys.path.insert(0, str(MODEL_DIR))

from common import DENSE_COLS, dense_schema, select_dense_columns  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
PYTHON = sys.executable

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "meta.json").exists(),
    reason="artifacts/ cache is not present (it is gitignored, ~2.4 GB)")


@pytest.fixture(scope="module")
def overlay(tmp_path_factory) -> pathlib.Path:
    """Build a 33-column overlay from the first training shard only."""

    out = tmp_path_factory.mktemp("artifacts_v2_smoke")
    cmd = [PYTHON, str(MODEL_DIR / "preprocess.py"),
           "--feature-set", "v2", "--reuse", str(ARTIFACTS),
           "--out", str(out), "--limit-train", "1", "--no-interactions"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return out


def test_overlay_schema_is_26_plus_7(overlay):
    meta = json.loads((overlay / "meta.json").read_text())
    assert meta["version"] == 2
    assert meta["feature_set"] == "v2"
    assert meta["interactions"] is False
    assert meta["dense_columns"] == dense_schema("v2", False)
    assert len(meta["dense_columns"]) == 33
    assert meta["dense_columns"][:26] == DENSE_COLS
    assert (overlay / "stats.json").exists()
    assert meta["v2"]["columns"] == meta["dense_columns"][26:]
    assert len(meta["v2"]["base_fingerprint"]) == 64


def test_overlay_is_tiny(overlay):
    """The whole point of the overlay: a few KB, not a second 2.3 GB cache."""

    total = sum(p.stat().st_size for p in overlay.rglob("*") if p.is_file())
    assert total < 256 * 1024, f"overlay is {total / 1024:.0f} KB"
    assert not (overlay / "train").exists(), "the overlay must not copy shards"


def test_batcher_yields_33_columns(overlay):
    from dataset import ShardBatcher

    columns = select_dense_columns("all", "v2", False)
    assert len(columns) == 33
    loader = ShardBatcher(str(overlay), "train", columns,
                          batch_size=512, shuffle=False, max_shards=1)
    assert loader.is_v2
    batch = next(loader.batches(epoch=0))
    dense = batch["dense"].numpy()
    assert dense.shape == (512, 33)
    assert np.isfinite(dense).all()


def test_shuffled_batches_keep_the_v2_block_aligned(overlay):
    """The V2 columns are computed after row permutation, so they must follow it.

    Both per-row standardisation and the V2 blocks are functions of a single row,
    so a shuffled batch must contain exactly the same row tensors as the
    unshuffled scan - if the block were computed before the permutation, or
    aligned to the wrong rows, the bytes would differ.
    """

    from dataset import ShardBatcher

    columns = select_dense_columns("all", "v2", False)
    plain = ShardBatcher(str(overlay), "train", columns, batch_size=4096,
                         shuffle=False, max_shards=1)
    shuffled = ShardBatcher(str(overlay), "train", columns, batch_size=4096,
                            shuffle=True, seed=7, max_shards=1)

    plain_rows = set()
    for batch in plain.batches(epoch=0):
        for row in batch["dense"].numpy():
            plain_rows.add(row.tobytes())
    assert len(plain_rows) > 0

    a = next(shuffled.batches(epoch=0))["dense"].numpy()
    b = next(shuffled.batches(epoch=0))["dense"].numpy()
    assert np.array_equal(a, b), "shuffling is not reproducible at a fixed seed"

    unmatched = [row for row in a if row.tobytes() not in plain_rows]
    assert not unmatched, (
        f"{len(unmatched)}/{a.shape[0]} shuffled rows do not exist in the "
        "unshuffled scan: the V2 block is misaligned with its rows")


def test_one_training_step(overlay):
    """One forward/backward/step on a 33-column batch, with a finite loss."""

    import torch
    import torch.nn.functional as F

    from dataset import ShardBatcher
    from model import HybridRatingModel

    torch.manual_seed(0)
    columns = select_dense_columns("all", "v2", False)
    loader = ShardBatcher(str(overlay), "train", columns,
                          batch_size=1024, shuffle=False, max_shards=1)
    batch = next(loader.batches(epoch=0))
    y = batch.pop("y")

    model = HybridRatingModel(
        vocab_title=loader.vocab_title, vocab_comment=loader.vocab_comment,
        n_dense=len(columns), text_dim=32, dense_dim=16, hidden_dim=32,
        dropout=0.0, emb_dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pred = model(batch["t_idx"], batch["t_off"], batch["t_val"],
                 batch["c_idx"], batch["c_off"], batch["c_val"],
                 batch["dense"])
    loss = F.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item(), f"loss is {loss.item()}"
    assert pred.shape == y.shape


def test_train_cli_writes_a_v2_checkpoint(overlay, tmp_path):
    """``train.py`` must record the dense schema in the checkpoint."""

    import torch

    out = tmp_path / "run"
    cmd = [PYTHON, str(MODEL_DIR / "train.py"),
           "--data", str(overlay), "--out", str(out), "--epochs", "1",
           "--batch-size", "1024", "--valid-batch-size", "1024",
           "--limit-train-rows", "2048", "--limit-valid-rows", "2048",
           "--text-dim", "32", "--dense-dim", "16", "--hidden-dim", "32",
           "--num-threads", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]

    summary = json.loads((out / "summary.json").read_text())
    assert np.isfinite(summary["best_val_rmse"])
    assert summary["config"]["feature_set"] == "v2"
    assert summary["config"]["interactions"] is False

    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert len(checkpoint["dense_columns"]) == 33
    assert checkpoint["dense_columns"] == dense_schema("v2", False)
    assert checkpoint["dense_schema"]["feature_set"] == "v2"
    assert checkpoint["dense_schema"]["interactions"] is False
    assert checkpoint["dense_schema"]["columns"] == checkpoint["dense_columns"]
