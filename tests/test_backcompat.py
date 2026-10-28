"""Guards that the V2 work did not disturb the published V1 model.

The plan's hard requirement (G1) is that the existing artifacts, checkpoints and
reported numbers stay valid.  These tests pin that down three ways:

1. the V1 column registry is unchanged, in content and in order;
2. a V1 batch built by the *current* code is bit-identical to the one the
   original (pre-V2) code produced, which is checksummed below;
3. the V2 code path never writes to ``artifacts/``.

Tests that need the real 2.4 GB cache or the 234 MB checkpoints are skipped when
those files are absent, so the file stays usable in a fresh clone.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "task2_model"
sys.path.insert(0, str(MODEL_DIR))

from common import DENSE_COLS, dense_schema  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_V2 = ROOT / "artifacts_v2"
FULL3 = ROOT / "runs" / "full3" / "best.pt"

# Fingerprint of the first validation batch (8192 rows, V1 = 26 columns) as
# produced by the pre-V2 code at commit 8e0b62a.  Standardisation is plain
# float32 NumPy arithmetic, so this is portable across NumPy versions.
V1_FIRST_BATCH_SHA256 = "26633e0c1474e8e0f4f3e44bd1d47f12dbf891c424dbafddc5f159f8f1e75c83"

needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "meta.json").exists(),
    reason="artifacts/ cache is not present (it is gitignored, ~2.4 GB)")
needs_checkpoint = pytest.mark.skipif(
    not FULL3.exists(), reason="runs/full3/best.pt is not present (gitignored)")
needs_v2 = pytest.mark.skipif(
    not (ARTIFACTS_V2 / "meta.json").exists(),
    reason="artifacts_v2/ overlay is not built")


# ---------------------------------------------------------------------------
# 1. the V1 registry is frozen
# ---------------------------------------------------------------------------
V1_COLUMNS = [
    "title_char_count", "comment_char_count", "title_word_count",
    "comment_word_count", "votes", "votes_log", "purchased", "review_year",
    "review_month", "comment_missing", "votes_invalid",
    "user_review_count", "user_review_count_log", "user_unseen",
    "product_review_count", "product_review_count_log", "product_unseen",
    "parent_product_review_count", "parent_product_review_count_log",
    "parent_product_unseen",
    "log_title_char", "log_comment_char", "title_words_per_char",
    "comment_words_per_char", "has_votes", "user_minus_product_log",
]


def test_dense_cols_is_still_the_v1_registry():
    assert isinstance(DENSE_COLS, list)
    assert len(DENSE_COLS) == 26
    assert DENSE_COLS == V1_COLUMNS
    assert dense_schema("v1") == V1_COLUMNS


def test_ablation_groups_are_unchanged():
    from common import FEATURE_GROUPS, select_dense_columns

    assert len(FEATURE_GROUPS["numeric"]) == 17
    assert len(FEATURE_GROUPS["entity"]) == 9
    # "all" must stay the 26 V1 columns, not the V2 schema
    assert select_dense_columns("all") == V1_COLUMNS
    assert select_dense_columns("all", "v1") == V1_COLUMNS
    assert select_dense_columns("none") == []


# ---------------------------------------------------------------------------
# 2. a V1 batch is bit-identical to the pre-V2 output
# ---------------------------------------------------------------------------
@needs_artifacts
def test_meta_json_still_describes_the_v1_schema():
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    assert meta["version"] == 1
    assert meta["dense_columns"] == V1_COLUMNS
    assert len(meta["dense_stats"]["mean"]) == 26
    assert "base_dir" not in meta
    assert "v2" not in meta


@needs_artifacts
def test_first_validation_batch_is_bit_identical():
    from dataset import ShardBatcher

    loader = ShardBatcher(str(ARTIFACTS), "valid", list(DENSE_COLS),
                          batch_size=8192, shuffle=False, max_shards=1)
    batch = next(loader.batches(epoch=0))
    dense = batch["dense"].numpy()

    assert dense.shape == (8192, 26)
    assert dense.dtype == np.float32
    digest = hashlib.sha256(np.ascontiguousarray(dense).tobytes()).hexdigest()
    assert digest == V1_FIRST_BATCH_SHA256, (
        "the V1 dense batch changed; the published 1.1543 is no longer "
        "reproducible with this code")


@needs_artifacts
@needs_checkpoint
def test_full3_checkpoint_still_loads_and_scores():
    """The existing checkpoint must load into the unchanged model and run."""

    import torch

    from dataset import ShardBatcher
    from predict import load_model

    device = torch.device("cpu")
    model = load_model(str(FULL3), device)
    loader = ShardBatcher(str(ARTIFACTS), "valid", list(DENSE_COLS),
                          batch_size=2048, shuffle=False, max_shards=1)
    batch = next(loader.batches(epoch=0))
    batch.pop("y")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        pred = model(batch["t_idx"], batch["t_off"], batch["t_val"],
                     batch["c_idx"], batch["c_off"], batch["c_val"],
                     batch["dense"])
    values = pred.numpy()
    assert values.shape == (2048,)
    assert np.isfinite(values).all()
    assert values.min() > 1.0 and values.max() < 5.0


# ---------------------------------------------------------------------------
# 3. the V2 path is read-only with respect to artifacts/
# ---------------------------------------------------------------------------
def _snapshot(folder: pathlib.Path) -> dict:
    state = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            stat = path.stat()
            state[str(path.relative_to(folder))] = (stat.st_size, stat.st_mtime_ns)
    return state


@needs_artifacts
def test_v2_loader_never_writes_to_the_v1_cache():
    from common import select_dense_columns
    from dataset import ShardBatcher

    before = _snapshot(ARTIFACTS)
    assert before, "artifacts/ looks empty"

    if (ARTIFACTS_V2 / "meta.json").exists():
        columns = select_dense_columns("all", "v2", True)
        loader = ShardBatcher(str(ARTIFACTS_V2), "valid", columns,
                              batch_size=512, shuffle=False, max_shards=1)
    else:
        loader = ShardBatcher(str(ARTIFACTS), "valid", list(DENSE_COLS),
                              batch_size=512, shuffle=False, max_shards=1)

    for _, batch in zip(range(3), loader.batches(epoch=0)):
        assert np.isfinite(batch["dense"].numpy()).all()

    after = _snapshot(ARTIFACTS)
    assert after == before, "the read-only V1 cache was modified"


# ---------------------------------------------------------------------------
# 4. the overlay is transparent for V1 models
# ---------------------------------------------------------------------------
@needs_v2
def test_v1_columns_through_the_overlay_match_the_v1_cache():
    """A V1 model reading a V2 overlay must see exactly the V1 features.

    This is the property that makes mixing a V1 checkpoint into a V2 stack safe.
    """

    from dataset import ShardBatcher

    v1 = ShardBatcher(str(ARTIFACTS), "valid", list(DENSE_COLS),
                      batch_size=1024, shuffle=False, max_shards=1)
    v2 = ShardBatcher(str(ARTIFACTS_V2), "valid", list(DENSE_COLS),
                      batch_size=1024, shuffle=False, max_shards=1)

    # the overlay resolves its shards through base_dir back to artifacts/
    assert os.path.realpath(v2.shard_dir) == os.path.realpath(v1.shard_dir)

    it1, it2 = v1.batches(epoch=0), v2.batches(epoch=0)
    for _ in range(3):
        a, b = next(it1), next(it2)
        assert np.array_equal(a["dense"], b["dense"])


@needs_v2
def test_overlay_statistics_cover_the_full_schema():
    meta = json.loads((ARTIFACTS_V2 / "meta.json").read_text())
    assert meta["version"] == 2
    assert meta["feature_set"] == "v2"
    assert meta["dense_columns"][:26] == V1_COLUMNS
    assert len(meta["dense_columns"]) == len(meta["dense_stats"]["mean"])
    assert len(meta["dense_columns"]) == len(meta["dense_stats"]["std"])
    assert meta["dense_stats"]["rows"] == 6104774
    assert meta["v2"]["fit_split"] == "train"

    # The overlay re-fits the V1 columns as part of one pass; they must come out
    # identical, otherwise V1 predictions would change under the overlay.
    v1_meta = json.loads((ARTIFACTS / "meta.json").read_text())
    assert np.allclose(meta["dense_stats"]["mean"][:26],
                       v1_meta["dense_stats"]["mean"], rtol=0, atol=1e-12)
    assert np.allclose(meta["dense_stats"]["std"][:26],
                       v1_meta["dense_stats"]["std"], rtol=0, atol=1e-12)
