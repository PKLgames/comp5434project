"""Hand-computed unit tests for the V2 feature blocks.

Every expectation here is derived by hand from tiny synthetic CSR arrays, so a
regression in the vectorised code cannot hide behind "the numbers look
plausible".  The vectors below are the exact failure modes the plan calls out:
empty rows, single-token rows, and ``data == 0`` vocabulary misses that must be
excluded from the non-zero count but must not corrupt the sum or the maximum.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "task2_model"))

from common import DENSE_COLS, V2_INTERACTION_COLS, V2_TEXT_SHAPE_COLS  # noqa: E402
from features_v2 import (  # noqa: E402
    all_columns,
    build_matrix,
    interaction_block,
    text_shape_block,
)


def _csr(rows):
    """Build ``(indptr, indices, data)`` from a list of ``{index: weight}`` rows."""

    indptr = [0]
    indices, data = [], []
    for row in rows:
        for idx in sorted(row):
            indices.append(idx)
            data.append(row[idx])
        indptr.append(len(indices))
    return (np.asarray(indptr, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(data, dtype=np.float32))


# ---------------------------------------------------------------------------
# text_shape_block
# ---------------------------------------------------------------------------
def test_column_names_are_the_documented_seven():
    assert V2_TEXT_SHAPE_COLS == [
        "title_nnz", "comment_nnz",
        "title_weight_sum", "comment_weight_sum",
        "title_weight_max", "comment_weight_max",
        "title_comment_cosine",
    ]
    assert all_columns(False) == V2_TEXT_SHAPE_COLS
    assert all_columns(True) == V2_TEXT_SHAPE_COLS + V2_INTERACTION_COLS


def test_text_shape_hand_computed():
    """row 0: two tokens each side, one shared bucket. row 1: both empty."""

    t = _csr([{0: 0.5, 1: 0.5}, {}, {2: 1.0}])
    c = _csr([{0: 0.5, 2: 0.5}, {}, {1: 1.0}])
    # stored id -> original hashed value.  Bucket 10 is shared by title and
    # comment, which is what makes row 0's cosine non-zero.
    t_map = np.asarray([10, 20, 30], dtype=np.int64)
    c_map = np.asarray([10, 40, 50], dtype=np.int64)

    block = text_shape_block(*t, *c, t_map=t_map, c_map=c_map)

    assert block.shape == (3, 7)
    assert block.dtype == np.float32

    # row 0 -------------------------------------------------------------
    assert block[0, 0] == pytest.approx(np.log1p(2), rel=1e-6)   # title_nnz
    assert block[0, 1] == pytest.approx(np.log1p(2), rel=1e-6)   # comment_nnz
    assert block[0, 2] == pytest.approx(np.log1p(1.0), rel=1e-6)  # 0.5 + 0.5
    assert block[0, 3] == pytest.approx(np.log1p(1.0), rel=1e-6)
    assert block[0, 4] == pytest.approx(0.5, rel=1e-6)
    assert block[0, 5] == pytest.approx(0.5, rel=1e-6)
    # only bucket 10 is common: 0.5 * 0.5
    assert block[0, 6] == pytest.approx(0.25, rel=1e-6)

    # row 1: empty on both sides -> every column is 0.0, cosine included
    assert np.all(block[1] == 0.0)

    # row 2: one token each, no shared bucket ---------------------------
    assert block[2, 0] == pytest.approx(np.log1p(1), rel=1e-6)
    assert block[2, 1] == pytest.approx(np.log1p(1), rel=1e-6)
    assert block[2, 4] == pytest.approx(1.0, rel=1e-6)
    assert block[2, 5] == pytest.approx(1.0, rel=1e-6)
    assert block[2, 6] == pytest.approx(0.0, abs=1e-7)


def test_zero_weight_entries_are_excluded_from_the_count():
    """``data == 0`` is a vocabulary miss, not a word.

    ``remap_to_vocab`` parks out-of-vocabulary hashed indices on slot 0 with
    weight 0 so ``EmbeddingBag`` ignores them.  Counting them as words would
    inflate ``title_nnz`` by the number of OOV tokens in the row.
    """

    rows = _csr
    t = rows([{0: 0.0}])          # one entry, but it is a miss -> zero words
    c = rows([{1: 0.0, 2: 0.0}])  # two misses -> zero words
    block = text_shape_block(*t, *c,
                             t_map=np.asarray([1, 2, 3]),
                             c_map=np.asarray([4, 5, 6]))

    assert block[0, 0] == 0.0     # log1p(0)
    assert block[0, 1] == 0.0
    assert block[0, 2] == 0.0     # weights are zero, so the sum is zero
    assert block[0, 3] == 0.0
    assert block[0, 4] == 0.0
    assert block[0, 5] == 0.0
    assert block[0, 6] == 0.0

    # A real token next to a miss: the miss must not raise the count.
    t = rows([{0: 0.0, 1: 0.6}])
    c = rows([{0: 0.0}])
    block = text_shape_block(*t, *c,
                             t_map=np.asarray([7, 8]),
                             c_map=np.asarray([9]))
    assert block[0, 0] == pytest.approx(np.log1p(1), rel=1e-6)
    assert block[0, 2] == pytest.approx(np.log1p(0.6), rel=1e-6)
    assert block[0, 4] == pytest.approx(0.6, rel=1e-6)


def test_all_rows_empty_gives_all_zero_block():
    t = _csr([{}, {}, {}])
    c = _csr([{}, {}, {}])
    block = text_shape_block(*t, *c,
                             t_map=np.asarray([1, 2]),
                             c_map=np.asarray([3, 4]))
    assert block.shape == (3, 7)
    assert np.all(block == 0.0)


def test_single_row_and_single_token_edges():
    t = _csr([{5: 1.0}])
    c = _csr([{5: 1.0}])
    block = text_shape_block(*t, *c,
                             t_map=np.asarray([100, 200, 300, 400, 500, 600]),
                             c_map=np.asarray([100, 200, 300, 400, 500, 600]))
    assert block[0, 0] == pytest.approx(np.log1p(1), rel=1e-6)
    assert block[0, 4] == pytest.approx(1.0, rel=1e-6)
    # a single identical token on both sides is a perfect cosine
    assert block[0, 6] == pytest.approx(1.0, rel=1e-6)


def test_cosine_uses_hashed_ids_not_stored_ids():
    """The two stored id spaces are independent, so ids must be mapped back.

    Here stored id 0 exists in both columns but maps to two *different* hashed
    buckets, so the correct cosine is 0 - a naive implementation that compares
    the stored ids would report 1.0.
    """

    t = _csr([{0: 1.0}])
    c = _csr([{0: 1.0}])
    block = text_shape_block(*t, *c,
                             t_map=np.asarray([11]),
                             c_map=np.asarray([22]))
    assert block[0, 6] == pytest.approx(0.0, abs=1e-7)

    block = text_shape_block(*t, *c,
                             t_map=np.asarray([33]),
                             c_map=np.asarray([33]))
    assert block[0, 6] == pytest.approx(1.0, rel=1e-6)


def test_maps_are_required():
    t = _csr([{0: 1.0}])
    c = _csr([{0: 1.0}])
    with pytest.raises(ValueError, match="t_map"):
        text_shape_block(*t, *c)


# ---------------------------------------------------------------------------
# interaction_block
# ---------------------------------------------------------------------------
def _dense_rows(n=2):
    dense = np.zeros((n, len(DENSE_COLS)), dtype=np.float32)
    dense[:, DENSE_COLS.index("review_year")] = 2023.0
    return dense


def test_interaction_block_hand_computed():
    dense = _dense_rows(2)
    dense[0, DENSE_COLS.index("comment_char_count")] = 100.0
    dense[0, DENSE_COLS.index("votes_log")] = 2.0
    dense[0, DENSE_COLS.index("purchased")] = 1.0
    dense[0, DENSE_COLS.index("user_unseen")] = 1.0
    dense[0, DENSE_COLS.index("product_unseen")] = 1.0
    dense[0, DENSE_COLS.index("parent_product_unseen")] = 0.0
    dense[0, DENSE_COLS.index("review_year")] = 2021.0

    block = interaction_block(dense)
    assert block.shape == (2, 4)
    comment_log = np.log1p(100.0)

    assert block[0, 0] == pytest.approx(2.0, rel=1e-6)
    assert block[0, 1] == pytest.approx(2.0 * comment_log, rel=1e-5)
    assert block[0, 2] == pytest.approx(1.0 * comment_log, rel=1e-5)
    assert block[0, 3] == pytest.approx(-2.0, rel=1e-6)

    # row 1 is the all-zero / year-2023 reference row
    assert block[1, 0] == 0.0
    assert block[1, 1] == 0.0
    assert block[1, 2] == 0.0
    assert block[1, 3] == pytest.approx(0.0, abs=1e-7)


def test_interaction_block_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="26"):
        interaction_block(np.zeros((3, 25), dtype=np.float32))


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------
def test_build_matrix_orders_columns_as_the_schema():
    dense = _dense_rows(2)
    t = _csr([{0: 1.0}, {}])
    c = _csr([{1: 1.0}, {}])
    t_map = np.asarray([1, 2])
    c_map = np.asarray([3, 4])

    core = build_matrix(dense, *t, *c, t_map, c_map, interactions=False)
    assert core.shape == (2, len(DENSE_COLS) + 7)
    assert np.array_equal(core[:, :len(DENSE_COLS)], dense)

    full = build_matrix(dense, *t, *c, t_map, c_map, interactions=True)
    assert full.shape == (2, len(DENSE_COLS) + 11)
    assert np.array_equal(full[:, :len(DENSE_COLS)], dense)
    assert np.array_equal(full[:, len(DENSE_COLS) + 7:],
                          interaction_block(dense))


def test_build_matrix_rejects_a_row_count_mismatch():
    dense = _dense_rows(3)
    t = _csr([{0: 1.0}])  # one row only
    c = _csr([{0: 1.0}])
    with pytest.raises(ValueError, match="rows"):
        build_matrix(dense, *t, *c, np.asarray([1]), np.asarray([1]))
