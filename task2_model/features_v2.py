"""V2 feature blocks: text-shape statistics and optional interactions.

Why these features
------------------
The V1 model consumes the two hashed TF-IDF rows only through their *values*
(``EmbeddingBag`` for the low-rank text tower, one free scalar per token for the
wide linear path).  It never sees their *shape*: how many distinct words survive
``minDocFreq``, how concentrated the weight mass is, and how title and comment
overlap.  Those are pure functions of the CSR arrays already stored in the cache,
cost a few milliseconds per shard, and cannot be recovered from a rank-256
projection of the TF-IDF row.

Every feature here is a function of a **single row** of already-materialised
arrays.  There is no label use, no cross-row statistic and no new upstream data,
so the block is trivially reusable for the test split.

Note on ``title_comment_cosine``
--------------------------------
The two cached CSR columns index *separate* compact vocabularies
(``vocab_title.npy`` : 55,731 entries, ``vocab_comment.npy`` : 182,382 entries),
so the stored column ids are not comparable across columns.  The dot product is
therefore taken in the **original hashed space**, by mapping the stored ids back
through the vocabulary arrays that ``preprocess.py`` wrote.  This is the only
space in which "the same hash bucket in title and in comment" is meaningful
(Part A hashed title into 2**16 and comment into 2**18 buckets, so 40,007 hashed
values are shared by both vocabularies).  Computing the product on the raw
stored ids instead would be semantically arbitrary and fires on only ~0.2 % of
rows, so :func:`text_shape_block` requires the two maps.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from common import DENSE_COLS, V2_INTERACTION_COLS, V2_TEXT_SHAPE_COLS

# Positions of the V1 columns needed by :func:`interaction_block`.  Resolved
# once from the canonical registry so the two files cannot drift apart.
_IX = {name: i for i, name in enumerate(DENSE_COLS)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _n_rows(t_indptr: np.ndarray, c_indptr: np.ndarray) -> int:
    """Number of rows implied by the CSR offset arrays (0-based, trailing +1)."""

    counts = [int(a.size) - 1 for a in (t_indptr, c_indptr) if a is not None and a.size]
    if not counts:
        return 0
    n = max(counts)
    for other in counts:
        if other not in (0, n):
            raise ValueError(f"inconsistent CSR row counts: {counts}")
    return n


def _row_ids(indptr: np.ndarray, n_rows: int) -> np.ndarray:
    """Row id of every stored non-zero, in storage order."""

    if n_rows == 0 or indptr.size < 2:
        return np.empty(0, dtype=np.int64)
    lengths = np.diff(indptr).astype(np.int64, copy=False)
    return np.repeat(np.arange(n_rows, dtype=np.int64), lengths)


def _column_shape(
    indptr: np.ndarray, data: np.ndarray, n_rows: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row ``(non-zero count, weight sum, weight max)`` of one sparse column.

    Entries whose weight is exactly ``0.0`` are vocabulary misses written by
    ``remap_to_vocab`` (an out-of-vocabulary hashed index is parked on slot 0
    with weight 0 so ``EmbeddingBag`` ignores it).  They are **not** words, so
    they are excluded from the non-zero count.  They contribute nothing to the
    sum, and cannot change the maximum because all TF-IDF weights are >= 0.
    """

    counts = np.zeros(n_rows, dtype=np.float64)
    sums = np.zeros(n_rows, dtype=np.float64)
    maxes = np.zeros(n_rows, dtype=np.float64)
    if n_rows == 0 or data is None or data.size == 0:
        return counts, sums, maxes

    values = np.asarray(data, dtype=np.float64)
    rows = _row_ids(indptr, n_rows)

    sums += np.bincount(rows, weights=values, minlength=n_rows)[:n_rows]

    positive = values > 0.0
    if not positive.any():
        return counts, sums, maxes

    counts += np.bincount(rows[positive], minlength=n_rows)[:n_rows]

    # Per-row maximum.  ``reduceat`` needs an index inside the array, so empty
    # rows (e.g. reviews with no title) are filled in separately with 0.0.
    lengths = np.diff(indptr).astype(np.int64, copy=False)
    nonempty = lengths > 0
    if nonempty.any():
        starts = indptr[:-1].astype(np.int64, copy=False)[nonempty]
        maxes[nonempty] = np.maximum.reduceat(values, starts)
    return counts, sums, maxes


def _keyed(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_rows: int,
    mapping: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rows, mapped column ids, weights)`` with zero weights dropped."""

    if n_rows == 0 or data is None or data.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float64)

    rows = _row_ids(indptr, n_rows)
    ids = np.asarray(indices, dtype=np.int64)
    if mapping is not None:
        ids = np.asarray(mapping, dtype=np.int64)[ids]
    values = np.asarray(data, dtype=np.float64)

    keep = values != 0.0
    if not keep.all():
        rows, ids, values = rows[keep], ids[keep], values[keep]
    return rows, ids, values


def _common_dot(
    t_indptr: np.ndarray,
    t_indices: np.ndarray,
    t_data: np.ndarray,
    c_indptr: np.ndarray,
    c_indices: np.ndarray,
    c_data: np.ndarray,
    n_rows: int,
    t_map: Optional[np.ndarray],
    c_map: Optional[np.ndarray],
) -> np.ndarray:
    """``t . c`` restricted to column ids present in both sparse rows.

    Both TF-IDF rows are L2-normalised by Part A, so the result is a cosine in
    ``[0, 1]`` (0.0 when either side is empty or the two rows share no bucket).
    """

    out = np.zeros(n_rows, dtype=np.float64)
    t_rows, t_ids, t_vals = _keyed(t_indptr, t_indices, t_data, n_rows, t_map)
    c_rows, c_ids, c_vals = _keyed(c_indptr, c_indices, c_data, n_rows, c_map)
    if t_ids.size == 0 or c_ids.size == 0:
        return out

    # One int64 key per entry: ``row * span + column id``.  ``span`` exceeds
    # every column id, so keys are strictly increasing across row boundaries and
    # can never overflow (rows <= 1e5, ids < 2**18).
    span = int(max(t_ids.max(), c_ids.max())) + 1
    t_keys = t_rows * span + t_ids
    c_keys = c_rows * span + c_ids

    # ``searchsorted`` needs the haystack sorted; CSR rows are stored in
    # increasing column order, so the check normally passes without a copy.
    if c_keys.size > 1 and (np.diff(c_keys) < 0).any():
        order = np.argsort(c_keys, kind="stable")
        c_keys, c_vals = c_keys[order], c_vals[order]

    pos = np.searchsorted(c_keys, t_keys)
    inside = pos < c_keys.size
    pos_clipped = np.minimum(pos, c_keys.size - 1)
    hit = inside & (c_keys[pos_clipped] == t_keys)
    if not hit.any():
        return out

    prod = t_vals[hit] * c_vals[pos_clipped[hit]]
    out += np.bincount(t_rows[hit], weights=prod, minlength=n_rows)[:n_rows]
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# public feature blocks
# ---------------------------------------------------------------------------
def text_shape_block(
    t_indptr: np.ndarray,
    t_indices: np.ndarray,
    t_data: np.ndarray,
    c_indptr: np.ndarray,
    c_indices: np.ndarray,
    c_data: np.ndarray,
    t_map: Optional[np.ndarray] = None,
    c_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the 7-column V2 text-shape block for a contiguous block of rows.

    Parameters
    ----------
    t_indptr, t_indices, t_data
        Title CSR arrays.  ``t_indptr`` must be 0-based and have ``rows + 1``
        entries, i.e. the offsets of the *slice* being encoded.
    c_indptr, c_indices, c_data
        Comment CSR arrays, same convention.
    t_map, c_map
        Stored-column-id -> original hashed-value lookup (i.e.
        ``vocab_title.npy`` / ``vocab_comment.npy``).  Required for the
        ``title_comment_cosine`` column, because the two stored id spaces are
        independent; see the module docstring.

    Returns
    -------
    ``float32[rows, 7]`` in :data:`common.V2_TEXT_SHAPE_COLS` order.
    """

    if t_map is None or c_map is None:
        raise ValueError(
            "text_shape_block needs t_map and c_map (the vocab_*.npy arrays): "
            "the title and comment stored id spaces are independent, so the "
            "cross-column cosine is undefined without mapping back to hashed ids"
        )

    n_rows = _n_rows(t_indptr, c_indptr)
    t_count, t_sum, t_max = _column_shape(t_indptr, t_data, n_rows)
    c_count, c_sum, c_max = _column_shape(c_indptr, c_data, n_rows)
    cosine = _common_dot(t_indptr, t_indices, t_data,
                         c_indptr, c_indices, c_data,
                         n_rows, t_map, c_map)

    block = np.empty((n_rows, len(V2_TEXT_SHAPE_COLS)), dtype=np.float32)
    block[:, 0] = np.log1p(t_count)
    block[:, 1] = np.log1p(c_count)
    block[:, 2] = np.log1p(t_sum)
    block[:, 3] = np.log1p(c_sum)
    block[:, 4] = t_max
    block[:, 5] = c_max
    block[:, 6] = cosine
    return block


def interaction_block(dense_v1: np.ndarray) -> np.ndarray:
    """Build the 4-column V2 interaction block from the 26 V1 dense columns.

    Returns ``float32[rows, 4]`` in :data:`common.V2_INTERACTION_COLS` order.
    """

    dense = np.asarray(dense_v1, dtype=np.float32)
    if dense.ndim != 2 or dense.shape[1] != len(DENSE_COLS):
        raise ValueError(
            f"interaction_block expects the {len(DENSE_COLS)} V1 dense columns, "
            f"got shape {dense.shape}"
        )

    comment_log = np.log1p(dense[:, _IX["comment_char_count"]], dtype=np.float32)
    unseen = (dense[:, _IX["user_unseen"]]
              + dense[:, _IX["product_unseen"]]
              + dense[:, _IX["parent_product_unseen"]]).astype(np.float32)
    votes_x = (dense[:, _IX["votes_log"]] * comment_log).astype(np.float32)
    purchased_x = (dense[:, _IX["purchased"]] * comment_log).astype(np.float32)
    year = (dense[:, _IX["review_year"]] - np.float32(2023.0)).astype(np.float32)

    block = np.empty((dense.shape[0], len(V2_INTERACTION_COLS)), dtype=np.float32)
    block[:, 0] = unseen
    block[:, 1] = votes_x
    block[:, 2] = purchased_x
    block[:, 3] = year
    return block


def all_columns(interactions: bool = False) -> List[str]:
    """Column names produced by :func:`build_matrix` for the given options."""

    columns = list(V2_TEXT_SHAPE_COLS)
    if interactions:
        columns.extend(V2_INTERACTION_COLS)
    return columns


def build_matrix(
    dense_v1: Optional[np.ndarray],
    t_indptr: np.ndarray,
    t_indices: np.ndarray,
    t_data: np.ndarray,
    c_indptr: np.ndarray,
    c_indices: np.ndarray,
    c_data: np.ndarray,
    t_map: Optional[np.ndarray],
    c_map: Optional[np.ndarray],
    interactions: bool = False,
) -> np.ndarray:
    """Full V2 dense matrix: the 26 V1 columns + text shape [+ interactions].

    ``dense_v1`` must be the raw (un-standardised) V1 dense block of the same
    rows, so the caller can append the new columns *before* standardisation and
    keep a single mean/std vector for the whole schema.
    """

    shape = text_shape_block(t_indptr, t_indices, t_data,
                             c_indptr, c_indices, c_data, t_map, c_map)
    parts = []
    if dense_v1 is not None:
        base = np.asarray(dense_v1, dtype=np.float32)
        if base.shape[0] != shape.shape[0]:
            raise ValueError(
                f"dense block has {base.shape[0]} rows but the CSR block has "
                f"{shape.shape[0]}"
            )
        parts.append(base)
    parts.append(shape)
    if interactions:
        if dense_v1 is None:
            raise ValueError("interactions need the V1 dense block")
        parts.append(interaction_block(dense_v1))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
