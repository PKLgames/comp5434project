"""Shared constants and helpers for the COMP5434 Task-2 rating model.

The handover feature files contain exactly 20 numeric columns plus two
hashed TF-IDF sparse-vector columns.  Keeping the column order in one
place guarantees that preprocessing, training and inference always agree.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Numeric columns (order matches the parquet schema written by Part A)
# ---------------------------------------------------------------------------
BASE_NUMERIC_COLS: List[str] = [
    "title_char_count",
    "comment_char_count",
    "title_word_count",
    "comment_word_count",
    "votes",
    "votes_log",
    "purchased",
    "review_year",
    "review_month",
    "comment_missing",
    "votes_invalid",
    # user / product / parent-product statistics
    "user_review_count",
    "user_review_count_log",
    "user_unseen",
    "product_review_count",
    "product_review_count_log",
    "product_unseen",
    "parent_product_review_count",
    "parent_product_review_count_log",
    "parent_product_unseen",
]

# Derived columns are computed from the base columns only (no labels), so the
# exact same transformation can be applied to the future test features.
DERIVED_NUMERIC_COLS: List[str] = [
    "log_title_char",
    "log_comment_char",
    "title_words_per_char",
    "comment_words_per_char",
    "has_votes",
    "user_minus_product_log",
]

DENSE_COLS: List[str] = BASE_NUMERIC_COLS + DERIVED_NUMERIC_COLS

# ---------------------------------------------------------------------------
# V2 columns (see features_v2.py for the definitions and their motivation).
#
# These are *not* part of ``DENSE_COLS``: the V1 schema must stay byte-identical
# so that ``artifacts/``, ``runs/full3/best.pt``, ``runs/full_tmlp/best.pt`` and
# every previously reported number remain valid.  A V2 cache is an overlay that
# appends these columns *after* the 26 V1 ones.
# ---------------------------------------------------------------------------
V2_TEXT_SHAPE_COLS: List[str] = [
    "title_nnz",
    "comment_nnz",
    "title_weight_sum",
    "comment_weight_sum",
    "title_weight_max",
    "comment_weight_max",
    "title_comment_cosine",
]

V2_INTERACTION_COLS: List[str] = [
    "unseen_sum",
    "votes_x_comment_log",
    "purchased_x_comment_log",
    "review_year_centered",
]

# Column groups used by the feature-ablation experiments required by the
# project brief ("text only", "text + numeric", "text + numeric + entity").
# ``all`` deliberately stays the 26 V1 columns.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "numeric": BASE_NUMERIC_COLS[0:11] + DERIVED_NUMERIC_COLS,
    "entity": BASE_NUMERIC_COLS[11:20],
}
FEATURE_GROUPS["all"] = FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["entity"]

# Ablation groups for the V2 blocks so that the existing ablation machinery
# (``select_dense_columns`` -> ``--dense-mode``) works unchanged.
FEATURE_GROUPS["text_shape"] = list(V2_TEXT_SHAPE_COLS)
FEATURE_GROUPS["interactions"] = list(V2_INTERACTION_COLS)

FEATURE_SETS = ("v1", "v2")

TEXT_COLS = ["title_tfidf", "comment_tfidf"]
TARGET_COL = "rating"
ID_COL = "id"


def dense_schema(feature_set: str = "v1", interactions: bool = False) -> List[str]:
    """Ordered dense column names for a feature set.

    ``v1`` -> the 26 columns stored in every existing ``meta.json``.
    ``v2`` -> those 26 plus the 7 text-shape columns, plus the 4 interaction
    columns when ``interactions`` is true.  The order must match what
    ``preprocess.py --feature-set v2`` writes into ``meta["dense_columns"]`` and
    what ``features_v2.build_matrix`` concatenates.
    """

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set!r}")
    columns = list(DENSE_COLS)
    if feature_set == "v1":
        if interactions:
            # The interaction block is a pure function of the V1 dense columns,
            # but no V1 cache carries statistics for it, so refuse early rather
            # than produce a schema whose mean/std cannot be resolved.
            raise ValueError("interactions require feature_set='v2'")
        return columns
    columns += V2_TEXT_SHAPE_COLS
    if interactions:
        columns += V2_INTERACTION_COLS
    return columns


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def build_dense_matrix(table) -> np.ndarray:
    """Materialise the 26-column dense feature matrix for one parquet shard.

    ``table`` is a :class:`pyarrow.Table` that contains all base numeric
    columns.  Vectorised NumPy arithmetic keeps this fast enough for six
    million rows.
    """

    n_rows = table.num_rows
    cols: List[np.ndarray] = []
    for name in BASE_NUMERIC_COLS:
        arr = table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        cols.append(arr.astype(np.float32, copy=False))

    title_char = cols[0]
    comment_char = cols[1]
    title_words = cols[2]
    comment_words = cols[3]
    votes = cols[4]
    user_log = cols[12]
    product_log = cols[15]

    derived = [
        np.log1p(title_char, dtype=np.float32),
        np.log1p(comment_char, dtype=np.float32),
        (title_words / (title_char + 1.0)).astype(np.float32),
        (comment_words / (comment_char + 1.0)).astype(np.float32),
        (votes > 0).astype(np.float32),
        (user_log - product_log).astype(np.float32),
    ]
    cols.extend(derived)
    dense = np.stack(cols, axis=1)
    assert dense.shape == (n_rows, len(DENSE_COLS)), dense.shape
    return dense


def select_dense_columns(
    mode: str, feature_set: str = "v1", interactions: bool = False
) -> List[str]:
    """Return the dense column names for an ablation mode.

    ``mode`` is one of ``all``, ``numeric``, ``entity``, ``text_shape``,
    ``interactions`` or ``none``.

    ``all`` means "the whole schema of this cache": 26 columns for a V1 artifact
    directory, 33 (or 37 with ``interactions``) for a V2 overlay.  The V1
    behaviour is therefore unchanged, while a V2 run does not silently drop the
    new columns.
    """

    allowed = ("all", "numeric", "entity", "text_shape", "interactions", "none")
    if mode not in allowed:
        raise ValueError(f"unknown dense mode: {mode}")
    if mode == "none":
        return []
    if mode == "all":
        return dense_schema(feature_set, interactions)
    if mode in ("text_shape", "interactions") and feature_set != "v2":
        raise ValueError(f"dense mode {mode!r} requires a V2 cache (--feature-set v2)")
    return list(FEATURE_GROUPS[mode])


def text_flags(mode: str) -> Sequence[bool]:
    """Return ``(use_title, use_comment)`` for a text ablation mode."""

    mapping = {
        "both": (True, True),
        "title": (True, False),
        "comment": (False, True),
        "none": (False, False),
    }
    if mode not in mapping:
        raise ValueError(f"unknown text mode: {mode}")
    return mapping[mode]
