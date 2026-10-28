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

# Column groups used by the feature-ablation experiments required by the
# project brief ("text only", "text + numeric", "text + numeric + entity").
FEATURE_GROUPS: Dict[str, List[str]] = {
    "numeric": BASE_NUMERIC_COLS[0:11] + DERIVED_NUMERIC_COLS,
    "entity": BASE_NUMERIC_COLS[11:20],
}
FEATURE_GROUPS["all"] = FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["entity"]

TEXT_COLS = ["title_tfidf", "comment_tfidf"]
TARGET_COL = "rating"
ID_COL = "id"


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


def select_dense_columns(mode: str) -> List[str]:
    """Return the dense column names for an ablation mode.

    ``mode`` is one of ``all``, ``numeric``, ``entity`` or ``none``.
    """

    if mode not in ("all", "numeric", "entity", "none"):
        raise ValueError(f"unknown dense mode: {mode}")
    if mode == "none":
        return []
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
