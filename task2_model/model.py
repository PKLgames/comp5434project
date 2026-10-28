"""Hybrid sparse-text / dense-feature rating regressor.

Design notes
------------
* **Sparse text tower.**  The two hashed TF-IDF vectors are fed to
  ``nn.EmbeddingBag(mode="sum")`` with the TF-IDF value as a per-sample weight.
  This computes ``W^T x`` for a sparse row while touching only its non-zeros,
  and it avoids materialising a 327k-column dense matrix.
* **Dense tower.**  A small MLP over the 26 numeric features captures
  non-linear effects such as "many reviews but few helpful votes".
* **Late fusion.**  Both representations are concatenated and passed through a
  two-layer head.  The output is mapped to ``1 + 4 * sigmoid(logit)`` so the
  prediction can never leave the legal rating range; this is a soft constraint
  that helps RMSE compared with an unbounded linear output.
* **Optional FiLM gating** (``film=True``).  The numeric features generate a
  per-dimension ``(gamma, beta)`` that modulates each text tower before fusion:
  ``h' = h * (1 + gamma) + beta``.  Concatenation can only express a
  text x numeric interaction additively through the head's first layer; FiLM
  makes it multiplicative, so "which words matter" can depend on the numeric
  context (length, votes, entity counts).  The last layer is zero-initialised and
  created after every other parameter, so enabling it is an exact identity at
  step 0 and leaves all other parameters at the same seed unchanged.
* **Optional ordinal head** (``ordinal=True``).  A 5-way classifier over the
  rating levels adds an auxiliary cross-entropy term and yields
  ``E[rating] = sum_k k * p_k``, a second estimator of the same target.  Its
  convex combination with the regression output is fitted out of fold by
  ``train.py``; see ``predict.py --blend-alpha``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HybridRatingModel(nn.Module):
    def __init__(
        self,
        vocab_title: int,
        vocab_comment: int,
        n_dense: int,
        text_dim: int = 256,
        dense_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        emb_dropout: float = 0.1,
        use_title: bool = True,
        use_comment: bool = True,
        use_wide: bool = True,
        ordinal: bool = False,
        text_mlp: bool = False,
        film: bool = False,
        init_rating: float = 3.8,
    ) -> None:
        super().__init__()
        self.use_title = use_title
        self.use_comment = use_comment
        self.use_dense = n_dense > 0
        self.use_wide = use_wide
        self.ordinal = ordinal
        self.text_mlp = bool(text_mlp and use_title and use_comment)
        self.text_dim = text_dim
        self.emb_dropout = nn.Dropout(emb_dropout)
        n_text_towers = int(use_title) + int(use_comment)
        self.n_text_towers = n_text_towers
        # FiLM needs a numeric context to generate the modulation from, and at
        # least one text tower to modulate.
        self.film = bool(film) and self.use_dense and n_text_towers > 0

        if use_title:
            self.title_emb = nn.EmbeddingBag(vocab_title, text_dim, mode="sum",
                                             include_last_offset=True)
            self.title_norm = nn.LayerNorm(text_dim)
            nn.init.normal_(self.title_emb.weight, mean=0.0, std=0.02)
        if use_comment:
            self.comment_emb = nn.EmbeddingBag(vocab_comment, text_dim, mode="sum",
                                               include_last_offset=True)
            self.comment_norm = nn.LayerNorm(text_dim)
            nn.init.normal_(self.comment_emb.weight, mean=0.0, std=0.02)

        # Wide (linear) text path.  The low-rank embedding tower above is a
        # rank-``text_dim`` approximation of a linear model; this path removes
        # that bottleneck by learning one free weight per hashed token, which
        # is exactly TF-IDF linear regression.  It is cheap (one scalar per
        # token) and is the single most important accuracy component.
        if self.use_wide:
            if use_title:
                self.lin_title = nn.EmbeddingBag(vocab_title, 1, mode="sum",
                                                 include_last_offset=True)
                nn.init.zeros_(self.lin_title.weight)
            if use_comment:
                self.lin_comment = nn.EmbeddingBag(vocab_comment, 1, mode="sum",
                                                   include_last_offset=True)
                nn.init.zeros_(self.lin_comment.weight)

        if self.use_dense:
            self.dense_mlp = nn.Sequential(
                nn.Linear(n_dense, dense_dim),
                nn.LayerNorm(dense_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dense_dim, dense_dim),
                nn.SiLU(),
            )
        else:
            self.dense_mlp = None

        # FiLM parameters are created at the *end* of __init__ (see below), so
        # that enabling ``--film`` does not shift the random stream of any other
        # parameter: a FiLM model and a plain model built from the same seed share
        # every other weight exactly.
        self.film_gen = None

        in_dim = text_dim * int(use_title) + text_dim * int(use_comment)
        if self.text_mlp:
            # A small non-linear layer over the concatenated title/comment
            # representation captures token interactions (e.g. negation) that
            # a purely additive bag-of-words cannot express.
            self.text_fuse = nn.Sequential(
                nn.Linear(in_dim, text_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            in_dim = text_dim
        in_dim += dense_dim if self.use_dense else 0

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 4, 16)),
            nn.SiLU(),
        )
        self.out = nn.Linear(max(hidden_dim // 4, 16), 1)
        if ordinal:
            # Auxiliary 5-way head: the cross-entropy term regularises the
            # shared representation and can later provide an expectation
            # estimate E[rating] for blending.
            self.ord_head = nn.Linear(max(hidden_dim // 4, 16), 5)

        self._init_rating_bias(init_rating)

        # FiLM: let the numeric features *modulate* the text representation
        # instead of only being concatenated next to it.  ``h_text' = h*(1+γ)+β``
        # with ``(γ, β) = f(dense)``.  Late concatenation can express such an
        # interaction only through the first head layer's additive weights; FiLM
        # makes it multiplicative and per-dimension.
        #
        # The final layer is zero-initialised, so γ = β = 0 and the block is an
        # exact identity at step 0.  Combined with creating these parameters last,
        # this means ``--film`` adds capacity without disturbing anything else:
        # same seed -> identical shared weights -> identical initial predictions.
        if self.film:
            self.film_gen = nn.Sequential(
                nn.Linear(n_dense, dense_dim),
                nn.SiLU(),
                nn.Linear(dense_dim, 2 * text_dim * n_text_towers),
            )
            nn.init.zeros_(self.film_gen[2].weight)
            nn.init.zeros_(self.film_gen[2].bias)

    def _init_rating_bias(self, rating: float) -> None:
        """Start the model at the global mean rating instead of a random value."""

        rating = min(max(rating, 1.0 + 1e-3), 5.0 - 1e-3)
        logit = math.log((rating - 1.0) / (5.0 - rating))
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, logit)

    # ------------------------------------------------------------------
    def _encode(self, t_idx, t_off, t_val, c_idx, c_off, c_val, dense):
        """Shared encoder; returns ``(representation, wide_score)``."""

        text_parts = []
        wide = None
        if self.use_title:
            h_title = self.title_emb(t_idx, t_off, per_sample_weights=t_val)
            text_parts.append(self.emb_dropout(self.title_norm(h_title)))
            if self.use_wide:
                wide = self.lin_title(t_idx, t_off, per_sample_weights=t_val).squeeze(-1)
        if self.use_comment:
            h_comment = self.comment_emb(c_idx, c_off, per_sample_weights=c_val)
            text_parts.append(self.emb_dropout(self.comment_norm(h_comment)))
            if self.use_wide:
                contrib = self.lin_comment(c_idx, c_off, per_sample_weights=c_val).squeeze(-1)
                wide = contrib if wide is None else wide + contrib

        # FiLM modulation happens per text tower, before any nonlinear fusion, so
        # that "which words matter" can depend on the numeric context.
        if self.film_gen is not None:
            gamma, beta = self.film_gen(dense).chunk(2, dim=1)
            modulated = []
            for i, part in enumerate(text_parts):
                lo, hi = i * self.text_dim, (i + 1) * self.text_dim
                modulated.append(part * (1.0 + gamma[:, lo:hi]) + beta[:, lo:hi])
            text_parts = modulated

        parts = list(text_parts)
        if self.text_mlp:
            # A small non-linear layer over the concatenated title/comment
            # representation captures token interactions (e.g. negation) that
            # a purely additive bag-of-words cannot express.
            parts = [self.text_fuse(torch.cat(text_parts, dim=1))]
        if self.use_dense:
            parts.append(self.dense_mlp(dense))

        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return h, wide

    def _heads(self, h, wide):
        hidden = self.head(h)
        logit = self.out(hidden).squeeze(-1)
        if wide is not None:
            logit = logit + wide
        # Soft bound: predictions live strictly inside [1, 5].
        pred = 1.0 + 4.0 * torch.sigmoid(logit)
        logits = self.ord_head(hidden) if self.ordinal else None
        return pred, logits

    def forward(self, t_idx, t_off, t_val, c_idx, c_off, c_val, dense):
        h, wide = self._encode(t_idx, t_off, t_val, c_idx, c_off, c_val, dense)
        pred, _ = self._heads(h, wide)
        return pred

    def forward_with_logits(self, t_idx, t_off, t_val, c_idx, c_off, c_val, dense):
        h, wide = self._encode(t_idx, t_off, t_val, c_idx, c_off, c_val, dense)
        return self._heads(h, wide)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        self.eval()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "y"}
        pred = self(
            batch["t_idx"], batch["t_off"], batch["t_val"],
            batch["c_idx"], batch["c_off"], batch["c_val"],
            batch["dense"],
        )
        return pred.float()
