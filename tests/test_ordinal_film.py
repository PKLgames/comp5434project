"""Tests for the two changes that make up the new baseline and architecture.

1. **Ordinal baseline** - the ``E[rating]`` blend weight is fitted out of fold,
   in closed form.  ``optimal_blend_alpha`` must agree with a brute-force search,
   and ``blend_out_of_fold`` must not report the in-sample optimum.
2. **FiLM architecture** - enabling it must (a) leave every other parameter
   bit-identical at the same seed, (b) start as an exact identity, and (c) not
   change the state_dict of a non-FiLM model, so the published V1 checkpoints
   keep loading.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "task2_model"))

from model import HybridRatingModel  # noqa: E402
from train import (blend_out_of_fold, optimal_blend_alpha,  # noqa: E402
                   resolve_selection_key)


# ---------------------------------------------------------------------------
# ordinal blend weight
# ---------------------------------------------------------------------------
def _blend_problem(n=200_000, seed=0):
    """A case whose optimal alpha is strictly interior (~0.71)."""

    rng = np.random.default_rng(seed)
    pred = rng.normal(3.8, 1.0, n)
    residual = rng.normal(0.0, 1.0, n)
    y = pred + residual
    exp = pred + 0.7 * residual + 0.7 * rng.normal(0.0, 1.0, n)
    return pred, exp, y


def test_optimal_blend_alpha_matches_brute_force():
    pred, exp, y = _blend_problem()

    alpha = optimal_blend_alpha(pred, exp, y)
    grid = np.linspace(0.0, 1.0, 2001)
    scores = [np.sqrt(np.mean(((1 - a) * pred + a * exp - y) ** 2)) for a in grid]
    best = grid[int(np.argmin(scores))]

    assert 0.05 < alpha < 0.95, "this fixture should give an interior optimum"
    assert alpha == pytest.approx(best, abs=2e-3)
    assert np.sqrt(np.mean(((1 - alpha) * pred + alpha * exp - y) ** 2)) == \
        pytest.approx(min(scores), abs=1e-6)


def test_optimal_blend_alpha_is_clamped_to_the_unit_interval():
    pred, _, y = _blend_problem()
    # an expectation that is pure noise can only hurt -> weight 0
    assert optimal_blend_alpha(pred, pred.copy(), y) == 0.0
    # an expectation identical to the target wants weight 1, not more
    assert optimal_blend_alpha(pred, y.copy(), y) == pytest.approx(1.0)
    # degenerate input must not divide by zero
    assert optimal_blend_alpha(pred, pred, y) == 0.0


def test_blend_out_of_fold_is_honest():
    pred, exp, y = _blend_problem()
    metrics, alphas = blend_out_of_fold(pred, exp, y, folds=5)

    assert len(alphas) == 5
    assert all(0.0 <= a <= 1.0 for a in alphas)
    assert metrics["blend_alpha"] == pytest.approx(float(np.mean(alphas)))

    rmse_pred = float(np.sqrt(np.mean((pred - y) ** 2)))
    rmse_exp = float(np.sqrt(np.mean((exp - y) ** 2)))
    # a two-estimator blend must beat the worse one, and must not be worse than
    # the in-sample-optimal blend (it is fitted on other folds)
    assert metrics["rmse_blend"] <= max(rmse_pred, rmse_exp)
    alpha_full = optimal_blend_alpha(pred, exp, y)
    in_sample = float(np.sqrt(np.mean(((1 - alpha_full) * pred + alpha_full * exp - y) ** 2)))
    assert metrics["rmse_blend"] >= in_sample - 1e-6


def test_blend_out_of_fold_handles_fewer_rows_than_folds():
    metrics, alphas = blend_out_of_fold(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                        np.array([1.0, 2.0]), folds=5)
    assert np.isfinite(metrics["rmse_blend"])
    assert len(alphas) == 5


# ---------------------------------------------------------------------------
# --select-metric
# ---------------------------------------------------------------------------
def test_select_metric_auto_reproduces_the_historical_rule():
    """The default must not change any existing behaviour."""

    # with an MSE term -> regression RMSE (what every published run did)
    assert resolve_selection_key("auto", 1.0, False) == "rmse"
    assert resolve_selection_key("auto", 1.0, True) == "rmse"
    # pure classification -> E[rating], also the historical rule
    assert resolve_selection_key("auto", 0.0, True) == "rmse_exp"


def test_select_metric_explicit_values():
    assert resolve_selection_key("rmse", 1.0, False) == "rmse"
    assert resolve_selection_key("exp", 1.0, True) == "rmse_exp"
    assert resolve_selection_key("blend", 1.0, True) == "rmse_blend"


def test_select_metric_requires_an_ordinal_head():
    for metric in ("exp", "blend"):
        with pytest.raises(ValueError, match="ordinal head"):
            resolve_selection_key(metric, 1.0, False)


def test_select_metric_rejects_unknown_values():
    with pytest.raises(ValueError, match="unknown"):
        resolve_selection_key("mae", 1.0, True)


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------
KW = dict(vocab_title=100, vocab_comment=200, n_dense=26, text_dim=32,
          dense_dim=16, hidden_dim=32, dropout=0.0, emb_dropout=0.0)


def _batch(n=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        t_idx=torch.randint(0, 100, (n * 4,), generator=g),
        t_off=torch.arange(0, n * 4 + 1, 4),
        t_val=torch.rand(n * 4, generator=g),
        c_idx=torch.randint(0, 200, (n * 6,), generator=g),
        c_off=torch.arange(0, n * 6 + 1, 6),
        c_val=torch.rand(n * 6, generator=g),
        dense=torch.randn(n, 26, generator=g),
    )


def test_film_off_keeps_the_published_state_dict():
    """No FiLM parameters may exist unless requested (V1 checkpoints must load)."""

    model = HybridRatingModel(**KW)
    assert model.film is False
    assert model.film_gen is None
    assert not any("film_gen" in key for key in model.state_dict())


def test_film_on_adds_exactly_the_film_parameters():
    plain = HybridRatingModel(**KW)
    film = HybridRatingModel(film=True, **KW)

    extra = set(film.state_dict()) - set(plain.state_dict())
    assert extra == {"film_gen.0.weight", "film_gen.0.bias",
                     "film_gen.2.weight", "film_gen.2.bias"}
    assert set(plain.state_dict()) - set(film.state_dict()) == set()


def test_film_same_seed_does_not_disturb_other_parameters():
    """Creating the FiLM block last means the shared weights are identical."""

    torch.manual_seed(0)
    plain = HybridRatingModel(**KW)
    torch.manual_seed(0)
    film = HybridRatingModel(film=True, **KW)

    for key, value in plain.state_dict().items():
        assert torch.equal(value, film.state_dict()[key]), key


def _film_model_with_a_live_head(seed=0):
    """A FiLM model whose head actually reads its representation.

    ``_init_rating_bias`` zeroes ``out.weight`` so that an untrained model emits
    the global mean for every row.  That would make any identity assertion pass
    trivially, so the output layer is given random weights here.
    """

    torch.manual_seed(seed)
    model = HybridRatingModel(film=True, **KW)
    with torch.no_grad():
        model.out.weight.normal_(0.0, 0.1)
    model.eval()
    return model


def test_film_starts_as_an_exact_identity():
    """gamma = beta = 0 at init, so the first forward pass equals the plain model."""

    torch.manual_seed(0)
    plain = HybridRatingModel(**KW)
    torch.manual_seed(0)
    film = HybridRatingModel(film=True, **KW)
    with torch.no_grad():
        plain.out.weight.normal_(0.0, 0.1)
        film.out.weight.copy_(plain.out.weight)
    plain.eval()
    film.eval()

    batch = _batch()
    with torch.no_grad():
        assert torch.equal(plain(**batch), film(**batch))


def test_film_really_modulates_after_a_step():
    """Once gamma/beta move, FiLM must change the output (i.e. it is wired in)."""

    film = _film_model_with_a_live_head()
    batch = _batch()
    with torch.no_grad():
        reference = film(**batch)
        # nudge only the FiLM output layer, as a training step would
        film.film_gen[2].weight.normal_(0.0, 0.05)
        moved = film(**batch)
    assert not torch.allclose(reference, moved, atol=1e-6)


def test_film_is_disabled_without_a_dense_tower():
    model = HybridRatingModel(vocab_title=100, vocab_comment=200, n_dense=0,
                              text_dim=32, dense_dim=16, hidden_dim=32,
                              film=True)
    assert model.film is False
    assert model.film_gen is None
