#!/usr/bin/env python
"""Train the hybrid sparse-text rating regressor.

Examples
--------
# Full run on a GPU machine
python train.py --data ../artifacts --out ../runs/nn_all --epochs 6 --batch-size 4096

# Quick smoke test on the CPU (first 4 shards of each split)
python train.py --data ../artifacts --out ../runs/smoke --epochs 1 \
                --limit-train-shards 4 --limit-valid-shards 1

# Feature ablation: text only
python train.py --data ../artifacts --out ../runs/nn_text --dense-mode none
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from common import load_json, save_json, select_dense_columns, text_flags
from dataset import ShardBatcher
from model import HybridRatingModel


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_batch(model, batch, want_logits: bool = False):
    args = (batch["t_idx"], batch["t_off"], batch["t_val"],
            batch["c_idx"], batch["c_off"], batch["c_val"], batch["dense"])
    return model.forward_with_logits(*args) if want_logits else model(*args)


# ---------------------------------------------------------------------------
def optimal_blend_alpha(pred: np.ndarray, exp: np.ndarray, y: np.ndarray) -> float:
    """Closed-form least-squares weight for ``(1-a)*pred + a*exp``.

    ``a`` minimises ``||(1-a)*pred + a*exp - y||^2``, i.e. it is the coefficient
    of ``y - pred`` regressed on ``exp - pred``.  Clamped to ``[0, 1]`` so the
    blend can only interpolate between the two estimators, never extrapolate.
    """

    delta = exp - pred
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return 0.0
    alpha = float(np.dot(y - pred, delta)) / denominator
    return float(np.clip(alpha, 0.0, 1.0))


def blend_out_of_fold(pred: np.ndarray, exp: np.ndarray, y: np.ndarray,
                      folds: int = 5) -> Tuple[Dict[str, float], list]:
    """Out-of-fold blend of the regression head and ``E[rating]``.

    The blend weight is fitted on the *other* folds and applied to the held-out
    fold, so the reported RMSE is not the in-sample optimum.  With one parameter
    over 1.5M rows the two differ by ~1e-6, but reporting the honest one costs
    nothing because the optimum is closed-form.
    """

    n = y.size
    oof = np.empty(n, dtype=np.float64)
    bounds = np.linspace(0, n, folds + 1).astype(np.int64)
    alphas = []
    for k in range(folds):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        held = np.zeros(n, dtype=bool)
        held[lo:hi] = True
        alpha = optimal_blend_alpha(pred[~held], exp[~held], y[~held])
        alphas.append(alpha)
        oof[held] = (1.0 - alpha) * pred[held] + alpha * exp[held]
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    return ({"rmse_blend": rmse, "blend_alpha": float(np.mean(alphas))},
            [float(a) for a in alphas])


def resolve_selection_key(select_metric: str, mse_weight: float,
                          has_ordinal: bool) -> str:
    """Map ``--select-metric`` to the metrics key used for early stopping.

    ``auto`` reproduces the historical rule exactly: the regression RMSE when
    there is an MSE term, otherwise ``E[rating]`` (which is all a pure
    classification model has).  ``exp`` and ``blend`` require an ordinal head.

    Selecting on ``blend`` is the honest choice when the blend is what gets
    submitted, but it reuses the validation split twice: once to fit the
    out-of-fold blend weight and once to pick the epoch.  That is one extra
    parameter over 1.5 M rows, so it is mild, but it is not free - say so when
    reporting the number.
    """

    if select_metric == "auto":
        return "rmse" if mse_weight > 0 else "rmse_exp"
    key = {"rmse": "rmse", "exp": "rmse_exp", "blend": "rmse_blend"}.get(select_metric)
    if key is None:
        raise ValueError(f"unknown --select-metric: {select_metric!r}")
    if key != "rmse" and not has_ordinal:
        raise ValueError(
            f"--select-metric {select_metric} needs an ordinal head "
            f"(--ordinal-weight > 0)")
    return key


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader: ShardBatcher, device: torch.device,
             max_batches: int | None = None,
             blend_folds: int = 5) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Return RMSE/MAE metrics plus the concatenated predictions and labels.

    When the model has an auxiliary ordinal head we additionally report the
    RMSE of ``E[rating] = sum_k k * p_k`` and of the convex blend between the
    regression output and that expectation.  The single blend weight is fitted
    **out of fold** (5 contiguous folds over the validation rows), so the
    reported ``rmse_blend`` is not the in-sample optimum; see
    :func:`blend_out_of_fold`.
    """

    model.eval()
    sq_err = 0.0
    abs_err = 0.0
    count = 0
    per_rating_sq = np.zeros(6, dtype=np.float64)
    per_rating_n = np.zeros(6, dtype=np.int64)
    preds, labels, exps = [], [], []

    for i, batch in enumerate(loader.batches(epoch=0)):
        if max_batches and i >= max_batches:
            break
        y = batch["y"].to(device)
        batch = move_batch(batch, device)
        if getattr(model, "ordinal", False):
            pred, logits = forward_batch(model, batch, want_logits=True)
            probs = torch.softmax(logits.float(), dim=1)
            levels = torch.arange(1, 6, device=device, dtype=probs.dtype)
            exp = (probs * levels).sum(dim=1)
            exps.append(exp.detach().cpu().numpy())
        else:
            pred = forward_batch(model, batch)
        diff = pred - y
        sq_err += float((diff * diff).sum())
        abs_err += float(diff.abs().sum())
        count += y.numel()

        y_np = y.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()
        labels.append(y_np)
        preds.append(p_np)
        r = np.clip(np.rint(y_np), 1, 5).astype(np.int64)
        per_rating_sq += np.bincount(r, weights=diff.detach().cpu().numpy() ** 2, minlength=6)
        per_rating_n += np.bincount(r, minlength=6)

    labels = np.concatenate(labels) if labels else np.empty(0)
    preds = np.concatenate(preds) if preds else np.empty(0)
    rmse = float(np.sqrt(sq_err / max(count, 1)))
    mae = float(abs_err / max(count, 1))
    per_rating = {}
    for k in range(1, 6):
        if per_rating_n[k]:
            per_rating[str(k)] = float(np.sqrt(per_rating_sq[k] / per_rating_n[k]))
    metrics = {"rmse": rmse, "mae": mae, "count": int(count),
               "per_rating_rmse": per_rating}

    exp = np.concatenate(exps) if exps else None
    if exp is not None:
        metrics["rmse_exp"] = float(np.sqrt(np.mean((exp - labels) ** 2)))
        blend, alphas = blend_out_of_fold(preds.astype(np.float64),
                                          exp.astype(np.float64),
                                          labels.astype(np.float64),
                                          folds=max(2, blend_folds))
        metrics.update(blend)
        metrics["blend_alphas_per_fold"] = alphas
    return metrics, preds, labels, exp


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="artifact directory from preprocess.py")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--valid-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--dense-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--ordinal-weight", type=float, default=0.0,
                        help="weight of the auxiliary 5-class cross-entropy loss")
    parser.add_argument("--mse-weight", type=float, default=1.0,
                        help="weight of the MSE loss (0 = pure classification)")
    parser.add_argument("--text-mode", default="both",
                        choices=["both", "title", "comment", "none"])
    parser.add_argument("--no-wide", dest="use_wide", action="store_false",
                        help="disable the linear (wide) TF-IDF path")
    parser.add_argument("--text-mlp", action="store_true",
                        help="add a non-linear fusion layer over the text towers")
    parser.add_argument("--film", action="store_true",
                        help="let the numeric features modulate the text towers "
                             "(FiLM: h' = h*(1+gamma)+beta). Zero-initialised, so "
                             "it starts as an exact identity and only adds capacity")
    parser.add_argument("--blend-folds", type=int, default=5,
                        help="folds used to fit the ordinal E[rating] blend weight")
    parser.add_argument("--select-metric", default="auto",
                        choices=["auto", "rmse", "exp", "blend"],
                        help="metric for early stopping / best-checkpoint selection. "
                             "auto reproduces the historical rule (regression RMSE, or "
                             "E[rating] when --mse-weight is 0). 'blend' selects on "
                             "what you would actually submit; it needs an ordinal head")
    parser.add_argument("--dense-mode", default="all",
                        choices=["all", "numeric", "entity", "text_shape",
                                 "interactions", "none"])
    parser.add_argument("--feature-set", default=None, choices=["v1", "v2"],
                        help="override the feature set declared by meta.json "
                             "(default: take it from the cache)")
    parser.add_argument("--interactions", dest="interactions",
                        action="store_true", default=None,
                        help="force-enable the V2 interaction block")
    parser.add_argument("--no-interactions", dest="interactions",
                        action="store_false",
                        help="force-disable the V2 interaction block")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--limit-train-shards", type=int, default=None)
    parser.add_argument("--limit-valid-shards", type=int, default=None)
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--limit-valid-rows", type=int, default=None)
    parser.add_argument("--save-val-preds", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.set_num_threads(max(1, min(args.num_threads, os.cpu_count() or 1)))
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device={device} threads={torch.get_num_threads()}", flush=True)

    meta = load_json(os.path.join(args.data, "meta.json"))
    v2_info = meta.get("v2") or {}
    feature_set = args.feature_set or str(
        meta.get("feature_set", v2_info.get("feature_set", "v1")))
    interactions = (bool(meta.get("interactions", v2_info.get("interactions", False)))
                    if args.interactions is None else bool(args.interactions))
    args.feature_set = feature_set
    args.interactions = interactions
    dense_columns = select_dense_columns(args.dense_mode, feature_set, interactions)
    print(f"feature_set={feature_set} interactions={interactions} "
          f"dense={len(dense_columns)} columns", flush=True)
    use_title, use_comment = text_flags(args.text_mode)
    if not dense_columns and not use_title and not use_comment:
        raise SystemExit("at least one feature group must be enabled")

    try:
        selection_key = resolve_selection_key(
            args.select_metric, args.mse_weight, args.ordinal_weight > 0)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"selecting the best epoch on {selection_key}", flush=True)

    # Written after resolution so the file records the feature set actually used
    # (``--feature-set``/``--interactions`` default to the cache's declaration).
    save_json(vars(args), os.path.join(args.out, "args.json"))

    train_loader = ShardBatcher(
        args.data, "train", dense_columns, batch_size=args.batch_size,
        shuffle=True, seed=args.seed, max_shards=args.limit_train_shards,
        max_rows=args.limit_train_rows, pin_memory=torch.cuda.is_available(),
    )
    valid_loader = ShardBatcher(
        args.data, "valid", dense_columns, batch_size=args.valid_batch_size,
        shuffle=False, seed=args.seed, max_shards=args.limit_valid_shards,
        max_rows=args.limit_valid_rows, pin_memory=torch.cuda.is_available(),
    )
    print(f"train rows={train_loader.num_rows:,} batches={len(train_loader):,}; "
          f"valid rows={valid_loader.num_rows:,}", flush=True)

    model = HybridRatingModel(
        vocab_title=train_loader.vocab_title,
        vocab_comment=train_loader.vocab_comment,
        n_dense=len(dense_columns),
        text_dim=args.text_dim,
        dense_dim=args.dense_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        use_title=use_title,
        use_comment=use_comment,
        use_wide=args.use_wide,
        ordinal=args.ordinal_weight > 0,
        text_mlp=args.text_mlp,
        film=args.film,
        init_rating=float(meta.get("train_rating_mean", 3.8)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(train_loader.steps_per_epoch(), 1)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, div_factor=10.0, final_div_factor=50.0)
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    use_amp = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Constant-mean baseline on the validation split (predict train mean).
    mean_rating = float(meta.get("train_rating_mean", 3.8))
    baseline_rmse = float("nan")

    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        running_sq = 0.0
        running_n = 0
        for step, batch in enumerate(train_loader.batches(epoch=epoch)):
            y = batch["y"].to(device, non_blocking=True)
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if args.ordinal_weight > 0:
                    pred, logits = forward_batch(model, batch, want_logits=True)
                    mse = F.mse_loss(pred, y)
                    loss = args.mse_weight * mse + args.ordinal_weight * F.cross_entropy(
                        logits, y.round().clamp(1, 5).long() - 1)
                else:
                    pred = forward_batch(model, batch)
                    mse = F.mse_loss(pred, y)
                    loss = mse
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_sq += float(mse.detach()) * y.numel()
            running_n += y.numel()
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                      f"train_rmse={np.sqrt(running_sq / running_n):.4f}", flush=True)

        train_rmse = float(np.sqrt(running_sq / max(running_n, 1)))
        metrics, preds, labels, exp = evaluate(model, valid_loader, device,
                                               blend_folds=args.blend_folds)
        if np.isnan(baseline_rmse) and labels.size:
            baseline_rmse = float(np.sqrt(np.mean((labels - mean_rating) ** 2)))
        elapsed = time.time() - epoch_started
        record = {"epoch": epoch, "train_rmse": train_rmse, "val_rmse": metrics["rmse"],
                  "val_mae": metrics["mae"], "seconds": round(elapsed, 2),
                  "lr": optimizer.param_groups[0]["lr"]}
        if "rmse_exp" in metrics:
            record["val_rmse_exp"] = metrics["rmse_exp"]
            record["val_rmse_blend"] = metrics["rmse_blend"]
            record["blend_alpha"] = metrics["blend_alpha"]
        history.append(record)
        extra = ""
        if "rmse_exp" in metrics:
            extra = (f" exp_rmse={metrics['rmse_exp']:.4f} "
                     f"blend_rmse={metrics['rmse_blend']:.4f} "
                     f"alpha={metrics['blend_alpha']:.2f}")
        print(f"epoch {epoch}: train_rmse={train_rmse:.4f} "
              f"val_rmse={metrics['rmse']:.4f} val_mae={metrics['mae']:.4f} "
              f"({elapsed:.1f}s){extra}", flush=True)

        selection_score = metrics[selection_key]

        if selection_score < best_rmse - 1e-5:
            best_rmse = selection_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state": best_state,
                "args": vars(args),
                "dense_columns": dense_columns,
                # Additive key: every V1 checkpoint predates it and simply does
                # not carry it, while predict.py can now refuse a checkpoint that
                # was trained with a different dense schema.
                "dense_schema": {"feature_set": feature_set,
                                 "interactions": interactions,
                                 "columns": dense_columns},
                "text_mode": args.text_mode,
                "vocab_title": train_loader.vocab_title,
                "vocab_comment": train_loader.vocab_comment,
                # Out-of-fold ordinal blend weight of this epoch, so that
                # ``predict.py --blend-alpha auto`` reproduces exactly what was
                # measured here.  ``None`` for a regression-only model.
                "blend_alpha": metrics.get("blend_alpha"),
                "val_rmse": best_rmse,
                "epoch": epoch,
            }, os.path.join(args.out, "best.pt"))
            if args.save_val_preds:
                payload = {"y": labels, "pred": preds}
                if exp is not None:
                    payload["exp"] = exp.astype(np.float32)
                np.savez(os.path.join(args.out, "val_preds.npz"), **payload)
        elif epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    total_seconds = time.time() - started
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": selection_key,
        "best_val_rmse": best_rmse,
        "baseline_mean_rmse": baseline_rmse,
        "improvement_vs_mean": baseline_rmse - best_rmse,
        "total_seconds": round(total_seconds, 2),
        "history": history,
        "n_parameters": n_params,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(),
        "config": vars(args),
    }
    save_json(summary, os.path.join(args.out, "summary.json"))
    print(f"best val RMSE={best_rmse:.4f} (epoch {best_epoch}); "
          f"mean baseline={baseline_rmse:.4f}; total {total_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
