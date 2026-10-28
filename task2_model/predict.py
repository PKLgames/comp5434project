#!/usr/bin/env python
"""Run a trained checkpoint on the validation or test split.

For validation the script reports RMSE/MAE and optionally stores
``id, y, pred`` arrays.  For the test split (once Part A produces
``test_features_with_entity``) it writes a Kaggle submission CSV.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from common import load_json, save_json
from dataset import ShardBatcher
from model import HybridRatingModel


def load_checkpoint(checkpoint: str) -> dict:
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 has no weights_only
        return torch.load(checkpoint, map_location="cpu")


def build_model(ckpt: dict, device: torch.device) -> HybridRatingModel:
    cfg = ckpt["args"]
    model = HybridRatingModel(
        vocab_title=ckpt["vocab_title"],
        vocab_comment=ckpt["vocab_comment"],
        n_dense=len(ckpt["dense_columns"]),
        text_dim=cfg["text_dim"],
        dense_dim=cfg["dense_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        emb_dropout=cfg.get("emb_dropout", 0.1),
        use_title=ckpt["text_mode"] in ("both", "title"),
        use_comment=ckpt["text_mode"] in ("both", "comment"),
        use_wide=cfg.get("use_wide", True),
        ordinal=cfg.get("ordinal_weight", 0.0) > 0,
        text_mlp=cfg.get("text_mlp", False),
        film=cfg.get("film", False),
        init_rating=3.8,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def load_model(checkpoint: str, device: torch.device) -> HybridRatingModel:
    return build_model(load_checkpoint(checkpoint), device)


@torch.no_grad()
def predict_loader(model: HybridRatingModel, loader: ShardBatcher,
                   device: torch.device):
    """Return ``(pred, exp, y)`` over one loader.

    ``exp`` is ``E[rating] = sum_k k * p_k`` from the auxiliary ordinal head, or
    ``None`` when the model has no such head.  ``y`` is ``None`` for the test
    split.  Row order is the loader's, so every model and every array here line
    up exactly.
    """

    ordinal = bool(getattr(model, "ordinal", False))
    preds: List[np.ndarray] = []
    exps: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader.batches(epoch=0):
        label = batch.pop("y", None)
        if label is not None:
            labels.append(label.numpy())
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        args = (batch["t_idx"], batch["t_off"], batch["t_val"],
                batch["c_idx"], batch["c_off"], batch["c_val"], batch["dense"])
        if ordinal:
            pred, logits = model.forward_with_logits(*args)
            levels = torch.arange(1, 6, device=device, dtype=torch.float32)
            exps.append((torch.softmax(logits.float(), dim=1) * levels)
                        .sum(dim=1).cpu().numpy().astype(np.float32))
        else:
            pred = model(*args)
        preds.append(pred.detach().cpu().numpy().astype(np.float32))

    pred = np.concatenate(preds) if preds else np.empty(0, np.float32)
    exp = np.concatenate(exps) if exps else None
    y = np.concatenate(labels) if labels else None
    return pred, exp, y


def predict_split(model: HybridRatingModel, loader: ShardBatcher,
                  device: torch.device):
    pred, exp, _ = predict_loader(model, loader, device)
    return pred, exp


def resolve_blend_alpha(spec, ckpt: dict):
    """Resolve ``--blend-alpha`` into ``(alpha, source_description)``.

    ``auto`` (the default) reads the out-of-fold weight that ``train.py`` stored
    in the checkpoint, so scoring reproduces exactly what was measured.  A
    regression-only checkpoint has no such key and gets ``0.0``, which makes the
    call a no-op and keeps every V1 prediction bit-identical.
    """

    if spec is None or (isinstance(spec, str) and spec.lower() == "auto"):
        stored = ckpt.get("blend_alpha")
        if stored is None:
            return 0.0, "none (regression-only checkpoint)"
        return float(stored), "checkpoint (out-of-fold)"
    return float(spec), "command line"


def check_dense_schema(ckpt: dict, loader: ShardBatcher) -> None:
    """Refuse to score a checkpoint against the wrong dense columns.

    The dense tower's first layer is sized by ``len(dense_columns)``, so a width
    mismatch crashes inside the first matmul.  The dangerous case is a *silent*
    one: two checkpoints with the same width but trained on different column
    subsets (e.g. ``--dense-mode numeric`` vs ``all``).  Comparing the full
    ordered column list catches both.
    """

    trained = [str(c) for c in ckpt["dense_columns"]]
    provided = [str(c) for c in loader.dense_index_names]
    if trained != provided:
        raise SystemExit(
            "checkpoint/cache mismatch: the checkpoint was trained on "
            f"{len(trained)} dense columns {trained[:4]}... but {loader.artifacts} "
            f"resolves to {len(provided)} {provided[:4]}...; point --data at the "
            "cache this checkpoint was trained with")


def check_checkpoints_agree(checkpoints: Sequence[str]) -> List[str]:
    """All members of an ensemble must use the same dense column list."""

    first = [str(c) for c in load_checkpoint(checkpoints[0])["dense_columns"]]
    for path in checkpoints[1:]:
        other = [str(c) for c in load_checkpoint(path)["dense_columns"]]
        if other != first:
            raise SystemExit(
                f"cannot average {checkpoints[0]} and {path}: they expect "
                f"different dense columns ({len(first)} vs {len(other)}). Blend "
                "checkpoints with stack.py, which builds one loader per model")
    return first


def read_ids(parquet_dir: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet parts under {parquet_dir}")
    ids = []
    for path in files:
        table = pq.read_table(path, columns=["id"])
        ids.append(np.asarray(table.column("id").combine_chunks().to_pylist(), dtype=object))
    return np.concatenate(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="artifact dir used for training")
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--checkpoint", required=True, action="append",
                        help="checkpoint path; repeat the flag to average an ensemble")
    parser.add_argument("--out", required=True, help="output .npz (valid) or .csv (test)")
    parser.add_argument("--ids-dir", default=None,
                        help="original parquet dir used to recover ids (test requires it)")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--weights", default=None,
                        help="comma-separated blend weights, one per checkpoint "
                             "(default: equal weights)")
    parser.add_argument("--intercept", type=float, default=0.0,
                        help="optional intercept added after blending")
    parser.add_argument("--blend-alpha", default="auto",
                        help="weight of E[rating] in the final prediction, i.e. "
                             "(1-a)*regression + a*expectation. 'auto' (default) uses "
                             "the out-of-fold value stored in the checkpoint; a "
                             "regression-only checkpoint resolves to 0.0, which "
                             "leaves its predictions bit-identical")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))

    # One loader serves every member, so they must agree on the dense columns;
    # blending models trained on different schemas needs stack.py instead.
    dense_columns = check_checkpoints_agree(args.checkpoint)
    models = [load_model(path, device) for path in args.checkpoint]
    ckpt = load_checkpoint(args.checkpoint[0])

    try:
        loader = ShardBatcher(args.data, args.split, dense_columns,
                              batch_size=args.batch_size, shuffle=False,
                              max_shards=args.max_shards,
                              pin_memory=torch.cuda.is_available())
    except ValueError as exc:
        raise SystemExit(f"cannot score this checkpoint with {args.data}: {exc}")
    check_dense_schema(ckpt, loader)
    print(f"predicting {loader.num_rows:,} rows on {device} "
          f"with {len(models)} model(s)", flush=True)

    outputs = [predict_split(model, loader, device) for model in models]
    parts = [pred for pred, _ in outputs]
    exps = [exp for _, exp in outputs]
    if args.weights:
        weights = np.asarray([float(w) for w in args.weights.split(",")], dtype=np.float32)
        if weights.size != len(parts):
            raise SystemExit("--weights must have one value per checkpoint")
        # Used verbatim: the published ensemble recipe relies on the weights
        # summing to < 1 together with an explicit intercept (0.477 / 0.484 with
        # +0.162), so normalising them here would change the reported number.
        weight_of = [float(w) for w in weights]
    else:
        weight_of = [1.0 / len(parts)] * len(parts)
    preds = sum(w * part for w, part in zip(weight_of, parts))

    # The ordinal expectation is blended the same way as the regression output,
    # so an ensemble of ordinal models stays consistent.  If any member lacks the
    # head, the expectation is unavailable and the blend is skipped.
    exp_preds = None
    if all(exp is not None for exp in exps):
        exp_preds = sum(w * exp for w, exp in zip(weight_of, exps))

    alpha, alpha_source = resolve_blend_alpha(args.blend_alpha, ckpt)
    if exp_preds is not None and alpha > 0.0:
        print(f"blending with E[rating]: alpha={alpha:.5f} ({alpha_source})", flush=True)
        preds = (1.0 - alpha) * preds + alpha * exp_preds
    elif exp_preds is not None:
        print(f"using the regression head only (alpha={alpha:.5f}, {alpha_source})",
              flush=True)

    preds = np.clip(preds + args.intercept, 1.0, 5.0).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.split == "valid":
        meta = load_json(os.path.join(args.data, "meta.json"))
        # ``loader.shard_dir`` may point at the read-only V1 cache when the
        # checkpoint was trained on a V2 overlay, so never rebuild the path here.
        y = []
        for shard in meta["splits"]["valid"]["shards"][: args.max_shards]:
            with np.load(os.path.join(loader.shard_dir, shard["file"])) as npz:
                y.append(npz["y"])
        y = np.concatenate(y).astype(np.float32)
        assert y.shape == preds.shape, (y.shape, preds.shape)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        mae = float(np.mean(np.abs(preds - y)))
        payload = {"pred": preds, "y": y}
        if exp_preds is not None:
            payload["exp"] = exp_preds.astype(np.float32)
        if args.ids_dir:
            payload["id"] = read_ids(args.ids_dir)
        np.savez(args.out, **payload)
        metrics = {"rmse": rmse, "mae": mae, "rows": int(y.size),
                   "blend_alpha": alpha, "blend_alpha_source": alpha_source}
        if exp_preds is not None:
            metrics["rmse_exp_only"] = float(np.sqrt(np.mean((exp_preds - y) ** 2)))
        save_json(metrics, os.path.splitext(args.out)[0] + "_metrics.json")
        print(f"val RMSE={rmse:.4f} MAE={mae:.4f} rows={y.size:,}")
    else:
        if not args.ids_dir:
            raise SystemExit("--ids-dir is required for test predictions")
        ids = read_ids(args.ids_dir)
        assert ids.size == preds.size, (ids.size, preds.size)
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "rating"])
            for review_id, rating in zip(ids, preds):
                writer.writerow([review_id, f"{float(rating):.4f}"])
        print(f"wrote {preds.size:,} predictions to {args.out}")


if __name__ == "__main__":
    main()
