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
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch

from common import load_json, save_json
from dataset import ShardBatcher
from model import HybridRatingModel


def load_model(checkpoint: str, device: torch.device) -> HybridRatingModel:
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 has no weights_only
        ckpt = torch.load(checkpoint, map_location="cpu")
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
        init_rating=3.8,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_split(model: HybridRatingModel, loader: ShardBatcher,
                  device: torch.device) -> np.ndarray:
    preds: List[np.ndarray] = []
    for batch in loader.batches(epoch=0):
        y = batch.pop("y", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model(batch["t_idx"], batch["t_off"], batch["t_val"],
                     batch["c_idx"], batch["c_off"], batch["c_val"],
                     batch["dense"])
        preds.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds) if preds else np.empty(0, np.float32)


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
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    models = [load_model(path, device) for path in args.checkpoint]

    try:
        ckpt = torch.load(args.checkpoint[0], map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint[0], map_location="cpu")
    dense_columns = ckpt["dense_columns"]

    loader = ShardBatcher(args.data, args.split, dense_columns,
                          batch_size=args.batch_size, shuffle=False,
                          max_shards=args.max_shards)
    print(f"predicting {loader.num_rows:,} rows on {device} "
          f"with {len(models)} model(s)", flush=True)
    parts = [predict_split(model, loader, device) for model in models]
    if args.weights:
        weights = np.asarray([float(w) for w in args.weights.split(",")], dtype=np.float32)
        if weights.size != len(parts):
            raise SystemExit("--weights must have one value per checkpoint")
        preds = sum(weight * part for weight, part in zip(weights, parts))
    else:
        preds = np.mean(parts, axis=0)
    preds = np.clip(preds + args.intercept, 1.0, 5.0).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.split == "valid":
        meta = load_json(os.path.join(args.data, "meta.json"))
        y = []
        for shard in meta["splits"]["valid"]["shards"][: args.max_shards]:
            with np.load(os.path.join(args.data, "valid", shard["file"])) as npz:
                y.append(npz["y"])
        y = np.concatenate(y).astype(np.float32)
        assert y.shape == preds.shape, (y.shape, preds.shape)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        mae = float(np.mean(np.abs(preds - y)))
        payload = {"pred": preds, "y": y}
        if args.ids_dir:
            payload["id"] = read_ids(args.ids_dir)
        np.savez(args.out, **payload)
        metrics = {"rmse": rmse, "mae": mae, "rows": int(y.size)}
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
