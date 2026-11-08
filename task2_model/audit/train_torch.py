"""Streaming trainer / predictor over the sharded hashed-text caches.

Model: EmbeddingBag(sum) on comment hashed 1-2 grams + EmbeddingBag(sum) on title
       hashed 1-2 grams + small MLP on the 9 review numerics -> scalar rating.

No fitted vocabulary, no identity features -> nothing to leak across splits.

  train: python -m task2_model.audit.train_torch --train /tmp/abl2/train_full \
             --valid /tmp/abl2/valid --epochs 2 --out /tmp/abl2/model.pt
  pred : python -m task2_model.audit.train_torch --predict /tmp/abl2/test \
             --ckpt /tmp/abl2/model.pt --submission /tmp/abl2/submission.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn


DENSE_MU = np.zeros(9, np.float32)
DENSE_SD = np.ones(9, np.float32)


def shard_files(d, pattern="shard_*.npz"):
    fs = sorted(glob.glob(os.path.join(d, pattern)))
    if not fs:
        raise SystemExit(f"no shards in {d}")
    return fs


def iter_batches(path, batch_rows, shuffle=False, seed=0):
    d = np.load(path)
    n = d["y"].shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if shuffle else np.arange(n)
    c_ptr, c_idx, c_dat = d["c_indptr"], d["c_indices"], d["c_data"]
    t_ptr, t_idx, t_dat = d["t_indptr"], d["t_indices"], d["t_data"]
    dense, y, ids = d["dense"], d["y"], d["ids"]
    for s in range(0, n, batch_rows):
        rows = order[s:s + batch_rows]
        # rows are contiguous only when not shuffled; handle both
        lo, hi = int(rows.min()), int(rows.max()) + 1
        sl = slice(lo, hi)
        keep = np.zeros(hi - lo, dtype=bool)
        keep[rows - lo] = True
        sub = np.nonzero(keep)[0]
        # coalesce contiguous runs
        breaks = np.nonzero(np.diff(sub) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(sub)]])
        for a, b in zip(starts, ends):
            r0, r1 = sub[a], sub[b - 1] + 1
            gi = np.arange(r0, r1) + lo
            yield {
                "c": (c_idx[c_ptr[r0]:c_ptr[r1]], c_dat[c_ptr[r0]:c_ptr[r1]], c_ptr[r0:r1 + 1] - c_ptr[r0]),
                "t": (t_idx[t_ptr[r0]:t_ptr[r1]], t_dat[t_ptr[r0]:t_ptr[r1]], t_ptr[r0:r1 + 1] - t_ptr[r0]),
                "dense": dense[gi], "y": y[gi].astype(np.float32), "ids": ids[gi],
            }


def to_tensors(b):
    def bag(arrs):
        idx, dat, ptr = arrs
        return (torch.from_numpy(idx.astype(np.int64)),
                torch.from_numpy(dat.astype(np.float32)),
                torch.from_numpy(ptr[:-1].astype(np.int64)))
    dense = (b["dense"] - DENSE_MU) / DENSE_SD
    return bag(b["c"]), bag(b["t"]), torch.from_numpy(dense), \
        torch.from_numpy(b["y"]), b["ids"]


class RatingNet(nn.Module):
    def __init__(self, c_dim, t_dim, dense_dim, dim=128, hidden=128, dropout=0.1):
        super().__init__()
        self.emb_c = nn.EmbeddingBag(c_dim, dim, mode="sum")
        self.emb_t = nn.EmbeddingBag(t_dim, dim, mode="sum")
        for e in (self.emb_c, self.emb_t):
            nn.init.normal_(e.weight, std=0.02)
        self.wc = nn.EmbeddingBag(c_dim, 1, mode="sum")
        self.wt = nn.EmbeddingBag(t_dim, 1, mode="sum")
        for w in (self.wc.weight, self.wt.weight):
            nn.init.zeros_(w)          # wide path starts at zero; it owns no upstream layer
        self.dn = nn.Sequential(nn.Linear(dim * 2 + dense_dim, hidden), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.normal_(self.dn[-1].weight, std=0.01)   # zero-init here would kill the
        nn.init.constant_(self.dn[-1].bias, 3.842)       # gradient to every upstream layer

    def forward(self, c, t, dense):
        cw = self.emb_c(c[0], c[2], per_sample_weights=c[1])
        tw = self.emb_t(t[0], t[2], per_sample_weights=t[1])
        wide = self.wc(c[0], c[2], per_sample_weights=c[1]) + \
            self.wt(t[0], t[2], per_sample_weights=t[1])
        return self.dn(torch.cat([cw, tw, dense], dim=1)).squeeze(1) + wide.squeeze(1)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 1, 5) - y) ** 2)))


def evaluate(model, files, batch_rows=32768):
    model.eval()
    err, n = 0.0, 0
    with torch.no_grad():
        for f in files:
            for b in iter_batches(f, batch_rows):
                c, t, dense, y, _ = to_tensors(b)
                p = torch.clamp(model(c, t, dense), 1, 5)
                err += float(((p - y) ** 2).sum())
                n += len(y)
    return math.sqrt(err / max(n, 1))


def train(args):
    torch.manual_seed(0)
    torch.set_num_threads(8)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = RatingNet(2 ** args.hash_comment, 2 ** args.hash_title, 9,
                      dim=args.dim, hidden=args.hidden, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    files = shard_files(args.train)
    print(f"[train] {len(files)} shards", flush=True)
    rng = np.random.default_rng(0)
    global DENSE_MU, DENSE_SD
    sample = np.concatenate([np.load(f)["dense"] for f in files[:2]]).astype(np.float64)
    DENSE_MU = sample.mean(0).astype(np.float32)
    DENSE_SD = np.maximum(sample.std(0), 0.05).astype(np.float32)
    print(f"[norm] dense mu {np.round(DENSE_MU,2)}", flush=True)
    best = 9.9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, cnt = 0.0, 0
        order = list(files)
        rng.shuffle(order)
        for f in order:
            for b in iter_batches(f, args.batch_rows):
                c, t, dense, y, _ = to_tensors(b)
                opt.zero_grad()
                p = model(c, t, dense)
                loss = ((p - y) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss) * len(y)
                cnt += len(y)
        msg = f"[ep{ep}] train rmse {math.sqrt(tot/cnt):.4f} ({time.time()-t0:.0f}s)"
        if args.valid:
            v = evaluate(model, shard_files(args.valid))
            msg += f" | val rmse {v:.4f}"
            if v < best:
                best = v
                torch.save({"state": model.state_dict(), "args": vars(args), "val": v,
                            "dense_mu": DENSE_MU, "dense_sd": DENSE_SD}, args.out)
                msg += " *saved*"
        print(msg, flush=True)
    print(f"[done] best val {best:.4f}", flush=True)


def predict(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model = RatingNet(2 ** a.hash_comment, 2 ** a.hash_title, 9,
                      dim=a.dim, hidden=a.hidden, dropout=a.dropout)
    model.load_state_dict(ck["state"])
    global DENSE_MU, DENSE_SD
    DENSE_MU, DENSE_SD = ck["dense_mu"], ck["dense_sd"]
    model.eval()
    rows = []
    with torch.no_grad():
        for f in shard_files(args.predict):
            for b in iter_batches(f, 32768):
                c, t, dense, _, ids = to_tensors(b)
                p = torch.clamp(model(c, t, dense), 1, 5).numpy()
                rows.append(np.column_stack([ids, p]))
    out = np.concatenate(rows)
    out = out[np.argsort(out[:, 0])]
    with open(args.submission, "w") as fh:
        fh.write("id,rating\n")
        for i, p in out:
            fh.write(f"{int(i)},{p:.6f}\n")
    print(f"[pred] {len(out)} rows -> {args.submission}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="")
    ap.add_argument("--valid", default="")
    ap.add_argument("--predict", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--submission", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-rows", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hash-comment", type=int, default=17)
    ap.add_argument("--hash-title", type=int, default=15)
    args = ap.parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
