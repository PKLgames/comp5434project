"""Encode a subsample of the corpus with a sentence encoder (CPU pilot).

Writes float16 embedding matrices for
  * train rows with id % 20 == 0   (384,232 rows)
  * validation rows with id % 40 == 0 (192,117 rows)
so that an embeddings-vs-bag-of-words comparison can be run at equal sample size
before paying for a full 7.5M-row encode.

Usage:
  python -m task2_model.audit.encode_emb --out /tmp/emb
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pyarrow.csv as pcsv
import torch
from transformers import AutoModel, AutoTokenizer

NAME = "sentence-transformers/all-MiniLM-L6-v2"
CSV = "/home/pklgames/coding/comp5434project/Dataset2_metadata/train.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/emb")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--train-mod", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)

    tk = AutoTokenizer.from_pretrained(NAME)
    mdl = AutoModel.from_pretrained(NAME).eval()

    rdr = pcsv.open_csv(
        CSV,
        read_options=pcsv.ReadOptions(block_size=1 << 24),
        convert_options=pcsv.ConvertOptions(include_columns=["id", "title", "comment"],
                                            column_types={"title": "string", "comment": "string"}),
    )
    t0 = time.time()
    bufs = {"train": [], "valid": []}
    ids = {"train": [], "valid": []}
    done = 0

    def flush(texts, key):
        enc = tk(texts, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        with torch.inference_mode():
            out = mdl(**enc)
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (h * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, dim=1)
        bufs[key].append(emb.numpy().astype(np.float16))

    for b in rdr:
        idv = b["id"].to_numpy(zero_copy_only=False)
        ti = b["title"].to_pylist()
        co = b["comment"].to_pylist()
        for key, mask_fn in (("train", lambda v: v % a.train_mod == 0), ("valid", lambda v: v % 40 == 0)):
            sel = mask_fn(idv)
            if sel.any():
                texts = [((t or "") + " " + (c or "")).strip() for t, c in zip(ti, co)]
                texts = [t for t, s in zip(texts, sel) if s]
                ids[key].append(idv[sel])
                for i in range(0, len(texts), a.batch):
                    flush(texts[i:i + a.batch], key)
        done += len(idv)
        print("[%6.0fs] %d rows scanned, train %d valid %d" % (
            time.time() - t0, done, sum(len(x) for x in bufs["train"]),
            sum(len(x) for x in bufs["valid"])), flush=True)

    for key in ("train", "valid"):
        E = np.concatenate(bufs[key])
        I = np.concatenate(ids[key])
        np.save(os.path.join(a.out, f"{key}_emb.npy"), E)
        np.save(os.path.join(a.out, f"{key}_ids.npy"), I)
        print("[out] %s %s %s" % (key, E.shape, I.shape), flush=True)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
