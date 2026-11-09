"""Fine-tune a HuggingFace encoder on the recovered clean ratings.

Training recipe for the four-member ensemble: an `AutoModel`
encoder + a single linear head on the CLS token, MSE against `cy`, trained on all
non-validation rows that have a recovered label.

The four members of the final ensemble (each command is ~35-60 min on a GPU):

  python finetune.py --model sentence-transformers/all-MiniLM-L6-v2 \
      --maxlen 96  --bs 512  --seed 1 --tag s1
  python finetune.py --model sentence-transformers/all-MiniLM-L6-v2 \
      --maxlen 128 --bs 1024 --seed 2 --tag s2
  python finetune.py --model distilbert-base-uncased \
      --maxlen 128 --bs 768  --seed 1 --tag d1
  python finetune.py --model BAAI/bge-small-en-v1.5 \
      --maxlen 128 --bs 768  --seed 1 --tag b1

Outputs in --out-dir
  ft3_<tag>.pt        model state (state_dict + args)
  ftv_<tag>.npy       predictions on the validation slice (id % 40 == 0)
  ftvids_<tag>.npy    validation ids for ftv_<tag>.npy
  ftp_<tag>.npy       predictions on the 10,000 real test rows
  ft3_<tag>.json      metrics + environment snapshot

Validation protocol: rows with id % 40 == 0 are held out, so train/valid never
overlap and the test set is only ever read for inference.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

P = argparse.ArgumentParser()
P.add_argument("--train-text", default="work/train_text.parquet")
P.add_argument("--test-text", default="work/test_text.parquet")
P.add_argument("--clean", "--labels", dest="clean", required=True)
P.add_argument("--out-dir", default="work")
P.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
P.add_argument("--epochs", type=int, default=2)
P.add_argument("--bs", type=int, default=512)
P.add_argument("--lr", type=float, default=2e-5)
P.add_argument("--maxlen", type=int, default=96)
P.add_argument("--seed", type=int, default=1)
P.add_argument("--tag", default="s1")
P.add_argument("--evalsub", type=int, default=40000, help="size of the per-epoch validation subsample")
P.add_argument("--limit", type=int, default=0, help="cap training rows (smoke test)")
P.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
P.add_argument("--predict-bs", type=int, default=1024)
A = P.parse_args()

os.makedirs(A.out_dir, exist_ok=True)
if Path(A.out_dir, "ft3_%s.pt" % A.tag).exists(): raise FileExistsError("Checkpoint already exists")
for filename in (A.train_text, A.test_text, A.clean):
    if not Path(filename).is_file(): raise FileNotFoundError(filename)
torch.manual_seed(A.seed)
np.random.seed(A.seed)
dev = ("cuda" if torch.cuda.is_available() else "cpu") if A.device == "auto" else A.device
print("torch", torch.__version__, "| python", platform.python_version(), "| transformers",
      __import__("transformers").__version__, "| device", dev, flush=True)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    print("gpu", torch.cuda.get_device_name(0), flush=True)

T0 = time.time()
EPOCH_TIMINGS = []
def sync():
    if dev == "cuda": torch.cuda.synchronize()


def log(*x):
    print("[%6.0fs]" % (time.time() - T0), *x, flush=True)


log("loading text parquet")
tab = pq.read_table(A.train_text)
ids = tab["id"].to_numpy()
texts = tab["text"].to_pylist()
log("rows", len(texts))

d = np.load(A.clean)
cy_full = d["cy"].astype(np.float32)
matched_full = d["matched"]
# Source labels are indexed by original review ID, including subset runs.
if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(cy_full):
    raise ValueError("Review IDs do not align with the source label array")
cy, matched = cy_full[ids], matched_full[ids]
log("clean labels", int(matched.sum()), "matched")

va = (ids % 40) == 0
tr = (~va) & matched & (cy > 0)
tr_idx = np.nonzero(tr)[0]
if A.limit:
    tr_idx = tr_idx[: A.limit]
va_idx = np.nonzero(va)[0]
va_m = np.nonzero(va & matched & (cy > 0))[0]
log("train", len(tr_idx), "valid_all", len(va_idx), "valid_matched", len(va_m))
if len(va_m) == 0: raise ValueError("No matched validation records")
if len(tr_idx) == 0:
    raise SystemExit("no training rows: is clean_y.npz covering these ids?")

rng = np.random.default_rng(A.seed)
va_eval = rng.choice(va_m, size=min(A.evalsub, len(va_m)), replace=False)

NAME = A.model
tok = AutoTokenizer.from_pretrained(NAME)


class FT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(NAME)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 4.12)  # clean-label mean is ~4.12

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(out[:, 0]).squeeze(1)


model = FT().to(dev)


def rmse(p, t):
    return float(np.sqrt(np.mean((np.clip(p, 1, 5) - t) ** 2)))


@torch.no_grad()
def predict(idx, bs=None):
    bs = A.predict_bs if bs is None else bs
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), bs):
        j = idx[s : s + bs]
        b = tok([texts[i] for i in j], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            y = model(**b)
        out[s : s + len(j)] = y.float().cpu().numpy()
    return out


steps_per_epoch = (len(tr_idx) + A.bs - 1) // A.bs
total = steps_per_epoch * A.epochs
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
lossf = nn.MSELoss()
log("steps/epoch", steps_per_epoch, "total", total)

step = 0
for ep in range(A.epochs):
    model.train()
    sync()
    epoch_start = time.perf_counter()
    order = rng.permutation(tr_idx)
    t_ep = time.time()
    run = 0.0
    for s in range(0, len(order), A.bs):
        j = order[s : s + A.bs]
        b = tok([texts[i] for i in j], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        t = torch.from_numpy(cy[j]).to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            y = model(**b)
            loss = lossf(y.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run = 0.95 * run + 0.05 * float(loss.detach())
        if step % 200 == 0:
            dt = (time.time() - t_ep) / max(1, step - ep * steps_per_epoch)
            log("ep%d step %d/%d loss %.4f %.2f it/s eta %.0f min"
                % (ep, step, total, run, 1 / dt, (total - step) * dt / 60))
    sync()
    train_seconds = time.perf_counter() - epoch_start
    ev = predict(va_eval)
    sync()
    EPOCH_TIMINGS.append({"epoch": ep + 1, "train_seconds": train_seconds, "with_validation_seconds": time.perf_counter() - epoch_start})
    Path(A.out_dir, "timing_%s.json" % A.tag).write_text(json.dumps(EPOCH_TIMINGS, indent=2), encoding="utf-8")
    log("EPOCH %d done in %.0fs | loss %.4f | evalsub clean rmse %.4f"
        % (ep, time.time() - t_ep, run, rmse(ev, cy[va_eval])))

log("saving weights")
torch.save({"state": model.state_dict(), "args": vars(A)}, os.path.join(A.out_dir, "ft3_%s.pt" % A.tag))

log("predicting validation slice")
pv = predict(va_idx)
np.save(os.path.join(A.out_dir, "ftv_%s.npy" % A.tag), pv)
np.save(os.path.join(A.out_dir, "ftvids_%s.npy" % A.tag), ids[va_idx])
m = matched[va_idx] & (cy[va_idx] > 0)
final = rmse(pv[m], cy[va_idx][m])
log("FINAL valid masked rmse %.4f (n=%d)" % (final, int(m.sum())))

log("predicting test set")
tt = pq.read_table(A.test_text)
ttexts = tt["text"].to_pylist()
tp = np.empty(len(ttexts), dtype=np.float32)
model.eval()
with torch.no_grad():
    for s in range(0, len(ttexts), A.predict_bs):
        j = list(range(s, min(s + A.predict_bs, len(ttexts))))
        b = tok([ttexts[i] for i in j], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            tp[s : s + len(j)] = model(**b).float().cpu().numpy()
np.save(os.path.join(A.out_dir, "ftp_%s.npy" % A.tag), tp)
log("test preds", len(tp), "mean %.3f sd %.3f" % (tp.mean(), tp.std()))

json.dump(
    {
        "tag": A.tag,
        "model": NAME,
        "epochs": A.epochs,
        "bs": A.bs,
        "lr": A.lr,
        "maxlen": A.maxlen,
        "seed": A.seed,
        "train_rows": int(len(tr_idx)),
        "valid_rows": int(len(va_idx)),
        "valid_matched": int(len(va_m)),
        "valid_rmse_masked": final,
        "wall_s": round(time.time() - T0, 1),
        "epoch_timings": EPOCH_TIMINGS,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": dev,
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if dev == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
        },
    },
    open(os.path.join(A.out_dir, "ft3_%s.json" % A.tag), "w"),
    indent=2,
)
log("DONE")
