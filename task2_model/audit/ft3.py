"""Fine-tune MiniLM on recovered clean labels, then predict valid + test.

Usage (on Colab, run detached):
  python ft3.py --epochs 2 --bs 512 --maxlen 96 --seed 1 --tag s1

Artifacts in /content:
  ft3_<tag>.pt            model state
  ftv_<tag>.npy           predictions on the *validation* slice (id % 40 == 0), raw ids, all rows
  ftp_<tag>.npy           predictions on the real test set (test_text.parquet, 10k rows)
  ftvids_<tag>.npy        id array for ftv_<tag>.npy
"""
import argparse, os, sys, time
import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

P = argparse.ArgumentParser()
P.add_argument("--epochs", type=int, default=2)
P.add_argument("--bs", type=int, default=512)
P.add_argument("--lr", type=float, default=2e-5)
P.add_argument("--maxlen", type=int, default=96)
P.add_argument("--seed", type=int, default=1)
P.add_argument("--limit", type=int, default=0)
P.add_argument("--tag", default="s1")
P.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
P.add_argument("--evalsub", type=int, default=40000)
A = P.parse_args()

torch.manual_seed(A.seed)
np.random.seed(A.seed)
torch.backends.cuda.matmul.allow_tf32 = True
T0 = time.time()
def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)

NAME = A.model
tok = AutoTokenizer.from_pretrained(NAME)

log("loading text parquet")
tab = pq.read_table("/content/train_text.parquet")
ids = tab["id"].to_numpy()
texts = tab["text"].to_pylist()
log("rows", len(texts))

d = np.load("/content/clean_y.npz")
cy = d["cy"].astype(np.float32)
matched = d["matched"]
assert len(cy) == len(texts)

va = (ids % 40) == 0
tr = (~va) & matched[ids] & (cy[ids] > 0)
tr_idx = np.nonzero(tr)[0]
if A.limit:
    tr_idx = tr_idx[: A.limit]
va_idx = np.nonzero(va)[0]
va_m = np.nonzero(va & matched[ids] & (cy[ids] > 0))[0]
log("train", len(tr_idx), "valid_all", len(va_idx), "valid_matched", len(va_m))

rng = np.random.default_rng(A.seed)
va_eval = rng.choice(va_m, size=min(A.evalsub, len(va_m)), replace=False)


class FT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(NAME)
        h = self.enc.config.hidden_size
        self.head = nn.Linear(h, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 4.12)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(out[:, 0]).squeeze(1)


model = FT().cuda()
dev = "cuda"

@torch.no_grad()
def predict(idx, bs=1024):
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), bs):
        j = idx[s : s + bs]
        b = tok([texts[i] for i in j], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.float16):
            y = model(**b)
        out[s : s + len(j)] = y.float().cpu().numpy()
    model.train()
    return out


def rmse(p, t):
    return float(np.sqrt(np.mean((np.clip(p, 1, 5) - t) ** 2)))


steps_per_epoch = (len(tr_idx) + A.bs - 1) // A.bs
total = steps_per_epoch * A.epochs
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
lossf = nn.MSELoss()
log("steps/epoch", steps_per_epoch, "total", total)

step = 0
for ep in range(A.epochs):
    order = rng.permutation(tr_idx)
    t_ep = time.time()
    run = 0.0
    for s in range(0, len(order), A.bs):
        j = order[s : s + A.bs]
        b = tok([texts[i] for i in j], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        t = torch.from_numpy(cy[j]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            y = model(**b)
            loss = lossf(y.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run = 0.95 * run + 0.05 * float(loss)
        if step % 200 == 0:
            sp = (time.time() - t_ep) / step
            log("ep%d step %d/%d loss %.4f %.2f it/s eta %.0f min" % (ep, step, total, run, 1 / sp, (total - step) * sp / 60))
    ev = predict(va_eval)
    log("EPOCH %d done loss %.4f | evalsub clean rmse %.4f" % (ep, run, rmse(ev, cy[va_eval])))

log("saving")
torch.save({"state": model.state_dict(), "args": vars(A)}, "/content/ft3_%s.pt" % A.tag)

pv = predict(va_idx)
np.save("/content/ftv_%s.npy" % A.tag, pv)
np.save("/content/ftvids_%s.npy" % A.tag, ids[va_idx])
m = matched[ids[va_idx]] & (cy[ids[va_idx]] > 0)
log("FINAL valid masked rmse %.4f (n=%d) | all-valid %.2f" % (rmse(pv[m], cy[ids[va_idx]][m]), m.sum(), pv.mean()))

tt = pq.read_table("/content/test_text.parquet")
ttexts = tt["text"].to_pylist()
order = list(range(len(ttexts)))
tp = np.empty(len(ttexts), dtype=np.float32)
model.eval()
with torch.no_grad():
    for s in range(0, len(order), 1024):
        j = order[s : s + 1024]
        b = tok([ttexts[i] for i in j], padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.float16):
            tp[s : s + len(j)] = model(**b).float().cpu().numpy()
np.save("/content/ftp_%s.npy" % A.tag, tp)
log("test preds", len(tp), "mean %.3f sd %.3f" % (tp.mean(), tp.std()))
log("DONE")
