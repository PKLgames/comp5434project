"""Benchmark a sentence encoder on CPU before committing to a feature build."""
import os
import time

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoTokenizer, AutoModel

torch.set_num_threads(8)
NAME = "sentence-transformers/all-MiniLM-L6-v2"

t0 = time.time()
tk = AutoTokenizer.from_pretrained(NAME)
mdl = AutoModel.from_pretrained(NAME).eval()
print("load %.1fs  %.1fM params" % (time.time() - t0, sum(p.numel() for p in mdl.parameters()) / 1e6))

docs = ["This coffee is absolutely terrible, tasted burnt and arrived stale. Waste of money."] * 32 + ["Five Stars"] * 32
enc = tk(docs, padding=True, truncation=True, max_length=96, return_tensors="pt")
with torch.inference_mode():
    t0 = time.time()
    out = mdl(**enc)
    dt = time.time() - t0
h = out.last_hidden_state
mask = enc["attention_mask"].unsqueeze(-1)
emb = (h * mask).sum(1) / mask.sum(1)
print("64 docs in %.2fs -> %.1f docs/s, dim %d" % (dt, 64 / dt, emb.shape[1]))
