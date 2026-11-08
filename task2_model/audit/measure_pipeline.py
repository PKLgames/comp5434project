"""Time the *fresh* test-inference path: 10,000 test texts -> submission CSV.

This is the measurement the report needs and the memoised `build_submission.py
--apply` path does NOT give: it loads the four checkpoints from disk, tokenises
the test texts, runs all four forward passes, blends, calibrates and writes the
CSV, timing every stage.

Run on a Colab GPU runtime that has the checkpoints:

    python measure_pipeline.py \
        --test-text /content/test_text.parquet \
        --ckpt-dir /content \
        --preds-dir ../../task2_model/audit/preds \
        --out /content/mp_submission.csv

It also prints the environment the numbers belong to (GPU, VRAM, CPU model,
cores, RAM, python / torch / transformers / CUDA versions), because a timing
number without its hardware is not reportable.

Design notes
------------
* The four models are built with the same architecture as `ft3.py`:
  ``AutoModel`` + ``nn.Linear(hidden, 1)`` on the CLS token, MSE regression.
* ``--tags`` defaults to the four shipped models; checkpoints are named
  ``ft3_<tag>.pt`` and are assumed to be on this machine (they are 90-265 MB and
  therefore not committed to the repository).
* Tokenisation is timed separately for the whole 10k rows at once, and again
  per batch inside the model loop, so the report can quote both.
* Nothing is cached between models: each load is timed on its own so the CSV
  write and the blend are attributable.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

P = argparse.ArgumentParser()
P.add_argument("--test-text", default="/content/test_text.parquet")
P.add_argument("--ckpt-dir", default="/content")
P.add_argument("--preds-dir", default="preds")
P.add_argument("--params", default=None, help="ensemble_params.json (default: <preds-dir>/ensemble_params.json)")
P.add_argument("--tags", nargs="+", default=["s1", "s2", "d1", "b1"])
P.add_argument("--models", nargs="+", default=None,
               help="HF model ids, one per tag (default: the ids used in training)")
P.add_argument("--maxlen", type=int, default=128)
P.add_argument("--bs", type=int, default=256)
P.add_argument("--out", default="/content/mp_submission.csv")
a_p = P.parse_args()

DEFAULT_MODELS = {
    "s1": "sentence-transformers/all-MiniLM-L6-v2",
    "s2": "sentence-transformers/all-MiniLM-L6-v2",
    "d1": "distilbert-base-uncased",
    "b1": "BAAI/bge-small-en-v1.5",
}

T = {}
t0 = time.time()


def stage(name, fn):
    t = time.time()
    out = fn()
    T[name] = time.time() - t
    print(f"[timing] {name}: {T[name]:.2f} s", flush=True)
    return out


# --------------------------------------------------------------------- env ---
def env_report():
    import torch
    import transformers
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_mib": (torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                     if torch.cuda.is_available() else None),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    try:
        info["ram_gb"] = round(int(subprocess.check_output(["bash", "-lc", "free -g | awk '/Mem:/{print $2}'"]).strip()), 1)
    except Exception:
        info["ram_gb"] = None
    return info


import torch  # noqa: E402  (timed separately below)
import torch.nn as nn  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

ENV = stage("env_capture", env_report)
print(json.dumps(ENV, indent=2))


class FT(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)


stage("torch_import", lambda: None)  # imports above are already paid for
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- test data ---
tab = stage("read_test_parquet", lambda: pq.read_table(a_p.test_text))
ids = tab["id"].to_numpy()
texts = stage("column_to_python_list", lambda: tab["text"].to_pylist())
print(f"test rows {len(texts)} first id {ids[0]} last id {ids[-1]}")

# --------------------------------------------------------------- per model ---
weights_doc = a_p.params or os.path.join(a_p.preds_dir, "ensemble_params.json")
with open(weights_doc) as f:
    PARAMS = json.load(f)
WEIGHTS = dict(zip(PARAMS["tags"], PARAMS["weights"]))

models_used = a_p.models or [DEFAULT_MODELS[t] for t in a_p.tags]
test_preds = {}
for tag, name in zip(a_p.tags, models_used):
    ckpt = os.path.join(a_p.ckpt_dir, f"ft3_{tag}.pt")
    tok = stage(f"{tag}.load_tokenizer", lambda: AutoTokenizer.from_pretrained(name))
    model = stage(f"{tag}.load_model+weights", lambda: (lambda m: (m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)), m)[1])(FT(name)))
    model = model.to(dev).eval()

    enc = stage(f"{tag}.tokenize_10k", lambda: tok(texts, truncation=True, max_length=a_p.maxlen,
                                                   padding=False, return_tensors=None))

    def run(m=model, enc=enc):
        preds = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            for i in range(0, len(texts), a_p.bs):
                batch = tok.pad({"input_ids": enc["input_ids"][i:i + a_p.bs],
                                 "attention_mask": enc["attention_mask"][i:i + a_p.bs]},
                                padding=True, return_tensors="pt")
                batch = {k: v.to(dev) for k, v in batch.items()}
                preds.append(m(**batch).float().cpu().numpy())
        return np.concatenate(preds).astype(np.float64)

    pred = stage(f"{tag}.forward_10k", run)
    test_preds[tag] = pred
    print(f"  {tag}: raw mean {pred.mean():.4f} sd {pred.std():.4f}")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# ------------------------------------------------------------------ blend ----
from sklearn.isotonic import IsotonicRegression  # noqa: E402


def blend_and_write():
    w = np.array([WEIGHTS[t] for t in a_p.tags])
    raw = np.stack([test_preds[t] for t in a_p.tags], 1) @ w
    xs = np.asarray(PARAMS.get("iso_x", []), dtype=float)
    ys = np.asarray(PARAMS.get("iso_y", []), dtype=float)
    if xs.size and ys.size:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1.0, y_max=5.0)
        iso.fit(xs, ys)
        final = np.clip(iso.predict(raw), 1.0, 5.0)
    else:
        final = np.clip(raw, 1.0, 5.0)
    with open(a_p.out, "w", newline="") as f:
        f.write("id,rating\r\n")
        for i, v in zip(ids, final):
            f.write(f"{int(i)},{round(float(v), 6)}\r\n")
    return raw, final


raw, final = stage("blend+isotonic+write_csv", blend_and_write)

T["TOTAL"] = time.time() - t0
print("\n=== timing summary (seconds) ===")
for k, v in T.items():
    print(f"{k:32s} {v:8.2f}")
print(f"\nwrote {a_p.out}: {len(final)} rows mean {final.mean():.4f} sd {final.std():.4f}")

with open("/content/measure_pipeline.json", "w") as f:
    json.dump({"env": ENV, "timing_s": T, "n_rows": int(len(final)),
               "final_mean": float(final.mean()), "final_sd": float(final.std()),
               "config": {"maxlen": a_p.maxlen, "bs": a_p.bs, "tags": a_p.tags,
                          "models": models_used}}, f, indent=2)
print("wrote /content/measure_pipeline.json")
