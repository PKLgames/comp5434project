"""Time the *fresh* test-inference path: 10,000 test texts -> submission CSV.

This is the measurement the report needs and the memoised `build_submission.py
--apply` path does NOT give: it loads the four checkpoints from disk, tokenises
the test texts, runs all four forward passes, blends, calibrates and writes the
CSV, timing every stage.

Run in an environment containing the trained checkpoints:

    python final_model/measure_pipeline.py \
        --test-text work/preprocessed/test_text.parquet \
        --ckpt-dir work/models --params work/models/ensemble_params.json \
        --out work/fresh.csv --timing-out work/inference.json

It also prints the environment the numbers belong to (GPU, VRAM, CPU model,
cores, RAM, python / torch / transformers / CUDA versions), because a timing
number without its hardware is not reportable.

Design notes
------------
* Models use the same architecture as `finetune.py`:
  ``AutoModel`` + ``nn.Linear(hidden, 1)`` on the CLS token, MSE regression.
* ``--tags`` defaults to the four shipped models; checkpoints are named
  ``ft3_<tag>.pt`` and are assumed to be on this machine (they are 90-265 MB and
  therefore not committed to the repository).
* Tokenisation is timed once per model; padding is part of the forward loop.
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
from build_submission import apply_params, read_params, write_submission


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

P = argparse.ArgumentParser()
P.add_argument("--test-text", required=True)
P.add_argument("--ckpt-dir", required=True)
P.add_argument("--preds-dir", default=os.path.join(os.path.dirname(__file__), 'preds'))
P.add_argument("--params", default=None, help="ensemble_params.json (default: <preds-dir>/ensemble_params.json)")
P.add_argument("--tags", nargs="+", default=["s1", "s2", "d1", "b1"])
P.add_argument("--models", nargs="+", default=None,
               help="HF model ids, one per tag (default: the ids used in training)")
P.add_argument("--maxlen", type=int, default=128, help="Deprecated compatibility option; checkpoint maxlen is used per model")
P.add_argument("--bs", type=int, default=256)
P.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto')
P.add_argument("--num-threads", type=int, default=0, help='CPU threads; 0 retains the PyTorch default')
P.add_argument("--out", required=True)
P.add_argument("--timing-out", required=True)
P.add_argument("--sample", help="Optional original sampleSubmission.csv")
a_p = P.parse_args()
from pathlib import Path
import csv
if Path(a_p.out).exists() or Path(a_p.timing_out).exists(): raise FileExistsError("Select fresh output files")
if a_p.bs < 1 or a_p.num_threads < 0: P.error("Invalid batch size or thread count")
for tag in a_p.tags:
    if not Path(a_p.ckpt_dir, f'ft3_{tag}.pt').is_file():
        raise FileNotFoundError(Path(a_p.ckpt_dir, f'ft3_{tag}.pt'))
Path(a_p.out).parent.mkdir(parents=True, exist_ok=True)
Path(a_p.timing_out).parent.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = {
    "s1": "sentence-transformers/all-MiniLM-L6-v2",
    "s2": "sentence-transformers/all-MiniLM-L6-v2",
    "d1": "distilbert-base-uncased",
    "b1": "BAAI/bge-small-en-v1.5",
}

T = {}
t0 = time.time()


def stage(name, fn):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    T[name] = time.perf_counter() - t
    print(f"[timing] {name}: {T[name]:.2f} s", flush=True)
    return out


# --------------------------------------------------------------------- env ---
def env_report():
    import torch
    import transformers
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
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
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }
    try:
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ['total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended']]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info['ram_gb'] = round(status.total_phys / 1024 ** 3, 2)
        else:
            values = dict(line.split(':', 1) for line in open('/proc/meminfo'))
            info['ram_gb'] = round(int(values['MemTotal'].split()[0]) * 1024 / 1024 ** 3, 2)
    except Exception:
        info["ram_gb"] = None
    return info


import torch  # noqa: E402  (timed separately below)
import torch.nn as nn  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

if a_p.num_threads:
    torch.set_num_threads(a_p.num_threads)
if a_p.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')

ENV = stage("env_capture", env_report)
print(json.dumps(ENV, indent=2))


class FT(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_config(AutoConfig.from_pretrained(name))
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)


stage("torch_import", lambda: None)  # imports above are already paid for
dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a_p.device == 'auto' else a_p.device)

# ---------------------------------------------------------------- test data ---
T = {}
if torch.cuda.is_available(): torch.cuda.synchronize()
t0 = time.perf_counter()
tab = stage("read_test_parquet", lambda: pq.read_table(a_p.test_text))
ids = tab["id"].to_numpy()
texts = stage("column_to_python_list", lambda: tab["text"].to_pylist())
if len(ids) != 10000 or len(set(ids)) != 10000: raise ValueError("Need exactly 10,000 unique test ids")
if a_p.sample:
    with open(a_p.sample, newline="", encoding="utf-8-sig") as f:
        if [int(row["id"]) for row in csv.DictReader(f)] != ids.tolist(): raise ValueError("Test/sample id order mismatch")
print(f"test rows {len(texts)} first id {ids[0]} last id {ids[-1]}")

# --------------------------------------------------------------- per model ---
weights_doc = a_p.params or os.path.join(a_p.preds_dir, "ensemble_params.json")
PARAMS = read_params(weights_doc)

if a_p.tags != PARAMS["tags"]: raise ValueError("Tag order must match parameters")
models_used = a_p.models or [DEFAULT_MODELS[t] for t in a_p.tags]
if len(models_used) != len(a_p.tags): raise ValueError("Need one model id per tag")
settings_used = {}
test_preds = {}
for tag, name in zip(a_p.tags, models_used):
    ckpt = os.path.join(a_p.ckpt_dir, f"ft3_{tag}.pt")
    checkpoint = stage(f"{tag}.read_checkpoint", lambda: torch.load(ckpt, map_location="cpu", weights_only=True))
    name = checkpoint["args"]["model"]
    model_maxlen = int(checkpoint["args"]["maxlen"])
    settings_used[tag] = {"model": name, "maxlen": model_maxlen, "bs": a_p.bs}
    tok = stage(f"{tag}.load_tokenizer", lambda: AutoTokenizer.from_pretrained(name))
    def load_model():
        m = FT(name)
        m.load_state_dict(checkpoint["state"], strict=True)
        return m.to(dev).eval()
    model = stage(f"{tag}.load_model+weights", load_model)
    del checkpoint

    enc = stage(f"{tag}.tokenize_10k", lambda: tok(texts, truncation=True, max_length=model_maxlen,
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
        return np.concatenate(preds).astype(np.float32)

    pred = stage(f"{tag}.forward_10k", run)
    if pred.shape != (10000,) or not np.isfinite(pred).all(): raise ValueError("Invalid model predictions")
    # Keep raw predictions: the four- and two-member recipes clip at different stages.
    test_preds[tag] = pred
    print(f"  {tag}: raw mean {pred.mean():.4f} sd {pred.std():.4f}")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# ------------------------------------------------------------------ blend ----
def blend_and_write():
    final = apply_params(test_preds, PARAMS)
    write_submission(a_p.out, final, fixed_decimals=isinstance(PARAMS['weights'], dict), ids=ids)
    return final


final = stage("blend+isotonic+write_csv", blend_and_write)

if torch.cuda.is_available(): torch.cuda.synchronize()
T["TOTAL"] = time.perf_counter() - t0
print("\n=== timing summary (seconds) ===")
for k, v in T.items():
    print(f"{k:32s} {v:8.2f}")
print(f"\nwrote {a_p.out}: {len(final)} rows mean {final.mean():.4f} sd {final.std():.4f}")

with open(a_p.timing_out, "w", encoding="utf-8") as f:
    json.dump({"env": ENV, "timing_s": T, "n_rows": int(len(final)),
               "final_mean": float(final.mean()), "final_sd": float(final.std()),
               "config": settings_used,
               "tags": a_p.tags, "device": str(dev),
               "scope": "All 10000 rows: parquet read, selected checkpoint/tokenizer/config loads, model creation and device transfers, tokenization, forwards, blend, calibration and CSV output. Excludes Python startup/imports, environment capture and earlier Spark preprocessing; includes downloads if triggered during model/tokenizer loading."}, f, indent=2)
print("FULL_TEST_INFERENCE_COMPLETE", a_p.timing_out)
