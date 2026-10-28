# GPU runbook — strict execution order

Every phase has a **gate**: do not start the next phase until the gate passes.
The order is not arbitrary — phase 5 exists because the absolute validation RMSE
shifts with the PyTorch build (~3e-4 observed between 2.9.1 and 2.14.1), so a
number measured on another machine is **not** a valid baseline. Establish the
reference inside the new environment first, then compare only within it.

| phase | what | cost | gate |
| --- | --- | --- | --- |
| 0 | device detection | seconds | CUDA visible, correct GPU |
| 1 | environment + record versions | ~5 min | imports OK, versions recorded |
| 2 | get the code and the data onto the box | minutes | dataset present, 75 parquet files |
| 3 | build / verify the cache | 1 min | row counts + vocab match |
| 4 | tests + smoke | ~2 min | 35 tests pass, both methods run |
| 5 | **reference reproduction (mandatory)** | ~30 min | reference RMSE recorded |
| 6 | method A / method B | ~10 min each | `compare.py` verdict |
| 7 | decision, then optional ensemble | hours | ΔRMSE rule |

---

## Phase 0 — device detection

```bash
nvidia-smi
python -c "import sys; print(sys.version)"
```

```bash
# after the venv of phase 1 exists
python -c "
import torch
print('torch      :', torch.__version__)
print('cuda build :', torch.version.cuda)
print('available  :', torch.cuda.is_available())
print('device     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('capability :', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')
print('count      :', torch.cuda.device_count())
"
```

Gate: `available` must be `True`. If it is `False`, the CPU wheel was installed —
reinstall with the CUDA index URL in phase 1. `capability` matters for the wheel
choice: `(8, 0)` / `(8, 6)` / `(8, 9)` / `(9, 0)` are all served by the `cu124`
index.

**Write these three values down** — they go into every later comparison:
`torch.__version__`, `torch.version.cuda`, `device name`.

---

## Phase 1 — environment, and record what you built

```bash
cd comp5434project
python -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r task2_model/requirements.txt
pip install -r task2_model/requirements-dev.txt
```

```bash
python -c "import torch, numpy, scipy, pyarrow, sklearn; print(torch.__version__, numpy.__version__, scipy.__version__, pyarrow.__version__, sklearn.__version__)"
pip check                       # must print "No broken requirements found"
```

Gate: the import line prints without error and `pip check` is clean.

Record the five versions plus the phase-0 values. **Do not compare RMSE numbers
across two different version sets.** If the GPU box differs from the machine that
produced 1.1543, that is expected and phase 5 handles it.

`requirements.txt` is unpinned on purpose (it accepts `torch>=2.1`); it is the
recording, not the lock, that makes results interpretable. If you need
bit-reproducibility later, write `pip freeze > task2_model/requirements-frozen.txt`
now and keep it with the run outputs.

---

## Phase 2 — get the code **and** the data onto the box

`git clone` gives you the code, the tests and the small JSON summaries — and
**not** the data or the checkpoints. Check what is deliberately excluded:

```bash
cat .gitignore
```

| item | in git? | size | needed for |
| --- | --- | --- | --- |
| `task2_model/`, `tests/`, `README.md`, `GPU_RUNBOOK.md` | **yes** | KB | everything |
| `Task2_Token_Dataset/tokenized/` (or `Task2_OriginData/output/`) | no | 2.02 GB | rebuilding the cache |
| `artifacts/` | no | 2.4 GB | training (or rebuild from the line above) |
| `runs/**/best.pt` | no | ~234 MB each | scoring / stacking old models |
| `runs/**/val_preds.npz` | no | 12 MB each | `compare.py --base` |
| `task2_model/results/*.json`, `runs/*/summary.json` | **yes** | KB | the reported numbers |

Consequences, in order of preference:

1. **Bring the parquet dataset and rebuild the cache on the GPU box.** Nothing
   model-related needs copying, and phase 5 recreates the reference there.
   ```bash
   # copy the folder in, then:
   export TOK=Task2_Token_Dataset/tokenized      # or Task2_OriginData/output
   ls "$TOK"/train_features_with_entity/*.parquet | wc -l        # expect 64
   ls "$TOK"/validation_features_with_entity/*.parquet | wc -l   # expect 11
   ```
2. **Alternatively copy `artifacts/`** (2.4 GB) — valid, because the cache is
   built from the identical bytes either way. Verify it arrived intact with the
   row counts in phase 3.
3. **Copy `runs/full3/best.pt` and `runs/full_tmlp/best.pt` only if** you want to
   re-score the old models without retraining. They are *not* required for
   training, and after a torch change they are not the right baseline anyway —
   phase 5 supersedes them.

Do **not** copy `.venv/` or `.pylibs/`. `.pylibs/` in this repository contains
Windows binaries (`*_win_amd64.pyd`) and is unusable on Linux.

Gate: either the parquet folders are present (64 + 11 files) or `artifacts/`
exists with `meta.json`.

---

## Phase 3 — build or verify the cache

```bash
python task2_model/preprocess.py \
  --train-dir "$TOK/train_features_with_entity" \
  --valid-dir "$TOK/validation_features_with_entity" \
  --out artifacts
```

```bash
python -c "
import json; m = json.load(open('artifacts/meta.json'))
print('vocab   :', m['vocab'])
print('train   :', m['splits']['train']['rows'], 'rows in', len(m['splits']['train']['shards']), 'shards')
print('valid   :', m['splits']['valid']['rows'], 'rows in', len(m['splits']['valid']['shards']), 'shards')
print('columns :', len(m['dense_columns']), 'mean/std:', len(m['dense_stats']['mean']))
"
```

Gate — must match exactly:

```
vocab   : {'title_tfidf': 55731, 'comment_tfidf': 182382}
train   : 6104774 rows in 64 shards
valid   : 1526196 rows in 11 shards
columns : 26 mean/std: 26
```

If the two possible source folders disagree on any of this, stop — they are
byte-identical and you have the wrong path. Cross-check by pointing `--train-dir`
/ `--valid-dir` at the other folder: the numbers must be identical.

---

## Phase 4 — tests and smoke (before any GPU hours)

```bash
python -m pytest tests -q
```

Gate: `35 passed`.

```bash
python task2_model/train.py --data artifacts --out runs/smoke \
  --epochs 1 --batch-size 1024 --valid-batch-size 1024 \
  --limit-train-rows 8192 --limit-valid-rows 8192 \
  --text-dim 32 --dense-dim 16 --hidden-dim 32 \
  --ordinal-weight 0.5 --film --save-val-preds
```

Gate: the epoch line prints `exp_rmse=`, `blend_rmse=` and `alpha=`, and the
checkpoint carries the blend weight:

```bash
python -c "import torch; c=torch.load('runs/smoke/best.pt',map_location='cpu',weights_only=False); print(c.get('blend_alpha'))"
```

Delete `runs/smoke` afterwards — it is a throwaway with tiny dimensions.

---

## Phase 5 — reference reproduction on this GPU (mandatory)

This is the step people skip and then regret. The headline 1.1543 was measured
with torch 2.9.1 on CPU. On this box it will differ. Retrain the **same two
members** with the exact published configuration, then stack them — the stacked
out-of-fold number becomes the only baseline you compare against from here on.

```bash
# member 1 — plain hybrid (published: runs/full3, best at epoch 2 of 3)
python task2_model/train.py --data artifacts --out runs/gpu_ref_base --epochs 3 \
  --batch-size 8192 --valid-batch-size 8192 \
  --lr 0.003 --weight-decay 0.0001 \
  --text-dim 256 --dense-dim 128 --hidden-dim 256 \
  --dropout 0.15 --emb-dropout 0.1 \
  --seed 42 --num-threads 16 --save-val-preds

# member 2 — text-fusion MLP (published: runs/full_tmlp, 2 epochs)
python task2_model/train.py --data artifacts --out runs/gpu_ref_tmlp --epochs 2 \
  --batch-size 8192 --valid-batch-size 8192 \
  --lr 0.003 --weight-decay 0.0001 \
  --text-dim 256 --dense-dim 128 --hidden-dim 256 \
  --dropout 0.15 --emb-dropout 0.1 --text-mlp \
  --seed 42 --num-threads 16 --save-val-preds

# the reference artefact you will compare against
python task2_model/stack.py --data artifacts --out runs/gpu_ref_stack \
  --checkpoint runs/gpu_ref_base/best.pt \
  --checkpoint runs/gpu_ref_tmlp/best.pt
```

Do **not** use `--amp` here: the reference is fp32, and mixing in fp16 would
confound the version difference with a precision difference.

Gate: `runs/gpu_ref_stack/val_preds_metrics.json` exists and its `rmse` is your
baseline. Expect it to land within a few 1e-4 of 1.1543 — if it is off by more
than ~1e-3, stop and re-check that `--batch-size`, `--lr` and `--seed` match the
block above (a different batch size changes the number materially).

Sanity-check that the GPU and CPU agree on the *architecture* even when the last
decimals differ:

```bash
python task2_model/compare.py --base runs/ensemble_stack/val_preds.npz \
  --candidate runs/gpu_ref_stack/val_preds.npz --out runs/gpu_ref_stack/vs_published.json
```

`vs_published.json` is a **diagnostic**, not a decision: a CI that just barely
excludes 0 with |ΔRMSE| ~1e-4 is the expected signature of a torch build change.

---

## Phase 6 — the two candidate methods

Both are flags on `train.py`; both are off by default; neither changes phase 5's
result. Run them against the phase-5 baseline.

### Method A — ordinal baseline + out-of-fold `E[rating]` blend

```bash
python task2_model/train.py --data artifacts --out runs/gpu_ord --epochs 4 \
  --batch-size 8192 --valid-batch-size 16384 \
  --lr 0.003 --weight-decay 0.0001 \
  --text-dim 256 --dense-dim 128 --hidden-dim 256 \
  --dropout 0.15 --emb-dropout 0.1 \
  --ordinal-weight 0.5 --blend-folds 5 \
  --seed 42 --num-threads 16 --save-val-preds

python task2_model/predict.py --data artifacts --split valid \
  --checkpoint runs/gpu_ord/best.pt --out runs/gpu_ord/val_preds.npz
```

Optional: select the checkpoint on the metric you will actually submit with
(`blend`) instead of the regression head. Default is `auto`, which reproduces the
historical behaviour:

```bash
  --select-metric blend
```

Read in `summary.json["history"]`: `val_rmse`, `val_rmse_exp`, `val_rmse_blend`,
`blend_alpha`. `predict.py --blend-alpha auto` reads the stored out-of-fold
weight, so the scored number equals `val_rmse_blend`.

### Method B — FiLM architecture

```bash
python task2_model/train.py --data artifacts --out runs/gpu_film --epochs 4 \
  --batch-size 8192 --valid-batch-size 16384 \
  --lr 0.003 --weight-decay 0.0001 \
  --text-dim 256 --dense-dim 128 --hidden-dim 256 \
  --dropout 0.15 --emb-dropout 0.1 \
  --film \
  --seed 42 --num-threads 16 --save-val-preds

python task2_model/predict.py --data artifacts --split valid \
  --checkpoint runs/gpu_film/best.pt --out runs/gpu_film/val_preds.npz
```

### The combined configuration

```bash
python task2_model/train.py --data artifacts --out runs/gpu_ord_film --epochs 4 \
  --batch-size 8192 --valid-batch-size 16384 \
  --ordinal-weight 0.5 --film --select-metric blend \
  --seed 42 --num-threads 16 --save-val-preds
```

### Gate: the decision rule

```bash
for r in gpu_ord gpu_film gpu_ord_film; do
  python task2_model/compare.py --base runs/gpu_ref_stack/val_preds.npz \
    --candidate runs/$r/val_preds.npz --out runs/$r/comparison.json
done
```

```
adopt  <=>  delta_rmse <= -0.001  AND  the 95% paired-bootstrap CI excludes 0
```

This is not a judgement call; `comparison.json["decision"]["adopt"]` stores it.
A candidate that improves RMSE by less than the phase-5 noise is not an
improvement, and one that trades RMSE for a much worse MAE (as the V2
interaction block did) is not worth adopting either.

---

## Phase 7 — only after a winner exists: diversity ensemble

Ensembling multiplies cost by the number of members, so it comes last, on the
winning configuration. Vary seeds and one structural knob at a time
(`--text-dim 128` vs `256`, with/without `--text-mlp`, `--dropout`), then:

```bash
python task2_model/stack.py --data artifacts --out runs/gpu_stack \
  --checkpoint runs/gpu_ref_base/best.pt \
  --checkpoint runs/gpu_ref_tmlp/best.pt \
  --checkpoint runs/gpu_ord_film/best.pt \
  --checkpoint runs/gpu_seed2/best.pt \
  --checkpoint runs/gpu_seed3/best.pt

python task2_model/compare.py --base runs/gpu_ref_stack/val_preds.npz \
  --candidate runs/gpu_stack/val_preds.npz --out runs/gpu_stack/comparison.json
```

`stack.py` scores every member through the same path as `predict.py` (including
each member's own `E[rating]` blend), so a member's contribution cannot silently
disagree with its standalone number. Same decision rule as phase 6.

---

## Parameter reference

The published configuration, and what is safe to vary.

| parameter | published value | notes |
| --- | --- | --- |
| `--epochs` | 3 (base) / 2 (text-MLP) | early stopping with `--patience 2` usually ends sooner |
| `--batch-size` | 8192 | **changes the result materially** — keep it fixed when reproducing, vary only in dedicated experiments; 4096 if OOM |
| `--valid-batch-size` | 8192 → 16384 | evaluation only, free to raise |
| `--lr` | 0.003 | OneCycleLR peaks here; safe range 1e-3 … 5e-3 |
| `--weight-decay` | 0.0001 | |
| `--text-dim` | 256 | 128 halves the checkpoint and is the first thing to try on a small GPU |
| `--dense-dim` | 128 | |
| `--hidden-dim` | 256 | |
| `--dropout` / `--emb-dropout` | 0.15 / 0.1 | |
| `--seed` | 42 | vary across ensemble members, keep fixed for A/B comparison |
| `--num-threads` | 16 | CPU knob, negligible on GPU |
| `--amp` | off | leave off for reference/AB runs; only use once an fp32 number is recorded |
| `--device` | auto | picks CUDA when available |

Memory: the default model is ~61.4 M parameters, so weights + AdamW state are
roughly 0.75 GB; the rest of the budget is the 238k-row embedding tables and the
batch. An 8 GB card runs `--batch-size 8192` with `--text-dim 256`; on 4–6 GB use
`--batch-size 4096` or `--text-dim 128`.

Determinism: CPU is bit-exact at a fixed seed; GPU reduction order differs, so
treat ±1e-4 as noise and use the paired bootstrap instead of eyeballing.

---

## Troubleshooting by phase

| symptom | phase | cause |
| --- | --- | --- |
| `available: False` | 0–1 | CPU wheel installed; reinstall from the `cu124` index |
| `meta.json` missing | 3 | `--out` path wrong, or preprocess failed on the parquet |
| row counts differ from the gate | 3 | wrong `$TOK`, or a truncated copy |
| `checkpoint/cache mismatch` | 4–6 | `--data` points at a cache with a different dense schema |
| `cannot score this checkpoint` | 6 | same, raised by `predict.py` |
| FiLM run much worse at epoch 1 | 6 | should not happen (γ/β start at 0); re-run `tests/test_ordinal_film.py` |
| `blend_alpha` is `None` | 6 | trained without `--ordinal-weight`; expected for a regression model |
