# COMP5434 Task 2 — Review Rating Regression

A reproducible pipeline that predicts the 1–5 star rating of a review (RMSE
metric) from the features produced by Part A.  It contains:

1. **`preprocess.py`** — turns the Spark parquet shards into compact CSR
   caches (vocabulary compaction, dense feature construction, train-fitted
   standardisation statistics).
2. **`train.py`** — trains a hybrid sparse-text / dense-feature neural
   regressor on CPU or GPU.
3. **`predict.py`** — evaluates on the validation split or writes a Kaggle
   submission for the test split.
4. **`linear_baseline.py`** — streaming SGD linear reference model used for the
   ablation table and for sanity checks.

The model, the design rationale and the measured results are described below.

---

## 1. Data

The handover folder contains the fixed 80/20 split produced by Part A
(6,104,774 train rows, 1,526,196 validation rows).  Each row has:

| group | columns |
| --- | --- |
| target | `rating` (1–5) |
| base numeric (11) | title/comment character and word counts, `votes`, `votes_log`, `purchased`, `review_year`, `review_month`, `comment_missing`, `votes_invalid` |
| entity statistics (9) | user / product / parent-product review counts, their logs and unseen flags |
| text (2) | `title_tfidf` (65,536-d) and `comment_tfidf` (262,144-d) hashed TF-IDF, L2-normalised |

`id` is only used to align predictions and is **never** a model input.

### 1.1 Where the features come from, and why `tokenized/` is not new data

The handover contains two feature folders:

| folder | feature parquet | auxiliary tables |
| --- | --- | --- |
| `Task2_OriginData/output/` | 75 files | `text_models/`, `entity_statistics/`, `split_assignments/` |
| `Task2_Token_Dataset/tokenized/` | 75 files | none |

The second folder was described as a re-tokenised dataset, but the two are
**byte-for-byte identical** for every feature file.  Measured on this checkout:

| check | result |
| --- | --- |
| parquet file names (train / valid) | 64 / 11 in both, identical names (same part UUIDs) |
| parquet bytes (train + valid) | 1,726,687,142 + 430,329,320 in both |
| `part-00000…143f1625…` SHA-256 | `2a5118a0b46c0d2c…` in both |
| last train shard and last valid shard SHA-256 | identical in both |
| `text_vectorization_summary.json` SHA-256 | `589e13d1f124bbf2…` in both |

The ~148 MB size difference between the two folders is entirely the auxiliary
tables that only exist under `Task2_OriginData/output/`
(`entity_statistics` 90.9 MB, `split_assignments` 56.0 MB, `text_models` 0.8 MB).
The Parquet footer column lists agree as well: `id`, `rating`, 11 base numeric
columns, 9 entity statistics columns, `title_tfidf`, `comment_tfidf` — no column
exists in one folder and not the other.

**Consequence.** Switching the data source cannot improve the score, because the
rows are the same rows.  `tokenized/` is kept as the canonical source (it is the
name the repository and its scripts use), and
`preprocess.py --feature-set v2` records a SHA-256 *fingerprint* of the cache it
was built from, so a genuinely different future data drop is detectable instead
of silently reused.  All subsequent gains therefore have to come from features
derived **within** an existing row, which is what section 10 does.

### 1.2 A property of the entity flags worth knowing

`user_unseen`, `product_unseen` and `parent_product_unseen` are **identically
zero on the training split** (dense `mean = 0`, `std` falls back to 1.0 because
the variance is 0 over all 6,104,774 rows).  This is structural, not a bug: an
unseen entity is by definition one absent from the entity statistics, and those
statistics were fitted on the training split, so every training row is "seen".
The flags only fire on validation/test (`std` 0.50 / 0.18 / 0.16 there).

The practical implication is that these three columns carry **no learnable
signal**: the dense tower never sees a non-zero value during training, so it can
only learn a near-zero weight.  They are retained for schema compatibility, and
any "unseen" interaction feature built on them is dead on the current data (see
section 10.4).

---

## 2. Model design

```
title_tfidf ──► EmbeddingBag(sum, tfidf weights) ──► LayerNorm ──► 256-d ─┐
comment_tfidf ─► EmbeddingBag(sum, tfidf weights) ──► LayerNorm ──► 256-d ─┤
                                                                          ├─► concat ─► MLP(256) ─► 64 ─► logit
26 dense features ────────────────────────────────► MLP(128→128) ─────────┘
title_tfidf ──► linear token weights (wide path) ──────────────────────────┘  (+ logit)
                                                                                    │
                                                          rating = 1 + 4·sigmoid(logit)
```

**Why this architecture**

* **Sparse-aware text tower.** `EmbeddingBag(mode="sum", per_sample_weights=…)`
  computes `Wᵀx` for a TF-IDF row while touching only its ~26 non-zero entries.
  A 238k-column dense matrix is never materialised.
* **Wide linear path.** The low-rank tower is a rank-256 approximation of a
  linear model.  One free scalar per token (`lin_*` tables) restores the full
  linear capacity that plain TF-IDF regression has; this is the most important
  accuracy component of the text branch and costs only 238k extra parameters.
* **Dense tower.** A two-layer MLP lets the model use non-linear effects such as
  "many reviews but few helpful votes".
* **Bounded output.** `1 + 4·sigmoid(logit)` keeps predictions inside the legal
  rating range, which is a soft prior that consistently helps RMSE.
* **Regularisation.** Dropout on the text embeddings and on the dense/fusion
  layers, weight decay, gradient clipping and early stopping on validation RMSE.

The model is intentionally trained with a single objective (MSE) so that it
optimises the competition metric directly.

---

## 3. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# CPU (works everywhere)
pip install -r requirements.txt

# GPU machine: install the CUDA build of PyTorch instead of the CPU one
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The code is tested with Python 3.12, PyTorch 2.x, NumPy 2.x and PyArrow 14+.
No GPU is required to run the pipeline; the code automatically uses CUDA when
available (`--device auto`).

### 3.1 Checklist for a GPU machine

The pipeline is platform-independent Python (no hard-coded Windows paths in the
code).  On a Linux/Windows GPU server:

1. `python -m venv .venv && source .venv/bin/activate`
2. Install the CUDA build **first**, then the remaining requirements:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
3. Either copy the 2 GB `artifacts/` cache from the CPU machine, or rerun
   `preprocess.py` on the server (it only needs the Part-A parquet files).
4. `python train.py --data artifacts --out runs/gpu --epochs 6 --batch-size 8192`
   — `--device auto` selects `cuda` automatically.
5. On a GPU with less than ~8 GB, use `--batch-size 4096` and/or `--text-dim 128`.
6. Mixed precision (`--amp`) is optional and disabled by default; enable it
   after a first successful fp32 run.  `torch.cuda.amp` fallbacks are handled
   for older PyTorch versions.

The only runtime difference is speed: the local CPU runs one epoch in 8–9
minutes, a recent GPU typically finishes the same epoch in a couple of minutes.

---

## 4. Step-by-step execution

### 4.1 Unpack and preprocess (once)

```bash
# from the folder that contains Task2_Token_Dataset/
python task2_model/preprocess.py \
  --train-dir "Task2_Token_Dataset/tokenized/train_features_with_entity" \
  --valid-dir "Task2_Token_Dataset/tokenized/validation_features_with_entity" \
  --out artifacts
```

This writes `artifacts/` (~2 GB): the compact vocabulary, per-shard CSR caches
and the train-fitted dense mean/std.  It takes about one minute on 16 CPU
threads.  The vocabulary is fitted **on the training split only**; validation
rows are mapped with the same vocabulary.

### 4.2 Train

```bash
# Full training (GPU recommended; ~2 min/epoch on a modern GPU, ~8 min/epoch on 16 CPU threads)
python task2_model/train.py --data artifacts --out runs/full --epochs 6 --batch-size 8192

# Quick smoke test (about one minute)
python task2_model/train.py --data artifacts --out runs/smoke --epochs 1 \
  --limit-train-shards 4 --limit-valid-shards 1
```

Outputs in `runs/full/`:

| file | content |
| --- | --- |
| `best.pt` | best checkpoint (validation RMSE) |
| `summary.json` | best RMSE, mean baseline, per-epoch history, timings, hardware |
| `args.json` | exact configuration used |

### 4.3 Evaluate / predict

```bash
# validation RMSE + per-row predictions
python task2_model/predict.py --data artifacts --split valid \
  --checkpoint runs/full/best.pt --out runs/full/val_preds.npz

# test submission (after Part A delivers test_features_with_entity)
python task2_model/preprocess.py --test-dir <test_features_dir> \
  --reuse artifacts --out artifacts

python task2_model/predict.py --data artifacts --split test \
  --checkpoint runs/full/best.pt \
  --ids-dir <test_features_dir> --out submission.csv
```

### 4.4 Feature ablations (required by the brief)

```bash
python task2_model/train.py --data artifacts --out runs/ablate_text \
  --text-mode both --dense-mode none
python task2_model/train.py --data artifacts --out runs/ablate_text_num \
  --text-mode both --dense-mode numeric
python task2_model/train.py --data artifacts --out runs/ablate_all \
  --text-mode both --dense-mode all
```

---

## 5. Reproducibility

* All shuffling uses NumPy `RandomState(seed)`; `--seed 42` is the default.
* PyTorch, NumPy and Python seeds are set at start-up.
* The train/validation split is fixed by Part A and is never regenerated here.
* The vocabulary and dense statistics are fitted on the training split only and
  reused for validation/test, so there is no validation leakage.
* Re-running the same command with the same checkpoint/seed reproduces the same
  RMSE (CPU is bit-exact; GPU results can differ in the last decimals due to
  non-deterministic reduction order — set `torch.use_deterministic_algorithms`
  if exact GPU reproducibility is required).

---

## 6. Measured results

### 6.1 Main results (full 6.10M train / 1.53M validation rows)

All numbers are RMSE on the fixed validation split.  The validation split is
used for early stopping and for the (out-of-fold) ensemble weights, so the
single-model numbers are slightly optimistic; the ensemble number is computed
with 5-fold out-of-fold weight fitting and is therefore an honest estimate.

| # | model | val RMSE | val MAE | notes |
| --- | --- | ---: | ---: | --- |
| 0 | train-mean constant | 1.5149 | — | reference |
| 1 | hybrid NN, seed 42 (3 epochs) | 1.1573 | 0.8646 | best epoch 2, epoch 3 overfits |
| 2 | hybrid NN + text-fusion MLP (2 epochs) | 1.1572 | 0.8681 | best epoch 2 |
| 3 | equal-weight average of (1) and (2) | 1.1550 | 0.8653 | no fitted parameters |
| 4 | weighted stack of (1) and (2) | **1.1543** | 0.8728 | weights 0.477 / 0.484, intercept 0.162; 5-fold OOF |

Per-rating error of the final stacked ensemble:

| rating | rows | RMSE | mean prediction | bias |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 224,366 | 1.9615 | 2.596 | +1.596 |
| 2 | 131,896 | 1.6574 | 3.247 | +1.247 |
| 3 | 147,756 | 1.0427 | 3.439 | +0.439 |
| 4 | 176,874 | 0.7846 | 3.876 | −0.124 |
| 5 | 845,304 | 0.7980 | 4.331 | −0.669 |

**Interpretation.**  The model explains
`1 − (1.1543/1.5149)² ≈ 42 %` of the rating variance and its average absolute
error is 0.87 stars.  Errors are concentrated on 1–2 star reviews: the model
systematically over-predicts them (bias ≈ +1.6 / +1.2) because the conditional
mean of a very positive corpus is high and text alone is an ambiguous signal
for the minority classes.  This is the main remaining source of error.

### 6.2 Feature ablation (1.52M-row subset, 1 epoch, identical settings)

The full-data ablation costs ~50 minutes per row on the CPU-only test machine,
so the ablation was run on the first 16 training shards (1.52M rows, 2 validation
shards) with exactly the same hyper-parameters.  The ranking is what matters.

| features | val RMSE | Δ vs previous |
| --- | ---: | ---: |
| text only | 1.1689 | — |
| text + numeric | 1.1668 | −0.0021 |
| text + numeric + entity counts | 1.1666 | −0.0002 |
| text + numeric + entity, Ridge linear | 1.2333 | reference |

The entity count features add very little on top of text + base numeric
features; the text representation carries almost all of the signal.  The
neural model beats a well-tuned linear TF-IDF model by ~0.066 RMSE on the same
subset, which justifies the non-linear fusion design.

### 6.3 Timing and hardware

Local test machine: **Intel Core i7-1260P (4P + 8E, 16 threads), no GPU,
~7 GB RAM available, Python 3.12, PyTorch 2.9.1 CPU, PyArrow 26**.

| stage | time |
| --- | ---: |
| preprocessing (vocabulary + 6.10M train + 1.53M valid shards) | ~46 s |
| training, 1 epoch, full 6.1M rows | 485–561 s |
| total offline, single model (preprocess + 2–3 epochs) | 17–25 min |
| total offline, two-model ensemble | ~43 min |
| inference, 1.53M rows, one model | 9.5 s (~160k rows/s) |
| inference, 1.53M rows, two-model ensemble | 14.9 s |
| model size | 61.4M parameters, ~246 MB checkpoint |

On a CUDA GPU the same script is used with `--device auto` (the default); the
sparse `EmbeddingBag` forward/backward is the only heavy operation and is
memory-bound, so a modern 8 GB GPU should run batch size 8192 comfortably.
The CPU numbers above are the reproducible fallback.

### 6.4 Ensemble command

```bash
python task2_model/predict.py --data artifacts --split valid \
  --checkpoint runs/full3/best.pt \
  --checkpoint runs/full_tmlp/best.pt \
  --weights 0.47725399,0.48444617 --intercept 0.16193599 \
  --out runs/ensemble_stack/val_preds.npz
```

For a submission, replace `--split valid` with `--split test` and add
`--ids-dir <test_features_dir> --out submission.csv`.

---

## 7. Project layout

```
task2_model/
  common.py            column definitions, feature groups, dense derivation
  preprocess.py        parquet → compact CSR shards; --feature-set v2 writes the overlay
  dataset.py           shard-wise mini-batch iterator (resolves V1/V2 overlays)
  model.py             hybrid sparse-text / dense model
  features_v2.py       V2 text-shape and interaction feature blocks
  train.py             training loop, early stopping, metrics
  predict.py           validation / test inference and submission writer
  stack.py             N-model stacked ensemble, 5-fold out-of-fold weights
  compare.py           paired bootstrap ΔRMSE / ΔMAE between two prediction files
  linear_baseline.py   streaming linear reference model
  analyze.py           per-rating / calibration error analysis
  requirements.txt
  requirements-dev.txt pytest, only needed for the tests
  README.md
  GPU_RUNBOOK.md       CUDA setup, both token-dataset paths, both methods, cost
tests/
  test_v2_features.py  hand-computed unit tests for the V2 blocks
  test_backcompat.py   the V1 model, schema and cache are unchanged
  test_smoke_v2.py     end-to-end overlay + one training step
  test_ordinal_film.py out-of-fold blend weight and the FiLM invariants
```

## 8. Notes on the big-data requirement

Part A's feature pipeline is implemented with **Apache Spark 4.0.1**
(`clean_data.py`, `split_train.py`, `feature_processing.py`,
`text_vectorization.py`, `entity_features.py`).  This repository consumes the
Spark-produced parquet features; therefore the end-to-end solution incorporates
a distributed processing framework for cleaning, splitting, TF-IDF/IDF fitting,
entity aggregation and feature materialisation.  Model training is
single-node PyTorch because the sparse model fits comfortably in GPU memory and
trains faster than a Spark MLlib job for this feature shape.

## 9. Cleaning up

Everything installed for the local smoke test lives in folders that can simply
be deleted: `.venv/` (or `.pylibs/`) and `artifacts/`, `runs/`.  Nothing is
installed globally and no system package is modified.

## 10. V2 features: text-shape overlay (optional)

An **additive** experiment.  Nothing here changes the V1 model, the existing
`artifacts/` cache, the existing checkpoints or any number in section 6.  If the
numbers below do not beat 1.1543, the V1 results simply stand.

### 10.1 Motivation

V1 uses the two hashed TF-IDF rows only through their *values*: `EmbeddingBag`
for the rank-256 text tower and one free scalar per token for the wide linear
path.  It never sees their **shape** — how many distinct words survived
`minDocFreq`, how concentrated the weight mass is, or whether title and comment
describe the same thing.  A rank-256 projection cannot recover those quantities,
they need no new data, and they cost a few milliseconds per shard.

### 10.2 The columns

Seven core columns, each a function of a single row (`features_v2.py`):

| column | definition | motivation |
| --- | --- | --- |
| `title_nnz`, `comment_nnz` | `log1p(#{data > 0})` | distinct words actually retained; **not** the same as `*_word_count`, which counts every token including `minDocFreq`-filtered ones |
| `title_weight_sum`, `comment_weight_sum` | `log1p(Σ data)` | dispersion: ≈1 for a single dominant word, small for spread-out content |
| `title_weight_max`, `comment_weight_max` | `max(data)` ∈ (0, 1] | how much the strongest single word dominates |
| `title_comment_cosine` | `t · c` over shared hashed buckets | cross-column agreement; the two `EmbeddingBag`s are separate and the wide path is additive, so the model cannot construct this itself |

Four optional interaction columns (off by default, measured separately):
`unseen_sum`, `votes_x_comment_log`, `purchased_x_comment_log`,
`review_year_centered`.

Two details that are easy to get wrong, both covered by unit tests:

* `data == 0` entries are **vocabulary misses**, not words.  `remap_to_vocab`
  parks an out-of-vocabulary hashed index on slot 0 with weight 0 so
  `EmbeddingBag` ignores it.  They are excluded from the non-zero count.
* The two cached CSR columns index **separate** compact vocabularies
  (55,731 title / 182,382 comment), so stored ids are not comparable across
  columns.  `title_comment_cosine` therefore maps stored ids back to the original
  hashed values via `vocab_*.npy`.  Part A hashed title into 2^16 and comment
  into 2^18 buckets, and 40,007 hashed values are shared by both vocabularies.
  Measured on 95,570 training rows: the hashed-space cosine is non-zero for
  **26.8 %** of rows, whereas taking the dot product of the raw stored ids would
  be non-zero for **0.22 %** — i.e. a dead feature.

### 10.3 Zero-copy overlay

`artifacts/` stays read-only.  `artifacts_v2/` adds only `meta.json` +
`stats.json` (≈13 KB) and points back at the V1 shards:

```
artifacts/  (unchanged, read-only)          artifacts_v2/  (new, KB)
  vocab_*.npy, {train,valid}/shard_*.npz      meta.json   v1 meta + base_dir + 7/11 columns
                                              stats.json  mean/std fitted in one pass over train
        │                                          │
        └────────────►  ShardBatcher  ◄────────────┘
             reads the V1 shard, recomputes the new columns from its
             CSR arrays, appends them, then standardises all 37 together
```

Everything is reproducible in ~20 s on 16 CPU threads:

```bash
python task2_model/preprocess.py --feature-set v2 --reuse artifacts --out artifacts_v2

python task2_model/train.py --data artifacts_v2 --out runs/v2_full \
  --epochs 3 --batch-size 8192

python task2_model/predict.py --data artifacts_v2 --split valid \
  --checkpoint runs/v2_full/best.pt --out runs/v2_full/val_preds.npz
```

Feature ablations reuse the existing `--dense-mode` machinery:

```bash
# 26 V1 columns only (identical to the V1 model)
python task2_model/train.py --data artifacts_v2 --out runs/v2_core --dense-mode all --no-interactions
# 33 columns: 26 V1 + 7 text-shape
python task2_model/train.py --data artifacts_v2 --out runs/v2_shape --dense-mode all --no-interactions
# 37 columns: + the 4 interactions
python task2_model/train.py --data artifacts_v2 --out runs/v2_inter
```

Stacking and the objective comparison:

```bash
python task2_model/stack.py --data artifacts_v2 --out runs/v2_stack \
  --checkpoint runs/full3/best.pt --checkpoint runs/full_tmlp/best.pt \
  --checkpoint runs/v2_full/best.pt

python task2_model/compare.py --base runs/ensemble_stack/val_preds.npz \
  --candidate runs/v2_stack/val_preds.npz --out runs/v2_stack/comparison.json
```

`stack.py` generalises the hand-fitted two-model weights (section 6.4) to N
checkpoints with the same 5-fold out-of-fold methodology.  It builds **one
loader per checkpoint**, so a V1 model (26 dense columns) and a V2 model (33) can
be stacked; `predict.py` cannot do that, and now refuses loudly instead of
silently feeding the wrong columns.  `compare.py` reports ΔRMSE / ΔMAE, the
per-rating deltas and a 5,000-resample paired bootstrap CI — the candidate is
only adopted when ΔRMSE ≤ −0.001 **and** the CI excludes 0.

### 10.4 Notes and caveats found while implementing

* **`unseen_sum` is a dead column.**  The three `*_unseen` flags are identically
  zero on the training split (section 1.2), so this interaction is 0 for every
  training row and cannot be learned.  It is implemented as specified and
  harmless, but it cannot contribute, which is why the interaction block as a
  whole is measured rather than claimed.
* **V1 statistics are reproduced exactly by the overlay.**  The overlay fits
  mean/std for all 37 columns in one pass over the training shards; the first 26
  come out with `max|Δmean| = max|Δstd| = 0` against `artifacts/meta.json`, so a
  V1 model reading `artifacts_v2/` sees bit-identical features (asserted in
  `tests/test_backcompat.py`).
* **`--dense-mode all` is schema-aware.**  On a V2 cache it means the whole
  schema (33 or 37), so the new columns cannot be dropped by accident; on a V1
  cache it still means exactly the 26 V1 columns.

### 10.5 Measured result: the gate is negative

The V2 features were tested with the same methodology as the published ablation
(section 6.2): the first **16 training shards (1,524,869 rows)** and **2
validation shards (286,852 rows)**, `--epochs 1 --batch-size 8192 --lr 0.003
--weight-decay 0.0001 --seed 42`, identical in all three arms.  The V1 arm was
re-run on this machine rather than compared against the published 1.166614,
because a different PyTorch build changes the last decimals (here the identical
V1 configuration yields 1.166958, a 3.4e-4 offset).

| arm | dense columns | val RMSE | val MAE |
| --- | ---: | ---: | ---: |
| V1 (control) | 26 | 1.166958 | 0.889986 |
| V2 core (26 + 7 text-shape) | 33 | 1.167099 | 0.889837 |
| V2 core + interactions | 37 | 1.166775 | 0.891777 |
| train-mean constant | — | 1.5151 | — |

Paired 5,000-resample bootstrap on the same 286,852 rows:

| comparison | ΔRMSE | 95 % CI | ΔMAE | 95 % CI |
| --- | ---: | --- | ---: | --- |
| V2 core − V1 | **+0.000142** | [−0.000154, +0.000431] | −0.000150 | [−0.000402, +0.000100] |
| V2 core+inter − V1 | **−0.000183** | [−0.000473, +0.000104] | **+0.001791** | [+0.001544, +0.002034] |

**Verdict: not adopted.**  Neither arm clears the bar, and neither interval
excludes 0:

* The 7 text-shape columns move RMSE by **+0.00014** (slightly *worse*) with a CI
  spanning zero.  The feature block is genuinely informative about the text —
  but the model already extracts that information from the TF-IDF values
  themselves, so the columns are redundant rather than additive.
* Adding the 4 interactions gives a RMSE change of −0.00018 whose CI includes 0,
  while **MAE degrades significantly** (+0.00179, CI excludes 0).  That is not a
  trade worth making, and it is consistent with `unseen_sum` being a dead column
  (section 10.4) and the rest duplicating what the dense tower already sees.
* The plan's adoption threshold was ΔRMSE ≤ −0.001 with a CI excluding 0.  The
  observed effect is roughly **5× smaller** than the threshold and has the wrong
  sign in the core arm.

Per the plan's G2/G4 rules the work stops here: the full 6.1 M-row / 3-epoch run
and the three-model stack are **not** executed, since there is no evidence they
could reach the −0.001 target, and a negative result recorded honestly is a
valid deliverable.  Section 6 is unchanged, so **1.1543 remains the reported
number**, and `compare.py` is left in place so any future candidate can be judged
by exactly this criterion.

Caveat worth stating: this is a one-epoch, quarter-data gate.  A feature that
only helps at convergence could hide here, and the gate was designed as a cheap
filter, not a proof.  The infrastructure to run the full comparison
(`preprocess.py --feature-set v2`, `train.py`, `stack.py`, `compare.py`) is in
place and documented in section 10.3, so the full run costs one command per step
if a GPU becomes available.

### 10.6 Tests

```bash
pip install -r task2_model/requirements-dev.txt
python -m pytest tests -q
```

| file | what it pins down |
| --- | --- |
| `tests/test_v2_features.py` | hand-computed nnz / sum / max / cosine, empty and single-token rows, `data == 0` exclusion, hashed-vs-stored id space |
| `tests/test_backcompat.py` | `DENSE_COLS` is still the 26-column V1 registry; the first validation batch is bit-identical to the pre-V2 code (SHA-256); `runs/full3/best.pt` still loads; the V2 path never writes `artifacts/` |
| `tests/test_smoke_v2.py` | one-shard overlay builds with 33 columns, overlay is < 256 KB, shuffled batches stay row-aligned, one optimiser step has a finite loss, `train.py` records `dense_schema` in the checkpoint |

`test_backcompat.py` needs the 2.4 GB cache and the 234 MB checkpoints; those
tests skip automatically when the gitignored files are absent.

### 10.7 Rollback

V2 is entirely additive: delete `artifacts_v2/` and `runs/v2_*`, and remove
`features_v2.py`, `compare.py`, `stack.py` and `tests/`.  Because `model.py` and
`artifacts/` were never touched, the V1 commands still reproduce 1.1543.

## 11. Next-round methods: ordinal baseline and FiLM architecture

Two further changes were added after the V2 word was settled.  Both are **flags,
off by default**, and neither alters any number in section 6 when unused.  The
full operational procedure — environment, both token-dataset paths, GPU settings,
cost and the decision rule — is in [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md).

| # | method | flag | mechanism |
| --- | --- | --- | --- |
| A | ordinal baseline | `--ordinal-weight 0.5` | auxiliary 5-class head; its `E[rating] = Σ k·p_k` is a second estimator, blended with the regression output |
| B | FiLM architecture | `--film` | numeric features generate `(γ, β)` and modulate each text tower: `h' = h·(1+γ) + β` |

**Why A is the baseline.** It changes a single model, so an architecture
experiment costs the same as today.  Ensembling first (#1 in the roadmap) would
multiply the cost of every subsequent experiment by the number of members and
confound attribution; nonlinear calibration (#3) would require an OOF harness
around every experiment.

**What is actually new in A.**  The `ordinal` head and the `rmse_exp` /
`rmse_blend` reporting already existed, but every published run used
`ordinal_weight: 0.0`, so the path was never exercised.  Two things were needed to
make it usable:

* the blend weight is now fitted **out of fold** in closed form
  (`train.optimal_blend_alpha` is the exact least-squares solution, not a grid
  search), so `val_rmse_blend` is not the in-sample optimum;
* `predict.py --blend-alpha auto` reads that out-of-fold weight from the
  checkpoint, so scoring reproduces what was measured.  A regression-only
  checkpoint resolves to `0.0`, which is why every V1 prediction is still
  bit-identical.

**Why B is the architecture step.** Late concatenation can express a
text × numeric interaction only additively, through the head's first layer.  FiLM
makes it multiplicative and per-dimension, so "which words matter" can depend on
review length, votes or entity counts — the interaction the V2 hand-built
statistics failed to supply.  It is deliberately cheap and safe:

* `γ`/`β` start at zero, so the block is an **exact identity** at step 0;
* its parameters are created **last**, so at the same seed every other parameter
  is bit-identical to a plain run — the two configurations start from the same
  point and differ only in added capacity;
* `film=False` adds no parameters at all, so the existing checkpoints keep
  loading (asserted in `tests/test_ordinal_film.py`).

Neither method has been measured on full data yet: this machine has no GPU, and
only smoke runs were done here (both run end-to-end, `tests` cover the
invariants).  Measure them with the runbook's commands and judge them with
`compare.py`'s fixed rule — ΔRMSE ≤ −0.001 **and** a 95 % CI excluding 0.

Early stopping and best-checkpoint selection still use the **regression** RMSE by
default (`--select-metric auto`), which is the historical rule.  When the blend is
what you would actually submit, `--select-metric blend` selects on it instead;
note that this reuses the validation split twice (once for the out-of-fold blend
weight, once for the epoch), so report it as such.
