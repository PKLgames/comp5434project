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
  --train-dir "Task2_Token_Dataset/新token化/train_features_with_entity" \
  --valid-dir "Task2_Token_Dataset/新token化/validation_features_with_entity" \
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
  preprocess.py        parquet → compact CSR shards
  dataset.py           shard-wise mini-batch iterator
  model.py             hybrid sparse-text / dense model
  train.py             training loop, early stopping, metrics
  predict.py           validation / test inference and submission writer
  linear_baseline.py   streaming linear reference model
  analyze.py           per-rating / calibration error analysis
  requirements.txt
  README.md
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
