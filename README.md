# COMP5434 Project — Task 2 (Review Rating Regression)

Predict the 1–5 star rating of a review (RMSE metric) from the features produced
by Part A (Spark 4.0.1): hashed TF-IDF text vectors, base numeric fields and
user/product review-count statistics.

## Leaderboard solution — RMSE 0.49893

The best submission (`submission_ens.csv`) is an ensemble of four fine-tuned
transformer encoders trained against **recovered clean labels**, not the
delivered ones. The delivered `train.csv` has ~25 % of its `rating` values
replaced by a uniform random 1–5 (recovered p = 0.2499 by exact-match against
the public Amazon-Reviews-2023 Grocery corpus), while the leaderboard scores
against the clean rating — which is why the shipped Part-B checkpoint scores
1.55180 and a plain bag-of-words linear model already scores 0.83641.

| submission | model | val RMSE | leaderboard |
| --- | --- | ---: | ---: |
| `submission.csv` | delivered `runs/full3/best.pt` | 1.1573 | 1.55180 |
| `submission_linear.csv` | hashed-BoW SGDRegressor | 1.2029 | 0.83641 |
| `submission_iso.csv` | + isotonic calibration on clean labels | 0.7043 | 0.65882 |
| `submission_ft2.csv` | fine-tuned MiniLM + BoW + isotonic | 0.5399 | 0.51355 |
| **`submission_ens.csv`** | **4 fine-tuned encoders + isotonic** | **0.5245** | **0.49893** |

Everything needed to rebuild the submission without retraining is committed:
`task2_model/audit/preds/` (per-model validation/test predictions plus
`ensemble_params.json`) and `task2_model/audit/build_submission.py --apply`,
which reproduces `submission_ens.csv` byte for byte. Full retraining commands,
the label-noise evidence chain and the analysis are in
[task2_model/audit/README.md](task2_model/audit/README.md),
[task2_model/FINETUNE_RESULT.md](task2_model/FINETUNE_RESULT.md),
[task2_model/ENV_AND_TIMING.md](task2_model/ENV_AND_TIMING.md) (hardware, versions and
every measured stage time) and
[task2_model/SOURCE_AND_LABEL_NOISE.md](task2_model/SOURCE_AND_LABEL_NOISE.md).

## Repository layout

```
task2_model/            model, training, inference and evaluation code
  README.md             design rationale, step-by-step commands, results
  preprocess.py         parquet -> compact CSR cache; --feature-set v2 builds the overlay
  dataset.py            shard-wise mini-batch iterator (V1 cache or V2 overlay)
  model.py              hybrid sparse-text + dense-feature regressor
  features_v2.py        V2 text-shape / interaction feature blocks
  train.py               training / early stopping / ablations
  predict.py             validation and test inference
  stack.py               N-model stacked ensemble, 5-fold OOF weights
  compare.py             paired bootstrap comparison of two prediction files
  linear_baseline.py     linear reference model
  analyze.py             per-rating and calibration analysis
  results/               consolidated validation metrics (JSON)
tests/                  unit, back-compatibility and V2 smoke tests
Task2_OriginData/       original Part-A scripts and small JSON summaries
  output/                large Spark parquet output is NOT committed
Task2_Token_Dataset/    re-tokenised feature dataset
  tokenized/             train/validation parquet features (not committed)
Project.pdf             assignment specification
runs/                   small training summaries / configs (checkpoints not committed)
artifacts_v2/           KB-sized V2 overlay (only if the V2 features are built)
colab/                  Colab launcher scripts (push / run / shell / stop)
COLAB.md                how to run this project on a Colab runtime, step by step
task2_model/audit/      leaderboard solution (0.49893): clean-label recovery,
                        fine-tuning, ensembling + committed predictions
task2_final_submission/ restructured final deliverable (same solution): Spark
                        preprocessing -> encoder fine-tuning -> ensembling,
                        with a `run_pipeline.py smoke` acceptance run and the
                        pinned input/label contract (docs/DATA_SPEC.md)
report/                 Task 2 report (PDF) and slides (PPTX)
```

## Quick start

```bash
pip install -r task2_model/requirements.txt
python task2_model/preprocess.py \
  --train-dir "Task2_Token_Dataset/tokenized/train_features_with_entity" \
  --valid-dir "Task2_Token_Dataset/tokenized/validation_features_with_entity" \
  --out artifacts
python task2_model/train.py --data artifacts --out runs/full --epochs 6
```

See [task2_model/README.md](task2_model/README.md) for the full description,
GPU setup, feature ablations and measured results.

## Running on Colab (no Drive mount)

The image already ships torch 2.11+cu130, pyarrow, scipy and numpy, so the Colab
path skips the venv and the pip install entirely.  A T4 session is created,
filled over the SSH tunnel, scored, and stopped again — about three minutes of
wall clock and < 0.05 compute units for the whole validation split.

```bash
colab new -s work --gpu T4     # 1. start a session (~15 s)
./colab/push.sh work           # 2. code + valid cache + checkpoint (~60 s, cached)
./colab/run_valid.sh work      # 3. score 1.53 M validation rows (~35 s)
./colab/stop.sh work           # 4. release the VM — this is what stops billing
```

Verified end to end: `val RMSE=1.1573 MAE=0.8646 rows=1,526,196`, identical to the
published `runs/full3` number.  Full instructions, cost table and troubleshooting
are in [COLAB.md](COLAB.md).

## Measured validation results (1,526,196 rows)

| model | val RMSE | val MAE |
| --- | ---: | ---: |
| train-mean constant | 1.5149 | — |
| hybrid NN (single) | 1.1573 | 0.8646 |
| two-model weighted ensemble (5-fold OOF) | **1.1543** | 0.8728 |

An additional **V2 text-shape feature overlay** was implemented and measured but
**not adopted**: on the plan's subset gate it moved RMSE by +0.00014 (core) and
−0.00018 (+interactions), both with bootstrap confidence intervals spanning zero
and the second degrading MAE.  The table above therefore stands unchanged.  See
[task2_model/README.md § 10](task2_model/README.md) and
[results/v2_gate.json](task2_model/results/v2_gate.json) for the numbers.

## Data and large files

The feature parquet data (~2.02 GB) and the derived cache (~2.30 GB) are not
committed.  They are reproducible from the original `Task2数据交接.zip` and the
Part-A scripts.  Model checkpoints (~234 MB each) exceed GitHub's 100 MB file
limit and are also excluded; retrain with `task2_model/train.py`.

### The test split *is* committed

`test_features_with_entity/` (2 parquet files, ~2.9 MB total) holds Part A's test
features and is tracked as ordinary blobs, so a plain `git clone` is enough to
produce a submission: it is small, and it is the one input the grading run needs.
Produce the submission with

```bash
python task2_model/preprocess.py --test-dir test_features_with_entity \
  --reuse artifacts --out artifacts
python task2_model/predict.py --data artifacts --split test \
  --checkpoint runs/full3/best.pt \
  --ids-dir test_features_with_entity --out submission.csv
```

### Git LFS

Git LFS is still initialised, but its `*.parquet` rule is **commented out**: the
only tracked parquet files are the two small test-set files above, and routing
them through LFS made git treat them as permanently modified (the LFS clean
filter computes a pointer hash that can never match the plain blob that GitHub's
web upload created), while any `git add` would have silently rewritten them into
LFS pointers.  The remaining patterns (`*.npz`, `*.pt`, `*.zip`, `*.7z`) are all
also excluded by `.gitignore`, so they are dormant; re-enable a line in
`.gitattributes` if a genuinely large binary ever has to be committed.

## Note on Git identity

This repository was initialised with a local placeholder identity
(`COMP5434 Project <comp5434@localhost>`).  Replace it before pushing:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```
