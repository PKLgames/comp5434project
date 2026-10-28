# COMP5434 Project — Task 2 (Review Rating Regression)

Predict the 1–5 star rating of a review (RMSE metric) from the features produced
by Part A (Spark 4.0.1): hashed TF-IDF text vectors, base numeric fields and
user/product review-count statistics.

## Repository layout

```
task2_model/            model, training, inference and evaluation code
  README.md             design rationale, step-by-step commands, results
  preprocess.py         parquet -> compact CSR cache
  model.py              hybrid sparse-text + dense-feature regressor
  train.py               training / early stopping / ablations
  predict.py             validation and test inference
  linear_baseline.py     linear reference model
  analyze.py             per-rating and calibration analysis
  results/               consolidated validation metrics (JSON)
Task2_OriginData/       original Part-A scripts and small JSON summaries
  output/                large Spark parquet output is NOT committed
Task2_Token_Dataset/    re-tokenised feature dataset
  新token化/             train/validation parquet features (not committed)
Project.pdf             assignment specification
runs/                   small training summaries / configs (checkpoints not committed)
```

## Quick start

```bash
pip install -r task2_model/requirements.txt
python task2_model/preprocess.py \
  --train-dir "Task2_Token_Dataset/新token化/train_features_with_entity" \
  --valid-dir "Task2_Token_Dataset/新token化/validation_features_with_entity" \
  --out artifacts
python task2_model/train.py --data artifacts --out runs/full --epochs 6
```

See [task2_model/README.md](task2_model/README.md) for the full description,
GPU setup, feature ablations and measured results.

## Measured validation results (1,526,196 rows)

| model | val RMSE | val MAE |
| --- | ---: | ---: |
| train-mean constant | 1.5149 | — |
| hybrid NN (single) | 1.1573 | 0.8646 |
| two-model weighted ensemble (5-fold OOF) | **1.1543** | 0.8728 |

## Data and large files

The feature parquet data (~2.02 GB) and the derived cache (~2.30 GB) are not
committed.  They are reproducible from the original `Task2数据交接.zip` and the
Part-A scripts.  Model checkpoints (~234 MB each) exceed GitHub's 100 MB file
limit and are also excluded; retrain with `task2_model/train.py`.

Git LFS is initialised.  `.gitattributes` routes parquet/npz/checkpoint/archive
files through LFS so that future 50–100 MB artifacts can be committed safely.

## Note on Git identity

This repository was initialised with a local placeholder identity
(`COMP5434 Project <comp5434@localhost>`).  Replace it before pushing:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```
