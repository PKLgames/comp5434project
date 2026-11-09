# COMP5434 Project - Task 2 Submission Code

This folder combines both members' essential Task 2 work: Spark data preparation and feature experiments, source-rating recovery, encoder fine-tuning, four-member ensembling, two-member pruning, calibration, submission checks and timing. The README is in English. Commands run **one stage at a time**; opening this folder runs nothing.

## 0. Package layout

The submission archive keeps the report and the code in two sibling folders:

```text
comp5434project_submission/
  code/     everything in this folder (this README lives here)
  report/   Task2_Report_English.pdf + slides_Task2.pptx
```

Commands in this README are run from `code/`; the report is at `../report/`.

## 1. Package map

| Location | Purpose |
| --- | --- |
| `spark_preprocessing/` | Final Spark preprocessing, environment/input checks, pinned input acquisition and small CSV compatibility checks. |
| `final_model/recover_clean_labels.py` | Recover source ratings from public Amazon reviews when starting from the public source file. |
| `final_model/finetune.py` | Train the encoder and regression head jointly; save checkpoints, predictions and epoch timings. |
| `final_model/build_submission.py` | Fit/apply the four-member ensemble and isotonic calibration. |
| `final_model/build_submission_pruned.py` | Fit/apply the MiniLM+BGE two-member ensemble. |
| `final_model/measure_pipeline.py` | Fresh inference on all 10,000 test records, using actual trained checkpoints. |
| `final_model/check_submission.py`, `evaluate_saved.py` | Check CSVs and evaluate archived validation predictions. |
| `final_model/preds/`, `reference/` | Small saved arrays, parameters, validation-label slice and both archived submissions. |
| `spark_experiments/`, `baseline_model/` | Reproduce the earlier feature-dimension, hybrid-model and parallel-processing experiments. |
| `run_pipeline.py` | Separate `verify`, `reproduce`, `smoke`, `preprocess`, `train`, `calibrate`, `infer` commands. |
| `summarize_timings.py` | Summarize a new run's epoch, offline and full-test inference timings. |
| `smoke_test.py` | Quick acceptance run (truncated inputs): real Spark preprocessing, one small CPU training step, prediction for all 10,000 test rows and a submission check. |
| `docs/` | Recorded results, timing evidence, data specification and experiment instructions. |
| `SOURCE_MANIFEST.json`, `MANIFEST.sha256` | Source provenance and delivered-file checksums. |

Raw CSVs, the 6 GB public source, large Parquet/features, model weights, virtual environments, caches and old run directories are not copied into this package. The complete source-label archive is fetched explicitly in Step 3 rather than duplicated. The saved-submission route needs no trained checkpoints. Full retraining creates them.

## 2. Environment and quick verification

Use Python 3.11 and a separate environment. In PowerShell, from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py verify
```

`verify` compiles source in memory, checks hashes/import dependencies and compares both CSV reconstructions byte-for-byte with the supplied references. It reads small saved arrays and removes its temporary CSVs. It does not launch Spark, download files, train or run neural networks.

The saved route requires NumPy only. On Linux/macOS use `source .venv/bin/activate` for activation; the Python stage commands are the same. Windows-specific Hadoop setup does not apply there.

To reconstruct either archived submission explicitly:

```powershell
python run_pipeline.py reproduce --variant full
python run_pipeline.py reproduce --variant pruned
```

Outputs are `work/submission_saved_full.csv` and `work/submission_saved_pruned.csv`. Reference public-leaderboard RMSE values are **0.49893** and **0.49914**. This route combines stored predictions; it is separate from fresh inference. Output files are not overwritten. For a repeat, choose a fresh `--work-dir`.

Optional saved-validation evaluation:

```powershell
python final_model/evaluate_saved.py
```

The small `validation_labels.npz` contains the archived validation slice only. It is sufficient for archived evaluation/calibration and is not a substitute for the complete training-label archive.

## 3. Prepare inputs for the final Spark pipeline

Install Spark dependencies, Java 21, and set `JAVA_HOME`. On Windows also set `HADOOP_HOME` to a compatible installation containing `bin/winutils.exe` and `bin/hadoop.dll`, and add its `bin` to `PATH`.

```powershell
python -m pip install -r requirements_spark.txt
```

Put the competition files in `data/`:

```text
data/
  train.csv
  test.csv
  sampleSubmission.csv
```

The raw files are provided separately through the course Kaggle competition. Do not replace the unlabelled competition test set with external reviews.

Obtain the complete, pinned source-rating archive and verify its size/content hash:

```powershell
python spark_preprocessing/02_prepare_inputs.py --data-dir data
```

This downloads `clean_y.npz`, `DATA_SPEC.md` and the upstream reference audit script into `spark_preprocessing/input/`, fixed to repository commit `261e9e648cc0b1080fae0fe192f9ab1fa7635946`. The upstream audit script is reference material and is not executed automatically. The checked input archive contains `cy`, `matched`, `ag`, indexed by original review ID.

Repeat the input/environment check independently if needed:

```powershell
python spark_preprocessing/01_check_inputs.py --data-dir data
```

You can use an existing CSV directory through `--data-dir` without copying the 2 GB training file. For example, when this folder remains inside the original project, `--data-dir ..` refers to the parent directory. Use the same data directory for preprocessing, calibration and inference.

## 4. Run full Spark preprocessing

```powershell
python run_pipeline.py preprocess --data-dir data --threads 4
```

This first checks 200 train/test records and synthetic quoting/newline cases, then runs the full Spark job. The full job fills missing title/comment with empty strings, constructs `title + ". " + comment`, joins source ratings by ID, checks eligibility and writes ordered model inputs. Case, whitespace and embedded newlines are retained; model tokenizers control sequence truncation. Spark computes the text, join and ordering; PyArrow packages the ordered Spark fragments.

```text
work/preprocessed/
  train_text.parquet          # id:int64, text:string; 7,684,645 rows
  test_text.parquet           # id:int64, text:string; 10,000 rows
  labels_by_id.parquet        # id, cy, matched, ag
  clean_y.npz                 # arrays reconstructed from the Spark join
  preprocessing_summary.json  # counts, stage times and full-content checks
  MODEL_INPUTS.json
```

For the pinned labels, the actual counts are 7,108,463 valid source matches, 177,607 matched validation records and **6,930,856 eligible training records**. Eligibility is `id % 40 != 0` with a valid matched source rating; validation uses `id % 40 == 0`. All original text rows are retained in the model files. Text and IDs are checked against every original CSV record, and exported labels are compared element-by-element.

`work/preprocessed` must not already exist. For another run choose a fresh `--work-dir`; the corresponding output becomes `<work-dir>/preprocessed`. A Windows file lock can leave that run's `_scratch` folder after success; `preprocessing_summary.json` must still report `SPARK_FULL_PREPROCESSING_PASS` before training.

### Optional: recover source ratings from the public reviews

Instead of fetching the published archive, download the public JSONL separately:

[Amazon Reviews 2023 Grocery_and_Gourmet_Food](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Grocery_and_Gourmet_Food.jsonl).

Install the training dependencies from Step 5, then run:

```powershell
python final_model/recover_clean_labels.py --source data/grocery.jsonl --train data/train.csv --out work/recovered_y.npz
python run_pipeline.py preprocess --data-dir data --labels work/recovered_y.npz --label-origin recovered
```

Recovery matches a blake2b text fingerprint followed by timestamp, retaining the original candidate-order rule and skipping texts with more than 64 candidates. It streams multiline competition CSVs correctly and does not save large recovery-index arrays. The `recovered` option records the new archive's SHA-256; the default `published` option requires the exact pinned archive hash. Do not run both preprocessing alternatives into the same output directory.

## 5. Train the encoders using Spark outputs

Select a suitable PyTorch CUDA installation for the GPU, then install:

```powershell
python -m pip install -r requirements_train.txt
```

The reported configurations target cloud GPUs, not a 6 GB laptop GPU. Training uses an `AutoModel` encoder, its first token's final hidden state and a linear regression head. The encoder/head are jointly optimized with MSE, AdamW at 2e-5, weight decay 0.01, 5% warmup, cosine scheduling, gradient clipping and CUDA autocast. The supplied trainer configurations use seeds 1, 2, 1, 1.

Run one member at a time:

```powershell
python run_pipeline.py train --tags s1 --device cuda
python run_pipeline.py train --tags s2 --device cuda
python run_pipeline.py train --tags d1 --device cuda
python run_pipeline.py train --tags b1 --device cuda
```

| Tag | Pretrained encoder | Maximum tokens | Batch | Epochs |
| --- | --- | --- | --- | --- |
| s1 | sentence-transformers/all-MiniLM-L6-v2 | 96 | 512 | 2 |
| s2 | sentence-transformers/all-MiniLM-L6-v2 | 128 | 1,024 | 2 |
| d1 | distilbert-base-uncased | 128 | 768 | 2 |
| b1 | BAAI/bge-small-en-v1.5 | 128 | 768 | 2 |

Each command reads `work/preprocessed/train_text.parquet`, `test_text.parquet`, and `clean_y.npz`. Outputs in `work/models/` are `ft3_<tag>.pt` (state/args), `ftv_<tag>.npy`, `ftvids_<tag>.npy`, `ftp_<tag>.npy`, `ft3_<tag>.json`, and `timing_<tag>.json`. Epoch timings separate training from training-plus-validation. The driver also records the successful child process under `work/timings/`.

The original fine-tuned checkpoints are not included or downloadable from the pinned repository. These commands retrain them; archived CSVs remain exactly reproducible from Step 2. A new training run can differ numerically with library/model revisions and hardware. Do not mix its prediction arrays with archived arrays. To change memory-related parameters deliberately, invoke `final_model/finetune.py --help` and use a separate output directory.

## 6. Fit ensemble weights and calibration

After the required members finish:

```powershell
python run_pipeline.py calibrate --variant full --data-dir data
python run_pipeline.py calibrate --variant pruned --data-dir data
```

The four-member variant searches 400 Dirichlet draws on matched development validation records and fits isotonic calibration on even positions, evaluating odd positions. The two-member variant selects s1+b1 weights by a 0.01 grid on even positions, fits calibration there and evaluates odd positions. Saved reported two-member weights are 0.33/0.67. Validation scores are development metrics; the Kaggle figures are public-leaderboard results.

Each variant writes fresh parameters in `work/models/`, a separate `work/submission_refit_<variant>.csv` and process timing. Applying fitted parameters uses the same arithmetic in rebuilding and fresh inference. Four-member blending clips members then uses float64 and three-decimal weights; two-member blending preserves the source float32 arithmetic and clips the blend. CSV formatting follows each archived submission.

## 7. Measure fresh inference for all test records

With the new trained checkpoints and new parameters:

```powershell
python run_pipeline.py infer --variant full --data-dir data --device cuda
python run_pipeline.py infer --variant pruned --data-dir data --device cuda
```

The default inference batch is 256; use `--inference-bs` explicitly to change it. CPU timing can be selected with `--device cpu --num-threads 16`.

Each run processes all 10,000 records and records Parquet reading, checkpoint/config/tokenizer loading, device transfer, tokenization, every selected encoder forward, blend, calibration and CSV output. CUDA is synchronized at boundaries. The total excludes Python startup/imports, environment capture and earlier Spark preprocessing; downloads triggered during model loading are included. Per-model maximum lengths come from checkpoints. Outputs are `work/submission_fresh_<variant>.csv` and `work/full_test_inference_<variant>.json`.

Use the **actual trained weights** for new final-model inference timings. The report's archived approximately 403/157.2-second measurements are same-architecture base-weight CPU benchmarks, not the result of running this new package with the missing original fine-tuned checkpoints.

Summarize a completed new run:

```powershell
python summarize_timings.py --variant full --out work/efficiency_full.json
python summarize_timings.py --variant pruned --out work/efficiency_pruned.json
```

This collects epoch records, Spark preprocessing, successful training-process times, calibration and all-test inference. Offline time is the sum of those preparation/training/calibration stages; fresh inference is separate. Recovery/download time is separate when that optional route is used. A pruned-only summary counts its two selected training members.

## 8. Earlier Spark/feature and hybrid-model experiments

These scripts support the report's early exploration, dimension comparison and thread benchmark. They use the delivered competition ratings and the earlier cleaning/split rules. They are separate from the final source-rating training inputs.

See **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** for the exact execution order and baseline commands. They are optional for producing the final submission, and their large outputs are not packaged.

## 9. Verification and source records

The source repository snapshot is `261e9e648cc0b1080fae0fe192f9ab1fa7635946`. The copied trainer comes from the previously integrated team submission package; provenance is retained in `SOURCE_MANIFEST.json`. `docs/DATA_SPEC.md`, `FINETUNE_RESULT.md`, `ENV_AND_TIMING.md` preserve upstream records. Counts for the pinned input come from the full archive rather than older prose totals. `docs/spark_preprocessing_record.json` records the completed local Spark run (61.435 seconds; 66.985 seconds with independent text checks).

Packaging changes are limited to portable paths/environment setup, Spark integration, explicit stage commands, CSV/parameter consistency, support for the pruned parameters, input validation, multiline CSV recovery and timing capture. No raw data, pretrained downloads, full Spark jobs, training or neural inference is executed during packaging. `verify_package.py` documents the lightweight checks. No ZIP is created.


## 10. Quick acceptance run (Spark → train → predict)

`smoke_test.py` proves the whole chain works without the full 2 GB training file or a GPU. It
truncates the training set (default: the first 20,000 records), runs **real Spark**
preprocessing with the same text rule (`title + ". " + comment`), the same label join and the
same `id % 40` split rule, writes the same Parquet schemas, trains one encoder member on
4,000 rows for one epoch on CPU, predicts all 10,000 real test rows and validates the CSV.

```bash
# Java 17+ must be available (JAVA_HOME or `java` on PATH)
python -m pip install -r requirements_spark.txt -r requirements_train.txt
python run_pipeline.py smoke --data-dir data --work-dir work_smoke
```

The pinned source-label archive is bundled at `spark_preprocessing/input/clean_y.npz`
(SHA-256 `5c133167…dee6f`), so the acceptance run needs no download; Step 3 is only
required if you want to re-fetch it. Typical runtime on a 16-thread CPU: ~3 min with
`--rows 5000 --limit 1000`, ~9 min with the defaults.

Outputs live in `work_smoke/`:

| Path | Content |
| --- | --- |
| `data_small/` | truncated `train.csv` + full `test.csv` / `sampleSubmission.csv` + `clean_y_small.npz` |
| `preprocessed/train_text/train_text.parquet` | `id,text` rows from the Spark job (all 20,000) |
| `preprocessed/test_text/test_text.parquet` | `id,text` for the 10,000 test rows |
| `preprocessed/clean_y.npz`, `smoke_preprocessing_summary.json` | labels and the `SMOKE_PREPROCESS_PASS` record |
| `models/ft3_smoke.pt`, `ftp_smoke.npy`, `ftv_smoke.npy` | the trained member and its predictions |
| `submission_smoke.csv` | `id,rating` for all 10,000 test rows |
| `smoke_result.json` | `SMOKE_PASS` plus the stage summary |

The smoke path deliberately does not call `04_preprocess_full.py`: that script pins the exact
full-data counts and the complete label domain, which a truncated input cannot satisfy. The
text rule, eligibility filter, split rule, id alignment and Parquet schemas are identical.
Useful options: `--rows`, `--limit`, `--model`, `--epochs`, `--maxlen`, `--device`,
`--preprocess-only`, `--force`. Smoke numbers are a plumbing check, not a score estimate.
