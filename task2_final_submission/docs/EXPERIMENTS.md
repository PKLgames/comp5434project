# Earlier feature and parallel experiments

Run from the package root with the Spark and training requirements installed. Set
the CSV directory explicitly; the scripts write only to `spark_experiments/output/`.

```powershell
$env:TASK2_DATA_DIR = (Resolve-Path data).Path
python spark_experiments/analyze_data.py
python spark_experiments/clean_data.py
python spark_experiments/split_train.py
python spark_experiments/feature_processing.py
python spark_experiments/text_vectorization_old.py
python spark_experiments/entity_features.py
python spark_experiments/text_vectorization.py --title-dim 4096 --comment-dim 16384
```

The earlier cleaning retains 7,630,970 records. Its fixed stratified split has
6,104,774 training and 1,526,196 validation records. IDF and entity counts are
fitted on training data only. Entity inputs are counts, log counts and unseen
flags for user, product and parent product; they contain no mean-rating statistic.

`text_vectorization_old.py` is retained specifically for the report's high-dimension
comparison (65,536 title / 262,144 body). `text_vectorization.py` writes a separate,
timestamped low-dimension experiment (4,096 / 16,384), reusing entity statistics
when available. Its printed output path identifies the reduced feature dataset.

## Thread benchmark

After cleaning/splitting:

```powershell
python spark_experiments/parallel_benchmark.py --threads 1
python spark_experiments/parallel_benchmark.py --threads 2
python spark_experiments/parallel_benchmark.py --threads 4
python spark_experiments/parallel_benchmark.py --summary
```

Each setting uses one warm-up and three measured repetitions. The workload reads
the training split, constructs numeric features and aggregates the results.
Reported medians are 83.47 / 43.46 / 26.97 seconds, giving approximately 3.10x at
four threads. Startup, warm-up and output writing are outside this benchmark.

## Hybrid model

High-dimensional features:

```powershell
python baseline_model/preprocess.py --train-dir spark_experiments/output/train_features_with_entity --valid-dir spark_experiments/output/validation_features_with_entity --out work/baseline_high_cache
python baseline_model/train.py --data work/baseline_high_cache --out work/baseline_high --epochs 3 --batch-size 8192 --valid-batch-size 8192 --lr 0.003 --weight-decay 0.0001 --device cpu --num-threads 16 --seed 42
```

For the low-dimensional version, replace the feature paths with the
`train_features_with_entity` and `validation_features_with_entity` directories in
the timestamped low-dimension run. Use fresh cache and run output directories.

The report's high-/low-dimensional validation RMSE values are 1.157312/1.158683.
The low-dimensional run's epoch times are 214.34, 220.69 and 230.24 seconds.
These earlier results use their own delivered-rating split and are not combined
with the final source-rating validation scores.

## Final-model experiment records

The essential final encoder, ensemble and pruning implementations are in
`final_model/`. `FINETUNE_RESULT.md` and `ENV_AND_TIMING.md` retain the team's
recorded intermediate comparisons and resource measurements. Superseded trial
scripts, notebooks and caches are omitted; the package reproduces the final
pipeline and the earlier feature/parallel experiments above.
