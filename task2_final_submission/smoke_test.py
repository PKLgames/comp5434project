"""Quick end-to-end acceptance run: Spark preprocessing -> training -> prediction.

This is the small, non-full-scale version of the pipeline. It

1. builds a truncated input set (the first N training records, the full 10,000
   test records, the submission sample and a matching slice of the source
   labels) under ``<work-dir>/data_small``;
2. runs a real Spark job that constructs the model text exactly as the full
   pipeline does (``title + ". " + comment``), joins the recovered source
   ratings by id and writes the same parquet inputs the full run produces;
3. trains one encoder member with ``final_model/finetune.py --limit`` (CPU is
   fine for this size);
4. writes a submission from that member's 10,000 test predictions and validates
   it with ``final_model/check_submission.py``.

It is deliberately independent of ``04_preprocess_full.py``: that script pins
the exact full-data row counts and label domain, which a truncated set cannot
satisfy. Everything else (text rule, id split ``id % 40``, eligibility filter,
parquet schemas, submission format) is identical.

Requirements: Java 17+ (set ``JAVA_HOME`` or have ``java`` on PATH) and
``pyspark`` from ``requirements_spark.txt``; the training step needs
``requirements_train.txt``. Nothing runs on import.

    python smoke_test.py --data-dir data --labels spark_preprocessing/input/clean_y.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TRAIN_COLUMNS = ['id', 'title', 'user_id', 'comment', 'prod_id',
                 'parent_prod_id', 'time', 'votes', 'purchased', 'rating']


def log(message: str) -> None:
    print('[smoke] ' + message, flush=True)


def check_java() -> None:
    java_home = os.environ.get('JAVA_HOME')
    java = Path(java_home) / 'bin' / 'java' if java_home else None
    candidate = str(java) if java and java.exists() else shutil.which('java')
    if not candidate:
        raise SystemExit('Java not found. Install a JRE 17+ and set JAVA_HOME, '
                         'or put `java` on PATH (Spark needs it).')
    out = subprocess.run([candidate, '-version'], capture_output=True, text=True)
    version = (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else ''
    log('java: %s (%s)' % (candidate, version))
    if java and java.exists():
        os.environ.setdefault('JAVA_HOME', str(Path(java).parent.parent))


def prepare_inputs(data: Path, work: Path, rows: int, labels: Path) -> Path:
    """Truncate the competition CSVs and the label archive for a small run."""
    import numpy as np

    small = work / 'data_small'
    small.mkdir(parents=True, exist_ok=True)
    src = data / 'train.csv'
    if not src.exists():
        raise SystemExit('missing training file: %s' % src)
    written = 0
    with src.open(encoding='utf-8-sig', newline='') as stream, \
            (small / 'train.csv').open('w', encoding='utf-8', newline='') as out:
        reader = csv.reader(stream)
        writer = csv.writer(out)
        header = next(reader)
        if header != TRAIN_COLUMNS:
            raise SystemExit('train.csv column order differs from DATA_SPEC: %s' % header)
        writer.writerow(header)
        for record in reader:
            if len(record) != len(TRAIN_COLUMNS):
                raise SystemExit('unexpected record length at row %d' % (written + 1))
            writer.writerow(record)
            written += 1
            if written >= rows:
                break
    shutil.copyfile(data / 'test.csv', small / 'test.csv')
    shutil.copyfile(data / 'sampleSubmission.csv', small / 'sampleSubmission.csv')
    log('inputs: %d training records + 10,000 test records -> %s' % (written, small))

    with np.load(labels, allow_pickle=False) as archive:
        cy, matched, ag = archive['cy'], archive['matched'], archive['ag']
    if len(cy) < rows:
        raise SystemExit('label archive has %d rows, need at least %d' % (len(cy), rows))
    trimmed = small / 'clean_y_small.npz'
    np.savez_compressed(trimmed, cy=cy[:rows], matched=matched[:rows], ag=ag[:rows])
    valid = int(np.count_nonzero(matched[:rows] & (cy[:rows] >= 1)))
    log('labels: %d rows kept, %d matched with a valid source rating' % (rows, valid))
    if valid < 1000:
        raise SystemExit('not enough matched labels in the truncated slice: %d' % valid)
    return small


def spark_preprocess(small: Path, out: Path) -> dict:
    """Small Spark job with the same text rule and eligibility filter as the full run."""
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession, functions as F, types as T

    spark = (SparkSession.builder.appName('Task2SmokePreprocess').master('local[2]')
             .config('spark.driver.memory', '2g')
             .config('spark.sql.shuffle.partitions', '2')
             .config('spark.ui.enabled', 'false')
             .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    started = time.perf_counter()
    try:
        text_schema = T.StructType([
            T.StructField('id', T.StringType()), T.StructField('title', T.StringType()),
            T.StructField('user_id', T.StringType()), T.StructField('comment', T.StringType()),
            T.StructField('prod_id', T.StringType()), T.StructField('parent_prod_id', T.StringType()),
            T.StructField('time', T.StringType()), T.StructField('votes', T.StringType()),
            T.StructField('purchased', T.StringType()), T.StructField('rating', T.StringType())])

        def load(name, with_rating):
            schema = text_schema if with_rating else T.StructType([f for f in text_schema.fields if f.name != 'rating'])
            frame = (spark.read.option('multiLine', 'true').option('escape', '"')
                     .option('header', 'true').schema(schema).csv(str(small / name)))
            text = F.concat_ws('. ', F.coalesce(F.col('title'), F.lit('')),
                               F.coalesce(F.col('comment'), F.lit('')))
            frame = frame.withColumn('id', F.col('id').cast('long')).withColumn('text', text)
            if with_rating:
                frame = frame.withColumn('rating', F.col('rating').cast('int'))
            return frame

        train = load('train.csv', True)
        test = load('test.csv', False)

        import numpy as np
        with np.load(small / 'clean_y_small.npz', allow_pickle=False) as archive:
            cy, matched, ag = archive['cy'], archive['matched'], archive['ag']
        labels = spark.createDataFrame(
            [(int(i), int(cy[i]), bool(matched[i]), int(ag[i])) for i in range(len(cy))],
            schema=T.StructType([T.StructField('id', T.LongType()), T.StructField('cy', T.IntegerType()),
                                 T.StructField('matched', T.BooleanType()), T.StructField('ag', T.IntegerType())]))

        joined = train.join(labels, on='id', how='inner')
        # All training rows are kept in the text file, exactly as in the full run:
        # the trainer selects the id % 40 validation slice and the eligible
        # training rows itself. Only the summary counts the eligible rows.
        train_rows = joined.select('id', 'text', 'cy', 'ag')
        test_rows = test.select('id', 'text')
        n_all = train_rows.count()
        n_eligible = joined.where((joined['id'] % 40 != 0) & joined['matched']
                                  & (joined['cy'] >= 1) & (joined['cy'] <= 5)).count()
        n_test = test_rows.count()
        if n_all == 0 or n_test != 10000:
            raise SystemExit('unexpected smoke counts: train=%d test=%d' % (n_all, n_test))

        out.mkdir(parents=True, exist_ok=True)
        for frame, name in ((train_rows, 'train_text'), (test_rows, 'test_text')):
            target = out / name
            frame.orderBy('id').coalesce(1).write.mode('overwrite').parquet(str(target))
            parts = sorted((target).glob('part-*.parquet'))
            if len(parts) != 1:
                raise SystemExit('expected one parquet part for %s, got %d' % (name, len(parts)))
            data = pq.read_table(parts[0])
            target.mkdir(exist_ok=True)
            pq.write_table(data, target / (name + '.parquet'), compression='zstd')
            parts[0].unlink()
            for leftover in ('_SUCCESS',):
                path = target / leftover
                if path.exists():
                    path.unlink()
        np.savez_compressed(out / 'clean_y.npz', cy=cy, matched=matched, ag=ag)
        seconds = time.perf_counter() - started
        summary = {'status': 'SMOKE_PREPROCESS_PASS', 'spark_version': spark.version,
                   'text_rows': n_all, 'eligible_training_rows': n_eligible,
                   'test_rows': n_test, 'seconds': round(seconds, 2)}
        (out / 'smoke_preprocessing_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        log('spark: %d text rows (%d eligible for training), %d test rows in %.1f s'
            % (n_all, n_eligible, n_test, seconds))
        return summary
    finally:
        spark.stop()


def train(inputs: Path, models: Path, args) -> Path:
    command = [sys.executable, '-B', str(HERE / 'final_model' / 'finetune.py'),
               '--train-text', str(inputs / 'train_text' / 'train_text.parquet'),
               '--test-text', str(inputs / 'test_text' / 'test_text.parquet'),
               '--clean', str(inputs / 'clean_y.npz'),
               '--out-dir', str(models), '--model', args.model, '--epochs', str(args.epochs),
               '--bs', str(args.bs), '--lr', str(args.lr), '--maxlen', str(args.maxlen),
               '--seed', '1', '--tag', 'smoke', '--evalsub', '2000', '--limit', str(args.limit),
               '--device', args.device, '--predict-bs', '256']
    log('train: ' + subprocess.list2cmdline(command))
    models.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=HERE, check=True)
    state = models / 'ft3_smoke.pt'
    if not state.exists():
        raise SystemExit('training did not produce %s' % state)
    return models / 'ftp_smoke.npy'


def write_submission(preds: Path, out_csv: Path) -> None:
    import numpy as np
    ratings = np.clip(np.load(preds).astype(np.float64), 1.0, 5.0)
    with out_csv.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'rating'])
        for i, value in enumerate(ratings):
            writer.writerow([i, round(float(value), 6)])
    log('submission: %d rows, mean %.4f -> %s' % (len(ratings), float(ratings.mean()), out_csv))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data')
    parser.add_argument('--work-dir', type=Path, default=HERE / 'work_smoke')
    parser.add_argument('--labels', type=Path, default=None,
                        help='full clean_y.npz (default: spark_preprocessing/input/clean_y.npz, '
                             'falling back to final_model/preds/clean_y.npz)')
    parser.add_argument('--rows', type=int, default=20000, help='training records kept (default 20000)')
    parser.add_argument('--limit', type=int, default=4000, help='training rows used by the encoder (default 4000)')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--maxlen', type=int, default=64)
    parser.add_argument('--device', default='cpu', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--force', action='store_true', help='remove an existing --work-dir')
    parser.add_argument('--preprocess-only', action='store_true', help='stop after the Spark stage')
    args = parser.parse_args()

    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        if not args.force:
            raise SystemExit('work dir is not empty: %s (choose another --work-dir or pass --force)' % work)
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    labels = args.labels
    if labels is None:
        for candidate in (HERE / 'spark_preprocessing' / 'input' / 'clean_y.npz',
                          HERE / 'final_model' / 'preds' / 'clean_y.npz'):
            if candidate.exists():
                labels = candidate
                break
        else:
            raise SystemExit('no clean_y.npz found; pass --labels or run '
                             'spark_preprocessing/02_prepare_inputs.py')
    check_java()
    small = prepare_inputs(args.data_dir.resolve(), work, args.rows, labels.resolve())
    summary = spark_preprocess(small, work / 'preprocessed')
    if args.preprocess_only:
        print('SMOKE_PREPROCESS_PASS ' + json.dumps(summary), flush=True)
        return
    preds = train(work / 'preprocessed', work / 'models', args)
    submission = work / 'submission_smoke.csv'
    write_submission(preds, submission)
    check = [sys.executable, '-B', str(HERE / 'final_model' / 'check_submission.py'), str(submission),
             '--sample', str(small / 'sampleSubmission.csv')]
    subprocess.run(check, cwd=HERE, check=True)
    result = {'status': 'SMOKE_PASS', 'submission': str(submission), 'preprocessing': summary,
              'model': args.model, 'limit': args.limit, 'rows': args.rows}
    (work / 'smoke_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('SMOKE_PASS ' + json.dumps({k: result[k] for k in ('rows', 'limit', 'model')}), flush=True)


if __name__ == '__main__':
    main()
