"""Step 3: small Spark text-compatibility test and full-label count audit.

Reads only the first 200 records of each original CSV and a few synthetic
edge cases. Does not preprocess the full dataset, modify inputs, or train.
"""
import argparse
import csv
import ctypes
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent


def short_path(path):
    value = str(Path(path).resolve())
    if os.name != 'nt':
        return value
    size = ctypes.windll.kernel32.GetShortPathNameW(value, None, 0)
    if not size:
        return value
    buffer = ctypes.create_unicode_buffer(size)
    if not ctypes.windll.kernel32.GetShortPathNameW(value, buffer, size):
        return value
    return buffer.value


def configure_spark():
    spec = importlib.util.find_spec('pyspark')
    if spec is None or spec.origin is None:
        raise RuntimeError('PySpark is not installed in this interpreter')
    python = short_path(sys.executable)
    os.environ['PYSPARK_PYTHON'] = python
    os.environ['PYSPARK_DRIVER_PYTHON'] = python
    os.environ['SPARK_HOME'] = short_path(Path(spec.origin).parent)
    if os.name == 'nt':
        hadoop = Path(os.environ.get('HADOOP_HOME', r'D:\HADOOP'))
        os.environ['HADOOP_HOME'] = short_path(hadoop)
        os.environ['PATH'] = str(hadoop / 'bin') + os.pathsep + os.environ.get('PATH', '')


def source_head(path, limit=200):
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            rows.append((int(row['id']), row['title'], row['comment']))
            if i + 1 == limit:
                break
    return rows


def record_separator(path):
    """Inspect only the header; keep record endings separate from text endings."""
    with path.open('rb') as stream:
        header = stream.readline()
    return '\r\n' if header.endswith(b'\r\n') else '\n'


def audit_labels(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as labels:
        cy, matched, ag = labels['cy'], labels['matched'], labels['ag']
        if any(x.shape != (7_684_645,) for x in (cy, matched, ag)):
            raise ValueError('Unexpected full-label shape')
        valid = matched & (cy >= 1) & (cy <= 5)
        total = int(np.count_nonzero(valid))
        valid_count = int(np.count_nonzero(valid[::40]))
        agreement = float(np.mean(cy[valid] == ag[valid]))
        return {
            'source': str(path), 'rows': len(cy),
            'matched_flag_rows': int(np.count_nonzero(matched)),
            'valid_source_label_rows': total,
            'matched_with_invalid_rating_rows': int(np.count_nonzero(matched & ~valid)),
            'valid_rating_without_match_rows': int(np.count_nonzero(~matched & (cy > 0))),
            'matched_validation_rows': valid_count,
            'training_rows': total - valid_count,
            'delivered_source_agreement': agreement,
            'policy': 'Counts are derived from the full label artifact; no rows are removed to match prose statistics.',
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PACKAGE / 'data')
    parser.add_argument('--labels', type=Path, default=HERE / 'input' / 'clean_y.npz')
    args = parser.parse_args()
    project = args.data_dir.resolve()
    label_audit = audit_labels(args.labels.resolve())
    print('LABEL_COUNT_AUDIT =', json.dumps(label_audit), flush=True)
    configure_spark()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession, functions as F, types as T

    run = HERE / 'work' / ('spark_smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    schema = T.StructType([T.StructField('id', T.StringType()), T.StructField('title', T.StringType()), T.StructField('comment', T.StringType())])
    cases = {
        'train_head': source_head(project / 'train.csv'),
        'test_head': source_head(project / 'test.csv'),
        'edge_cases': [
            (0, 'Five Stars', 'It popped fluffy and big!'),
            (1, '', ''), (2, None, None),
            (3, '  Keep Spaces  ', ' \t Keep CASE and spacing \t '),
            (4, 'Quoted "title", comma', 'Text with "double quotes", commas.'),
            (5, 'LF', 'first line\nsecond line'),
            (6, 'CRLF', 'first line\r\nsecond line'),
            (7, 'NULL', 'NA'),
            (8, 'Unicode \u4e2d\u6587', 'Caf\u00e9 \U0001f600 <b>Great!</b>'),
            (9, 'No normalisation', 'DO NOT lower-case, strip HTML, or remove stop words.'),
        ],
    }
    cases['edge_cases_crlf_records'] = list(cases['edge_cases'])
    separators = {
        'train_head': record_separator(project / 'train.csv'),
        'test_head': record_separator(project / 'test.csv'),
        'edge_cases': '\n',
        'edge_cases_crlf_records': '\r\n',
    }
    started = time.perf_counter()
    spark = SparkSession.builder.appName('Task2SparkInputCompatibility').master('local[2]').config('spark.driver.memory', '2g').config('spark.sql.shuffle.partitions', '2').config('spark.ui.enabled', 'false').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    results = {}
    try:
        print('SPARK_VERSION =', spark.version, 'MASTER =', spark.sparkContext.master, flush=True)
        workers = spark.sparkContext.parallelize([1, 2, 3], 2).map(lambda x: x * 2).collect()
        if workers != [2, 4, 6]:
            raise ValueError('Spark Python-worker check failed')
        for name, rows in cases.items():
            sample = run / (name + '.csv')
            with sample.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, lineterminator=separators[name])
                writer.writerow(['id', 'title', 'comment'])
                writer.writerows(rows)
            data = (spark.read.schema(schema).option('header', True).option('multiLine', True)
                    # Spark 4.0.1 only disables quoted-newline normalization when
                    # lineSep is explicitly supplied together with multiLine.
                    .option('lineSep', separators[name])
                    .option('quote', '"').option('escape', '"').option('mode', 'FAILFAST')
                    .option('ignoreLeadingWhiteSpace', False).option('ignoreTrailingWhiteSpace', False)
                    .csv(str(sample)))
            text = data.select(F.col('id').cast('long').alias('id'),
                               F.concat(F.coalesce('title', F.lit('')), F.lit('. '),
                                        F.coalesce('comment', F.lit(''))).alias('text'))
            out = run / (name + '.parquet')
            text.coalesce(1).sortWithinPartitions('id').write.mode('errorifexists').option('compression', 'zstd').parquet(str(out))
            table = pq.read_table(out)
            if table.column_names != ['id', 'text'] or table.schema.field('id').type != pa.int64() or table.schema.field('text').type != pa.string():
                raise ValueError('Model-input schema differs from id:int64,text:string')
            ids = table['id'].to_numpy()
            values = table['text'].to_pylist()
            expected = sorted([(i, (title or '') + '. ' + (comment or '')) for i, title, comment in rows])
            if not np.array_equal(ids, np.array([row[0] for row in expected])):
                raise ValueError(name + ': IDs/row order changed')
            for i, actual, reference in zip(ids, values, expected):
                if actual.encode('utf-8') != reference[1].encode('utf-8'):
                    raise ValueError(name + ': text mismatch at id=' + str(i) + '; actual=' + repr(actual) + '; expected=' + repr(reference[1]))
            results[name] = {'rows': len(rows), 'text_byte_equality': True,
                             'record_separator': repr(separators[name]), 'parquet': str(out)}
            print('TEXT_COMPATIBILITY_PASS =', name, 'rows =', len(rows), flush=True)
        summary = {'environment': {'python': sys.version, 'spark': spark.version, 'master': spark.sparkContext.master},
                   'label_audit': label_audit, 'text_tests': results,
                   'smoke_seconds': time.perf_counter() - started,
                   'scope': 'Small compatibility check only, not full-data preprocessing timing.'}
        (run / 'check_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('CHECK_SUMMARY =', run / 'check_summary.json', flush=True)
        print('STEP_3_RESULT = SPARK_COMPATIBILITY_PASS', flush=True)
        print('NO_FULL_DATA_PREPROCESSING_OR_TRAINING_WAS_RUN', flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
