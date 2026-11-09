"""Full Spark preprocessing with deterministic model inputs and complete audits.

No sentiment rules, deduplication, new split, training, or source-rating
recovery are performed. Verified source ratings are joined by original ID.
Only this script's new run directory is written.
"""
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE_COMMIT = '261e9e648cc0b1080fae0fe192f9ab1fa7635946'
PUBLISHED_LABEL_SHA256 = '5c133167aceb845de1612af7a3555910934fb934736731a8b3f3a426667dee6f'
N_TRAIN = 7_684_645
N_TEST = 10_000
TRAIN_COLUMNS = ['id', 'title', 'user_id', 'comment', 'prod_id',
                 'parent_prod_id', 'time', 'votes', 'purchased', 'rating']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns}


def load_compatibility_helper():
    spec = importlib.util.spec_from_file_location('task2_spark_compat', HERE / '03_check_spark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_headers(project):
    for name, expected in [('train.csv', TRAIN_COLUMNS), ('test.csv', TRAIN_COLUMNS[:-1]),
                           ('sampleSubmission.csv', ['id', 'rating'])]:
        with (project / name).open(encoding='utf-8-sig', newline='') as stream:
            if next(csv.reader(stream)) != expected:
                raise ValueError('Input column order differs from DATA_SPEC: ' + name)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4, choices=[1, 2, 4])
    parser.add_argument('--data-dir', type=Path, default=PACKAGE / 'data')
    parser.add_argument('--labels', type=Path, default=HERE / 'input' / 'clean_y.npz')
    parser.add_argument('--label-origin', choices=['published', 'recovered'], default='published',
                        help='published checks the pinned SHA-256; recovered accepts a fresh recovery output')
    parser.add_argument('--output-dir', type=Path, default=PACKAGE / 'work' / 'preprocessed',
                        help='New output directory; existing directories are never overwritten')
    args = parser.parse_args()
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    project = args.data_dir.resolve()
    verify_headers(project)
    checks = sorted((HERE / 'work').glob('spark_smoke_*/check_summary.json'))
    if not checks:
        raise RuntimeError('Run Step 3 successfully before full preprocessing')
    compatibility = json.loads(checks[-1].read_text(encoding='utf-8'))
    for name in ('train_head', 'test_head', 'edge_cases', 'edge_cases_crlf_records'):
        if compatibility['text_tests'].get(name, {}).get('text_byte_equality') is not True:
            raise RuntimeError('Step 3 did not pass all required text compatibility cases')

    inputs = {name: project / name for name in ('train.csv', 'test.csv', 'sampleSubmission.csv')}
    labels_path = args.labels.resolve()
    inputs['clean_y.npz'] = labels_path
    before = {name: stamp(path) for name, path in inputs.items()}
    expected_digest = sha256(labels_path)
    if args.label_origin == 'published' and expected_digest != PUBLISHED_LABEL_SHA256:
        raise ValueError('Source-label archive differs from the verified download')

    run = args.output_dir.resolve()
    if run.exists():
        raise FileExistsError('Select a new output directory: ' + str(run))
    scratch = run / '_scratch'
    scratch.mkdir(parents=True, exist_ok=False)
    local = scratch / 'spark_local'
    local.mkdir()
    timings = {}
    started = time.perf_counter()

    def stage(name, function):
        print('STAGE_START =', name, flush=True)
        start = time.perf_counter()
        result = function()
        timings[name] = time.perf_counter() - start
        print('STAGE_COMPLETE =', name, 'seconds =', round(timings[name], 3), flush=True)
        return result

    # Keep the full source arrays available for the later elementwise output audit.
    with np.load(labels_path, allow_pickle=False) as labels:
        cy, matched, ag = labels['cy'], labels['matched'], labels['ag']
    if any(array.shape != (N_TRAIN,) for array in (cy, matched, ag)):
        raise ValueError('Source-label arrays do not cover the full training ID domain')
    if cy.dtype != np.int8 or ag.dtype != np.int8 or matched.dtype != np.bool_:
        raise ValueError('Expected cy:int8, matched:bool, ag:int8')
    valid = matched & (cy >= 1) & (cy <= 5)
    if not np.array_equal(matched, valid) or np.any((~matched) & (cy != 0)):
        raise ValueError('Source-label match flags and rating values are inconsistent')
    if not np.isin(ag, [1, 2, 3, 4, 5]).all():
        raise ValueError('Delivered labels outside 1..5')
    source_count = int(np.count_nonzero(valid))
    matched_valid_count = int(np.count_nonzero(valid[::40]))
    expected_train_count = source_count - matched_valid_count
    print('ACTUAL_LABEL_COUNTS =', json.dumps({'matched': source_count,
          'matched_validation': matched_valid_count, 'training': expected_train_count,
          'label_origin': args.label_origin}), flush=True)

    source_parquet = scratch / 'source_labels.parquet'

    def prepare_source_table():
        table = pa.table({'id': pa.array(np.arange(N_TRAIN, dtype=np.int64)),
                          'source_rating': pa.array(cy), 'source_matched': pa.array(matched),
                          'archived_delivered_rating': pa.array(ag)})
        pq.write_table(table, source_parquet, compression='zstd', row_group_size=262144)

    stage('source_label_bridge_seconds', prepare_source_table)
    helper = load_compatibility_helper()
    helper.configure_spark()
    os.environ['SPARK_LOCAL_DIRS'] = helper.short_path(local)
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession, functions as F, types as T

    spark = None
    joined = None
    test = None
    completed = False

    def create_spark():
        session = (SparkSession.builder.appName('Task2FinalSparkPreprocessing')
                   .master('local[' + str(args.threads) + ']')
                   .config('spark.driver.memory', '4g')
                   .config('spark.sql.shuffle.partitions', '32')
                   .config('spark.sql.autoBroadcastJoinThreshold', '-1')
                   .config('spark.rdd.compress', 'true')
                   .config('spark.sql.session.timeZone', 'UTC')
                   .config('spark.ui.enabled', 'false').getOrCreate())
        session.sparkContext.setLogLevel('ERROR')
        return session

    def read_csv(name):
        columns = TRAIN_COLUMNS if name == 'train.csv' else TRAIN_COLUMNS[:-1]
        schema = T.StructType([T.StructField(column, T.StringType(), True) for column in columns])
        return (spark.read.schema(schema).option('header', True).option('multiLine', True)
                .option('lineSep', helper.record_separator(project / name))
                .option('quote', '"').option('escape', '"').option('mode', 'FAILFAST')
                .option('ignoreLeadingWhiteSpace', False).option('ignoreTrailingWhiteSpace', False)
                .csv(str(project / name)))

    def model_text(raw):
        return raw.select(F.expr('try_cast(id as bigint)').alias('id'),
                          F.concat(F.coalesce('title', F.lit('')), F.lit('. '),
                                   F.coalesce('comment', F.lit(''))).alias('text'))

    def count_if(condition):
        return F.sum(F.when(condition, 1).otherwise(0))

    def write_model_file(frame, name, expected_rows, partitions=16):
        """Spark computes/sorts the data; Arrow only packages ordered fragments."""
        parts = scratch / (name + '_parts')
        ordered = (frame.repartitionByRange(partitions, 'id') if partitions > 1 else frame.coalesce(1))
        ordered.sortWithinPartitions('id').write.mode('errorifexists').option('compression', 'zstd').parquet(str(parts))
        fragments = sorted(parts.glob('part-*.parquet'))
        if not fragments:
            raise ValueError('Spark did not write Parquet fragments for ' + name)
        output = run / name
        partial = output.with_name(output.name + '.partial')
        writer = None
        writer_schema = None
        position = 0
        try:
            for path in fragments:
                parquet = pq.ParquetFile(path)
                schema = parquet.schema_arrow.remove_metadata()
                if writer is None:
                    writer = pq.ParquetWriter(partial, schema, compression='zstd')
                    writer_schema = schema
                elif not schema.equals(writer_schema, check_metadata=False):
                    raise ValueError('Spark fragment schema mismatch')
                for batch in parquet.iter_batches(batch_size=65536):
                    ids = batch.column(batch.schema.get_field_index('id')).to_numpy(zero_copy_only=False)
                    if not np.array_equal(ids, np.arange(position, position + len(ids), dtype=np.int64)):
                        raise ValueError(name + ': missing, repeated or out-of-order IDs near ' + str(position))
                    writer.write_batch(batch)
                    position += len(ids)
        finally:
            if writer is not None:
                writer.close()
        if position != expected_rows:
            raise ValueError(name + ': output row count is ' + str(position))
        partial.rename(output)
        return {'path': str(output), 'rows': position, 'bytes': output.stat().st_size,
                'schema': str(pq.ParquetFile(output).schema_arrow), 'all_ids_sequential_unique': True}

    def export_model_labels():
        """Rebuild the model's ID-indexed NPZ from the actual Spark join output."""
        reconstructed = {'cy': np.zeros(N_TRAIN, dtype=np.int8),
                         'matched': np.zeros(N_TRAIN, dtype=bool),
                         'ag': np.zeros(N_TRAIN, dtype=np.int8)}
        position = 0
        for batch in pq.ParquetFile(run / 'labels_by_id.parquet').iter_batches(batch_size=262144):
            stop = position + batch.num_rows
            for key in reconstructed:
                values = batch.column(batch.schema.get_field_index(key)).to_numpy(zero_copy_only=False)
                reconstructed[key][position:stop] = values
            position = stop
        if position != N_TRAIN:
            raise ValueError('Spark label export is incomplete')
        for key, source in [('cy', cy), ('matched', matched), ('ag', ag)]:
            if not np.array_equal(reconstructed[key], source):
                raise ValueError('Spark label association changed ' + key)
        path = run / 'clean_y.npz'
        np.savez(path, **reconstructed)
        return {'path': str(path), 'rows': N_TRAIN, 'bytes': path.stat().st_size,
                'all_arrays_equal_verified_source': True}

    class ParquetCursor:
        def __init__(self, path):
            self.iterator = iter(pq.ParquetFile(path).iter_batches(batch_size=65536))
            self.current = None
            self.offset = 0

        def take(self, length):
            batches = []
            remaining = length
            while remaining:
                if self.current is None or self.offset == self.current.num_rows:
                    self.current = next(self.iterator, None)
                    self.offset = 0
                if self.current is None:
                    raise ValueError('Spark text output ended before original CSV')
                amount = min(remaining, self.current.num_rows - self.offset)
                batches.append(self.current.slice(self.offset, amount))
                remaining -= amount
                self.offset += amount
            return pa.Table.from_batches(batches)

        def assert_finished(self):
            if self.current is not None and self.offset < self.current.num_rows:
                raise ValueError('Spark output has extra rows')
            if next(self.iterator, None) is not None:
                raise ValueError('Spark output has extra batches')

    def audit_all_text(csv_name, output_name, expected_rows):
        """Independent streaming CSV reference; no full strings table in memory."""
        path = run / output_name
        schema = pq.ParquetFile(path).schema_arrow
        if schema.names != ['id', 'text'] or schema.field('id').type != pa.int64() or schema.field('text').type != pa.string():
            raise ValueError('Model text schema differs from DATA_SPEC')
        cursor = ParquetCursor(path)
        reference = pacsv.open_csv(str(project / csv_name),
            read_options=pacsv.ReadOptions(block_size=1 << 24),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(include_columns=['id', 'title', 'comment'],
                column_types={'id': pa.int64(), 'title': pa.string(), 'comment': pa.string()},
                strings_can_be_null=False))
        rows = 0
        for batch in reference:
            expected_text = pc.binary_join_element_wise(pc.fill_null(batch.column('title'), ''),
                                                         pc.fill_null(batch.column('comment'), ''), '. ')
            actual = cursor.take(batch.num_rows)
            equal_ids = pc.equal(actual['id'], batch.column('id'))
            equal_text = pc.equal(actual['text'], expected_text)
            equal = pc.fill_null(pc.and_(equal_ids, equal_text), False)
            if not pc.all(equal).as_py():
                index = int(pc.indices_nonzero(pc.invert(equal))[0].as_py())
                bad_id = batch.column('id')[index].as_py()
                raise ValueError(csv_name + ': full text/ID equivalence failed at id=' + str(bad_id))
            rows += batch.num_rows
            if rows // 1_000_000 != (rows - batch.num_rows) // 1_000_000:
                print('TEXT_AUDIT_PROGRESS =', csv_name, rows, flush=True)
        cursor.assert_finished()
        if rows != expected_rows:
            raise ValueError('CSV reference row count changed: ' + csv_name)
        if csv_name == 'test.csv':
            with (project / 'sampleSubmission.csv').open(encoding='utf-8-sig', newline='') as stream:
                sample_ids = [int(row['id']) for row in csv.DictReader(stream)]
            if sample_ids != list(range(N_TEST)):
                raise ValueError('Submission sample ID order is not 0..9999')
        return {'rows_compared': rows, 'all_text_equal_by_id': True,
                'reference': 'Independent PyArrow CSV reader and exact title + separator + comment expression',
                'null_whitespace_case_and_newlines_preserved': True}

    outputs = {}
    try:
        spark = stage('spark_startup_seconds', create_spark)

        def prepare_train():
            raw = read_csv('train.csv')
            features = raw.select(
                F.expr('try_cast(id as bigint)').alias('id'),
                F.concat(F.coalesce('title', F.lit('')), F.lit('. '), F.coalesce('comment', F.lit(''))).alias('text'),
                F.expr('try_cast(rating as int)').alias('delivered_rating'))
            source = spark.read.parquet(str(source_parquet))
            frame = features.join(source, on='id', how='left').persist(StorageLevel.DISK_ONLY)
            in_valid = (F.pmod(F.col('id'), F.lit(40)) == 0)
            eligible = F.col('source_matched') & F.col('source_rating').between(1, 5)
            stats = frame.agg(F.count('*').alias('rows'), F.min('id').alias('min_id'), F.max('id').alias('max_id'),
                count_if(F.col('id').isNull() | (F.col('id') < 0) | (F.col('id') >= N_TRAIN)).alias('invalid_ids'),
                count_if(F.col('text').isNull()).alias('null_texts'),
                count_if(F.col('source_matched').isNull() | F.col('archived_delivered_rating').isNull()).alias('missing_label_links'),
                count_if(F.col('delivered_rating').isNull() | ~F.col('delivered_rating').between(1, 5)).alias('invalid_delivered_ratings'),
                count_if(~F.col('delivered_rating').eqNullSafe(F.col('archived_delivered_rating'))).alias('csv_vs_archive_rating_mismatches'),
                count_if(in_valid).alias('validation_rows'), count_if(eligible).alias('matched_rows'),
                count_if(in_valid & eligible).alias('matched_validation_rows'),
                count_if(~in_valid & eligible).alias('eligible_training_rows')).first().asDict()
            expected = {'rows': N_TRAIN, 'min_id': 0, 'max_id': N_TRAIN - 1, 'invalid_ids': 0,
                        'null_texts': 0, 'missing_label_links': 0, 'invalid_delivered_ratings': 0,
                        'csv_vs_archive_rating_mismatches': 0, 'validation_rows': 192117,
                        'matched_rows': source_count, 'matched_validation_rows': matched_valid_count,
                        'eligible_training_rows': expected_train_count}
            if stats != expected:
                frame.unpersist()
                raise ValueError('Training association/count audit failed: ' + json.dumps(stats))
            return frame, stats

        joined, train_stats = stage('spark_train_read_text_join_materialize_audit_seconds', prepare_train)
        print('TRAIN_JOIN_AUDIT_PASS =', json.dumps(train_stats), flush=True)
        outputs['train_text'] = stage('spark_train_text_write_and_package_seconds', lambda: write_model_file(joined.select('id', 'text'), 'train_text.parquet', N_TRAIN))
        label_frame = joined.select('id', F.col('source_rating').cast('byte').alias('cy'),
                                     F.col('source_matched').alias('matched'),
                                     F.col('delivered_rating').cast('byte').alias('ag'))
        outputs['labels_by_id'] = stage('spark_label_write_and_package_seconds', lambda: write_model_file(label_frame, 'labels_by_id.parquet', N_TRAIN))
        joined.unpersist(blocking=True)
        joined = None
        outputs['clean_y'] = stage('model_npz_export_and_full_label_equivalence_seconds', export_model_labels)

        def prepare_test():
            frame = model_text(read_csv('test.csv')).persist(StorageLevel.DISK_ONLY)
            stats = frame.agg(F.count('*').alias('rows'), F.min('id').alias('min_id'), F.max('id').alias('max_id'),
                             count_if(F.col('id').isNull() | F.col('text').isNull()).alias('invalid_rows')).first().asDict()
            if stats != {'rows': N_TEST, 'min_id': 0, 'max_id': N_TEST - 1, 'invalid_rows': 0}:
                frame.unpersist()
                raise ValueError('Test-input audit failed: ' + json.dumps(stats))
            return frame, stats

        test, test_stats = stage('spark_test_read_text_materialize_audit_seconds', prepare_test)
        outputs['test_text'] = stage('spark_test_text_write_and_package_seconds', lambda: write_model_file(test, 'test_text.parquet', N_TEST, partitions=1))
        test.unpersist(blocking=True)
        test = None
        preprocessing_seconds = time.perf_counter() - started
        text_audits = {
            'train': stage('full_train_text_equivalence_seconds', lambda: audit_all_text('train.csv', 'train_text.parquet', N_TRAIN)),
            'test': stage('full_test_text_equivalence_seconds', lambda: audit_all_text('test.csv', 'test_text.parquet', N_TEST)),
        }
        after = {name: stamp(path) for name, path in inputs.items()}
        if before != after:
            raise ValueError('An input file changed during this run')
        total_seconds = time.perf_counter() - started
        summary = {
            'status': 'SPARK_FULL_PREPROCESSING_PASS', 'input_stamps': before,
            'label_origin': args.label_origin,
            'label_source_commit': SOURCE_COMMIT if args.label_origin == 'published' else None,
            'source_label_sha256': expected_digest,
            'spark': {'version': spark.version, 'master': spark.sparkContext.master,
                      'driver_memory': '4g', 'shuffle_partitions': 32},
            'environment': {'python': sys.version, 'numpy': np.__version__, 'pyarrow': pa.__version__,
                            'cpu': platform.processor(), 'logical_cpus': os.cpu_count(), 'platform': platform.platform()},
            'train_audit': train_stats, 'test_audit': test_stats, 'full_text_audits': text_audits,
            'model_input_files': outputs, 'stage_seconds': timings,
            'preprocessing_seconds': preprocessing_seconds,
            'total_seconds_including_full_text_audit': total_seconds,
            'timing_scope': 'Source-label bridge, Spark startup, CSV parsing, exact text construction, source-rating joins, materialized checks, range ordering, Spark Parquet writing, streaming model-file packaging, NPZ export and label equivalence. Full independent text audits reported separately. Excludes downloads, initial environment/input preflight, model training/inference, Spark shutdown and scratch cleanup.',
            'selection': {'validation': 'id % 40 == 0', 'training': 'id % 40 != 0 and matched and 1 <= cy <= 5',
                          'valid_source_labels': source_count, 'matched_validation': matched_valid_count,
                          'eligible_training': expected_train_count},
            'transformations': ['Fill null title/comment with empty strings', 'text = title + ". " + comment',
                                'Join verified full source ratings by original id', 'Preserve every original row and rating in ag',
                                'Order model files by original id; no stripping/lowercasing/HTML removal/deduplication'],
            'logical_equivalence': 'All original train/test id/text rows and all source-label entries compared; Parquet bytes/row groups may differ.',
            'model_consumption': 'Pass train_text.parquet, test_text.parquet and the exported clean_y.npz to the existing finetune.py CLI. Source-label eligibility and validation filtering remain in the trainer. Nothing is trained here.',
        }
        (run / 'preprocessing_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (run / 'MODEL_INPUTS.json').write_text(json.dumps({key: value['path'] for key, value in outputs.items()}, indent=2), encoding='utf-8')
        completed = True
        print('SPARK_FULL_PREPROCESSING_COMPLETE =', json.dumps({'train_csv_rows': N_TRAIN,
              'eligible_training_rows': expected_train_count, 'matched_validation_rows': matched_valid_count,
              'test_rows': N_TEST, 'preprocessing_seconds': round(preprocessing_seconds, 3),
              'total_seconds_with_full_audit': round(total_seconds, 3), 'output': str(run),
              'summary': str(run / 'preprocessing_summary.json')}), flush=True)
        print('NO_MODEL_TRAINING_OR_INFERENCE_WAS_RUN', flush=True)
    except Exception as exc:
        (run / 'FAILED.json').write_text(json.dumps({'error': str(exc), 'completed_stages_seconds': timings,
                                                    'output_is_not_approved_for_training': True}, indent=2), encoding='utf-8')
        raise
    finally:
        if joined is not None:
            joined.unpersist()
        if test is not None:
            test.unpersist()
        if spark is not None:
            spark.stop()
        if completed:
            # Only remove scratch files created inside this new, isolated run.
            if scratch.resolve().parent != run.resolve() or scratch.name != '_scratch' or scratch.is_symlink():
                raise RuntimeError('Unexpected scratch cleanup path')
            try:
                shutil.rmtree(scratch)
            except OSError as exc:
                print('SCRATCH_RETAINED_FOR_MANUAL_CLEANUP =', str(scratch), str(exc), flush=True)


if __name__ == '__main__':
    main()
