"""Explicit, separate stages of the Task 2 workflow. Nothing runs on import."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MEMBERS = {
    's1': ('sentence-transformers/all-MiniLM-L6-v2', 96, 512, 1),
    's2': ('sentence-transformers/all-MiniLM-L6-v2', 128, 1024, 2),
    'd1': ('distilbert-base-uncased', 128, 768, 1),
    'b1': ('BAAI/bge-small-en-v1.5', 128, 768, 1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify', 'reproduce', 'smoke', 'preprocess', 'train', 'calibrate', 'infer'])
    parser.add_argument('--variant', choices=['full', 'pruned'], default='full')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work')
    parser.add_argument('--inputs-dir', type=Path, help='Spark output; default: <work-dir>/preprocessed')
    parser.add_argument('--models-dir', type=Path, help='Fresh checkpoints/predictions; default: <work-dir>/models')
    parser.add_argument('--labels', type=Path, default=ROOT / 'spark_preprocessing/input/clean_y.npz')
    parser.add_argument('--label-origin', choices=['published', 'recovered'], default='published')
    parser.add_argument('--tags', nargs='+', choices=list(MEMBERS), help='Train only these members')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, choices=[1, 2, 4], default=4)
    parser.add_argument('--inference-bs', type=int, default=256)
    parser.add_argument('--num-threads', type=int, default=0)
    parser.add_argument('--rows', type=int, help='smoke: training records kept (default 20000)')
    parser.add_argument('--limit', type=int, help='smoke: rows actually trained on (default 4000)')
    args = parser.parse_args()
    work, data = args.work_dir.resolve(), args.data_dir.resolve()
    inputs = (args.inputs_dir or work / 'preprocessed').resolve()
    models = (args.models_dir or work / 'models').resolve()
    selected = list(MEMBERS) if args.variant == 'full' else ['s1', 'b1']
    builder = 'build_submission.py' if args.variant == 'full' else 'build_submission_pruned.py'
    params_name = 'ensemble_params.json' if args.variant == 'full' else 'ensemble_params_pruned.json'

    def run(script, *options, timing_name=None):
        command = [sys.executable, '-B', str(ROOT / script), *map(str, options)]
        timing_path = work / 'timings' / (timing_name + '.json') if timing_name else None
        if timing_path and timing_path.exists():
            raise FileExistsError(timing_path)
        print('RUN', subprocess.list2cmdline(command), flush=True)
        started = time.perf_counter()
        subprocess.run(command, cwd=ROOT, check=True)
        if timing_path:
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            with timing_path.open('x', encoding='utf-8') as stream:
                json.dump({'command': command, 'seconds': time.perf_counter() - started,
                           'scope': 'Successful child process including interpreter startup and imports.'}, stream, indent=2)

    if args.stage == 'verify':
        run('verify_package.py')
    elif args.stage == 'smoke':
        options = ['--data-dir', data, '--work-dir', work]
        label = args.labels.resolve()
        if label.is_file():
            options += ['--labels', label]
        if args.rows:
            options += ['--rows', args.rows]
        if args.limit:
            options += ['--limit', args.limit]
        if args.device != 'auto':
            options += ['--device', args.device]
        run('smoke_test.py', *options)
    elif args.stage == 'reproduce':
        out = work / f'submission_saved_{args.variant}.csv'
        run('final_model/' + builder, '--apply', '--out', out,
            '--sample', ROOT / 'reference/sampleSubmission.csv')
        reference = ROOT / ('reference/submission.csv' if args.variant == 'full' else 'reference/submission_pruned.csv')
        run('final_model/check_submission.py', out, '--sample', ROOT / 'reference/sampleSubmission.csv', '--reference', reference)
    elif args.stage == 'preprocess':
        run('spark_preprocessing/03_check_spark.py', '--data-dir', data, '--labels', args.labels.resolve())
        run('spark_preprocessing/04_preprocess_full.py', '--data-dir', data, '--labels', args.labels.resolve(),
            '--label-origin', args.label_origin, '--output-dir', inputs, '--threads', args.threads)
    elif args.stage == 'train':
        status = json.loads((inputs / 'preprocessing_summary.json').read_text(encoding='utf-8'))
        if status.get('status') != 'SPARK_FULL_PREPROCESSING_PASS':
            raise ValueError('Spark preprocessing must finish successfully before training')
        for tag in args.tags or selected:
            model, maxlen, batch, seed = MEMBERS[tag]
            run('final_model/finetune.py', '--train-text', inputs / 'train_text.parquet',
                '--test-text', inputs / 'test_text.parquet', '--clean', inputs / 'clean_y.npz',
                '--out-dir', models, '--model', model, '--maxlen', maxlen, '--bs', batch,
                '--seed', seed, '--tag', tag, '--epochs', 2, '--device', args.device,
                timing_name='train_' + tag)
    elif args.stage == 'calibrate':
        run('final_model/' + builder, '--preds-dir', models, '--labels', inputs / 'clean_y.npz',
            '--params-out', models / params_name, '--sample', data / 'sampleSubmission.csv',
            '--out', work / f'submission_refit_{args.variant}.csv', timing_name='calibrate_' + args.variant)
    elif args.stage == 'infer':
        out = work / f'submission_fresh_{args.variant}.csv'
        run('final_model/measure_pipeline.py', '--test-text', inputs / 'test_text.parquet',
            '--ckpt-dir', models, '--params', models / params_name, '--tags', *selected,
            '--device', args.device, '--num-threads', args.num_threads, '--bs', args.inference_bs,
            '--sample', data / 'sampleSubmission.csv', '--out', out,
            '--timing-out', work / f'full_test_inference_{args.variant}.json')
        run('final_model/check_submission.py', out, '--sample', data / 'sampleSubmission.csv')


if __name__ == '__main__':
    main()
