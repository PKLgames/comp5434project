"""Summarize new-run preprocessing, epoch, offline and complete-test timings."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).parent / 'work')
    parser.add_argument('--variant', choices=['full', 'pruned'], default='full')
    parser.add_argument('--out', type=Path, help='Optional new JSON output; otherwise print only')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    pre = read(work / 'preprocessed/preprocessing_summary.json')
    if pre.get('status') != 'SPARK_FULL_PREPROCESSING_PASS':
        raise ValueError('Preprocessing did not pass')
    tags = ['s1', 's2', 'd1', 'b1'] if args.variant == 'full' else ['s1', 'b1']
    jobs, epochs = {}, {}
    for tag in tags:
        jobs[tag] = read(work / 'timings' / f'train_{tag}.json')['seconds']
        epochs[tag] = read(work / 'models' / f'ft3_{tag}.json')['epoch_timings']
    blend = read(work / 'timings' / f'calibrate_{args.variant}.json')['seconds']
    inference = read(work / f'full_test_inference_{args.variant}.json')
    if inference['n_rows'] != 10000 or inference['tags'] != tags:
        raise ValueError('Inference does not cover all selected models and 10,000 rows')
    summary = {'variant': args.variant, 'epoch_timings': epochs, 'training_process_seconds': jobs,
               'preprocessing_seconds': pre['preprocessing_seconds'],
               'preprocessing_with_audit_seconds': pre['total_seconds_including_full_text_audit'],
               'calibration_process_seconds': blend,
               'offline_seconds': pre['preprocessing_seconds'] + sum(jobs.values()) + blend,
               'offline_with_audit_seconds': pre['total_seconds_including_full_text_audit'] + sum(jobs.values()) + blend,
               'all_10000_test_inference_seconds': inference['timing_s']['TOTAL'],
               'inference_scope': inference['scope'], 'inference_environment': inference['env'],
               'scope': 'Cumulative successful stage times. Training jobs include validation, checkpoint writing and post-training predictions. Excludes input downloads/source recovery, optional smoke checks and the separately reported fresh-inference run. A pruned-only summary counts the two selected members.'}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2)


if __name__ == '__main__':
    main()
