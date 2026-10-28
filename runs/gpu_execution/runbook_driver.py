"""Local phase 4-6 runner; commands follow task2_model/GPU_RUNBOOK.md."""
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
DATA = str(Path(sys.argv[1]).resolve())
LOG_DIR = ROOT / 'runs' / 'gpu_execution'
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE = {'data': DATA, 'completed': [], 'status': 'running'}


def save():
    (LOG_DIR / 'execution.json').write_text(json.dumps(STATE, indent=2), encoding='utf-8')


def run(name, args):
    STATE['current'] = name
    save()
    cmd = [sys.executable, '-u', *args]
    print('\nSTART ' + name, flush=True)
    started = time.perf_counter()
    with (LOG_DIR / (name + '.log')).open('w', encoding='utf-8') as log:
        log.write(json.dumps(cmd) + '\n')
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
    if code:
        raise RuntimeError(f'{name} exited with {code}; see {LOG_DIR / (name + ".log")}')
    STATE['completed'].append({'phase': name, 'seconds': round(time.perf_counter() - started, 2)})
    save()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train(name, epochs, extras=(), smoke=False):
    target = ROOT / 'runs' / name
    if target.exists():
        raise FileExistsError(target)
    cmd = ['task2_model/train.py', '--data', DATA, '--out', str(target),
           '--epochs', str(epochs), '--device', 'cuda']
    if smoke:
        cmd += ['--batch-size', '1024', '--valid-batch-size', '1024',
                '--limit-train-rows', '8192', '--limit-valid-rows', '8192',
                '--text-dim', '32', '--dense-dim', '16', '--hidden-dim', '32',
                '--ordinal-weight', '0.5', '--film', '--save-val-preds']
    else:
        cmd += list(extras)
    run(name, cmd)


try:
    run('phase4_tests', ['-m', 'pytest', 'tests', '-q'])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    train('gpu_runbook_smoke', 1, smoke=True)
    checkpoint = torch.load('runs/gpu_runbook_smoke/best.pt', map_location='cpu', weights_only=False)
    if checkpoint.get('blend_alpha') is None:
        raise RuntimeError('Smoke checkpoint has no blend_alpha')
    print('Smoke blend_alpha:', checkpoint['blend_alpha'], flush=True)
    del checkpoint
    gc.collect()
    smoke_dir = (ROOT / 'runs/gpu_runbook_smoke').resolve()
    if smoke_dir.parent != (ROOT / 'runs').resolve():
        raise RuntimeError('Unexpected smoke cleanup path')
    shutil.rmtree(smoke_dir)

    common = ['--batch-size', '8192', '--valid-batch-size', '8192',
              '--lr', '0.003', '--weight-decay', '0.0001',
              '--text-dim', '256', '--dense-dim', '128', '--hidden-dim', '256',
              '--dropout', '0.15', '--emb-dropout', '0.1',
              '--seed', '42', '--num-threads', '16', '--save-val-preds']
    train('gpu_ref_base', 3, common)
    train('gpu_ref_tmlp', 2, [*common, '--text-mlp'])
    run('gpu_ref_stack', ['task2_model/stack.py', '--data', DATA,
                         '--out', 'runs/gpu_ref_stack', '--device', 'cuda',
                         '--checkpoint', 'runs/gpu_ref_base/best.pt',
                         '--checkpoint', 'runs/gpu_ref_tmlp/best.pt'])
    baseline = read('runs/gpu_ref_stack/val_preds_metrics.json')['rmse']
    STATE['reference_rmse'] = baseline
    save()
    if abs(baseline - 1.1543) > 0.001:
        raise RuntimeError(f'Phase 5 gate: RMSE {baseline:.6f} differs from 1.1543 by more than 0.001')

    candidate = list(common)
    candidate[candidate.index('--valid-batch-size') + 1] = '16384'
    train('gpu_ord', 4, [*candidate, '--ordinal-weight', '0.5', '--blend-folds', '5'])
    run('gpu_ord_predict', ['task2_model/predict.py', '--data', DATA, '--split', 'valid',
                           '--checkpoint', 'runs/gpu_ord/best.pt', '--device', 'cuda',
                           '--out', 'runs/gpu_ord/val_preds.npz'])
    train('gpu_film', 4, [*candidate, '--film'])
    run('gpu_film_predict', ['task2_model/predict.py', '--data', DATA, '--split', 'valid',
                            '--checkpoint', 'runs/gpu_film/best.pt', '--device', 'cuda',
                            '--out', 'runs/gpu_film/val_preds.npz'])
    # The combined block in the runbook deliberately uses train.py defaults for
    # lr/decay, unlike the two individual methods. Preserve the published command.
    train('gpu_ord_film', 4, ['--batch-size', '8192', '--valid-batch-size', '16384',
                             '--ordinal-weight', '0.5', '--film', '--select-metric', 'blend',
                             '--seed', '42', '--num-threads', '16', '--save-val-preds'])
    for name in ['gpu_ord', 'gpu_film', 'gpu_ord_film']:
        run(name + '_compare', ['task2_model/compare.py',
                               '--base', 'runs/gpu_ref_stack/val_preds.npz',
                               '--candidate', f'runs/{name}/val_preds.npz',
                               '--out', f'runs/{name}/comparison.json'])
    STATE['comparisons'] = {name: read(f'runs/{name}/comparison.json')['decision']
                            for name in ['gpu_ord', 'gpu_film', 'gpu_ord_film']}
    STATE['status'] = 'phase6_complete'
    save()
    print('PHASES_4_TO_6_COMPLETE', flush=True)
except BaseException as error:
    STATE['status'] = 'failed'
    STATE['error'] = str(error)
    save()
    raise
