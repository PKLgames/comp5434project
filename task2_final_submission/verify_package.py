"""Check source integrity and exact saved-CSV reproduction, without model runs.

Reads small packaged arrays only. Two temporary CSVs are compared byte-for-byte
with the archived submissions and removed on exit. No raw CSV scan, Spark job,
download, training or neural-network inference is performed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'final_model'))
import numpy as np
from build_submission import (apply_params, load_predictions, load_validation_labels,
                              read_params, write_submission)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-labels', type=Path, help='Optional full archive to check against the compact slice')
    args = parser.parse_args()
    records = (ROOT / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
    for line in records:
        digest, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or invalid manifest path: ' + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('File hash mismatch: ' + relative)
    syntax_count = 0
    third_party = {'numpy', 'scipy', 'sklearn', 'torch', 'pyspark', 'pyarrow', 'transformers'}
    for path in sorted(ROOT.rglob('*.py')):
        if {'data', 'work', '.venv', '.hf'} & set(path.relative_to(ROOT).parts):
            continue
        source = path.read_text(encoding='utf-8-sig')
        compile(source, str(path), 'exec')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split('.')[0]
                if module not in sys.stdlib_module_names and module not in third_party:
                    if not (path.parent / (module + '.py')).is_file() and not (
                            path == ROOT / 'verify_package.py'
                            and (ROOT / 'final_model' / (module + '.py')).is_file()):
                        raise ValueError(f'Unresolved local import in {path.name}: {module}')
        syntax_count += 1
    preds = ROOT / 'final_model/preds'
    ids = np.load(preds / 'ftvids_s1.npy', allow_pickle=False)
    if not np.array_equal(ids, np.arange(0, 7684645, 40)):
        raise ValueError('Validation IDs differ from the fixed split')
    y, mask = load_validation_labels(preds, ids)
    if int(mask.sum()) != 177607:
        raise ValueError('Unexpected matched-validation count')
    if args.full_labels:
        full_y, full_mask = load_validation_labels(preds, ids, args.full_labels)
        if not np.array_equal(y, full_y) or not np.array_equal(mask, full_mask):
            raise ValueError('Compact labels differ from the full source archive')
    for tag in ['s1', 's2', 'd1', 'b1']:
        load_predictions(preds, [tag], 'ftv', len(ids))
    proofs = {}
    with tempfile.TemporaryDirectory(prefix='_package_check_', dir=ROOT) as directory:
        scratch = Path(directory).resolve()
        if not scratch.is_relative_to(ROOT):
            raise RuntimeError('Unexpected temporary-directory cleanup target')
        for variant, params_name, reference in [
            ('full', 'ensemble_params.json', 'submission.csv'),
            ('pruned', 'ensemble_params_pruned.json', 'submission_pruned.csv'),
        ]:
            cfg = read_params(preds / params_name)
            arrays = load_predictions(preds, cfg['tags'], 'ftp', 10000)
            out = scratch / f'{variant}.csv'
            write_submission(out, apply_params(arrays, cfg), fixed_decimals=variant == 'pruned')
            data = out.read_bytes()
            if data != (ROOT / 'reference' / reference).read_bytes():
                raise ValueError(variant + ': saved submission byte mismatch')
            try:
                write_submission(out, apply_params(arrays, cfg), fixed_decimals=variant == 'pruned')
            except FileExistsError:
                pass
            else:
                raise AssertionError('Existing submission was overwritten')
            proofs[variant] = {'rows': 10000, 'reference_bytes_equal': True,
                               'sha256': hashlib.sha256(data).hexdigest()}
    print(json.dumps({'status': 'PACKAGE_CHECK_PASS', 'files_verified': len(records),
                      'python_files_checked': syntax_count, 'submissions': proofs,
                      'matched_validation_rows': int(mask.sum()),
                      'compact_labels_checked_against_full': bool(args.full_labels),
                      'model_training_or_inference_run': False}, indent=2))


if __name__ == '__main__':
    main()
