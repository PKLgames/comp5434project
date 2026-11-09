"""Step 1: inspect dependencies and inputs without launching Spark or processing CSVs.

No downloads, installations, model runs, or output datasets are created.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PACKAGE / 'data',
                        help='Directory containing train.csv, test.csv and sampleSubmission.csv')
    parser.add_argument('--labels', type=Path, help='Optional explicit path to the full clean_y.npz')
    args = parser.parse_args()
    project = args.data_dir.resolve()
    missing = []
    issues = []
    print('STEP = 1: READ-ONLY INPUT AND ENVIRONMENT CHECK', flush=True)
    print('DATA_DIRECTORY =', project)
    print('PYTHON =', sys.executable)
    print('PYTHON_VERSION =', sys.version.split()[0])
    for package in ('numpy', 'pyarrow', 'pyspark'):
        try:
            print(package.upper() + '_VERSION =', importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
            print(package.upper() + '_VERSION = MISSING')

    java_home = os.environ.get('JAVA_HOME', '')
    java = Path(java_home) / 'bin' / ('java.exe' if os.name == 'nt' else 'java') if java_home else None
    if java is None or not java.is_file():
        fallback = shutil.which('java')
        java = Path(fallback) if fallback else None
    if java:
        try:
            result = subprocess.run([str(java), '-version'], capture_output=True, text=True, timeout=15)
            version_lines = (result.stderr + result.stdout).strip().splitlines()
            print('JAVA =', java)
            print('JAVA_VERSION =', version_lines[0] if version_lines else 'NO VERSION OUTPUT')
            if result.returncode:
                issues.append('java -version failed')
        except (OSError, subprocess.TimeoutExpired) as exc:
            issues.append('Java check failed: ' + str(exc))
    else:
        missing.append('java')
        print('JAVA = MISSING')

    if os.name == 'nt':
        hadoop = Path(os.environ.get('HADOOP_HOME', r'D:\HADOOP'))
        print('HADOOP_HOME_CANDIDATE =', hadoop)
        for filename in ('winutils.exe', 'hadoop.dll'):
            path = hadoop / 'bin' / filename
            print(filename.upper() + '_EXISTS =', path.is_file())
            if not path.is_file():
                missing.append(str(path))

    expected_headers = {
        'train.csv': {'id', 'title', 'comment', 'rating'},
        'test.csv': {'id', 'title', 'comment'},
        'sampleSubmission.csv': {'id', 'rating'},
    }
    for name, required in expected_headers.items():
        path = project / name
        print(name + '_EXISTS =', path.is_file())
        if not path.is_file():
            missing.append(str(path))
            continue
        print(name + '_BYTES =', path.stat().st_size)
        try:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                header = next(csv.reader(handle))
            print(name + '_HEADER =', json.dumps(header))
            if not required.issubset(header) or len(header) != len(set(header)):
                issues.append(name + ': missing or duplicate required columns')
            if name == 'test.csv' and 'rating' in header:
                issues.append('test.csv unexpectedly contains rating')
        except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
            issues.append(name + ': ' + str(exc))

    candidates = [args.labels.resolve()] if args.labels else [
        PACKAGE / 'spark_preprocessing' / 'input' / 'clean_y.npz',
        project / 'clean_y.npz',
    ]
    found = [path for path in candidates if path.is_file()]
    print('FULL_LABEL_CANDIDATES =', json.dumps([str(p) for p in candidates], ensure_ascii=False))
    print('FULL_LABELS_FOUND =', bool(found))
    if not found:
        missing.append('full clean_y.npz (the validation-only slice is not a substitute)')
    elif 'numpy' not in missing:
        import numpy as np
        labels = found[0]
        print('FULL_LABELS_PATH =', labels)
        print('FULL_LABELS_BYTES =', labels.stat().st_size)
        try:
            with np.load(labels, allow_pickle=False) as data:
                if not {'cy', 'matched', 'ag'}.issubset(data.files):
                    raise ValueError('Expected cy, matched and ag arrays')
                cy, matched, ag = data['cy'], data['matched'], data['ag']
                print('LABEL_ARRAY_SHAPES =', {k: list(data[k].shape) for k in ('cy', 'matched', 'ag')})
                if any(x.shape != (7_684_645,) for x in (cy, matched, ag)):
                    raise ValueError('Full labels must cover 7,684,645 original review IDs')
                if matched.dtype != np.bool_:
                    raise ValueError('matched must be a boolean array')
                if not np.isin(cy, [0, 1, 2, 3, 4, 5]).all() or not np.isin(ag, [1, 2, 3, 4, 5]).all():
                    raise ValueError('Unexpected source/delivered rating values')
                valid = matched & (cy > 0)
                print('VALID_SOURCE_LABEL_ROWS =', int(np.count_nonzero(valid)))
                valid_split = valid[::40]
                print('EXPECTED_MATCHED_VALIDATION_ROWS =', int(np.count_nonzero(valid_split)))
                print('EXPECTED_TRAINING_ROWS =', int(np.count_nonzero(valid) - np.count_nonzero(valid_split)))
        except (OSError, ValueError, KeyError) as exc:
            issues.append('Full label validation failed: ' + str(exc))

    print('FREE_DISK_GIB =', round(shutil.disk_usage(PACKAGE).free / (1024 ** 3), 2))
    print('MISSING_ITEMS =', json.dumps(missing, ensure_ascii=False))
    print('INPUT_ISSUES =', json.dumps(issues, ensure_ascii=False))
    ready = not missing and not issues
    print('STEP_1_RESULT =', 'READY_FOR_NEXT_STEP' if ready else 'NEEDS_ATTENTION')
    print('NO_PREPROCESSING_OR_DOWNLOAD_WAS_PERFORMED')
    return 0 if ready else 2


if __name__ == '__main__':
    raise SystemExit(main())
