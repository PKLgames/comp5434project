"""Step 2: download pinned input artifacts and validate them, without preprocessing."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


HERE = Path(__file__).resolve().parent
COMMIT = '261e9e648cc0b1080fae0fe192f9ab1fa7635946'
BASE_URL = 'https://raw.githubusercontent.com/PKLgames/comp5434project/' + COMMIT + '/'
ARTIFACTS = [
    ('task2_model/audit/preds/clean_y.npz', 'clean_y.npz', 23054675,
     'ae06b0662299c393c0bb7a40de35c99bde098a3c'),
    ('task2_model/audit/DATA_SPEC.md', 'DATA_SPEC.md', 8096,
     'd6cf4077916b476d225f926a9ba364bbd80f3e1d'),
    ('task2_model/audit/check_data.py', 'check_data.py', 8779,
     '3a7d7519a5b5dbac466409883033613686710e75'),
]


def verify(path, expected_size, expected_blob):
    if path.stat().st_size != expected_size:
        raise ValueError('Unexpected file size: ' + str(path))
    git_hash = hashlib.sha1()
    git_hash.update(b'blob ' + str(expected_size).encode('ascii') + b'\0')
    sha256 = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            git_hash.update(block)
            sha256.update(block)
    if git_hash.hexdigest() != expected_blob:
        raise ValueError('Git content checksum mismatch: ' + str(path))
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    args = parser.parse_args()
    # Stop before downloading if the selected interpreter still lacks dependencies.
    for package in ('numpy', 'pyarrow'):
        print(package.upper() + '_VERSION =', importlib.metadata.version(package), flush=True)
    destination = HERE / 'input'
    destination.mkdir(exist_ok=True)
    records = []
    print('SOURCE_COMMIT =', COMMIT, flush=True)
    for source, name, size, blob in ARTIFACTS:
        output = destination / name
        if output.exists():
            digest = verify(output, size, blob)
            print('REUSE_VERIFIED =', output, flush=True)
        else:
            partial = output.with_name(output.name + '.download')
            if partial.exists():
                raise FileExistsError('Incomplete prior download found; report this path: ' + str(partial))
            request = urllib.request.Request(BASE_URL + source, headers={'User-Agent': 'COMP5434-Input-Preparation'})
            print('DOWNLOADING =', name, 'bytes =', size, flush=True)
            with urllib.request.urlopen(request, timeout=90) as response, partial.open('xb') as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            digest = verify(partial, size, blob)
            partial.rename(output)
            print('DOWNLOAD_VERIFIED =', name, flush=True)
        records.append({'name': name, 'repository_path': source, 'bytes': size,
                        'git_blob_sha1': blob, 'sha256': digest})
    manifest = {'repository': 'PKLgames/comp5434project', 'commit': COMMIT, 'files': records}
    (destination / 'INPUT_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # The previous checker only reads headers, versions and label arrays; it does not
    # run check_data.py, launch Spark, or scan the complete CSV files.
    print('DOWNLOADS_COMPLETE; RECHECKING INPUTS', flush=True)
    result = subprocess.run([sys.executable, str(HERE / '01_check_inputs.py'),
                             '--labels', str(destination / 'clean_y.npz'),
                             '--data-dir', str(args.data_dir.resolve())], check=False)
    if result.returncode:
        print('STEP_2_RESULT = INPUT_CHECK_NEEDS_ATTENTION', flush=True)
        return result.returncode
    print('STEP_2_RESULT = INPUTS_READY', flush=True)
    print('NO_SPARK_JOB_OR_CSV_PREPROCESSING_WAS_RUN', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
