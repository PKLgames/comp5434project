"""Fit the four-member ensemble or rebuild a CSV from saved predictions.

--apply needs NumPy only. Fitting additionally needs scikit-learn.
The shared functions are also used by pruning and fresh inference.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ['s1', 's2', 'd1', 'b1']


def read_params(path):
    with Path(path).open(encoding='utf-8') as stream:
        cfg = json.load(stream)
    tags, weights = cfg['tags'], cfg['weights']
    if not tags or len(set(tags)) != len(tags):
        raise ValueError('Invalid model tags')
    values = np.asarray([weights[t] for t in tags] if isinstance(weights, dict) else weights, dtype=np.float64)
    if (values.shape != (len(tags),) or not np.isfinite(values).all()
            or (values < 0).any() or not np.isclose(values.sum(), 1, atol=.002)):
        raise ValueError('Invalid ensemble weights')
    x, y = np.asarray(cfg.get('iso_x', [])), np.asarray(cfg.get('iso_y', []))
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('Invalid calibration arrays')
    if x.size and (not np.isfinite(x).all() or not np.isfinite(y).all()
                   or (np.diff(x) <= 0).any() or (np.diff(y) < 0).any()):
        raise ValueError('Calibration thresholds must be finite and monotonic')
    return cfg


def load_predictions(directory, tags, prefix, expected_rows):
    arrays = {}
    for tag in tags:
        array = np.load(Path(directory) / f'{prefix}_{tag}.npy', allow_pickle=False)
        if array.shape != (expected_rows,) or not np.isfinite(array).all():
            raise ValueError(f'Invalid prediction array: {prefix}_{tag}')
        arrays[tag] = array
    return arrays


def load_validation_labels(directory, ids, labels=None):
    """Accept full ID-indexed source labels or the included lossless slice."""
    path = Path(labels) if labels else Path(directory) / 'validation_labels.npz'
    if not path.is_file() and labels is None:
        path = Path(directory) / 'clean_y.npz'
    with np.load(path, allow_pickle=False) as source:
        if 'ids' in source.files:
            if not np.array_equal(source['ids'], ids):
                raise ValueError('Compact validation-label ID order mismatch')
            y, matched = source['cy'].copy(), source['matched'].copy()
        else:
            if ids.dtype.kind not in 'iu' or ids.min() < 0 or ids.max() >= len(source['cy']):
                raise ValueError('Validation IDs outside the full label domain')
            y, matched = source['cy'][ids], source['matched'][ids]
    if y.shape != ids.shape or matched.shape != ids.shape or matched.dtype != np.bool_:
        raise ValueError('Invalid validation-label arrays')
    return y.astype(np.float64), matched & (y >= 1) & (y <= 5)


def raw_blend(predictions, cfg):
    """Preserve each submitted recipe's rounding, clipping and dtype order."""
    tags, weights = cfg['tags'], cfg['weights']
    if isinstance(weights, dict):
        if tags != ['s1', 'b1']:
            raise ValueError('Dictionary weights describe the s1+b1 recipe')
        w = float(weights['s1'])
        # The source pruned recipe blends float32 arrays BEFORE clipping.
        return np.clip(w * np.asarray(predictions['s1'], dtype=np.float32)
                       + (1 - w) * np.asarray(predictions['b1'], dtype=np.float32), 1, 5)
    weights = np.round(np.asarray(weights, dtype=np.float64), 3)
    return sum(weights[i] * np.clip(np.asarray(predictions[tag], dtype=np.float64), 1, 5)
               for i, tag in enumerate(tags))


def apply_params(predictions, cfg, no_iso=False):
    values = raw_blend(predictions, cfg)
    if not no_iso and cfg.get('iso_x'):
        # sklearn's isotonic transform retains the prediction-input dtype.
        # This float32 cast matters at CSV rounding boundaries for pruning.
        values = np.interp(values, cfg['iso_x'], cfg['iso_y']).astype(values.dtype)
    return np.clip(values, 1, 5)


def validate_sample(path):
    if path:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        if (not rows or rows[0] != ['id', 'rating']
                or [int(row[0]) for row in rows[1:]] != list(range(10000))):
            raise ValueError('Sample must contain IDs 0..9999 in order')


def write_submission(path, values, fixed_decimals=False, ids=None):
    values = np.asarray(values)
    if values.shape != (10000,) or not np.isfinite(values).all():
        raise ValueError('Need 10,000 finite test predictions')
    ids = np.arange(10000) if ids is None else np.asarray(ids)
    if not np.array_equal(ids, np.arange(10000)):
        raise ValueError('Test IDs must remain 0..9999 in order')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\r\n')
        writer.writerow(['id', 'rating'])
        for record_id, value in zip(ids, values):
            value = float(np.clip(value, 1, 5))
            writer.writerow([int(record_id), '%.6f' % value if fixed_decimals else round(value, 6)])


def write_params(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(cfg, stream, indent=2)


def rmse(predictions, targets):
    return float(np.sqrt(np.mean((np.clip(predictions, 1, 5) - targets) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preds-dir', type=Path, default=HERE / 'preds')
    parser.add_argument('--tags', nargs='+', default=TAGS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--params-in', type=Path)
    parser.add_argument('--params-out', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--no-iso', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    validate_sample(args.sample)
    test = load_predictions(args.preds_dir, args.tags, 'ftp', 10000)
    if args.apply:
        cfg = read_params(args.params_in or args.preds_dir / 'ensemble_params.json')
        if cfg['tags'] != args.tags or isinstance(cfg['weights'], dict):
            raise ValueError('Use build_submission_pruned.py for the two-member recipe')
    else:
        from sklearn.isotonic import IsotonicRegression
        params_out = args.params_out or args.out.with_suffix('.params.json')
        if params_out.exists():
            raise FileExistsError(params_out)
        ids = np.load(args.preds_dir / 'ftvids_s1.npy', allow_pickle=False)
        y, matched = load_validation_labels(args.preds_dir, ids, args.labels)
        if not matched.any():
            raise ValueError('No matched validation labels')
        valid = load_predictions(args.preds_dir, args.tags, 'ftv', len(ids))
        valid = {t: np.clip(p.astype(np.float64), 1, 5) for t, p in valid.items()}
        best_weights = np.full(len(args.tags), 1 / len(args.tags))
        def blend(weights):
            return sum(weights[i] * valid[t] for i, t in enumerate(args.tags))
        best_score = rmse(blend(best_weights)[matched], y[matched])
        rng = np.random.default_rng(0)
        for _ in range(400):
            weights = rng.dirichlet(np.ones(len(args.tags)))
            score = rmse(blend(weights)[matched], y[matched])
            if score < best_score:
                best_weights, best_score = weights, score
        cfg = {'tags': args.tags, 'weights': best_weights.tolist(),
               'validation_rmse_raw': best_score,
               'weight_selection': 'all matched development validation rows'}
        pred = blend(best_weights)
        if not args.no_iso:
            even = matched & (np.arange(len(ids)) % 2 == 0)
            odd = matched & (np.arange(len(ids)) % 2 == 1)
            if not even.any() or not odd.any():
                raise ValueError('Both calibration halves need matched labels')
            iso = IsotonicRegression(out_of_bounds='clip', y_min=1., y_max=5.).fit(pred[even], y[even])
            cfg.update(iso_x=iso.X_thresholds_.tolist(), iso_y=iso.y_thresholds_.tolist(),
                       validation_rmse_iso_heldout=rmse(iso.predict(pred[odd]), y[odd]),
                       fit_rows=int(even.sum()), eval_rows=int(odd.sum()))
        write_params(params_out, cfg)
        print('ENSEMBLE_FIT_COMPLETE', 'raw_rmse', best_score, 'params', params_out)
    write_submission(args.out, apply_params(test, cfg, no_iso=args.no_iso))
    print('SUBMISSION_COMPLETE', args.out, 'rows', 10000)


if __name__ == '__main__':
    main()
