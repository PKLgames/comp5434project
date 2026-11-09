"""Fit or apply the MiniLM+BGE two-member ensemble.

Weight search and calibration fitting use even validation positions; evaluation
uses odd positions. --apply uses only saved test predictions and parameters.
"""
import argparse
from pathlib import Path
import numpy as np
from build_submission import (apply_params, load_predictions, load_validation_labels,
                              read_params, rmse, validate_sample, write_params,
                              write_submission)

HERE = Path(__file__).resolve().parent
TAGS = ['s1', 'b1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preds-dir', type=Path, default=HERE / 'preds')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--params-in', type=Path)
    parser.add_argument('--params-out', type=Path)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    validate_sample(args.sample)
    test = load_predictions(args.preds_dir, TAGS, 'ftp', 10000)
    if args.apply:
        cfg = read_params(args.params_in or args.preds_dir / 'ensemble_params_pruned.json')
        if cfg['tags'] != TAGS or not isinstance(cfg['weights'], dict):
            raise ValueError('Expected pruned s1+b1 parameters')
    else:
        from sklearn.isotonic import IsotonicRegression
        params_out = args.params_out or args.out.with_suffix('.params.json')
        if params_out.exists():
            raise FileExistsError(params_out)
        ids = np.load(args.preds_dir / 'ftvids_s1.npy', allow_pickle=False)
        y, matched = load_validation_labels(args.preds_dir, ids, args.labels)
        valid = load_predictions(args.preds_dir, TAGS, 'ftv', len(ids))
        even = (np.arange(len(ids)) % 2 == 0) & matched
        odd = (np.arange(len(ids)) % 2 == 1) & matched
        if not even.any() or not odd.any():
            raise ValueError('Both calibration halves need matched labels')
        best = None
        # Preserve the original float32 arithmetic throughout the grid search.
        for w in np.arange(0., 1.0001, .01):
            p = np.clip(w * valid['s1'] + (1 - w) * valid['b1'], 1, 5)
            score = rmse(p[even], y[even])
            if best is None or score < best[1]:
                best = (round(float(w), 2), score)
        w, raw_even = best
        pred = np.clip(w * valid['s1'] + (1 - w) * valid['b1'], 1, 5)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=1., y_max=5.).fit(pred[even], y[even])
        cfg = {'tags': TAGS, 'weights': {'s1': w, 'b1': round(1 - w, 2)},
               'iso_x': iso.X_thresholds_.tolist(), 'iso_y': iso.y_thresholds_.tolist(),
               'validation_rmse_raw': rmse(pred[matched], y[matched]),
               'validation_rmse_iso_heldout': rmse(iso.predict(pred[odd]), y[odd]),
               'weight_fit_rmse': raw_even, 'fit_rows': int(even.sum()), 'eval_rows': int(odd.sum()),
               'weight_selection': 'even development-validation positions only'}
        write_params(params_out, cfg)
        print('PRUNED_FIT_COMPLETE', 'weight_s1', w, 'evaluation_rmse', cfg['validation_rmse_iso_heldout'])
    write_submission(args.out, apply_params(test, cfg), fixed_decimals=True)
    print('PRUNED_SUBMISSION_COMPLETE', args.out, 'rows', 10000)


if __name__ == '__main__':
    main()
