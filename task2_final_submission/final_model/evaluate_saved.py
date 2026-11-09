"""Evaluate the saved validation predictions without downloading data or running models."""
import argparse
import json
from pathlib import Path
import numpy as np
from build_submission import load_validation_labels, read_params


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preds-dir', type=Path, default=Path(__file__).parent / 'preds')
    a = p.parse_args()
    cfg = read_params(a.preds_dir / 'ensemble_params.json')
    ids = np.load(a.preds_dir / 'ftvids_s1.npy', allow_pickle=False)
    y, matched = load_validation_labels(a.preds_dir, ids)
    mask = matched & (y > 0)
    def rmse(pred, selection):
        return float(np.sqrt(np.mean((np.clip(pred[selection], 1, 5) - y[selection]) ** 2)))
    arrays = []
    print('validation_rows', len(ids), 'matched_rows', int(mask.sum()))
    for tag in cfg['tags']:
        pred = np.clip(np.load(a.preds_dir / f'ftv_{tag}.npy', allow_pickle=False).astype(np.float64), 1, 5)
        if pred.shape != ids.shape or not np.isfinite(pred).all():
            raise ValueError(f'Invalid validation array: {tag}')
        arrays.append(pred)
        print(tag, 'validation_rmse', rmse(pred, mask))
    # Full-precision weights are used for development validation, as in the source.
    blend = sum(float(w) * pred for w, pred in zip(cfg['weights'], arrays))
    print('weighted_validation_rmse', rmse(blend, mask))
    evaluate = mask & (np.arange(len(ids)) % 2 == 1)
    print('calibration_evaluation_rows', int(evaluate.sum()))
    print('calibration_half_rmse', rmse(np.interp(blend, cfg['iso_x'], cfg['iso_y']), evaluate))
    print('Note: ensemble weights were selected on the development validation set.')


if __name__ == '__main__':
    main()
