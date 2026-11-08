"""Build the two-member pruned submission (s1 MiniLM + b1 bge-small).

Protocol (leakage-safe):
  * weight w is selected on the EVEN half of the matched validation rows only;
  * the isotonic map is fitted on that same even half;
  * the reported RMSE is measured on the ODD half (never used for fitting).
The test blend reuses the stored fine-tuned test predictions, so no encoder is run.
"""
import argparse, csv, json, pathlib

import numpy as np
from sklearn.isotonic import IsotonicRegression

TAGS = ("s1", "b1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default=str(pathlib.Path(__file__).parent / "preds"))
    ap.add_argument("--out", default="submission_pruned.csv")
    ap.add_argument("--params-out", default=None)
    a = ap.parse_args()
    P = pathlib.Path(a.preds_dir)

    d = np.load(P / "clean_y.npz")
    vids = np.load(P / "ftvids_s1.npy")
    cy = d["cy"][vids].astype(np.float64)
    matched = d["matched"][vids] & (d["cy"][vids] > 0)
    ftv = [np.load(P / ("ftv_%s.npy" % t)) for t in TAGS]
    ftp = [np.load(P / ("ftp_%s.npy" % t)) for t in TAGS]

    n = len(vids)
    even = (np.arange(n) % 2 == 0) & matched
    odd = (np.arange(n) % 2 == 1) & matched

    best = None
    for w in np.arange(0.0, 1.0001, 0.01):
        p = np.clip(w * ftv[0] + (1 - w) * ftv[1], 1, 5)
        r = float(np.sqrt(np.mean((p[even] - cy[even]) ** 2)))
        if best is None or r < best[1]:
            best = (round(float(w), 2), r)
    w, raw_even = best

    v = np.clip(w * ftv[0] + (1 - w) * ftv[1], 1, 5)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1.0, y_max=5.0).fit(v[even], cy[even])
    raw_all = float(np.sqrt(np.mean((v[matched] - cy[matched]) ** 2)))
    iso_odd = float(np.sqrt(np.mean((np.clip(iso.predict(v[odd]), 1, 5) - cy[odd]) ** 2)))
    print("weights w_s1=%.2f (selected on even half)" % w)
    print("valid raw rmse  all=%.4f  even=%.4f" % (raw_all, raw_even))
    print("isotonic fit even=%d eval odd=%d -> heldout rmse %.4f" % (even.sum(), odd.sum(), iso_odd))

    test = np.clip(w * ftp[0] + (1 - w) * ftp[1], 1, 5)
    pred = np.clip(iso.predict(test), 1, 5)

    out = pathlib.Path(a.out)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "rating"])
        for i, r in enumerate(pred):
            wr.writerow([i, "%.6f" % float(r)])
    print("test preds n=%d mean %.4f sd %.4f min %.4f max %.4f"
          % (len(pred), pred.mean(), pred.std(), pred.min(), pred.max()))

    params = {
        "tags": list(TAGS),
        "weights": {TAGS[0]: w, TAGS[1]: round(1 - w, 2)},
        "validation_rmse_raw": round(raw_all, 4),
        "validation_rmse_iso_heldout": round(iso_odd, 4),
        "iso_x": [float(x) for x in iso.X_thresholds_],
        "iso_y": [float(y) for y in iso.y_thresholds_],
        "fit_rows": int(even.sum()),
        "eval_rows": int(odd.sum()),
    }
    pp = pathlib.Path(a.params_out) if a.params_out else P / "ensemble_params_pruned.json"
    pp.write_text(json.dumps(params, indent=1))
    print("wrote %s and %s" % (out, pp))


if __name__ == "__main__":
    main()
