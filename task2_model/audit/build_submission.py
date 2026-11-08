"""Reproduce the 0.49893 leaderboard submission from the committed predictions.

Inputs (all in ./preds/ by default):
  ftv_<tag>.npy      fine-tuned model predictions on the validation slice (id % 40 == 0)
  ftp_<tag>.npy      fine-tuned model predictions on the 10,000 test rows
  ftvids_s1.npy      the validation ids (shared by every tag)
  clean_y.npz        recovered clean labels: cy / matched / ag

Modes
  fit      (default) search blend weights on the matched validation rows, fit an
           isotonic map on the even half, report held-out RMSE, write the params
           JSON and the submission CSV.
  apply    skip the search, load --params-in and emit the identical submission.

Example
  python build_submission.py --tags s1 d1 b1 \
      --out ../../submission_ens.csv --params-out preds/ensemble_params.json
  python build_submission.py --apply --params-in preds/ensemble_params.json --out sub.csv
"""
import argparse
import csv
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = argparse.ArgumentParser()
P.add_argument("--preds-dir", default=os.path.join(HERE, "preds"))
P.add_argument("--tags", nargs="+", default=["s1", "d1", "b1"])
P.add_argument("--out", default="submission_ens.csv")
P.add_argument("--params-in")
P.add_argument("--params-out")
P.add_argument("--no-iso", action="store_true")
P.add_argument("--apply", action="store_true")
a = P.parse_args()

D = a.preds_dir
clean = np.load(os.path.join(D, "clean_y.npz"))
cy, matched = clean["cy"], clean["matched"]


def rmse(p, t):
    return float(np.sqrt(np.mean((np.clip(p, 1.0, 5.0) - t) ** 2)))


def load_valid(tag):
    p = np.clip(np.load(os.path.join(D, "ftv_%s.npy" % tag)).astype(np.float64), 1, 5)
    return p


ids = np.load(os.path.join(D, "ftvids_s1.npy"))
y = cy[ids].astype(np.float64)
m = matched[ids] & (cy[ids] > 0)
print("validation rows %d, matched with clean label %d" % (len(ids), int(m.sum())))

Pv = {t: load_valid(t) for t in a.tags}
for t in a.tags:
    print("  %-3s raw clean rmse %.4f" % (t, rmse(Pv[t][m], y[m])))

T = {t: np.clip(np.load(os.path.join(D, "ftp_%s.npy" % t)).astype(np.float64), 1, 5) for t in a.tags}

if a.apply:
    cfg = json.load(open(a.params_in))
    assert cfg["tags"] == a.tags, "tag order mismatch"
    w = np.array(cfg["weights"], dtype=np.float64)
    pred = sum(w[i] * Pv[t] for i, t in enumerate(a.tags))
    tp = sum(w[i] * T[t] for i, t in enumerate(a.tags))
    print("weights", dict(zip(a.tags, np.round(w, 4))), "validation rmse %.4f" % rmse(pred[m], y[m]))
else:
    best_w, best_r = np.full(len(a.tags), 1.0 / len(a.tags)), None
    pred = sum(best_w[i] * Pv[t] for i, t in enumerate(a.tags))
    best_r = rmse(pred[m], y[m])
    print("equal blend %.4f" % best_r)
    rng = np.random.default_rng(0)
    for _ in range(400):
        w = rng.dirichlet(np.ones(len(a.tags)))
        p = sum(w[i] * Pv[t] for i, t in enumerate(a.tags))
        r = rmse(p[m], y[m])
        if r < best_r:
            best_w, best_r = w, r
    print("best weights", dict(zip(a.tags, np.round(best_w, 4))), "validation rmse %.4f" % best_r)
    pred = sum(best_w[i] * Pv[t] for i, t in enumerate(a.tags))
    tp = sum(best_w[i] * T[t] for i, t in enumerate(a.tags))

if not a.apply:
    cfg = {"tags": a.tags, "weights": [float(x) for x in best_w]}

w_test = np.round(w, 3) if a.apply else np.round(best_w, 3)
if a.apply:
    tp = sum(w_test[i] * T[t] for i, t in enumerate(a.tags))

if not a.no_iso:
    from sklearn.isotonic import IsotonicRegression

    if a.apply:
        iso_x = np.array(cfg["iso_x"], dtype=np.float64)
        iso_y = np.array(cfg["iso_y"], dtype=np.float64)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1.0, y_max=5.0)
        iso.fit(iso_x, iso_y)
        print("iso loaded from params")
    else:
        half = np.arange(len(ids)) % 2 == 0
        fit, ev = m & half, m & ~half
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1.0, y_max=5.0).fit(pred[fit], y[fit])
        cfg["iso_x"] = [float(v) for v in iso.X_thresholds_]
        cfg["iso_y"] = [float(v) for v in iso.y_thresholds_]
        print("isotonic held-out rmse %.4f (fit %d / eval %d)" % (rmse(iso.predict(pred[ev]), y[ev]), int(fit.sum()), int(ev.sum())))
    pred = iso.predict(pred)
    tp = iso.predict(tp)

print("test preds n=%d mean %.3f sd %.3f min %.3f max %.3f" % (len(tp), tp.mean(), tp.std(), tp.min(), tp.max()))
with open(a.out, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["id", "rating"])
    for i, v in enumerate(tp):
        wcsv.writerow([i, round(float(np.clip(v, 1.0, 5.0)), 6)])
print("wrote", a.out)

if a.params_out:
    cfg["validation_rmse_raw"] = rmse(pred[m], y[m])
    json.dump(cfg, open(a.params_out, "w"), indent=2)
    print("wrote", a.params_out)
