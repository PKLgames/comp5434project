"""Input-data contract checker: verifies every artefact the model consumes.

Run it before training or before believing a submission:

    python check_data.py --train-csv ../Dataset2_metadata/train.csv \
        --test-csv ../Dataset2_metadata/test.csv

Nothing is trained here; every check is a cheap pass over the files. Exit code 0 = all pass.

The contract (see DATA_SPEC.md for the full prose version):

  train.csv         10 columns in this order:
                    id,title,user_id,comment,prod_id,parent_prod_id,time,votes,purchased,rating
  test.csv           9 columns: the same without rating
  ids                per-file contiguous 0..N-1 (train 7,684,645 / test 10,000)
  rating             1..5 integers, distribution ~[.147,.087,.097,.116,.553]  (corrupted labels)
  text               title + ". " + comment   (exactly this join, used by build_text.py)
  clean_y.npz        keys cy/matched/ag, len == train rows, coverage 92.50%, agreement 80.01%
  validation slice   id % 40 == 0 -> 192,117 rows; training uses the other 7,492,528
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

TRAIN_COLS = ["id", "title", "user_id", "comment", "prod_id", "parent_prod_id",
              "time", "votes", "purchased", "rating"]
TEST_COLS = TRAIN_COLS[:-1]
EXPECT_TRAIN_ROWS = 7_684_645
EXPECT_TEST_ROWS = 10_000
EXPECT_RATING_SHARE = np.array([0.147, 0.087, 0.097, 0.116, 0.553])

ok = True


def check(name, cond, detail=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else ""))
    ok = ok and bool(cond)


def stream(csv_path, columns, block=1 << 24):
    """Yield batches of the requested columns without materialising the file."""
    table = pcsv.open_csv(csv_path, read_options=pcsv.ReadOptions(block_size=block),
                          convert_options=pcsv.ConvertOptions(include_columns=columns))
    for batch in table:
        yield batch


def col(batch, name):
    return batch.column(batch.schema.get_field_index(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--sample-submission", default=None)
    ap.add_argument("--clean", default=None)
    ap.add_argument("--train-text", default=None)
    ap.add_argument("--test-text", default=None)
    ap.add_argument("--preds-dir", default=None)
    a = ap.parse_args()

    if a.train_csv:
        print("== train.csv ==")
        with open(a.train_csv, "rb") as f:
            header = f.readline().decode("utf-8").strip().split(",")
        check("column order", header == TRAIN_COLS, str(header))
        n = 0
        id_max, id_sum = -1, 0
        rating_hist = np.zeros(5, dtype=np.int64)
        bad = {"rating": 0, "purchased": 0, "votes": 0, "time": 0}
        votes_min = 0
        for b in stream(a.train_csv, ["id", "rating", "purchased", "votes", "time"]):
            ids = col(b, "id").to_numpy(zero_copy_only=False)
            n += len(ids)
            id_max = max(id_max, int(ids.max()) if len(ids) else -1)
            id_sum += int(ids.sum())
            r = col(b, "rating").to_numpy(zero_copy_only=False)
            bad["rating"] += int(((r < 1) | (r > 5)).sum())
            rating_hist += np.bincount(np.clip(r, 1, 5).astype(int) - 1, minlength=5)
            pur = col(b, "purchased").cast(pa.int8()).to_numpy(zero_copy_only=False)
            bad["purchased"] += int(((pur != 0) & (pur != 1)).sum())
            v = col(b, "votes").to_numpy(zero_copy_only=False)
            bad["votes"] += int((v < 0).sum())
            votes_min = min(votes_min, int(v.min()) if len(v) else 0)
            t = col(b, "time").to_numpy(zero_copy_only=False)
            bad["time"] += int((t <= 0).sum())
        share = rating_hist / max(n, 1)
        check("row count", n == EXPECT_TRAIN_ROWS, "%d rows" % n)
        check("ids contiguous 0..N-1", id_max == n - 1 and id_sum == n * (n - 1) // 2,
              "max id %d, expected %d" % (id_max, n - 1))
        check("rating in 1..5", bad["rating"] == 0, "%d violations" % bad["rating"])
        check("purchased in {0,1}", bad["purchased"] == 0, "%d violations" % bad["purchased"])
        # exactly one row in the delivered data has votes = -1 (id 1862497); Part A flags it as
        # votes_invalid, so a single negative sentinel is accepted and reported.
        check("votes >= 0 (one documented -1 sentinel allowed)", bad["votes"] <= 1 and votes_min >= -1,
              "%d negative, min %d" % (bad["votes"], votes_min))
        check("time > 0 (epoch ms)", bad["time"] == 0, "%d violations" % bad["time"])
        check("rating distribution (corrupted, ~25% U{1..5})",
              float(np.abs(share - EXPECT_RATING_SHARE).max()) < 0.01,
              "share " + " ".join("%.3f" % x for x in share))

        if a.train_text:
            print("== train_text.parquet vs train.csv ==")
            import pyarrow.parquet as pq
            tt = pq.read_table(a.train_text)
            check("columns", tt.column_names == ["id", "text"], str(tt.column_names))
            check("row count matches csv", tt.num_rows == n, "%d" % tt.num_rows)
            # spot-check the exact text formula on the first 200 rows of the CSV
            head = next(stream(a.train_csv, ["id", "title", "comment"], block=1 << 20)).slice(0, 200)
            expect = pa.array([(t or "") + ". " + (c or "") for t, c in
                               zip(head.column("title").to_pylist(), head.column("comment").to_pylist())])
            # build_text.py streams the CSV in order, so parquet row i corresponds to CSV row i
            got = tt.slice(0, 200)
            check("first 200 ids match the csv order", got["id"].to_pylist() == head.column("id").to_pylist())
            check("text == title + '. ' + comment (200 rows)", got["text"].to_pylist() == expect.to_pylist())

    if a.test_csv:
        print("== test.csv ==")
        with open(a.test_csv, "rb") as f:
            header = f.readline().decode("utf-8").strip().split(",")
        check("column order (no rating)", header == TEST_COLS, str(header))
        n = 0
        id_max = -1
        for b in stream(a.test_csv, ["id"]):
            ids = col(b, "id").to_numpy(zero_copy_only=False)
            n += len(ids)
            id_max = max(id_max, int(ids.max()))
        check("row count", n == EXPECT_TEST_ROWS, "%d rows" % n)
        check("ids 0..9999", id_max == n - 1, "max id %d" % id_max)

    if a.sample_submission:
        print("== sampleSubmission.csv ==")
        with open(a.sample_submission, "rb") as f:
            header = f.readline().decode("utf-8").strip().split(",")
        check("header id,rating", header == ["id", "rating"], str(header))

    if a.clean:
        print("== clean_y.npz ==")
        d = np.load(a.clean)
        check("keys cy/matched/ag", set(d.files) >= {"cy", "matched", "ag"}, str(d.files))
        cy, mm = d["cy"], d["matched"].astype(bool)
        check("length == train rows", len(cy) == EXPECT_TRAIN_ROWS, "%d" % len(cy))
        cov = float(mm.mean())
        check("coverage ~92.50%", abs(cov - 0.9250) < 0.01, "%.4f" % cov)
        ag = float((d["ag"][mm] == cy[mm]).mean())
        check("agreement ~80.01%", abs(ag - 0.8001) < 0.01, "%.4f" % ag)
        p = (1 - ag) / (4 / 5)
        check("implied corruption rate ~25%", abs(p - 0.25) < 0.02, "p = %.4f" % p)
        check("cy in 1..5 where matched", bool(((cy[mm] >= 1) & (cy[mm] <= 5)).all()))

    if a.test_text:
        print("== test_text.parquet ==")
        import pyarrow.parquet as pq
        tt = pq.read_table(a.test_text)
        check("columns", tt.column_names == ["id", "text"], str(tt.column_names))
        check("row count", tt.num_rows == EXPECT_TEST_ROWS, "%d" % tt.num_rows)
        check("first text is the expected test row",
              tt["text"][0].as_py().startswith("Five Stars. It popped fluffy"),
              tt["text"][0].as_py()[:48])

    if a.preds_dir:
        print("== stored predictions ==")
        P = Path(a.preds_dir)
        vids = np.load(P / "ftvids_s1.npy")
        check("validation slice size", len(vids) == 192_117, "%d" % len(vids))
        check("validation slice is id % 40 == 0", bool(((vids % 40) == 0).all()))
        check("validation ids unique", len(np.unique(vids)) == len(vids))
        for tag in ("s1", "s2", "d1", "b1"):
            check("ftv_%s length" % tag, len(np.load(P / ("ftv_%s.npy" % tag))) == len(vids))
            check("ftp_%s length" % tag, len(np.load(P / ("ftp_%s.npy" % tag))) == EXPECT_TEST_ROWS)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
