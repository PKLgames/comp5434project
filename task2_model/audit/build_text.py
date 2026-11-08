"""Stream train.csv / test.csv -> parquet(id, text) for the Colab fine-tuning stage.

text = title + ". " + comment  (same construction used by the 0.51355 run).
Writes zstd parquet, block-wise, never materialising the whole file.
"""
import argparse, os, time
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.compute as pcomp
import pyarrow.parquet as pq


def build(src, dst, limit=None):
    t0 = time.time()
    rdr = pc.open_csv(
        src,
        read_options=pc.ReadOptions(block_size=1 << 26),
        convert_options=pc.ConvertOptions(include_columns=["id", "title", "comment"]),
    )
    writer = None
    n = 0
    for batch in rdr:
        title = pcomp.fill_null(batch["title"], "")
        comment = pcomp.fill_null(batch["comment"], "")
        txt = pcomp.binary_join_element_wise(title, comment, ". ")
        tab = pa.table({"id": batch["id"], "text": txt})
        if writer is None:
            writer = pq.ParquetWriter(dst, tab.schema, compression="zstd")
        writer.write_table(tab)
        n += tab.num_rows
        if n % 2000000 < tab.num_rows:
            print("[%5.0fs] %d rows" % (time.time() - t0, n), flush=True)
        if limit and n >= limit:
            break
    if writer is not None:
        writer.close()
    print("wrote", dst, n, "rows", os.path.getsize(dst), "bytes", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(a.src, a.dst, a.limit or None)
