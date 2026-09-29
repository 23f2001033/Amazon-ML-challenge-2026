"""RAW-text pair files for cross-encoder 3 (different view: original names/addresses, accents,
legal forms and native scripts, instead of our canonical text).

  python export_ce_raw.py <train2.parquet> <band_dir> <out_dir> <n_train>
  -> out_dir/train3.parquet (first n_train rows of train2's shuffled pairs, raw text)
     out_dir/raw_band_{val,wide,test}.parquet
"""
import os
import sys

import polars as pl

from config import PARQUET_DIR


def raw(split):
    return pl.concat([pl.read_parquet(os.path.join(PARQUET_DIR, f"{split}_s{i}.parquet"),
                                      columns=["entity_id", "business_name", "business_address"]) for i in (1, 2, 3)]).with_columns(
        (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t")).select("entity_id", "t")


def attach(d, split):
    r = raw(split)
    return (d.join(r.rename({"entity_id": "s1", "t": "text_a"}), on="s1", how="left")
             .join(r.rename({"entity_id": "mid", "t": "text_b"}), on="mid", how="left")
             .with_columns(pl.col("text_a").fill_null(""), pl.col("text_b").fill_null("")))


def main(train2, band_dir, out, n_train):
    os.makedirs(out, exist_ok=True)
    tr = pl.read_parquet(train2, columns=["s1", "mid", "y"]).head(int(n_train))
    attach(tr, "train").write_parquet(os.path.join(out, "train3.parquet"))
    print(f"train3: {tr.height:,} pairs, positives {int(tr['y'].sum()):,}")
    for kind, split in (("val", "train"), ("wide", "train"), ("test", "test")):
        b = pl.read_parquet(os.path.join(band_dir, f"band_{kind}.parquet"), columns=["s1", "mid"])
        attach(b, split).write_parquet(os.path.join(out, f"raw_band_{kind}.parquet"))
        print(f"raw_band_{kind}: {b.height:,}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
