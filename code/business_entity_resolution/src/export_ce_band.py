"""Uncertain-band pairs (stage-1 0.02 <= p < 0.99) of val / wide / test for the heavier cross-encoders
(CE v2, CE 3), which score only the band; confident pairs (p >= 0.99) get p_ce = 1.0 in stage 2.

  python export_ce_band.py <tag> <out_dir>   -> out_dir/band_{val,wide,test}.parquet (s1, mid, text_a, text_b)
"""
import os
import sys

import polars as pl

from config import FEAT_DIR
from export_ce_data import attach_text

LO, HI = 0.02, 0.99


def main(tag, out):
    os.makedirs(out, exist_ok=True)
    for kind, split in (("val", "train"), ("wide", "train"), ("test", "test")):
        d = (pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet"), columns=["s1", "mid", "p"])
               .filter((pl.col("p") >= LO) & (pl.col("p") < HI)).select("s1", "mid").unique())
        d = attach_text(d, split).select("s1", "mid", pl.col("text_a").fill_null(""), pl.col("text_b").fill_null(""))
        d.write_parquet(os.path.join(out, f"band_{kind}.parquet"))
        print(f"{kind}: {d.height:,} band pairs")


if __name__ == "__main__":
    main(*sys.argv[1:3])
