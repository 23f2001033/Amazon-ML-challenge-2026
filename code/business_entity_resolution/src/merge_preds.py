"""Merge test stage-2 predictions by country: US/India from one stage-2 variant, France from another.
  python merge_preds.py <usin_s2name> <france_s2name> <out_s2name>
reads features/test_pred_<name>.parquet (s1, mid, p, country), writes features/test_pred_<out_s2name>.parquet.
Final (v6fr3x): US/India from ..._w2 (test-faithful weights incl. outside-owner records at 0.05), France from ..._w."""
import os
import sys

import polars as pl

from config import FEAT_DIR

usin, fr, out = sys.argv[1:4]
a = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{usin}.parquet"))
b = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{fr}.parquet"))
m = pl.concat([a.filter(pl.col("country") != "France"), b.filter(pl.col("country") == "France").select(a.columns)])
assert m.height == a.height
m.write_parquet(os.path.join(FEAT_DIR, f"test_pred_{out}.parquet"))
print(f"wrote test_pred_{out}.parquet: {m.height:,} pairs")
