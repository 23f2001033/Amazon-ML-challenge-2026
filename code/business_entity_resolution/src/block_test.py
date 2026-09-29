"""Blocking recall check on wide regions: current candidates vs v6 blocking (ER_COMB / ER_LEAKS).
  ER_COMB=1 ER_LEAKS=1 python block_test.py telangana maharashtra nc"""
import sys
import polars as pl
import run_dev as R
from blocking import generate

regs = sys.argv[1:]
uni = R.universes()
s1 = R.load_prep("train_s1", R.BLOCK_COLS)
q = s1.join(uni.filter(pl.col("is_query")), on="entity_id").filter(pl.col("region").is_in(regs)).select(R.BLOCK_COLS)
pool = R.load_prep(["train_s2", "train_s3"], R.BLOCK_COLS)
pos = R.gt_pairs().join(q.select(pl.col("entity_id").alias("s1"), "region"), on="s1")
old = pl.read_parquet(R.os.path.join(R.CAND_DIR, "dev_wide_wide.parquet")).join(q.select(pl.col("entity_id").alias("s1")), on="s1")
new = generate(q, pool, log=R.log, workers=6)
for nm, c in (("current", old), ("v6", new)):
    f = pos.join(c.select("s1", "mid"), on=["s1", "mid"], how="left", coalesce=True)
    found = pos.join(c.select("s1", "mid", "hit"), on=["s1", "mid"], how="left")
    per = c.join(q.select(pl.col("entity_id").alias("s1"), "region"), on="s1").group_by("region").agg(pl.len())
    rec = found.group_by("region").agg(pl.col("hit").is_not_null().mean().round(4).alias("recall"), pl.len().alias("gt")).join(
        per, on="region").join(q.group_by("region").len().rename({"len": "S1"}), on="region").with_columns(
        (pl.col("len") / pl.col("S1")).round(1).alias("cand/S1")).sort("region")
    print(f"== {nm}:", rec.select("region", "recall", "cand/S1", "gt").rows())
    if nm == "v6":
        for bit, lab in ((16, "combined"), (32, "leak")):
            only = found.filter(pl.col("hit").is_not_null() & ((pl.col("hit") & (bit | 1 | 2 | 4)) == bit) if bit == 16 else pl.col("hit").is_not_null() & ((pl.col("hit") & 32) > 0))
            print(f"   true pairs found via {lab}{' only' if bit == 16 else ''}: {only.height:,}")
