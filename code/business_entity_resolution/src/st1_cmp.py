"""Compare stage-1 models on test-faithful validation (variant B: records owned outside the universe removed,
dropped-S1 orphans thinned to test density): best-threshold macro F0.5 per country, pruning recall (true pairs with
p >= 0.02), and the confident set (p >= 0.99) precision.   python st1_cmp.py v6 v6t [kind=val]"""
import os
import sys
import numpy as np
import polars as pl
import run_dev as R
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05, truth_map_from_gt
from model import assign

KEEP = {"US": 0.47, "India": 0.105}
tags = sys.argv[1:3]
kind = sys.argv[3] if len(sys.argv) > 3 else "val"
uni = R.universes()
u = uni.filter(pl.col("universe") == kind)
qs = u.filter(pl.col("is_query"))["entity_id"]
gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
truth = truth_map_from_gt(gt, qs.to_list())
owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
cmap = dict(s1c.filter(pl.col("entity_id").is_in(qs.implode())).iter_rows())
for tag in tags:
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet"), columns=["s1", "mid", "p", "y"])
    m = d.select("mid").unique().join(owner, on="mid", how="left").join(s1c.rename({"entity_id": "owner"}), on="owner", how="left")
    outside = m.filter(pl.col("owner").is_not_null() & ~pl.col("owner").is_in(u["entity_id"].implode())).select("mid")
    dropped = m.filter(pl.col("owner").is_in(u["entity_id"].implode()) & ~pl.col("owner").is_in(qs.implode())).with_columns(
        pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("k"), (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h"))
    drop = pl.concat([outside, dropped.filter(pl.col("h") >= pl.col("k")).select("mid")]).unique()
    d = d.join(drop, on="mid", how="anti")
    tot_true = sum(len(v) for v in truth.values())
    print(f"[{tag}] true pairs with p>=0.02: {d.filter((pl.col('y') == 1) & (pl.col('p') >= 0.02)).height / tot_true:.4f} | "
          f"confident p>=0.99: {d.filter(pl.col('p') >= 0.99).height:,} precision {d.filter(pl.col('p') >= 0.99)['y'].mean():.5f}", flush=True)
    for t in (0.6, 0.65, 0.7, 0.75, 0.8):
        sc = macro_f05(assign(d, t), truth, qs.to_list())[1]
        per = {c: round(float(np.mean([s for s, q in zip(sc, qs.to_list()) if cmap[q] == c])), 5) for c in ("US", "India")}
        print(f"   thr {t}: all {sc.mean():.5f} {per}", flush=True)
