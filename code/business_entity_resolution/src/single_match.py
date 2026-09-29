"""S1s with exactly ONE predicted match: if the S1 is really a singleton, that one false match costs the whole S1
(1.0), five times a false match on an S1 with 4 matches. Profile the single match by pattern cell per country;
on labelled US/India data give P(match true) and P(S1 is a singleton) per cell.
  python single_match.py wide | test"""
import os
import sys
import polars as pl
import sizebias_scan as SB
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from model import assign

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
split = "train" if sys.argv[1] == "wide" else "test"
if split == "train":
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "y"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "wide_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    d = d.join(pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
else:
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "country"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
    d = d.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
m = assign(d, 0.725, SB.THR)
single = pl.DataFrame([(s, next(iter(xs))) for s, xs in m.items() if len(xs) == 1], schema=["s1", "mid"], orient="row")
x = SB.describe(SB.frame(split, single.join(d.drop("country"), on=["s1", "mid"])))
x = x.with_columns(pl.col("p").cut([0.8, 0.95, 0.99], labels=["<.8", ".8-.95", ".95-.99", ">=.99"]).alias("pb"))
if split == "train":
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")).rename({"source1_entity_id": "s1"})
    x = x.join(gt, on="s1", how="left").with_columns(pl.col("matched_entity_ids").fill_null("").eq("").alias("S1_singleton"))
    print("WIDE labelled: S1s with exactly one predicted match")
    print(x.group_by("country").agg(pl.len(), pl.col("y").mean().round(4).alias("P_true"), pl.col("S1_singleton").mean().round(4).alias("P_singleton")))
    print(x.group_by("arel", "nrel").agg(pl.len(), pl.col("y").mean().round(3).alias("P_true"), pl.col("S1_singleton").mean().round(3).alias("P_singleton"))
           .filter(pl.col("len") >= 100).sort("len", descending=True))
    print(x.group_by("pb").agg(pl.len(), pl.col("y").mean().round(3).alias("P_true"), pl.col("S1_singleton").mean().round(3).alias("P_singleton")).sort("pb"))
else:
    n1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country"]).group_by("country").len("S1")
    t = x.group_by("country", "arel", "nrel").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("S1") * 1000).round(2).alias("per1k"))
    print(t.pivot(on="country", index=["arel", "nrel"], values="per1k").fill_null(0).sort("France", descending=True).head(20))
    t2 = x.group_by("country", "pb").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("S1") * 1000).round(2).alias("per1k"))
    print(t2.pivot(on="country", index="pb", values="per1k").sort("pb"))
    x.select("s1", "mid", "country", "arel", "nrel", "p", "p1").write_parquet("/home/ubuntu/er/single_match_test.parquet")
