"""Validate the count-model group test on labelled US/India held-out data (wide universe):
estimated true share of each rejected best-candidate group vs the actual share (y)."""
import os
import numpy as np
import polars as pl
import count_fit as CF
import sizebias_scan as SB
from config import FEAT_DIR, PREPARED_DIR
from model import assign
import run_dev as R

gen = CF.generator_pmf()
uni = R.universes()
qs = uni.filter((pl.col("universe") == "wide") & pl.col("is_query")).select("entity_id")
s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country", "n_compact"]).join(qs, on="entity_id")
d = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "y"]).join(
    pl.read_parquet(os.path.join(FEAT_DIR, "wide_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
d = d.join(s1c.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
m = assign(d, 0.725, SB.THR)
acc = pl.DataFrame([(s, x) for s, xs in m.items() for x in xs], schema=["s1", "mid"], orient="row")
cnt = acc.with_columns(pl.col("mid").str.slice(0, 2).alias("src")).group_by("s1", "src").len("n").pivot(on="src", index="s1", values="n").fill_null(0)
base_all = s1c.select(pl.col("entity_id").alias("s1"), "country").join(cnt, on="s1", how="left").with_columns(pl.col("S2").fill_null(0), pl.col("S3").fill_null(0))
best = d.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first").filter(pl.col("p") >= 0.05)
rej = best.join(acc.with_columns(pl.lit(1).alias("a")), on=["s1", "mid"], how="left").filter(pl.col("a").is_null()).drop("a")
rej = SB.describe(SB.frame("train", rej.drop("country")))
rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"train_s{i}.parquet"), columns=["entity_id", "n_compact"]) for i in (2, 3)]).rename({"entity_id": "mid"})
kn = s1c.group_by("country", "n_compact").len("k")
rej = rej.join(rec, on="mid", how="left").join(kn, on=["country", "n_compact"], how="left").with_columns(pl.col("k").fill_null(0))
rej = rej.with_columns(pl.when(pl.col("arel") == "no_addr").then(pl.concat_str([pl.lit("no_addr k="), pl.col("k").cut([1, 3], labels=["1", "2-3", "4+"]).cast(pl.String)]))
                         .otherwise(pl.col("arel")).alias("a2"), pl.col("p").cut([0.2, 0.5], labels=["<.2", ".2-.5", ">=.5"]).alias("pb"))


def fit_c(b):
    e = {s: CF.fit(b[s].to_numpy(), gen[s]) for s in ("S2", "S3")}
    return (e["S2"][0] + e["S3"][0]) / 2, e["S2"][1] + e["S3"][1]


for ctry in ("US", "India"):
    b0 = base_all.filter(pl.col("country") == ctry)
    r0, x0 = fit_c(b0)
    print(f"[{ctry}] base r {r0:.4f} extra {x0:.4f}", flush=True)
    rc = rej.filter(pl.col("country") == ctry)
    groups = rc.group_by("a2", "nrel", "pb").len().filter(pl.col("len") >= 800).sort("len", descending=True).head(12)
    for a2, nrel, pb, n in groups.iter_rows():
        g = rc.filter((pl.col("a2") == a2) & (pl.col("nrel") == nrel) & (pl.col("pb") == pb))
        add = g.with_columns(pl.col("mid").str.slice(0, 2).alias("src")).group_by("s1", "src").len("n").pivot(on="src", index="s1", values="n").fill_null(0)
        for c in ("S2", "S3"):
            if c not in add.columns:
                add = add.with_columns(pl.lit(0).alias(c))
        b = b0.join(add.rename({"S2": "a2_", "S3": "a3_"}), on="s1", how="left").with_columns(
            (pl.col("S2") + pl.col("a2_").fill_null(0)).alias("S2"), (pl.col("S3") + pl.col("a3_").fill_null(0)).alias("S3"))
        r, x = fit_c(b)
        per = n / b0.height
        est = 1 - min(max((x - x0) / per, 0), 1)
        print(f"   {a2:<14} {nrel:<16} p{pb:<6} n={n:>6}: est. true share {est:.2f} | ACTUAL {g['y'].mean():.2f}", flush=True)
