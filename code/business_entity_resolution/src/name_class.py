"""Wrong owner inside a name class. For an accepted pair whose S1 compact name is shared by 2..MAXK S1s of the
country, compare the record's address with EVERY S1 of that name class (not only the retrieved candidates).
Flag = some other same-name S1 matches the record's address strictly better than the assigned S1.
  python name_class.py wide   -> labelled precision of flagged pairs (US/India held-out)
  python name_class.py test   -> flagged pairs per S1 by country (+ saved for France)"""
import os
import sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from config import FEAT_DIR, PREPARED_DIR
from model import assign

MAXK = 300
S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"
THR = {"US": 0.6, "India": 0.5, "France": 0.725}
pl.Config.set_tbl_rows(-1)
split = "train" if sys.argv[1] == "wide" else "test"
cols = ["entity_id", "country", "n_compact", "a_tokens", "a_house", "a_null"]
s1 = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=cols)
if split == "train":
    import run_dev as R
    uni = R.universes()
    s1 = s1.join(uni.filter((pl.col("universe") == "wide") & pl.col("is_query")).select("entity_id"), on="entity_id")   # the S1 table stage 2 saw
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{S2N}.parquet"), columns=["s1", "mid", "p", "y"])
    d = d.join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
else:
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S2N}.parquet"), columns=["s1", "mid", "p", "country"])
    veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
    d = d.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
m = assign(d, 0.725, THR)
acc = pl.DataFrame([(s, x) for s, xs in m.items() for x in xs], schema=["s1", "mid"], orient="row").join(d, on=["s1", "mid"])
cls = s1.with_columns(pl.len().over("country", "n_compact").alias("k")).filter((pl.col("k") >= 2) & (pl.col("k") <= MAXK) & (pl.col("a_null") == 0))
rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s{i}.parquet"), columns=["entity_id", "a_tokens", "a_house", "a_null"]) for i in (2, 3)]).rename(
    {"entity_id": "mid", "a_tokens": "ra", "a_house": "rh", "a_null": "rnull"})
a = acc.join(cls.select(pl.col("entity_id").alias("s1"), "n_compact", "k"), on="s1", how="inner").join(rec, on="mid").filter(pl.col("rnull") == 0)
print(f"accepted pairs in shared-name classes (2..{MAXK}): {a.height:,}", flush=True)
# every member of the class
x = a.select("s1", "mid", "country", "n_compact", "ra", "rh").join(
    cls.select(pl.col("entity_id").alias("o"), "country", "n_compact", pl.col("a_tokens").alias("oa"), pl.col("a_house").alias("oh")), on=["country", "n_compact"])
x = x.with_columns(pl.col(c).fill_null("") for c in ("ra", "rh", "oa", "oh"))
x = x.with_columns(pl.Series("sim", cpdist(x["ra"].to_list(), x["oa"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
x = x.with_columns((pl.col("sim") + 5.0 * ((pl.col("rh") == pl.col("oh")) & (pl.col("rh") != "")).cast(pl.Float32)).alias("score"))
own = x.filter(pl.col("o") == pl.col("s1")).select("s1", "mid", pl.col("score").alias("own"))
oth = x.filter(pl.col("o") != pl.col("s1")).group_by("s1", "mid").agg(pl.col("score").max().alias("best_other"))
f = a.join(own, on=["s1", "mid"]).join(oth, on=["s1", "mid"]).with_columns((pl.col("best_other") > pl.col("own")).alias("flag"),
                                                                          (pl.col("best_other") - pl.col("own")).alias("gap"))
if split == "train":
    print(f.group_by("country", "flag").agg(pl.len(), pl.col("y").mean().round(4).alias("P_true")).sort("country", "flag"))
    print(f.filter(pl.col("flag")).with_columns(pl.col("gap").cut([5, 15, 30], labels=["<=5", "5-15", "15-30", ">30"]).alias("g"))
           .group_by("g").agg(pl.len(), pl.col("y").mean().round(4).alias("P_true")).sort("g"))
else:
    n1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country"]).group_by("country").len("S1")
    print(f.group_by("country").agg(pl.len(), pl.col("flag").sum().alias("flagged")).join(n1, on="country").with_columns(
        (pl.col("flagged") / pl.col("S1") * 1000).round(2).alias("flag_per1k")))
    print(f.filter(pl.col("flag")).with_columns(pl.col("gap").cut([5, 15, 30], labels=["<=5", "5-15", "15-30", ">30"]).alias("g")).group_by("country", "g").len().sort("country", "g"))
    f.filter(pl.col("flag")).select("s1", "mid", "country", "gap", "p").write_parquet("/home/ubuntu/er/name_class_flags.parquet")
