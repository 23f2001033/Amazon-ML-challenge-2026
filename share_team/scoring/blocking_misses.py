"""Why are true pairs missing from the candidates? (wide universe, all query S1)"""
import os, sys
sys.path.insert(0, "/home/ubuntu/er/code/business_entity_resolution/src")
os.environ["ER_UNIVERSES"] = "big"
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
import run_dev as R
pl.Config.set_tbl_rows(-1); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(250)
F, Q, P = "/home/ubuntu/er/data/features", "/home/ubuntu/er/data/parquet", "/home/ubuntu/er/data/prepared"
uni = R.universes()
kind = "wide"
qs = uni.filter((pl.col("universe") == kind) & pl.col("is_query")).select(pl.col("entity_id").alias("s1"))
gt = (pl.read_parquet(f"{Q}/train_gt.parquet").drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
        .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("mid"))).join(qs, on="s1")
cand = pl.read_parquet(f"{F}/{kind}_pred_v6.parquet", columns=["s1", "mid"])
miss = gt.join(cand, on=["s1", "mid"], how="anti")
cols = ["entity_id", "country", "region", "n_core", "n_compact", "a_tokens", "a_house", "a_null"]
s1 = pl.read_parquet(f"{P}/train_s1.parquet", columns=cols)
rec = pl.concat([pl.read_parquet(f"{P}/train_s{i}.parquet", columns=cols) for i in (2, 3)])
raw = pl.concat([pl.read_parquet(f"{Q}/train_s{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (1, 2, 3)])
d = (miss.join(s1.rename({c: c + "_1" for c in cols[1:]}).rename({"entity_id": "s1"}), on="s1")
         .join(rec.rename({c: c + "_2" for c in cols[1:]}).rename({"entity_id": "mid"}), on="mid"))
d = d.with_columns(pl.col(c).fill_null("") for c in d.columns if d[c].dtype == pl.String)
d = d.with_columns(pl.Series("nm", cpdist(d["n_core_1"].to_list(), d["n_core_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.Series("ad", cpdist(d["a_tokens_1"].to_list(), d["a_tokens_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
print(f"missed true pairs: {miss.height:,} of {gt.height:,} ({miss.height / gt.height:.4f})")
d = d.with_columns(
    pl.when(pl.col("a_null_2").cast(pl.Int8) == 1).then(pl.lit("no_addr"))
      .when(pl.col("region_2") == "").then(pl.lit("rec_region_unknown"))
      .when(pl.col("region_2") != pl.col("region_1")).then(pl.lit("region_differs"))
      .otherwise(pl.lit("same_region")).alias("reg"),
    pl.col("nm").cut([49.99, 79.99, 99.99], labels=["n<50", "n50-80", "n80-99", "n100"]).alias("name"),
    pl.col("ad").cut([49.99, 79.99, 89.99], labels=["a<50", "a50-80", "a80-90", "a90+"]).alias("addr"))
print(d.group_by("country_1", "reg").len().sort("country_1", "len", descending=[False, True]))
print(d.filter(pl.col("reg") != "no_addr").group_by("reg", "name", "addr").len().sort("len", descending=True).head(20))
ex = d.filter(pl.col("reg") != "no_addr").sample(25, seed=3)
ex = (ex.join(raw.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("S1_name"), pl.col("business_address").alias("S1_addr")), on="s1")
        .join(raw.select(pl.col("entity_id").alias("mid"), pl.col("business_name").alias("rec_name"), pl.col("business_address").alias("rec_addr")), on="mid"))
for r in ex.select("reg", "region_1", "region_2", "nm", "ad", "S1_name", "rec_name", "S1_addr", "rec_addr").iter_rows():
    print(f"\n[{r[0]}] S1 region={r[1]!r} rec region={r[2]!r} name_sim={r[3]:.0f} addr_sim={r[4]:.0f}\n   S1 : {r[5]!r} | {r[7]!r}\n   rec: {r[6]!r} | {r[8]!r}")
nn = d.filter(pl.col("reg") == "no_addr")
print("\nno-address misses by name similarity:", nn.group_by("name").len().sort("name").rows())
ex2 = nn.sample(12, seed=5).join(raw.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("S1_name")), on="s1").join(
    raw.select(pl.col("entity_id").alias("mid"), pl.col("business_name").alias("rec_name")), on="mid")
for r in ex2.select("country_1", "nm", "S1_name", "rec_name").iter_rows():
    print(f"  {r[0]:<6} sim={r[1]:.0f}  S1 {r[2]!r:<50} rec {r[3]!r}")
