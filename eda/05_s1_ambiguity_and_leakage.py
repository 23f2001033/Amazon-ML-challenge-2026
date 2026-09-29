"""EDA 5: S1 name ambiguity (same name, different entity), address structure, and ID/order leakage."""
import numpy as np
import polars as pl
from common import load, gt_pairs, OUT
from norm import norm_name, core_name, basic

pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(100); pl.Config.set_tbl_width_chars(250)

for split in ["train", "test"]:
    s1 = load(f"{split}_s1")
    s1 = s1.with_columns(
        pl.col("business_name").map_elements(norm_name, return_dtype=pl.String).alias("nn"),
        pl.col("business_name").map_elements(core_name, return_dtype=pl.String).alias("cn"),
        pl.col("business_address").str.split(",").list.eval(pl.element().str.strip_chars()).alias("parts"),
    )
    print(f"\n######## {split} S1")
    for key in ["business_name", "nn", "cn"]:
        g = s1.group_by(["country", key]).len()
        amb = g.filter(pl.col("len") > 1)
        print(f"{key}: S1 rows sharing their (country,{key}) with >=1 other S1: "
              f"{amb['len'].sum():,} / {s1.height:,} ({amb['len'].sum()/s1.height:.3f}); max group={g['len'].max()}")
    print("most common core names:")
    print(s1.group_by(["country", "cn"]).len().sort("len", descending=True).head(15))
    # Address components: last part / first part patterns per country
    s1 = s1.with_columns(pl.col("parts").list.len().alias("np"))
    for c in s1["country"].unique().to_list():
        sc = s1.filter(pl.col("country") == c)
        print(f"\n[{c}] address parts count dist:", sc["np"].value_counts().sort("np").to_dicts()[:10])
        # candidate city/state vocab = alpha-only parts; how many distinct
        alpha = sc.select(pl.col("parts").explode()).filter(~pl.col("parts").str.contains(r"\d"))
        vc = alpha.group_by("parts").len().sort("len", descending=True)
        print(f"[{c}] distinct non-numeric address parts: {vc.height:,}; top:")
        print(vc.head(25))
        # name + address duplicates
        print(f"[{c}] dup (core name, first address part):",
              sc.height - sc.select("cn", pl.col("parts").list.first()).unique().height)

# ---- leakage checks ----
print("\n######## leakage checks (train)")
p = gt_pairs()
s1 = load("train_s1").with_row_index("r1")
s2 = load("train_s2").with_row_index("r2"); s3 = load("train_s3").with_row_index("r3")
q = (p.join(s1.select(pl.col("entity_id").alias("s1"), "r1"), on="s1")
      .join(pl.concat([s2.select(pl.col("entity_id").alias("mid"), pl.col("r2").alias("rm")),
                       s3.select(pl.col("entity_id").alias("mid"), pl.col("r3").alias("rm"))]), on="mid"))
q = q.with_columns(pl.col("s1").str.slice(3).cast(pl.Int64).alias("i1"), pl.col("mid").str.slice(3).cast(pl.Int64).alias("im"))
print("corr(S1 id num, match id num):", np.corrcoef(q["i1"], q["im"])[0, 1])
print("corr(S1 row, match row):", np.corrcoef(q["r1"].cast(pl.Float64), q["rm"].cast(pl.Float64))[0, 1])
print("S1 ids appear in S2/S3 id numbers? overlap of numeric parts:",
      len(set(q["i1"].to_list()) & set(q["im"].to_list())))
print("GT file row order vs S1 file row order corr:",
      np.corrcoef(load("train_gt").with_row_index("g").join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "r1"), on="source1_entity_id").select("g", "r1").to_numpy().T.astype(float))[0, 1])
