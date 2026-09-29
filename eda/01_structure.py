"""EDA 1: file structure, nulls, IDs, countries, ground-truth cardinality."""
import polars as pl
from common import load, gt_pairs, OUT

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(120)

names = ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]
print("## Per-file overview")
for n in names:
    df = load(n)
    pre = df["entity_id"].str.slice(0, 3).value_counts()
    print(f"\n### {n}: rows={df.height:,} unique_ids={df['entity_id'].n_unique():,} prefixes={pre.to_dicts()}")
    print("null counts:", {c: df[c].null_count() for c in df.columns})
    print("literal 'null'/'nan'/'None' strings:",
          {c: df.filter(pl.col(c).str.to_lowercase().is_in(["null", "nan", "none", "na", "n/a", "-"])).height
           for c in ["business_name", "business_address"]})
    print(df["country"].value_counts(sort=True))
    print("exact duplicate (name,address,country) rows:", df.height - df.select(["business_name", "business_address", "country"]).unique().height)
    print("duplicate names:", df.height - df["business_name"].n_unique())
    # ID numeric part
    num = df["entity_id"].str.slice(3).cast(pl.Int64, strict=False)
    print("id numeric range:", num.min(), num.max(), "non-numeric:", num.null_count())

print("\n## Ground truth")
gt = load("train_gt")
s1 = load("train_s1")
print("GT rows", gt.height, "unique s1", gt["source1_entity_id"].n_unique(),
      "all S1 in GT:", s1["entity_id"].is_in(gt["source1_entity_id"].implode()).all())
print("singletons (empty list):", gt["matched_entity_ids"].null_count(),
      f"= {gt['matched_entity_ids'].null_count() / gt.height:.4f}")

p = gt_pairs()
print("total positive pairs:", p.height, p["src"].value_counts().to_dicts())
per = p.group_by("s1").agg(n=pl.len(), n2=(pl.col("src") == "S2").sum(), n3=(pl.col("src") == "S3").sum())
allg = gt.select(pl.col("source1_entity_id").alias("s1")).join(per, on="s1", how="left").fill_null(0)
print("\nmatches per S1 (incl singletons):")
print(allg["n"].value_counts().sort("n"))
print("\nS2 matches per S1:"); print(allg["n2"].value_counts().sort("n2"))
print("\nS3 matches per S1:"); print(allg["n3"].value_counts().sort("n3"))
print("\nn2 x n3 crosstab (top):")
print(allg.group_by(["n2", "n3"]).len().sort("len", descending=True).head(25))
print("mean matches/S1:", allg["n"].mean(), "mean among non-singletons:", allg.filter(pl.col("n") > 0)["n"].mean())

# Is each S2/S3 record matched to at most one S1?
dup = p.group_by("mid").len().filter(pl.col("len") > 1)
print("\nS2/S3 ids appearing under >1 S1:", dup.height)
# Duplicate within list?
print("dup within list:", p.height - p.unique(["s1", "mid"]).height)

s2 = load("train_s2"); s3 = load("train_s3")
for nm, df in [("S2", s2), ("S3", s3)]:
    m = df["entity_id"].is_in(p["mid"].implode())
    print(f"{nm}: {df.height:,} records, matched to some S1: {m.sum():,} ({m.mean():.4f}); unmatched (distractors): {(~m).sum():,}")
print("GT ids not present in source files:",
      p.filter(~pl.col("mid").is_in(pl.concat([s2["entity_id"], s3["entity_id"]]).implode())).height)

# Country consistency across matched pairs
c = pl.concat([s2.select("entity_id", "country"), s3.select("entity_id", "country")])
pc = (p.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c1")), on="s1")
       .join(c.rename({"entity_id": "mid", "country": "cm"}), on="mid"))
print("\ncountry agreement in positive pairs:", (pc["c1"] == pc["cm"]).mean())
print(pc.filter(pl.col("c1") != pl.col("cm")).group_by(["c1", "cm"]).len())

# Singleton rate & cluster size by country
sc = allg.join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
print("\nby country:")
print(sc.group_by("country").agg(n=pl.len(), singleton_rate=(pl.col("n") == 0).mean(),
                                 mean_matches=pl.col("n").mean(), mean_s2=pl.col("n2").mean(),
                                 mean_s3=pl.col("n3").mean(), max_matches=pl.col("n").max()))
allg.join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1").write_parquet(OUT + "/gt_cardinality.parquet")
