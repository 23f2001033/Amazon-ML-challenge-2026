"""EDA 4b: which true pairs does name+address TF-IDF top-50 miss, and why?"""
import polars as pl
from common import load, OUT

pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(260)
nb = pl.read_parquet(f"{OUT}/04_neighbours.parquet")
F = pl.read_parquet(f"{OUT}/04_pair_features.parquet").filter(pl.col("y") == 1)
found = nb.filter(pl.col("is_match")).select("s1", "mid").unique().with_columns(pl.lit(True).alias("found"))
F = F.join(found, on=["s1", "mid"], how="left").with_columns(pl.col("found").fill_null(False))
print("positive pairs:", F.height, "missed@50:", (~F["found"]).sum(), f"recall={F['found'].mean():.4f}")
print("\nmiss rate by condition:")
conds = {
    "name_nonascii": pl.col("name_nonascii") == 1, "addr_null": pl.col("addr_null") == 1,
    "core_exact": pl.col("core_exact") == 1, "core_ratio<60": pl.col("core_ratio") < 60,
    "first_num_eq": pl.col("first_num_eq") == 1, "addr_tset<70": (pl.col("addr_tset") >= 0) & (pl.col("addr_tset") < 70),
}
for k, c in conds.items():
    sub = F.filter(c)
    print(f"  {k:<16} share_of_pos={sub.height / F.height:.3f}  miss_rate={1 - sub['found'].mean():.3f}  share_of_misses={(~sub['found']).sum() / (~F['found']).sum():.3f}")
for c in ["India", "US"]:
    for s in ["S2", "S3"]:
        sub = F.filter((pl.col("country") == c) & (pl.col("src") == s))
        print(f"  {c} {s}: recall={sub['found'].mean():.4f}")
# Rank at which the true match appears (best of name/addr)
best = nb.filter(pl.col("is_match")).group_by(["s1", "mid"]).agg(pl.col("rank").min().alias("best_rank"))
print("\nbest-rank distribution of found positives:", best["best_rank"].describe())
# examples of misses
miss = F.filter(~pl.col("found")).sample(25, seed=1)
ids = pl.concat([miss["s1"], miss["mid"]])
recs = pl.concat([load("train_s1"), load("train_s2"), load("train_s3")]).filter(pl.col("entity_id").is_in(ids.implode()))
rd = {r[0]: r[1:3] for r in recs.iter_rows()}
print("\nexamples of missed positives:")
for a, b, c in miss.select("s1", "mid", "country").iter_rows():
    print(f"[{c}] S1 | {rd[a][0]!s:<45} | {rd[a][1]}")
    print(f"     {b[:2]} | {rd[b][0]!s:<45} | {rd[b][1]}\n")
