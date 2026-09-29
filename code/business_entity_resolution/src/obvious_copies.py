"""Label-free pipeline audit: records that are OBVIOUSLY an S1's copy (identical canonical name AND identical
canonical address + house number, the S1 being the only S1 with that name at that address) should be
(a) blocking candidates, (b) kept by stage-1 pruning, (c) accepted. Shortfalls per country reveal
retrieval/decision bugs that no labelled validation can show for France.
Also a looser class: identical address + house and the same name tokens in any order (canonical n_core set)."""
import os
import polars as pl
from config import FEAT_DIR, PREPARED_DIR

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"
cols = ["entity_id", "country", "region", "n_compact", "n_core", "a_tokens", "a_house", "a_null"]
s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=cols)
rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"test_s{i}.parquet"), columns=cols) for i in (2, 3)])
key = lambda: pl.concat_str([pl.col("n_compact").fill_null(""), pl.col("a_tokens").fill_null(""), pl.col("a_house").fill_null("")], separator="|")
s1 = s1.with_columns(key().alias("k"), pl.col("n_core").fill_null("").str.split(" ").list.sort().list.join(" ").alias("tokset"))
rec = rec.with_columns(key().alias("k"), pl.col("n_core").fill_null("").str.split(" ").list.sort().list.join(" ").alias("tokset"))
s1 = s1.filter((pl.col("a_null") == 0) & (pl.col("a_house").fill_null("") != "") & (pl.col("n_compact").fill_null("").str.len_chars() >= 3))
uniq = s1.group_by("country", "k").len("n_s1").filter(pl.col("n_s1") == 1).drop("n_s1")
pairs = rec.filter(pl.col("a_null") == 0).select(pl.col("entity_id").alias("mid"), "country", "k", pl.col("region").alias("r_region")).join(
    s1.join(uniq, on=["country", "k"]).select(pl.col("entity_id").alias("s1"), "country", "k", pl.col("region").alias("s_region")), on=["country", "k"])
cand = pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"})
final = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S2N}.parquet"), columns=["s1", "mid", "p"])
acc = pl.read_csv("/home/ubuntu/er/sub_v6fr3s/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).drop_nulls("matched_entity_ids")
acc = acc.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"}).with_columns(pl.lit(1).alias("acc"))
owner = acc.select("mid", pl.col("s1").alias("owner"))
d = pairs.join(cand, on=["s1", "mid"], how="left").join(final, on=["s1", "mid"], how="left").join(acc, on=["s1", "mid"], how="left").join(owner, on="mid", how="left")
d = d.with_columns(pl.col("acc").fill_null(0),
                   pl.when(pl.col("p1").is_null()).then(pl.lit("1_not_candidate(p>=.02)"))
                     .when(pl.col("acc") == 1).then(pl.lit("4_accepted"))
                     .when(pl.col("owner").is_not_null()).then(pl.lit("3_given_to_other_S1"))
                     .otherwise(pl.lit("2_rejected")).alias("fate"),
                   (pl.col("r_region") == pl.col("s_region")).alias("same_region"))
n1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country"]).group_by("country").len("S1")
print("OBVIOUS copies (identical canonical name + address + house; unique S1 at that key):")
t = d.group_by("country", "fate").len().join(d.group_by("country").len("tot"), on="country").with_columns((pl.col("len") / pl.col("tot")).round(4).alias("share"))
print(t.sort("country", "fate"))
print(d.group_by("country").agg(pl.len().alias("obvious_pairs"), (~pl.col("same_region")).mean().round(4).alias("region_mismatch")).join(n1, on="country").with_columns((pl.col("obvious_pairs") / pl.col("S1")).round(3).alias("per_S1")))
miss = d.filter(pl.col("fate") != "4_accepted")
miss.select("s1", "mid", "country", "fate", "p1", "p", "same_region").write_parquet("/home/ubuntu/er/obvious_missed.parquet")
print("missed obvious copies by country/fate/region-match:")
print(miss.group_by("country", "fate", "same_region").len().sort("country", "fate"))
