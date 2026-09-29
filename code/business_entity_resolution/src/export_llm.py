"""Pairs for the LLM judge (Qwen2.5-7B-Instruct, Apache-2.0), raw text.
  llm_val.parquet : labelled US/India held-out (wide) best-candidate pairs in the uncertain range, stratified
  llm_fr.parquet  : France best-candidate pairs in the uncertain range (after the swap veto; vetoed pairs included,
                    flagged, as a sanity check)
Columns: s1, mid, country, p (stage-2), [y], vetoed, name1, addr1, name2, addr2."""
import os
import sys
import polars as pl
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from model import assign

S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def raw(split):
    return pl.concat([pl.read_parquet(os.path.join(PARQUET_DIR, f"{split}_s{i}.parquet"), columns=["entity_id", "business_name", "business_address"]) for i in (1, 2, 3)])


def attach(d, split):
    r = raw(split)
    return (d.join(r.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("name1"), pl.col("business_address").alias("addr1")), on="s1")
             .join(r.select(pl.col("entity_id").alias("mid"), pl.col("business_name").alias("name2"), pl.col("business_address").alias("addr2")), on="mid")
             .with_columns(pl.col(c).fill_null("") for c in ("name1", "addr1", "name2", "addr2")))


# ---- labelled US/India (wide, out-of-fold stage-2 p)
w = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{S2N}.parquet"), columns=["s1", "mid", "p", "y"])
w = w.join(pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
best = w.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True).filter((pl.col("p") >= 0.15) & (pl.col("p") < 0.995))
best = best.with_columns(pl.col("p").cut([0.3, 0.5, 0.725, 0.9], labels=["a", "b", "c", "d", "e"]).alias("pb"))
val = pl.concat([best.filter((pl.col("country") == c) & (pl.col("pb") == b)).sample(min(600, best.filter((pl.col("country") == c) & (pl.col("pb") == b)).height), seed=7)
                 for c in ("US", "India") for b in ("a", "b", "c", "d", "e")])
val = attach(val.drop("pb"), "train").with_columns(pl.lit(0).alias("vetoed"))
val.write_parquet(os.path.join(out, "llm_val.parquet"))
print("llm_val:", val.height, "pairs; positives", int(val["y"].sum()))

# ---- France test
t = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S2N}.parquet"), columns=["s1", "mid", "p", "country"]).filter(pl.col("country") == "France")
veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("vetoed"))
bt = t.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
bt = bt.join(veto, on=["s1", "mid"], how="left").with_columns(pl.col("vetoed").fill_null(0))
fr = bt.filter(((pl.col("p") >= 0.15) & (pl.col("p") < 0.99)) | (pl.col("vetoed") == 1))
fr_veto = fr.filter(pl.col("vetoed") == 1).sample(min(2000, fr.filter(pl.col("vetoed") == 1).height), seed=3)   # sanity-check subset
fr = pl.concat([fr.filter(pl.col("vetoed") == 0), fr_veto])
fr = attach(fr, "test")
fr.write_parquet(os.path.join(out, "llm_fr.parquet"))
print("llm_fr:", fr.height, "pairs (of which vetoed sanity pairs:", int(fr["vetoed"].sum()), ")")
print(fr.with_columns(pl.col("p").cut([0.3, 0.5, 0.725, 0.9], labels=["<.3", ".3-.5", ".5-.725", ".725-.9", ">=.9"]).alias("pb")).group_by("pb").len().sort("pb"))
