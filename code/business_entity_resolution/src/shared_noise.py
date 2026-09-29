"""Do copies of one business share NOISE tokens (typos / garbled words that appear in no S1 record)?
Train (labelled): for pairs of pool records sharing a rare noise token (not in any S1 text of the country,
pool df 2..DFMAX), how often do both belong to the same S1?  If high, an unmatched record can be linked
through an already-matched sibling ("sibling expansion")."""
import os
import polars as pl
from config import PARQUET_DIR

pl.Config.set_tbl_rows(-1)
DFMAX = int(os.environ.get("DFMAX", "5"))
TOK = r"[^0-9a-zÀ-ɏऀ-෿]+"


def toks(df, col):
    return df.select("entity_id", "country", pl.col(col).fill_null("").str.to_lowercase().str.split_exact(" ", 0).struct.field("field_0").alias("_") if False else
                     pl.col(col).fill_null("").str.to_lowercase().str.replace_all(TOK, " ").str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= 4)


s1 = pl.read_parquet(os.path.join(PARQUET_DIR, "train_s1.parquet"), columns=["entity_id", "business_name", "business_address", "country"])
s1v = pl.concat([toks(s1, "business_name"), toks(s1, "business_address")]).select("country", "t").unique()
pool = pl.concat([pl.read_parquet(os.path.join(PARQUET_DIR, f"train_s{i}.parquet"), columns=["entity_id", "business_name", "business_address", "country"]) for i in (2, 3)])
for field in ("business_name", "business_address"):
    pt = toks(pool, field).unique(["entity_id", "t"])
    noise = pt.join(s1v, on=["country", "t"], how="anti")
    df = noise.group_by("country", "t").len("df").filter((pl.col("df") >= 2) & (pl.col("df") <= DFMAX))
    nt = noise.join(df, on=["country", "t"])
    gt = (pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")).drop_nulls("matched_entity_ids")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .select(pl.col("matched_entity_ids").alias("entity_id"), pl.col("source1_entity_id").alias("owner")))
    nt = nt.join(gt, on="entity_id", how="left")
    pairs = nt.join(nt, on=["country", "t"], suffix="_b").filter(pl.col("entity_id") < pl.col("entity_id_b"))
    pairs = pairs.unique(["entity_id", "entity_id_b"])
    both = pairs.filter(pl.col("owner").is_not_null() & pl.col("owner_b").is_not_null())
    print(f"[{field}] records with a rare noise token (df 2..{DFMAX}): {nt['entity_id'].n_unique():,} of {pool.height:,}; "
          f"record pairs sharing one: {pairs.height:,}; both owned: {both.height:,}; SAME S1: {(both['owner'] == both['owner_b']).mean():.4f}; "
          f"one owned + one distractor: {pairs.filter(pl.col('owner').is_null() ^ pl.col('owner_b').is_null()).height:,}", flush=True)
    print("   by df:", both.join(df, on=["country", "t"]).group_by("df").agg(pl.len(), (pl.col("owner") == pl.col("owner_b")).mean().round(4)).sort("df").rows(), flush=True)
    ex = both.filter(pl.col("owner") == pl.col("owner_b")).sample(8, seed=1).select("t")["t"].to_list()
    print("   example shared noise tokens:", ex, flush=True)
