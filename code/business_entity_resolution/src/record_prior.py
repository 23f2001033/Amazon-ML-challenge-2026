"""Record prior: how likely is an S2/S3 record to be a TRUE record of some S1 (vs a generated distractor),
judged from the record alone (no S1 comparison). The generator builds distractors differently: they
almost always carry a full address with a house number and longer, composite names; records without a
house number or address are ~97% true (train).

Leak-free: the model is trained only on records from the stage-1 training regions (universe 'train'
of ER_UNIVERSES=big) and applied to every train and test record; val / wide / test records are never
trained on. Country-agnostic features only (France is unseen in training).

  python record_prior.py <out.parquet>   -> mid, rp  (train + test records)
"""
import os
import sys

import numpy as np
import polars as pl
import lightgbm as lgb

from config import PARQUET_DIR, PREPARED_DIR

FEATS = ["oov_share", "name_tokens", "name_len", "addr_len", "addr_commas", "has_legal", "has_house", "name_upper", "addr_upper",
         "name_digit", "name_punct", "native", "alias", "web", "addr_null", "is_s2"]


def record_frame(split):
    voc = set(" ".join(pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=["n_core"])["n_core"].fill_null("").to_list()).split())
    parts = []
    for src in ("s2", "s3"):
        p = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_{src}.parquet"),
                            columns=["entity_id", "country", "region", "n_core", "n_legal", "a_house", "a_null", "n_native", "n_alias", "n_web"])
        r = pl.read_parquet(os.path.join(PARQUET_DIR, f"{split}_{src}.parquet"), columns=["entity_id", "business_name", "business_address"])
        parts.append(p.join(r, on="entity_id").with_columns(pl.lit(1 if src == "s2" else 0, pl.Int8).alias("is_s2")))
    d = pl.concat(parts).with_columns(pl.col(c).fill_null("") for c in ("n_core", "n_legal", "a_house", "business_name", "business_address"))
    toks = d["n_core"].to_list()
    return d.with_columns(
        pl.Series("oov_share", [float(np.mean([t not in voc for t in x.split()])) if x else 1.0 for x in toks], dtype=pl.Float32),
        pl.col("n_core").str.split(" ").list.len().alias("name_tokens"), pl.col("business_name").str.len_chars().alias("name_len"),
        pl.col("business_address").str.len_chars().alias("addr_len"), pl.col("business_address").str.count_matches(",").alias("addr_commas"),
        (pl.col("n_legal") != "").cast(pl.Int8).alias("has_legal"), (pl.col("a_house") != "").cast(pl.Int8).alias("has_house"),
        (pl.col("business_name") == pl.col("business_name").str.to_uppercase()).cast(pl.Int8).alias("name_upper"),
        (pl.col("business_address") == pl.col("business_address").str.to_uppercase()).cast(pl.Int8).alias("addr_upper"),
        pl.col("business_name").str.contains(r"[0-9]").cast(pl.Int8).alias("name_digit"),
        pl.col("business_name").str.count_matches(r"[&+\-\.\(\)\[\]/]").alias("name_punct"),
        (pl.col("n_native").fill_null("") != "").cast(pl.Int8).alias("native"), pl.col("n_alias").cast(pl.Int8).alias("alias"),
        (pl.col("n_web").fill_null("") != "").cast(pl.Int8).alias("web"), pl.col("a_null").cast(pl.Int8).alias("addr_null"))


def main(out):
    import run_dev as R
    os.environ["ER_UNIVERSES"] = "big"
    uni = R.universes()
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country", "region"])
    train_regions = s1.join(uni.filter(pl.col("universe") == "train"), on="entity_id").select("country", "region").unique()
    gt = (pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")).drop_nulls("matched_entity_ids")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .select(pl.col("matched_entity_ids").alias("entity_id")).with_columns(pl.lit(1, pl.Int8).alias("true")))
    tr = record_frame("train").join(gt, on="entity_id", how="left").with_columns(pl.col("true").fill_null(0))
    fit = tr.join(train_regions, on=["country", "region"], how="semi")
    print(f"training on {fit.height:,} records from training regions (true share {fit['true'].mean():.3f})")
    m = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 200, "seed": 3},
                  lgb.Dataset(fit.select(FEATS).to_numpy().astype(np.float32), fit["true"].to_numpy(), feature_name=FEATS), 400)
    held = tr.join(train_regions, on=["country", "region"], how="anti")
    from sklearn.metrics import roc_auc_score
    print(f"AUC on records outside the training regions: {roc_auc_score(held['true'].to_numpy(), m.predict(held.select(FEATS).to_numpy().astype(np.float32))):.4f}")
    te = record_frame("test")
    outs = [f.select(pl.col("entity_id").alias("mid"), pl.Series("rp", m.predict(f.select(FEATS).to_numpy().astype(np.float32)), dtype=pl.Float32))
            for f in (tr, te)]
    pl.concat(outs).write_parquet(out)
    print("test mean rp by country:", te.with_columns(pl.Series("rp", outs[1]["rp"])).group_by("country").agg(pl.col("rp").mean().round(3)).sort("country").rows())


if __name__ == "__main__":
    main(sys.argv[1])
