"""Export pair text for the cross-encoder (CPU machine).

  python export_ce_data.py <tag_train_cands> <tag_val_pred> <tag_test_pred> <out_dir>

train.parquet : train-universe candidates -> all positives + the top-NEG_K negatives per S1 by
                blocking score (hard negatives from our own blocker), with labels
val.parquet   : val-universe pairs with stage-1 p >= 0.005 (labels) -> CE evaluation + stage-2 fit
test.parquet  : test pairs with stage-1 p >= 0.02 (as saved by run_test.py) -> CE inference
Text = canonical name and canonical address of both records (Indic translated, accents folded).
"""
import os
import sys

import polars as pl

from config import CAND_DIR, FEAT_DIR, PARQUET_DIR, PREPARED_DIR

NEG_K = 12
COLS = ["entity_id", "n_canon", "a_tokens", "country"]


def gt_pos():
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    return (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"})
              .with_columns(pl.lit(1, pl.Int8).alias("y")))


def attach_text(pairs, split):
    ids = pl.concat([pairs["s1"], pairs["mid"]]).unique()
    recs = (pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f"{split}_s{i}.parquet")) for i in (1, 2, 3)])
              .select(COLS).filter(pl.col("entity_id").is_in(ids.implode())).collect(engine="streaming"))
    a = recs.select(pl.col("entity_id").alias("s1"), (pl.col("n_canon") + " ; " + pl.col("a_tokens")).alias("text_a"), "country")
    b = recs.select(pl.col("entity_id").alias("mid"), (pl.col("n_canon") + " ; " + pl.col("a_tokens")).alias("text_b"))
    return pairs.join(a, on="s1", how="left").join(b, on="mid", how="left")


def main(tag_cands, tag_val, tag_test, out):
    os.makedirs(out, exist_ok=True)
    pos = gt_pos()
    c = pl.read_parquet(os.path.join(CAND_DIR, f"dev_train_{tag_cands}.parquet"))
    c = c.with_columns((pl.col("name_cos") + pl.col("addr_cos")).alias("pre"))
    c = c.join(pos, on=["s1", "mid"], how="left").with_columns(pl.col("y").fill_null(0))
    c = c.with_columns(pl.col("pre").rank("ordinal", descending=True).over("s1").alias("r"))
    tr = c.filter((pl.col("y") == 1) | (pl.col("r") <= NEG_K)).select("s1", "mid", "y")
    tr = attach_text(tr, "train").sample(fraction=1.0, seed=7, shuffle=True)
    tr.write_parquet(os.path.join(out, "train.parquet"))
    print("train pairs", tr.height, "positives", int(tr["y"].sum()))
    va = pl.read_parquet(os.path.join(FEAT_DIR, f"val_pred_{tag_val}.parquet")).filter(pl.col("p") >= 0.005).select("s1", "mid", "p", "y")
    va = attach_text(va, "train")
    va.write_parquet(os.path.join(out, "val.parquet"))
    print("val pairs", va.height, "positives", int(va["y"].sum()))
    te = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{tag_test}.parquet")).select("s1", "mid", "p")
    te = attach_text(te, "test")
    te.write_parquet(os.path.join(out, "test.parquet"))
    print("test pairs", te.height)


if __name__ == "__main__":
    main(*sys.argv[1:5])
