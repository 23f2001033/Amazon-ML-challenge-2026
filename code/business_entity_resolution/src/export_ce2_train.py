"""CE v2 training pairs from the v6 big training universe (never val or wide regions).

  python export_ce2_train.py <tag> <frac_s1> <neg_k> <out.parquet>

For a hash-sampled fraction of the training-universe S1 queries: all positives + the NEG_K hardest
negatives by blocking score (name_cos + addr_cos), plus up to 2 negatives found only by the v6
combined / leak passes (hit bits 16/32). Text = canonical name ; canonical address (as CE v1).
"""
import os
import sys

import polars as pl

from config import CAND_DIR
from export_ce_data import attach_text, gt_pos


def main(tag, frac, neg_k, out):
    frac, neg_k = float(frac), int(neg_k)
    c = pl.scan_parquet(os.path.join(CAND_DIR, f"dev_train_{tag}.parquet")).select("s1", "mid", "name_cos", "addr_cos", "hit")
    c = c.filter(((pl.col("s1").hash(seed=21) % 10_000) / 10_000.0) < frac).collect(engine="streaming")
    c = c.join(gt_pos(), on=["s1", "mid"], how="left").with_columns(pl.col("y").fill_null(0),
                                                                     (pl.col("name_cos") + pl.col("addr_cos")).alias("pre"))
    c = c.with_columns(pl.col("pre").rank("ordinal", descending=True).over("s1").alias("r"))
    new_only = (pl.col("hit") & (1 | 2 | 4)) == 0
    c = c.with_columns(pl.when(new_only & (pl.col("y") == 0)).then(pl.col("pre").rank("ordinal", descending=True).over(["s1", new_only]))
                         .otherwise(None).alias("r_new"))
    keep = (pl.col("y") == 1) | (pl.col("r") <= neg_k) | (pl.col("r_new") <= 2)
    tr = c.filter(keep).select("s1", "mid", "y")
    tr = attach_text(tr, "train").select("s1", "mid", "y", "country", pl.col("text_a").fill_null(""), pl.col("text_b").fill_null(""))
    tr = tr.sample(fraction=1.0, seed=11, shuffle=True)
    tr.write_parquet(out)
    print(f"S1 {tr['s1'].n_unique():,} | pairs {tr.height:,} | positives {int(tr['y'].sum()):,} | by country",
          tr.group_by("country").len().sort("country").rows())


if __name__ == "__main__":
    main(*sys.argv[1:5])
