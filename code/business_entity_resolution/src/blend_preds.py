"""Blend test stage-2 predictions of two stage-1 bases (blend_eval.py: v6 + v6t mean, val-B 0.99107 vs 0.99096).
  python blend_preds.py <out_s2name> <usin_a> <usin_b> <fr_a> [fr_b]
reads features/test_pred_<name>.parquet (s1, mid, p, country) for each input. Per pair: mean of the two p where both
bases keep it, else the one available. US/India from (usin_a, usin_b); France from (fr_a, fr_b), or fr_a alone.
Writes features/test_pred_<out_s2name>.parquet."""
import os
import sys

import polars as pl

from config import FEAT_DIR


def load(name):
    return pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{name}.parquet"), columns=["s1", "mid", "p", "country"])


def mix(a, b):
    if b is None:
        return load(a)
    x, y = load(a), load(b).rename({"p": "p_b", "country": "c_b"})
    d = x.join(y, on=["s1", "mid"], how="full", coalesce=True)
    return d.select("s1", "mid", pl.mean_horizontal("p", "p_b").alias("p"), pl.coalesce("country", "c_b").alias("country"))


def main(out, usin_a, usin_b, fr_a, fr_b=None):
    u = mix(usin_a, usin_b).filter(pl.col("country") != "France")
    f = mix(fr_a, fr_b).filter(pl.col("country") == "France")
    m = pl.concat([u, f]).sort("s1", "mid")
    m.write_parquet(os.path.join(FEAT_DIR, f"test_pred_{out}.parquet"))
    print(f"wrote test_pred_{out}.parquet: {m.height:,} pairs", m.group_by("country").len().sort("country").rows(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
