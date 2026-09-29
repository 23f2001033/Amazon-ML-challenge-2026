"""Cross-encoder pairs for a new stage-1 version (v6): every pair with stage-1 p >= 0.02 in the
val, wide and test universes gets a CE score. Pairs already scored by the first Kaggle run are
reused, only new pairs are exported for inference-only scoring (ce_train_infer.infer).

  python export_ce_v6.py export <tag> <have_dir> <out_dir>
      -> out_dir/{val,wide,test}_new.parquet (s1, mid, text_a, text_b)
  python export_ce_v6.py merge <tag> <have_dir> <new_dir> <out_dir>
      -> out_dir/{val,wide,test}_ce.parquet (s1, mid, p_ce) for run_stage2 (ER_S2_CE=out_dir)

have_dir holds val_ce.parquet / test_ce.parquet from the first run (the CE was trained on the old
10% train universe only, so val and wide scores are out-of-sample).
"""
import os
import sys

import polars as pl

from config import FEAT_DIR
from export_ce_data import attach_text

KINDS = (("val", "train"), ("wide", "train"), ("test", "test"))
P_MIN = 0.02


def wanted(kind, tag):
    return (pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet"), columns=["s1", "mid", "p"])
              .filter(pl.col("p") >= P_MIN).select("s1", "mid").unique())


def have(kind, have_dir):
    f = os.path.join(have_dir, f"{kind}_ce.parquet")
    return pl.read_parquet(f).select("s1", "mid", "p_ce").unique(["s1", "mid"]) if os.path.exists(f) else None


def export(tag, have_dir, out, kinds="val,wide,test"):
    os.makedirs(out, exist_ok=True)
    for kind, split in [k for k in KINDS if k[0] in kinds.split(",")]:
        w = wanted(kind, tag)
        h = have(kind, have_dir)
        new = w if h is None else w.join(h.select("s1", "mid"), on=["s1", "mid"], how="anti")
        new = attach_text(new, split).select("s1", "mid", pl.col("text_a").fill_null(""), pl.col("text_b").fill_null(""))
        new.write_parquet(os.path.join(out, f"{kind}_new.parquet"))
        print(f"{kind}: wanted {w.height:,}, already scored {w.height - new.height:,}, new {new.height:,}")


def merge(tag, have_dir, new_dir, out):
    os.makedirs(out, exist_ok=True)
    for kind, _ in KINDS:
        w = wanted(kind, tag)
        parts = [pl.read_parquet(os.path.join(new_dir, f"{kind}_new_ce.parquet")).select("s1", "mid", "p_ce")]
        h = have(kind, have_dir)
        if h is not None:
            parts.append(h.join(w, on=["s1", "mid"], how="semi"))
        m = pl.concat(parts).unique(["s1", "mid"])
        assert m.height == w.height, f"{kind}: {m.height} scored vs {w.height} wanted"
        m.write_parquet(os.path.join(out, f"{kind}_ce.parquet"))
        print(f"{kind}: {m.height:,} pairs with p_ce")


if __name__ == "__main__":
    {"export": lambda a: export(*a[:4]), "merge": lambda a: merge(*a[:4])}[sys.argv[1]](sys.argv[2:])
