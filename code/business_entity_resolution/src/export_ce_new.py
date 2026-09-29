"""New pairs of a re-trained stage 1 (tag) that the cross-encoders never scored.
  python export_ce_new.py <tag> <out_dir> [kinds=val,wide,test]
  -> <out_dir>/canon_<kind>.parquet (s1, mid, text_a, text_b)  canonical text, for CE v1 and CE v2: pairs with p >= 0.02
                                    lacking a CE v1 score, plus band pairs (0.02 <= p < 0.99) lacking a CE v2 score
  -> <out_dir>/raw_<kind>.parquet   (s1, mid, text_a, text_b)  raw 'name | address', for CE 3-FR and CE 4-FR: band pairs
                                    lacking either score
Scores (Kaggle, score_new_kaggle.py) are merged back with  export_ce_new.py merge <tag> <scores_dir>."""
import os
import sys

import polars as pl

from config import DATA_DIR, FEAT_DIR
from export_ce_data import attach_text
from export_ce_raw import attach as attach_raw

CE = os.environ.get("ER_CE_DIR", os.path.join(DATA_DIR, "ce"))   # cross-encoder score files: <CE>/v6, ce2v6, ce3frv6, ce4frv6 (README 3.2)
SPLIT = {"val": "train", "wide": "train", "test": "test"}
# score name on Kaggle -> (existing score dir, file pattern)
SRC = {"v1": ("v6", "{k}_ce"), "ce2": ("ce2v6", "band_{k}_ce"), "ce3fr": ("ce3frv6", "band_{k}_ce3fr"), "ce4fr": ("ce4frv6", "band_{k}_ce4fr")}


def have(name, kind):
    sub, fn = SRC[name]
    return pl.read_parquet(f"{CE}/{sub}/{fn.format(k=kind)}.parquet", columns=["s1", "mid"]).unique()


def export(tag, out, kinds):
    os.makedirs(out, exist_ok=True)
    for kind in kinds:
        p = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet"), columns=["s1", "mid", "p"])
        keep = p.filter(pl.col("p") >= 0.02).select("s1", "mid")
        band = p.filter((pl.col("p") >= 0.02) & (pl.col("p") < 0.99)).select("s1", "mid")
        canon = pl.concat([keep.join(have("v1", kind), on=["s1", "mid"], how="anti"),
                           band.join(have("ce2", kind), on=["s1", "mid"], how="anti")]).unique().sort("s1", "mid")
        raw = pl.concat([band.join(have(n, kind), on=["s1", "mid"], how="anti") for n in ("ce3fr", "ce4fr")]).unique().sort("s1", "mid")
        attach_text(canon, SPLIT[kind]).select("s1", "mid", "text_a", "text_b").write_parquet(os.path.join(out, f"canon_{kind}.parquet"))
        attach_raw(raw, SPLIT[kind]).select("s1", "mid", "text_a", "text_b").write_parquet(os.path.join(out, f"raw_{kind}.parquet"))
        print(f"{kind}: canonical-text pairs {canon.height:,} | raw-text band pairs {raw.height:,}", flush=True)


def merge(tag, sc):
    """scores dir holds <name>_<kind>.parquet (s1, mid, p_ce) for name in v1, ce2, ce3fr, ce4fr. Existing scores are kept;
    only pairs without one are added. Writes <CE>/<dir>_<tag>/<same file name>."""
    for kind in ("val", "wide", "test"):
        for name, (sub, fn) in SRC.items():
            g = os.path.join(sc, f"{name}_{kind}.parquet")
            if not os.path.exists(g):
                continue
            old = pl.read_parquet(f"{CE}/{sub}/{fn.format(k=kind)}.parquet").select("s1", "mid", pl.col("p_ce").cast(pl.Float32))
            new = pl.read_parquet(g).select("s1", "mid", pl.col("p_ce").cast(pl.Float32)).unique(["s1", "mid"], keep="first", maintain_order=True)
            add = new.join(old, on=["s1", "mid"], how="anti")
            os.makedirs(f"{CE}/{sub}_{tag}", exist_ok=True)
            pl.concat([old, add]).write_parquet(f"{CE}/{sub}_{tag}/{fn.format(k=kind)}.parquet")
            print(f"merged {name} {kind}: kept {old.height:,} + added {add.height:,}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    else:
        export(sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "val,wide,test").split(","))
