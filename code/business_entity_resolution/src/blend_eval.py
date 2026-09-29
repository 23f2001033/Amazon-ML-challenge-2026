"""Stage-2 blend of two stage-1 bases (v6 and v6t), US/India held-out val + wide, test-like variant B.
Each base: p = stage-2 w2 out-of-fold on its band, stage-1 p outside it (pairs with stage-1 p >= 0.02).
Blend: mean of the two where both keep the pair; a pair kept by one base only takes that base's p (or `fill`).
Reports macro F0.5 per country at the best threshold of each country, for v6, v6t and the blends.
  python blend_eval.py"""
import os

import numpy as np
import polars as pl

import run_dev as R
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05, truth_map_from_gt
from model import assign

KEEP = {"US": 0.47, "India": 0.105}
THRS = (0.55, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75)


def base(kind, tag):
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet"), columns=["s1", "mid", "p", "y"]).filter(pl.col("p") >= 0.02)
    o = pl.read_parquet(os.path.join(FEAT_DIR, f"s2tune_oof_w2_{tag}.parquet"))
    return d.join(o, on=["s1", "mid"], how="left").select("s1", "mid", "y", pl.coalesce("p2", "p").alias(f"p_{tag}"))


def main():
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    res = {}
    for kind in ("val", "wide"):
        u = uni.filter(pl.col("universe") == kind)
        qs = u.filter(pl.col("is_query"))["entity_id"]
        a, b = base(kind, "v6"), base(kind, "v6t")
        d = a.join(b, on=["s1", "mid"], how="full", coalesce=True).with_columns(pl.coalesce("y", "y_right").alias("y")).drop("y_right")
        m = d.select("mid").unique().join(owner, on="mid", how="left").join(s1c.rename({"entity_id": "owner"}), on="owner", how="left")
        outside = m.filter(pl.col("owner").is_not_null() & ~pl.col("owner").is_in(u["entity_id"].implode())).select("mid")
        dropped = m.filter(pl.col("owner").is_in(u["entity_id"].implode()) & ~pl.col("owner").is_in(qs.implode())).with_columns(
            pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("k"), (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h"))
        drop = pl.concat([outside, dropped.filter(pl.col("h") >= pl.col("k")).select("mid")]).unique()
        d = d.join(drop, on="mid", how="anti")
        truth = truth_map_from_gt(gt, qs.to_list())
        cmap = dict(s1c.filter(pl.col("entity_id").is_in(qs.implode())).iter_rows())
        q = qs.to_list()
        variants = {
            "v6": d.filter(pl.col("p_v6").is_not_null()).select("s1", "mid", pl.col("p_v6").alias("p")),
            "v6t": d.filter(pl.col("p_v6t").is_not_null()).select("s1", "mid", pl.col("p_v6t").alias("p")),
            "blend_other": d.select("s1", "mid", pl.mean_horizontal("p_v6", "p_v6t").alias("p")),
            "blend_zero": d.select("s1", "mid", ((pl.col("p_v6").fill_null(0.0) + pl.col("p_v6t").fill_null(0.0)) / 2).alias("p")),
        }
        for name, v in variants.items():
            for t in THRS:
                sc = macro_f05(assign(v, t), truth, q)[1]
                for c in ("US", "India"):
                    idx = [i for i, s in enumerate(q) if cmap[s] == c]
                    res.setdefault((name, c, t), []).append((float(np.sum(np.asarray(sc)[idx])), len(idx)))
        print(f"{kind} done", flush=True)
    for name in ("v6", "v6t", "blend_other", "blend_zero"):
        tot, n, per = 0.0, 0, {}
        for c in ("US", "India"):
            best = max(((sum(s for s, _ in res[(name, c, t)]) / sum(k for _, k in res[(name, c, t)]), t) for t in THRS))
            per[c] = (best[1], round(best[0], 5))
            k = sum(k for _, k in res[(name, c, THRS[0])])
            tot += best[0] * k; n += k
        print(f"{name:<12} test-like-B F0.5 {tot / n:.5f} {per}", flush=True)
    print("BLEND_DONE", flush=True)


if __name__ == "__main__":
    main()
