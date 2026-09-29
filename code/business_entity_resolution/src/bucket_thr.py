"""Per-country x per-bucket thresholds on the final US/India probabilities (v6 + v6t stage-2 blend), test-like variant B.
Bucket = number of pruned candidates of the S1 (pairs with p >= 0.02). An S1's F0.5 depends only on the records
assigned to it, so each bucket's threshold can be chosen independently. Cross-fit: tune on val, score on wide, and
the reverse; compare with one threshold per country tuned the same way.
  python bucket_thr.py"""
import os

import numpy as np
import polars as pl

import blend_eval as BE
import run_dev as R
from config import PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05, truth_map_from_gt
from model import assign

GRID = [0.45, 0.5, 0.55, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85]
EDGES = [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 10, "7-10"), (11, 10 ** 6, "11+")]


def per_s1_scores():
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    out = {}
    for kind in ("val", "wide"):
        u = uni.filter(pl.col("universe") == kind)
        qs = u.filter(pl.col("is_query"))["entity_id"]
        a, b = BE.base(kind, "v6"), BE.base(kind, "v6t")
        d = a.join(b, on=["s1", "mid"], how="full", coalesce=True).select("s1", "mid", pl.mean_horizontal("p_v6", "p_v6t").alias("p"))
        m = d.select("mid").unique().join(owner, on="mid", how="left").join(s1c.rename({"entity_id": "owner"}), on="owner", how="left")
        outside = m.filter(pl.col("owner").is_not_null() & ~pl.col("owner").is_in(u["entity_id"].implode())).select("mid")
        dropped = m.filter(pl.col("owner").is_in(u["entity_id"].implode()) & ~pl.col("owner").is_in(qs.implode())).with_columns(
            pl.col("country").replace_strict(BE.KEEP, default=1.0, return_dtype=pl.Float64).alias("k"), (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h"))
        drop = pl.concat([outside, dropped.filter(pl.col("h") >= pl.col("k")).select("mid")]).unique()
        d = d.join(drop, on="mid", how="anti")
        q = qs.to_list()
        ncand = dict(d.group_by("s1").len().iter_rows())
        truth = truth_map_from_gt(gt, q)
        cmap = dict(s1c.filter(pl.col("entity_id").is_in(qs.implode())).iter_rows())
        F = np.vstack([np.asarray(macro_f05(assign(d, t), truth, q)[1]) for t in GRID])   # thresholds x S1
        nc = np.array([ncand.get(s, 0) for s in q])
        bucket = np.full(len(q), "0", dtype=object)
        for lo, hi, name in EDGES:
            bucket[(nc >= lo) & (nc <= hi)] = name
        out[kind] = (F, np.array([cmap[s] for s in q]), bucket)
        print(f"{kind}: {len(q):,} S1 scored on {len(GRID)} thresholds", flush=True)
    return out


def main():
    S = per_s1_scores()
    tot = {"global": 0.0, "bucket": 0.0}
    n_all = 0
    for tune, test in (("val", "wide"), ("wide", "val")):
        Ft, ct, bt = S[tune]
        Fe, ce, be = S[test]
        for c in ("US", "India"):
            mt, me = ct == c, ce == c
            g = int(np.argmax(Ft[:, mt].mean(axis=1)))
            glob = Fe[g, me].sum()
            buck, rows = 0.0, []
            for name in ["0"] + [e[2] for e in EDGES]:
                st, se = mt & (bt == name), me & (be == name)
                if se.sum() == 0:
                    continue
                k = int(np.argmax(Ft[:, st].sum(axis=1))) if st.sum() else g
                buck += Fe[k, se].sum()
                rows.append(f"{name}:{GRID[k]}({int(se.sum())})")
            n = int(me.sum())
            tot["global"] += glob; tot["bucket"] += buck; n_all += n
            print(f"tune {tune} -> score {test} {c:<5}: global thr {GRID[g]} F {glob / n:.5f} | bucketed F {buck / n:.5f} "
                  f"(delta {(buck - glob) / n:+.5f}) | {' '.join(rows)}", flush=True)
    print(f"CROSS-FIT total: global {tot['global'] / n_all:.5f} | bucketed {tot['bucket'] / n_all:.5f} | delta {(tot['bucket'] - tot['global']) / n_all:+.5f}", flush=True)
    print("BUCKET_DONE", flush=True)


if __name__ == "__main__":
    main()
