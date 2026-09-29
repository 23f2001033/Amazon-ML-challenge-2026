"""Threshold sweep with the count model. validate: US/India wide (labelled) true recall / false-per-S1 vs fitted
(r, extra) at each threshold. test: France fitted (r, extra) per threshold, veto kept."""
import os, sys
import numpy as np
import polars as pl
import count_fit as CF
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from model import assign

gen = CF.generator_pmf()
GRID = [0.725, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"


def fitc(m, ids):
    rows = [(s, sum(x.startswith("S2") for x in m.get(s, ())), sum(x.startswith("S3") for x in m.get(s, ()))) for s in ids]
    f = pl.DataFrame(rows, schema=["s", "S2", "S3"], orient="row")
    e = {k: CF.fit(f[k].to_numpy(), gen[k]) for k in ("S2", "S3")}
    return (e["S2"][0] + e["S3"][0]) / 2, e["S2"][1] + e["S3"][1]


if sys.argv[1] == "validate":
    import run_dev as R
    from evaluate import truth_map_from_gt, macro_f05
    uni = R.universes()
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{S2N}.parquet"), columns=["s1", "mid", "p"]).join(s1c.rename({"entity_id": "s1"}), on="s1")
    qs = uni.filter((pl.col("universe") == "wide") & pl.col("is_query")).join(s1c, on="entity_id")
    truth = truth_map_from_gt(pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")), qs["entity_id"].to_list())
    for c in ("India",):
        ids = qs.filter(pl.col("country") == c)["entity_id"].to_list()
        dc = d.filter(pl.col("country") == c)
        for t in GRID:
            m = assign(dc, t)
            tp = sum(len(m.get(s, set()) & truth.get(s, set())) for s in ids); tt = sum(len(truth.get(s, ())) for s in ids)
            fp = sum(len(m.get(s, set()) - truth.get(s, set())) for s in ids)
            r, x = fitc(m, ids)
            print(f"[{c} thr {t}] TRUE recall {tp / tt:.4f} false/S1 {fp / len(ids):.4f} F {macro_f05(m, truth, ids)[0]:.5f} | FIT r {r:.4f} extra {x:.4f}", flush=True)
else:
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S2N}.parquet"), columns=["s1", "mid", "p", "country"])
    veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
    d = d.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
    ids = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"].to_list()
    dc = d.filter(pl.col("country") == "France")
    for t in GRID:
        m = assign(dc, t)
        n = sum(len(v) for v in m.values()) / len(ids)
        r, x = fitc(m, ids)
        print(f"[France thr {t}] matches/S1 {n:.4f} empty {1 - len([s for s in ids if m.get(s)]) / len(ids):.4f} | FIT r {r:.4f} extra {x:.4f}", flush=True)
