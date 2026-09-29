"""Group-specific threshold for ambiguous empty-address records, tuned on val + wide in test-like conditions
(orphan-like records at test density), everything else at the v6fr3t decision (US 0.60, India 0.50).
Group G: record has no address, its compact name equals the S1's compact name, and >= K S1s of the country carry
that name (K counted on the split's S1 table, scaled to test size)."""
import os
import sys
import numpy as np
import polars as pl
import run_dev as R
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05, truth_map_from_gt

S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"
THR = {"US": 0.6, "India": 0.5}
KEEP = {"US": 0.47, "India": 0.105}
SCALE = {"US": 663106 / 1323632, "India": 809986 / 806150}   # test S1 / train S1 per country


def decide(d, g_thr, e_thr=None):
    """d: s1, mid, p, country, g (bool). Partition by p, then per-pair threshold.
    e_thr: S1s left empty get their best still-unassigned candidate if p >= e_thr."""
    best = d.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first")
    t = pl.when(pl.col("g")).then(pl.lit(g_thr)).otherwise(pl.col("country").replace_strict(THR, return_dtype=pl.Float64))
    sel = best.filter(pl.col("p") >= t)
    out = {}
    for s, m in sel.select("s1", "mid").iter_rows():
        out.setdefault(s, set()).add(m)
    if e_thr is not None:
        taken = set(sel["mid"].to_list())
        extra = (d.filter((pl.col("p") >= e_thr) & ~pl.col("s1").is_in(list(out.keys())) & ~pl.col("mid").is_in(list(taken)))
                  .sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first").unique("s1", keep="first"))
        for s, m in extra.select("s1", "mid").iter_rows():
            out[s] = {m}
    return out


def main(kmin):
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country", "n_compact"])
    kn = s1.group_by("country", "n_compact").len("k").with_columns((pl.col("k") * pl.col("country").replace_strict(SCALE, return_dtype=pl.Float64)).alias("k_test"))
    rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"train_s{i}.parquet"), columns=["entity_id", "n_compact", "a_null"]) for i in (2, 3)]).rename(
        {"entity_id": "mid", "n_compact": "r_name"})
    grid = ([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.01] if os.environ.get("ER_EMPTY") else [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01])
    tot = {}
    for kind in ("val", "wide"):
        u = uni.filter(pl.col("universe") == kind)
        qs = u.filter(pl.col("is_query"))["entity_id"]
        d = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_s2oof_{S2N}.parquet"), columns=["s1", "mid", "p"])
        own = d.select("mid").unique().join(owner, on="mid").join(s1.select(pl.col("entity_id").alias("owner"), "country"), on="owner")
        orph = own.filter(~pl.col("owner").is_in(qs.implode())).with_columns(
            pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("kp"), (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h"))
        d = d.join(orph.filter(pl.col("h") >= pl.col("kp")).select("mid"), on="mid", how="anti")
        d = d.join(s1.select(pl.col("entity_id").alias("s1"), "country", pl.col("n_compact").alias("s_name")), on="s1").join(rec, on="mid", how="left")
        d = d.join(kn.select("country", pl.col("n_compact").alias("s_name"), "k_test"), on=["country", "s_name"], how="left")
        d = d.with_columns(((pl.col("a_null").fill_null(1) == 1) & (pl.col("r_name") == pl.col("s_name")) & (pl.col("k_test").fill_null(0) >= kmin)).alias("g"))
        truth = truth_map_from_gt(gt, qs.to_list())
        if os.environ.get("ER_EMPTY"):
            d = d.with_columns(pl.lit(False).alias("g"))
            curve = [macro_f05(decide(d, 0.5, t if t < 1 else None), truth, qs)[0] for t in grid]
        else:
            curve = [macro_f05(decide(d, t), truth, qs)[0] for t in grid]
        tot[kind] = (len(qs), curve)
        print(f"[{kind}] G pairs {int(d['g'].sum()):,}; " + " ".join(f"{t}:{c:.5f}" for t, c in zip(grid, curve)), flush=True)
    n = sum(a for a, _ in tot.values())
    w = [sum(a * c[i] for a, c in tot.values()) / n for i in range(len(grid))]
    base = w[grid.index(0.5)]   # G at the country thresholds ~ equals baseline only for India; report both
    j = int(np.argmax(w))
    print(f"K>={kmin}: weighted " + " ".join(f"{t}:{x:.5f}" for t, x in zip(grid, w)) + f" | best {grid[j]} ({w[j] - w[0]:+.5f} vs G at 0.5)")


if __name__ == "__main__":
    for k in sys.argv[1:]:
        main(float(k))
