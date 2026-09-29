"""Error budget with oracle ceilings on val + wide (US/India), in test-like conditions (orphan-like records at
test density) with the v6fr3t decision (US 0.60, India 0.50). Each oracle fixes ONE error class with the truth
and recomputes macro F0.5, so the gain column is the most that class can ever give."""
import os, sys
sys.path.insert(0, "/home/ubuntu/er/code/business_entity_resolution/src")
os.environ["ER_UNIVERSES"] = "big"
import numpy as np, polars as pl
import run_dev as R
from evaluate import macro_f05, truth_map_from_gt
from model import assign
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
F, Q, P = "/home/ubuntu/er/data/features", "/home/ubuntu/er/data/parquet", "/home/ubuntu/er/data/prepared"
S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"
THR = {"US": 0.6, "India": 0.5}
KEEP = {"US": 0.47, "India": 0.105}
uni = R.universes()
gt = pl.read_parquet(f"{Q}/train_gt.parquet")
pairs_gt = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("mid")))
s1p = pl.read_parquet(f"{P}/train_s1.parquet", columns=["entity_id", "country", "n_compact"])
recp = pl.concat([pl.read_parquet(f"{P}/train_s{i}.parquet", columns=["entity_id", "n_compact", "a_null"]) for i in (2, 3)])
tot = {}
for kind in ("val", "wide"):
    u = uni.filter(pl.col("universe") == kind)
    qs = u.filter(pl.col("is_query"))["entity_id"]
    q_set = qs.implode()
    d = (pl.read_parquet(f"{F}/{kind}_s2oof_{S2N}.parquet", columns=["s1", "mid", "p", "y"])
           .join(pl.read_parquet(f"{F}/{kind}_pred_v6.parquet", columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"], how="left"))
    # test-like: drop orphan-like records (owner outside universe or dropped S1) down to test density
    own = d.select("mid").unique().join(pairs_gt.rename({"s1": "owner"}), on="mid", how="inner").join(s1p.select(pl.col("entity_id").alias("owner"), "country"), on="owner")
    orph = own.filter(~pl.col("owner").is_in(q_set)).with_columns(
        pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("k"), (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h"))
    d = d.join(orph.filter(pl.col("h") >= pl.col("k")).select("mid"), on="mid", how="anti")
    d = d.join(s1p.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
    truth = truth_map_from_gt(gt, qs.to_list())
    tp = pairs_gt.filter(pl.col("s1").is_in(q_set)).join(s1p.select(pl.col("entity_id").alias("s1"), "country", pl.col("n_compact").alias("s1_name")), on="s1")
    tp = tp.join(recp.rename({"entity_id": "mid", "n_compact": "r_name"}), on="mid", how="left")
    # ambiguity: how many query S1s of this universe/country carry the record's exact compact name
    namecnt = s1p.filter(pl.col("entity_id").is_in(q_set)).group_by("country", "n_compact").len("k_name")
    tp = tp.join(namecnt.rename({"n_compact": "r_name"}), on=["country", "r_name"], how="left").with_columns(pl.col("k_name").fill_null(0))
    best = d.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first")
    thr_e = pl.col("country").replace_strict(THR, return_dtype=pl.Float64)
    acc = best.filter(pl.col("p") >= thr_e).select("s1", "mid", "p")
    tp = (tp.join(d.select("s1", "mid", "p", "p1"), on=["s1", "mid"], how="left")
            .join(acc.select("s1", "mid", pl.lit(1).alias("hit")), on=["s1", "mid"], how="left")
            .join(acc.select("mid", pl.col("s1").alias("got_s1")), on="mid", how="left"))
    fn = tp.filter(pl.col("hit").is_null()).with_columns(
        pl.when(pl.col("p").is_null()).then(pl.lit("1_not_candidate"))
          .when(pl.col("p1") < 0.02).then(pl.lit("2_pruned_p1<0.02"))
          .when(pl.col("got_s1").is_not_null()).then(pl.lit("4_taken_by_other_S1"))
          .otherwise(pl.lit("3_scored_below_thr")).alias("stage"),
        pl.when(pl.col("a_null").fill_null(1) == 1).then(pl.when(pl.col("k_name") >= 2).then(pl.lit("noaddr_ambig(k>=2)")).when(pl.col("k_name") == 1).then(pl.lit("noaddr_unique")).otherwise(pl.lit("noaddr_nameless")))
          .otherwise(pl.lit("with_addr")).alias("kind"))
    fp = acc.join(d.select("s1", "mid", "y", "p1", "country"), on=["s1", "mid"]).filter(pl.col("y") == 0).join(
        pairs_gt.rename({"s1": "owner"}), on="mid", how="left").with_columns(
        pl.when(pl.col("owner").is_null()).then(pl.lit("fp_distractor")).when(pl.col("owner").is_in(q_set)).then(pl.lit("fp_other_query_S1")).otherwise(pl.lit("fp_orphan")).alias("stage"),
        pl.lit("-").alias("kind"))

    def score(p_df):
        return macro_f05(assign(p_df, 0.5, THR), truth, qs)[0]
    base = score(d)
    n = len(qs)
    tot.setdefault("base", []).append((n, base))
    print(f"[{kind}] test-like macro F0.5 with v6fr3t thresholds: {base:.5f}  (S1 {n:,}; FN pairs {fn.height:,}; FP pairs {fp.height:,})", flush=True)
    groups = [("FN", s, k) for s, k in fn.select("stage", "kind").unique().sort("stage", "kind").iter_rows()] + [("FP", s, "-") for s in sorted(fp["stage"].unique().to_list())]
    for typ, s, k in groups:
        if typ == "FN":
            x = fn.filter((pl.col("stage") == s) & (pl.col("kind") == k)).select("s1", "mid")
            # oracle: give the true pair p=1 (add it if absent) and remove the record's competing claims
            dd = pl.concat([d.join(x.select("mid"), on="mid", how="anti").select("s1", "mid", "p", "country"),
                            d.join(x.select("mid"), on="mid", how="semi").join(x, on=["s1", "mid"], how="anti").select("s1", "mid", "country").with_columns(pl.lit(0.0).alias("p")).select("s1", "mid", "p", "country"),
                            x.join(s1p.select(pl.col("entity_id").alias("s1"), "country"), on="s1").with_columns(pl.lit(1.0).alias("p")).select("s1", "mid", "p", "country")])
            cnt = x.height
        else:
            x = fp.filter(pl.col("stage") == s).select("s1", "mid")
            dd = d.join(x.with_columns(pl.lit(1).alias("bad")), on=["s1", "mid"], how="left").with_columns(
                pl.when(pl.col("bad") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("bad")
            cnt = x.height
        g = score(dd) - base
        tot.setdefault((typ, s, k), []).append((n, g, cnt))
        print(f"[{kind}] {typ:<3} {s:<22} {k:<20} pairs {cnt:>7,}  oracle gain {g:+.5f}", flush=True)
nall = sum(a for a, _ in tot["base"])
print("\nWEIGHTED over val+wide  (test-like, v6fr3t thresholds)")
print(f"  base F0.5 {sum(a * b for a, b in tot['base']) / nall:.5f}")
rows = []
for key, v in tot.items():
    if key == "base":
        continue
    rows.append((*key, sum(c for _, _, c in v), sum(a * g for a, g, _ in v) / nall))
for typ, s, k, c, g in sorted(rows, key=lambda r: -r[4]):
    print(f"  {typ:<3} {s:<22} {k:<20} pairs {c:>7,}  ceiling {g:+.5f}")
