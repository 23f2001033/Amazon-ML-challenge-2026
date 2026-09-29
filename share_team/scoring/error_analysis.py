"""Where is the remaining macro-F0.5 loss? (validation, final stage-2 out-of-fold predictions)

  python error_analysis.py <s2oof parquet> <thr> [universe=val]

S1-level categories (loss = sum(1 - F) / N, i.e. points of macro F0.5):
  singleton_fp   truth empty, something predicted          (F = 0)
  all_missed     truth non-empty, nothing predicted         (F = 0)
  fn_only        some true matches missed, no false ones
  fp_only        all true matches found, extra false ones
  fn_and_fp      both
Pair-level causes for missed pairs (FN) and wrong pairs (FP).
"""
import os
import sys

import polars as pl

import run_dev as R
from config import PARQUET_DIR, PREPARED_DIR
from evaluate import f05, truth_map_from_gt
from model import assign


def main(path, thr, kind="val"):
    thr = float(thr)
    uni = R.universes()
    q = uni.filter((pl.col("universe") == kind) & pl.col("is_query"))["entity_id"].to_list()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    truth = truth_map_from_gt(gt, q)
    pred = pl.read_parquet(path).select("s1", "mid", "p")
    pm = assign(pred, thr)
    rows = []
    for s in q:
        t, p = truth.get(s, set()), pm.get(s, set())
        f = f05(p, t)
        tp, fn, fp = len(t & p), len(t - p), len(p - t)
        cat = ("ok" if f == 1.0 else "singleton_fp" if not t else "all_missed" if not p else
               "fn_only" if fp == 0 else "fp_only" if fn == 0 else "fn_and_fp")
        rows.append((s, f, cat, len(t), tp, fn, fp))
    S = pl.DataFrame(rows, schema=["s1", "f", "cat", "n_true", "tp", "fn", "fp"], orient="row")
    N = S.height
    print(f"macro F0.5 = {S['f'].mean():.4f} over {N:,} S1 at thr {thr}; S1 with F<1: {(S['f'] < 1).sum():,} ({(S['f'] < 1).mean():.2%})")
    print(S.group_by("cat").agg(pl.len().alias("S1"), ((1 - pl.col("f")).sum() / N).round(5).alias("loss_pts"),
                                pl.col("f").mean().round(3).alias("mean_F")).sort("loss_pts", descending=True))

    # ---- pair level
    s1r = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"),
                          columns=["entity_id", "country", "region", "n_compact", "a_house", "a_tokens"]).rename(
        {"entity_id": "s1", "n_compact": "s_name", "a_house": "s_house", "a_tokens": "s_addr"})
    s1r = s1r.with_columns(pl.len().over(["country", "region", "s_name"]).alias("amb_r"))
    pool = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"train_s{i}.parquet"),
                                      columns=["entity_id", "n_compact", "a_house", "a_null", "a_tokens"]) for i in (2, 3)]).rename(
        {"entity_id": "mid", "n_compact": "m_name", "a_house": "m_house", "a_null": "m_null", "a_tokens": "m_addr"})
    tp_pairs = pl.DataFrame([(s, m) for s in q for m in truth.get(s, ())], schema=["s1", "mid"], orient="row")
    pr_pairs = pl.DataFrame([(s, m) for s, ms in pm.items() for m in ms], schema=["s1", "mid"], orient="row")
    owner = gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode(
        "matched_entity_ids").rename({"source1_entity_id": "true_s1", "matched_entity_ids": "mid"})
    fn = tp_pairs.join(pr_pairs, on=["s1", "mid"], how="anti").join(pred, on=["s1", "mid"], how="left")
    fn = fn.join(pr_pairs.rename({"s1": "taken_by"}), on="mid", how="left")
    fn = fn.join(s1r, on="s1", how="left").join(pool, on="mid", how="left")
    fn = fn.with_columns(
        pl.when(pl.col("p").is_null()).then(pl.lit("not_in_candidates"))
          .when(pl.col("taken_by").is_not_null()).then(pl.lit("assigned_to_other_S1"))
          .otherwise(pl.lit("below_threshold")).alias("why"))
    print(f"\nMISSED true pairs (FN): {fn.height:,}")
    print(fn.group_by("why").agg(pl.len(), pl.col("m_null").cast(pl.Float64).mean().round(3).alias("null_addr"),
                                 (pl.col("s_name") == pl.col("m_name")).mean().round(3).alias("same_name"),
                                 (pl.col("amb_r") > 1).mean().round(3).alias("ambiguous_S1_name"),
                                 pl.col("p").median().round(3).alias("median_p")).sort("len", descending=True))
    fp = pr_pairs.join(tp_pairs, on=["s1", "mid"], how="anti").join(pred, on=["s1", "mid"], how="left")
    fp = fp.join(owner, on="mid", how="left").join(s1r, on="s1", how="left").join(pool, on="mid", how="left")
    fp = fp.with_columns(
        pl.when(pl.col("true_s1").is_null()).then(pl.lit("distractor_or_decoy"))
          .when(pl.col("true_s1").is_in(pl.Series(q).implode())).then(pl.lit("belongs_to_other_query_S1"))
          .otherwise(pl.lit("orphan_of_dropped_S1")).alias("what"),
        ((pl.col("s_name") == pl.col("m_name")) & (pl.col("s_house") != pl.col("m_house")) & (pl.col("m_house") != "")).alias("decoy_like"))
    print(f"\nWRONG pairs (FP): {fp.height:,}")
    print(fp.group_by("what").agg(pl.len(), pl.col("decoy_like").mean().round(3).alias("same_name_diff_house"),
                                  pl.col("m_null").cast(pl.Float64).mean().round(3).alias("null_addr"),
                                  pl.col("p").median().round(3).alias("median_p")).sort("len", descending=True))
    by_c = S.join(s1r.select("s1", "country"), on="s1").group_by("country", "cat").agg(pl.len(), ((1 - pl.col("f")).sum()).alias("lost"))
    tot = S.join(s1r.select("s1", "country"), on="s1").group_by("country").len().rename({"len": "n_c"})
    print("\nloss by country and category (points of that country's macro F):")
    print(by_c.join(tot, on="country").with_columns((pl.col("lost") / pl.col("n_c")).round(5).alias("pts"))
              .select("country", "cat", "len", "pts").filter(pl.col("cat") != "ok").sort("country", "pts", descending=[False, True]))
    out = os.path.splitext(path)[0]
    fn.write_parquet(out + "_FN.parquet")
    fp.write_parquet(out + "_FP.parquet")
    S.write_parquet(out + "_S1.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:4])
