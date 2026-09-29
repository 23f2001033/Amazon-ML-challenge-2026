"""Wide-region check: score an existing stage-1 model on labelled train regions that neither
training nor validation used (universe 'unused'), including giant regions (no 1.5x cap).

  ER_TAG=<feature tag> python eval_regions.py <model_tag> [frac=0.25] [thr=0.75]

Uses the same blocking, feature settings (env as for the tag) and 81% query / orphan simulation
as the val universe. Reports macro F0.5 overall, by country and by region, and writes
features/wide_pred_<model_tag>.parquet (s1, mid, hit, p, y).
"""
import os
import sys

import lightgbm as lgb
import polars as pl

import model as M
import run_dev as R
from blocking import generate
from config import ARTIFACT_DIR, FEAT_DIR, PARQUET_DIR
from evaluate import _h, f05, truth_map_from_gt


def pick_regions(frac, base=None):
    uni = R.universes() if base is None else base
    s1 = R.load_prep("train_s1", ["entity_id", "country", "region"]).join(uni, on="entity_id")
    blocks = s1.filter(pl.col("universe") == "unused").group_by("country", "region").len()
    tot = s1.group_by("country").len().rename({"len": "tot"})
    chosen = []
    for country, n_tot in tot.iter_rows():
        b = sorted(blocks.filter(pl.col("country") == country).iter_rows(), key=lambda r: _h(f"wide|{r[1]}", 7))
        filled = 0
        for _, region, n in b:
            if filled >= frac * n_tot:
                break
            chosen.append((country, region))
            filled += n
    ch = pl.DataFrame(chosen, schema=["country", "region"], orient="row")
    q = s1.join(ch, on=["country", "region"]).filter(pl.col("is_query"))
    return q, ch


def main(model_tag, frac="0.25", thr="0.75"):
    frac, thr = float(frac), float(thr)
    if os.environ.get("ER_UNIVERSES") == "big":  # v6+: 'wide' is a universe; candidates built by run_dev block
        uni = R.universes()
        s1r = R.load_prep("train_s1", ["entity_id", "country", "region"])
        q = s1r.join(uni.filter((pl.col("universe") == "wide") & pl.col("is_query")), on="entity_id")
        ch = q.select("country", "region").unique()
    else:
        R.cand_path = lambda u: os.path.join(R.CAND_DIR, f"dev_{u}_wide.parquet")  # one candidate set for all tags
        q, ch = pick_regions(frac)
    R.log(f"wide universe: {q.height:,} queries in {ch.height} regions: "
          + ", ".join(f"{c}:{r}" for c, r in ch.sort("country", "region").iter_rows()))
    if not os.path.exists(R.cand_path("wide")):
        s1 = R.load_prep("train_s1", R.BLOCK_COLS)
        pool = R.load_prep(["train_s2", "train_s3"], R.BLOCK_COLS)
        qq = s1.filter(pl.col("entity_id").is_in(q["entity_id"].implode()))
        cand = generate(qq, pool, log=R.log, workers=int(os.environ.get("ER_WORKERS", "4")))
        cand.write_parquet(R.cand_path("wide"))
        del s1, pool, cand
    m = lgb.Booster(model_file=os.path.join(ARTIFACT_DIR, f"lgbm_{model_tag}.txt"))
    feats = m.feature_name()
    pos = R.gt_pairs().select("s1", "mid").with_columns(pl.lit(1, pl.Int8).alias("y"))

    save = R.band_saver("wide")  # ER_SAVE_BAND=1: full stage-1 rows of the uncertain band (rich stage 2)

    def score(part):
        part = part.with_columns(pl.Series("p", M.predict(m, part, feats)))
        save(part)
        return (part.select("s1", "mid", "hit", "p").join(pos, on=["s1", "mid"], how="left")
                    .with_columns(pl.col("y").fill_null(0)))
    pred = R._batched_features("wide", score)
    pred.write_parquet(os.path.join(FEAT_DIR, f"wide_pred_{model_tag}.parquet"))
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    ids = q["entity_id"].to_list()
    truth = truth_map_from_gt(gt, ids)
    pm = M.assign(pred, thr)
    sc = pl.DataFrame({"entity_id": ids, "f": [f05(pm.get(s, set()), truth.get(s, set())) for s in ids]}).join(
        q.select("entity_id", "country", "region"), on="entity_id")
    R.log(f"[{model_tag}] WIDE macro F0.5 @thr {thr} = {sc['f'].mean():.4f} over {len(ids):,} S1")
    R.log("by country: " + str(sc.group_by("country").agg(pl.col("f").mean().round(4)).sort("country").rows()))
    by_r = sc.group_by("country", "region").agg(pl.col("f").mean().round(4), pl.len()).sort("f")
    pl.Config.set_tbl_rows(100)
    print(by_r)
    for t in (0.5, 0.6, 0.7, 0.8, 0.85):
        pm = M.assign(pred, t)
        print(f"  thr {t}: {sum(f05(pm.get(s, set()), truth.get(s, set())) for s in ids) / len(ids):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
