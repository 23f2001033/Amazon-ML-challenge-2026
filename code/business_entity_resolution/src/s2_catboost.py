"""Stage-2 CatBoost partner for the LightGBM stage 2, averaged on the uncertain band.

Same band, features and 4 region folds as run_stage2.dev; the LightGBM out-of-fold scores come from
{kind}_s2oof_<s2name>.parquet. Reports the plain held-out score and the score at test-like orphan
density (ER_R2_ORPHAN_KEEP semantics of s2_round2.py), then fits the final CatBoost and scores test.

  python s2_catboost.py <s2name>        # env as in v6x.sh (ER_S2_* of that stack)
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

import run_stage2 as RS
import stage2 as S2
from config import ARTIFACT_DIR, FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05
from model import assign

T0 = time.time()
KEEP = {"US": 0.47, "India": 0.105}
GRID = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.725, 0.75]
CB = dict(iterations=4000, learning_rate=0.08, depth=8, l2_leaf_reg=5, loss_function="Logloss", thread_count=16,
          random_seed=7, od_type="Iter", od_wait=150, verbose=500)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def testlike_drop(kind, pred, qs, uni, owner, s1c):
    """Records to drop for test-like orphan density: owners outside the universe, dropped-S1 orphans kept at KEEP."""
    u = uni.filter(pl.col("universe") == kind)
    mids = pred.select("mid").unique().join(owner, on="mid", how="inner").join(s1c, on="owner", how="left")
    outside = mids.join(u.select(pl.col("entity_id").alias("owner")), on="owner", how="anti").select("mid")
    dropped = mids.join(u.select(pl.col("entity_id").alias("owner")), on="owner", how="semi").filter(~pl.col("owner").is_in(qs.implode()))
    h = (dropped.with_columns(pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("k"),
                              (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h")).filter(pl.col("h") >= pl.col("k")).select("mid"))
    h_out = (outside.join(owner, on="mid").join(s1c, on="owner", how="left").unique("mid")
                    .with_columns(pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("k"),
                                  (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h")).filter(pl.col("h") >= pl.col("k")).select("mid"))
    return {"A": pl.concat([h, h_out]),      # both kinds at the test orphan rate
            "B": pl.concat([h, outside])}    # other-region owners removed (their S1 is present on test)


def main(s2name):
    import run_dev as R
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    D = {k: RS.universe_data(k, "v6", 2 * i) for i, k in enumerate(("val", "wide"))}
    feats = D["val"]["feats"]
    band = pl.concat([D[k]["band"].select(["s1", "mid", "y", "fold"] + feats) for k in D])
    lgb_oof = pl.concat([pl.read_parquet(os.path.join(FEAT_DIR, f"{k}_s2oof_{s2name}.parquet"), columns=["s1", "mid", "p"]) for k in D])
    band = band.join(lgb_oof.rename({"p": "p_lgb"}), on=["s1", "mid"], how="left")
    log(f"band {band.height:,} pairs, {len(feats)} features; lgb oof joined {band['p_lgb'].is_not_null().mean():.4f}")
    oof, iters = [], []
    for k in sorted(band["fold"].unique().to_list()):
        tr, te = band.filter(pl.col("fold") != k), band.filter(pl.col("fold") == k)
        es = tr["s1"].unique().sample(fraction=0.15, seed=3)
        a, b = tr.filter(~pl.col("s1").is_in(es.implode())), tr.filter(pl.col("s1").is_in(es.implode()))
        m = CatBoostClassifier(**CB)
        m.fit(a.select(feats).to_numpy().astype(np.float32), a["y"].to_numpy(),
              eval_set=(b.select(feats).to_numpy().astype(np.float32), b["y"].to_numpy()), use_best_model=True)
        oof.append(te.select("s1", "mid", "p_lgb").with_columns(pl.Series("p_cat", m.predict_proba(te.select(feats).to_numpy().astype(np.float32))[:, 1])))
        iters.append(m.get_best_iteration())
        log(f"fold {k}: best_iter {iters[-1]}")
    oof = pl.concat(oof)
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    oof = oof.with_columns(((pl.col("p_lgb") + pl.col("p_cat")) / 2).alias("p_avg"),
                           (1 / (1 + (-(lg("p_lgb") + lg("p_cat")) / 2).exp())).alias("p_logit"))
    oof.write_parquet(os.path.join(FEAT_DIR, f"band_oof_cat_{s2name}.parquet"))
    cmap = dict(s1c.iter_rows())
    res = {}
    for k in D:
        d = D[k]
        base = pl.read_parquet(os.path.join(FEAT_DIR, f"{k}_s2oof_{s2name}.parquet"), columns=["s1", "mid", "p"])
        drops = testlike_drop(k, base, pl.Series(d["q"]), uni, owner, s1c.rename({"entity_id": "owner"}))
        ctry = np.array([cmap[x] for x in d["q"]])
        for model in ("p_lgb", "p_cat", "p_avg", "p_logit"):
            p_ = base.join(oof.select("s1", "mid", pl.col(model).alias("p2")), on=["s1", "mid"], how="left").with_columns(
                pl.coalesce("p2", "p").alias("p")).drop("p2")
            for view in ("std", "A", "B"):
                pv = p_ if view == "std" else p_.join(drops[view], on="mid", how="anti")
                sc = [macro_f05(assign(pv, t), d["truth"], d["q"])[1] for t in GRID]
                for c in ("US", "India"):
                    msk = ctry == c
                    res.setdefault((model, view, c), []).append((int(msk.sum()), [float(v[msk].mean()) for v in sc]))
        log(f"[{k}] evaluated")
    for model in ("p_lgb", "p_cat", "p_avg", "p_logit"):
        line = []
        for view in ("std", "A", "B"):
            tot, best_thr = 0.0, {}
            n_all = 0
            for c in ("US", "India"):
                v = res[(model, view, c)]
                n = sum(a for a, _ in v)
                w = [sum(a * cv[i] for a, cv in v) / n for i in range(len(GRID))]
                j = int(np.argmax(w)) if view != "std" else GRID.index(0.725)
                best_thr[c] = GRID[j]
                tot += w[j] * n
                n_all += n
            line.append(f"{view} {tot / n_all:.5f} {best_thr}")
        log(f"{model}: " + " | ".join(line))
    final = CatBoostClassifier(**{**{k: v for k, v in CB.items() if k not in ("od_type", "od_wait")}, "iterations": int(np.mean(iters) * 1.1)})
    final.fit(band.select(feats).to_numpy().astype(np.float32), band["y"].to_numpy())
    final.save_model(os.path.join(ARTIFACT_DIR, f"cat_{s2name}.cbm"))
    json.dump({"features": feats, "iters": iters}, open(os.path.join(ARTIFACT_DIR, f"cat_{s2name}.json"), "w"), indent=1)
    log("final CatBoost saved; scoring test band")
    pred = pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet")).select("s1", "mid", "p")
    ids = pl.concat([pred["s1"], pred["mid"]]).unique()
    recs = RS.recs_for("test", ids)
    tband, tfeats = S2.build(pred, recs, RS.stage1_band("test", "v6"), RS.ce_scores("test"), RS.band_ce_scores("test"), RS.visible_s1("test"))
    assert tfeats == feats
    X = tband.select(feats).to_numpy().astype(np.float32)
    lgbm = S2.load_bag(os.path.join(ARTIFACT_DIR, f"{s2name}.txt"))
    t = tband.select("s1", "mid").with_columns(pl.Series("p_lgb", lgbm.predict(X)), pl.Series("p_cat", final.predict_proba(X)[:, 1]))
    t.write_parquet(os.path.join(FEAT_DIR, f"test_band_cat_{s2name}.parquet"))
    ref = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{s2name}.parquet"), columns=["s1", "mid", "p"])
    chk = ref.join(t, on=["s1", "mid"]).select((pl.col("p") - pl.col("p_lgb")).abs().max()).item()
    log(f"test band {t.height:,} pairs; max |p_lgb - saved stage-2 p| = {chk:.2e} (must be ~0)")


if __name__ == "__main__":
    main(sys.argv[1])
