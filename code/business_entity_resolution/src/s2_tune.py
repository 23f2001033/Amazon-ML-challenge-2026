"""Stronger stage 2: capacity and test-like training weights, evaluated out-of-fold on val + wide in test-like
conditions (orphan-like records thinned to test density; per-country thresholds tuned).
  python s2_tune.py dev              # compare configs
  python s2_tune.py final <config>   # refit on all folds + score test -> features/test_pred_<s2name>_<config>.parquet
Env as in v6x.sh for the v6fr3 stack."""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import run_stage2 as RS
import stage2 as S2
from config import ARTIFACT_DIR, FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05
from model import assign

T0 = time.time()
S1TAG = os.environ.get("ER_S1TAG", "v6")   # stage-1 tag whose predictions / bands stage 2 re-scores
KEEP = {k: float(v) for k, v in (x.split("=") for x in os.environ.get("ER_KEEP", "US=0.47,India=0.105").split(","))}
BASE = dict(S2.S2_PARAMS)
if os.environ.get("ER_NTHREADS"):
    BASE["num_threads"] = int(os.environ["ER_NTHREADS"])
CONFIGS = {
    "w2": (BASE, True),
    "w3": (BASE, True),
    "wfr": (BASE, True),
    "base": (BASE, False),
    "big": ({**BASE, "num_leaves": 255, "min_data_in_leaf": 40, "feature_fraction": 0.6, "lambda_l2": 10.0, "learning_rate": 0.03}, False),
    "w": (BASE, True),
    "big_w": ({**BASE, "num_leaves": 255, "min_data_in_leaf": 40, "feature_fraction": 0.6, "lambda_l2": 10.0, "learning_rate": 0.03}, True),
}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def orphan_keep(kind, mids, qs, uni, owner, s1c):
    """per record: keep-probability under test density (1 for normal records)."""
    u = uni.filter(pl.col("universe") == kind)
    m = mids.join(owner, on="mid", how="left").join(s1c.rename({"entity_id": "owner"}), on="owner", how="left")
    orph = pl.col("owner").is_not_null() & ~pl.col("owner").is_in(qs.implode())
    outside = pl.col("owner").is_not_null() & ~pl.col("owner").is_in(u["entity_id"].implode())
    out_w = float(os.environ.get("ER_OUT_W", "-1"))
    wk = pl.when(orph).then(pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64)).otherwise(1.0)
    if out_w >= 0:
        wk = pl.when(outside).then(pl.lit(out_w)).otherwise(wk)
    return m.with_columns(wk.alias("wk"), outside.alias("outside"),
                          (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h")).select("mid", "wk", "h", "outside")


def train(tr, feats, params, weighted, rounds=None, valid=None):
    ds = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["y"].to_numpy(), weight=tr["wk"].to_numpy() if weighted else None, feature_name=feats)
    if valid is None:
        return lgb.train(params, ds, rounds)
    dv = lgb.Dataset(valid.select(feats).to_numpy().astype(np.float32), valid["y"].to_numpy(), weight=valid["wk"].to_numpy() if weighted else None, reference=ds)
    return lgb.train(params, ds, 5000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])


def load():
    import run_dev as R
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    D = {k: RS.universe_data(k, S1TAG, 2 * i) for i, k in enumerate(("val", "wide"))}
    feats = D["val"]["feats"]
    for k in D:
        qs = pl.Series(D[k]["q"])
        ok = orphan_keep(k, D[k]["pred"].select("mid").unique(), qs, uni, owner, s1c)
        D[k]["ok"] = ok
        D[k]["band"] = D[k]["band"].join(ok.select("mid", "wk"), on="mid", how="left").with_columns(pl.col("wk").fill_null(1.0))
        D[k]["country"] = s1c.filter(pl.col("entity_id").is_in(qs.implode())).rename({"entity_id": "s1"})
    return D, feats


def evaluate(D, oof):
    """test-like: drop orphan-like records at test density (variant A), tune a threshold per country."""
    grid = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.725]
    acc = {}
    for k in D:
        d = D[k]
        new = d["pred"].join(oof, on=["s1", "mid"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        drop = d["ok"].filter(pl.col("h") >= pl.col("wk")).select("mid")
        if os.environ.get("ER_EVAL_B") == "1":
            drop = pl.concat([drop, d["ok"].filter(pl.col("outside")).select("mid")]).unique()
        new = new.join(drop, on="mid", how="anti").join(d["country"], on="s1")
        ctry = dict(d["country"].iter_rows())
        qs = d["q"]
        for t in grid:
            sc = macro_f05(assign(new, t), d["truth"], qs)[1]
            for c in ("US", "India"):
                msk = np.array([ctry[s] == c for s in qs])
                acc.setdefault((c, t), []).append((int(msk.sum()), float(sc[msk].mean())))
    res = {}
    for c in ("US", "India"):
        best = max(grid, key=lambda t: sum(a * b for a, b in acc[(c, t)]) / sum(a for a, _ in acc[(c, t)]))
        v = acc[(c, best)]
        res[c] = (best, sum(a * b for a, b in v) / sum(a for a, _ in v), sum(a for a, _ in v))
    tot = sum(r[1] * r[2] for r in res.values()) / sum(r[2] for r in res.values())
    return tot, {c: (r[0], round(r[1], 5)) for c, r in res.items()}


def dev():
    D, feats = load()
    band = pl.concat([D[k]["band"].select(["s1", "mid", "y", "fold", "wk"] + feats) for k in D])
    log(f"band {band.height:,} pairs, {len(feats)} features; weighted-down pairs {(band['wk'] < 1).sum():,}")
    for name, (params, weighted) in [(n, CONFIGS[n]) for n in os.environ.get("ER_CFGS", ",".join(CONFIGS)).split(",")]:
        oof, iters = [], []
        for k in sorted(band["fold"].unique().to_list()):
            tr, te = band.filter(pl.col("fold") != k), band.filter(pl.col("fold") == k)
            es = tr["s1"].unique().sample(fraction=0.15, seed=3)
            m = train(tr.filter(~pl.col("s1").is_in(es.implode())), feats, params, weighted, valid=tr.filter(pl.col("s1").is_in(es.implode())))
            oof.append(te.select("s1", "mid").with_columns(pl.Series("p2", m.predict(te.select(feats).to_numpy().astype(np.float32)))))
            iters.append(m.best_iteration)
        if os.environ.get("ER_SAVE_OOF") == "1":
            pl.concat(oof).write_parquet(os.path.join(FEAT_DIR, f"s2tune_oof_{name}.parquet"))
        tot, per = evaluate(D, pl.concat(oof))
        log(f"CONFIG {name:<6} test-like F0.5 {tot:.5f} {per} iters {iters}")
        json.dump({"iters": iters}, open(os.path.join(ARTIFACT_DIR, f"s2tune_{name}{'' if S1TAG == 'v6' else '_' + S1TAG}.json"), "w"))


def final(name):
    s2name = RS.s2name(S1TAG)
    shipped = os.environ.get("ER_S2_LOAD")   # directory with the shipped stage-2 models (../models): score test, no retraining
    if shipped:
        m = lgb.Booster(model_file=os.path.join(shipped, f"{s2name}_{name}.txt"))
        feats = m.feature_name()
        log(f"loaded shipped model {s2name}_{name}.txt ({len(feats)} features); scoring test")
    else:
        D, feats = load()
        params, weighted = CONFIGS[name]
        band = pl.concat([D[k]["band"].select(["s1", "mid", "y", "fold", "wk"] + feats) for k in D])
        iters = json.load(open(os.path.join(ARTIFACT_DIR, f"s2tune_{name}{'' if S1TAG == 'v6' else '_' + S1TAG}.json")))["iters"]
        m = train(band, feats, params, weighted, rounds=int(np.mean(iters) * 1.1))
        m.save_model(os.path.join(ARTIFACT_DIR, f"{s2name}_{name}.txt"))
        json.dump({"features": feats}, open(os.path.join(ARTIFACT_DIR, f"{s2name}_{name}.json"), "w"))
        log("final model saved; scoring test")
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S1TAG}.parquet")).select("s1", "mid", "p")
    ids = pl.concat([pred["s1"], pred["mid"]]).unique()
    recs = RS.recs_for("test", ids)
    tband, tfeats = S2.build(pred, recs, RS.stage1_band("test", S1TAG), RS.ce_scores("test"), RS.band_ce_scores("test"), RS.visible_s1("test"))
    assert tfeats == feats
    new = S2.rescore(pred, tband, m, feats)
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    new.join(s1.rename({"entity_id": "s1"}), on="s1", how="left").write_parquet(os.path.join(FEAT_DIR, f"test_pred_{s2name}_{name}.parquet"))
    log(f"wrote features/test_pred_{s2name}_{name}.parquet")


def rescue_grid(name):
    D, feats = load()
    oof = pl.read_parquet(os.path.join(FEAT_DIR, f"s2tune_oof_{name}.parquet"))
    res = {}
    for k in D:
        d = D[k]
        new = d["pred"].join(oof, on=["s1", "mid"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        drop = pl.concat([d["ok"].filter(pl.col("h") >= pl.col("wk")).select("mid"), d["ok"].filter(pl.col("outside")).select("mid")]).unique()
        new = new.join(drop, on="mid", how="anti").join(d["country"], on="s1")
        for thr in (0.65, 0.7, 0.75):
            for e in (None, 0.5, 0.4, 0.35, 0.3, 0.25):
                m = assign(new, thr)
                if e is not None:
                    taken = {x for xs in m.values() for x in xs}
                    extra = (new.filter((pl.col("p") >= e) & ~pl.col("s1").is_in(list(m.keys())) & ~pl.col("mid").is_in(list(taken)))
                                .sort(["p", "s1", "mid"], descending=[True, False, False]).unique("mid", keep="first", maintain_order=True)
                                .unique("s1", keep="first", maintain_order=True))
                    for s_, x in extra.select("s1", "mid").iter_rows():
                        m[s_] = {x}
                sc = macro_f05(m, d["truth"], d["q"])[1]
                ctry = dict(d["country"].iter_rows())
                for c in ("US", "India"):
                    msk = np.array([ctry[s_] == c for s_ in d["q"]])
                    res.setdefault((c, thr, e), []).append((int(msk.sum()), float(sc[msk].mean())))
    for c in ("US", "India"):
        rows = sorted(((sum(a * b for a, b in v) / sum(a for a, _ in v), thr, e) for (cc, thr, e), v in res.items() if cc == c), reverse=True)
        log(f"{c}: best " + " | ".join(f"thr {t} rescue {e}: {f:.5f}" for f, t, e in rows[:5]))
        base = [f for f, t, e in rows if t == 0.7 and e is None]
        log(f"{c}: reference thr 0.7 no rescue {base[0]:.5f}")


if __name__ == "__main__":
    if sys.argv[1] == "rescue":
        rescue_grid(sys.argv[2]); sys.exit()
    dev() if sys.argv[1] == "dev" else final(sys.argv[2])
