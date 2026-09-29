"""Development loop on TRAIN data: universes -> blocking -> features -> LightGBM -> macro F0.5.

Artifacts are versioned by the ER_TAG environment variable (default "v2a"), so experiments never
overwrite each other: candidates/dev_{u}_{tag}, features/dev_train_{tag}/, artifacts/lgbm_{tag}.txt,
artifacts/result_{tag}.json, features/val_pred_{tag}.parquet.

Usage:  python run_dev.py block      # candidates for val/train universes + recall report
        python run_dev.py features   # train-universe pair features (+ labels), streamed to disk
        python run_dev.py train      # train LightGBM, then score val and tune the decision rule
        python run_dev.py eval       # re-evaluate the saved model of this tag on val
        python run_dev.py all        # block + features + train
"""
import gc
import glob
import json
import os
import sys
import time

import polars as pl

from blocking import generate
from config import ARTIFACT_DIR, CAND_DIR, FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05, make_universes, truth_map_from_gt
from features import REC_COLS, build_chunked, compute_stats, feature_columns
import model as M

T0 = time.time()
TAG = os.environ.get("ER_TAG", "v2a")
BLOCK_COLS = ["entity_id", "country", "region", "n_compact", "a_tokens"]
BATCH_S1 = int(os.environ.get("ER_BATCH_S1", "40000"))  # region batches; v5 uses 100000 = test batch size


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_prep(names, cols=None, ids=None):
    """Lazy, column-projected load of one or more prepared files (optionally only some ids)."""
    names = [names] if isinstance(names, str) else names
    lf = pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f"{n}.parquet")) for n in names])
    if cols:
        lf = lf.select(cols)
    if ids is not None:
        lf = lf.filter(pl.col("entity_id").is_in(ids.implode() if isinstance(ids, pl.Series) else list(ids)))
    return lf.collect(engine="streaming")


def split_stats(split):
    """Statistics for the E2 features of one split ('train' or 'test'); no labels involved."""
    return compute_stats(load_prep(f"{split}_s1", ["country", "n_core"]),
                         load_prep([f"{split}_s2", f"{split}_s3"], ["country", "n_compact"]))


def gt_pairs():
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    return (gt.drop_nulls("matched_entity_ids")
              .with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"}))


def universes():
    path = os.path.join(ARTIFACT_DIR, "universes.parquet")
    if not os.path.exists(path):
        make_universes(load_prep("train_s1", ["entity_id", "country", "region"])).write_parquet(path)
    uni = pl.read_parquet(path)
    if os.environ.get("ER_UNIVERSES") == "big":
        # v6: same val universe; the wide-check regions (eval_regions.py) stay held out as 'wide';
        # every other region joins 'train' (~5x more training S1 than the 10% universe).
        from eval_regions import pick_regions
        _, wide = pick_regions(0.25, base=uni)
        reg = load_prep("train_s1", ["entity_id", "country", "region"]).join(
            wide.with_columns(pl.lit(True).alias("is_wide")), on=["country", "region"], how="left")
        uni = uni.join(reg.select("entity_id", "is_wide"), on="entity_id").with_columns(
            pl.when(pl.col("universe") == "val").then(pl.lit("val"))
              .when(pl.col("is_wide").fill_null(False)).then(pl.lit("wide"))
              .otherwise(pl.lit("train")).alias("universe")).drop("is_wide")
    return uni


def cand_path(u):
    return os.path.join(CAND_DIR, f"dev_{u}_{TAG}.parquet")


def region_batches(q, batch_s1):
    """Group regions (largest first) into batches of at most batch_s1 queries."""
    sizes = q.group_by("region").len().sort("len", descending=True).iter_rows()
    batches, cur, n = [], [], 0
    for reg, size in sizes:
        if cur and n + size > batch_s1:
            batches.append(cur)
            cur, n = [], 0
        cur.append(reg)
        n += size
    if cur:
        batches.append(cur)
    return batches


def step_block():
    uni = universes()
    s1 = load_prep("train_s1", BLOCK_COLS)
    pool = load_prep(["train_s2", "train_s3"], BLOCK_COLS)
    pos = gt_pairs()
    for u in os.environ.get("ER_BLOCK_UNIVERSES", "val,train").split(","):
        q_ids = uni.filter((pl.col("universe") == u) & pl.col("is_query"))["entity_id"]
        q = s1.filter(pl.col("entity_id").is_in(q_ids.implode()))
        log(f"[{u}] queries={q.height:,}")
        cand = generate(q, pool, log=log, workers=int(os.environ.get("ER_WORKERS", "2")))
        cand.write_parquet(cand_path(u))
        p = pos.filter(pl.col("s1").is_in(q_ids.implode()))
        found = p.join(cand.select("s1", "mid"), on=["s1", "mid"], how="semi").height
        per = cand.group_by("s1").len()["len"]
        log(f"[{u}] candidates={cand.height:,} per S1 mean={per.mean():.1f} max={per.max()}; "
            f"pair recall={found / p.height:.4f} ({found:,}/{p.height:,})")
        for bit, nm in ((1, "name"), (2, "addr"), (4, "fallback"), (8, "exact-name key")):
            sub = cand.filter((pl.col("hit") & bit) > 0)
            fb = p.join(sub.select("s1", "mid"), on=["s1", "mid"], how="semi").height
            log(f"     via {nm}: {fb / p.height:.4f}")
        only8 = cand.filter(pl.col("hit") == 8)
        gain = p.join(only8.select("s1", "mid"), on=["s1", "mid"], how="semi").height
        log(f"     recovered ONLY by exact-name key: {gain:,} pairs ({only8.height:,} extra candidates)")
        del cand
        gc.collect()


def _batched_features(u, extra, out_dir=None):
    """Build features for universe u region-batch by region-batch (bounded memory)."""
    stats = split_stats("train")
    s1 = load_prep("train_s1", REC_COLS)
    s1_all = s1
    if os.environ.get("ER_TESTLIKE_AMB") == "1":
        # v5: ambiguity counts (amb_country/amb_region) computed on a per-country S1 sample of the
        # same size as the TEST S1 set, so counts have the same scale in train/val and test.
        n_test = load_prep("test_s1", ["country"]).group_by("country").len()
        frac = {c: min(1.0, n / s1.filter(pl.col("country") == c).height) for c, n in n_test.iter_rows()
                if s1.filter(pl.col("country") == c).height}
        hu = (pl.col("entity_id").hash(seed=11) % 10_000) / 10_000.0
        s1_all = s1.filter(hu < pl.col("country").replace_strict(frac, default=1.0, return_dtype=pl.Float64))
        log(f"test-like ambiguity sample: {s1_all.height:,} of {s1.height:,} S1 ({ {k: round(v, 3) for k, v in frac.items()} })")
    cand_all = pl.read_parquet(cand_path(u))
    pool_all = load_prep(["train_s2", "train_s3"], REC_COLS, cand_all["mid"].unique())
    q = s1.filter(pl.col("entity_id").is_in(cand_all["s1"].unique().implode())).select("entity_id", "country", "region")
    outs, bi = [], 0
    for country in sorted(q["country"].unique().to_list()):
        qc = q.filter(pl.col("country") == country)
        for regs in region_batches(qc, BATCH_S1):
            marker = os.path.join(out_dir, f"b{bi:03d}.done") if out_dir else None
            if marker and os.path.exists(marker):  # resume: batch already written
                log(f"[{u}] features batch {bi}: done earlier, skipped")
                bi += 1
                continue
            if out_dir:  # remove partial parts of an interrupted batch
                for fp in glob.glob(os.path.join(out_dir, f"b{bi:03d}_*.parquet")):
                    os.remove(fp)
            ids = qc.filter(pl.col("region").is_in(regs))["entity_id"]
            cand = cand_all.filter(pl.col("s1").is_in(ids.implode()))
            pool = pool_all.filter(pl.col("entity_id").is_in(cand["mid"].unique().implode()))
            res = build_chunked(cand, s1, pool, s1_all=s1_all, log=lambda *a: None, extra=extra,
                                stats=stats, out_dir=out_dir, part_prefix=f"b{bi:03d}")
            if res is not None:
                outs.append(res)
            if marker:
                open(marker, "w").close()
            log(f"[{u}] features batch {bi}: {country} {regs[:3]}... ({cand.height:,} pairs)")
            bi += 1
            del cand, pool
            gc.collect()
    return pl.concat(outs) if outs else None


def step_features():
    pos = gt_pairs().select("s1", "mid").with_columns(pl.lit(1, pl.Int8).alias("y"))
    label = lambda part: part.join(pos, on=["s1", "mid"], how="left").with_columns(pl.col("y").fill_null(0))
    if os.environ.get("ER_PRESAMPLE") == "1":  # big universes: keep only the rows load_train_sample() would keep
        plain = label
        label = lambda part: plain(part).filter(_keep_expr())
    out = os.path.join(FEAT_DIR, f"dev_train_{TAG}")
    os.makedirs(out, exist_ok=True)
    _batched_features("train", label, out_dir=out)
    log(f"[train] features written to {out}")


NEG_KEEP_RANK, NEG_SAMPLE = 15, 0.25


def _keep_expr():
    u = (pl.col("s1") + pl.col("mid")).hash(seed=3) % 10000 / 10000.0
    return (pl.col("y") == 1) | (pl.col("pre_rank_s1") <= NEG_KEEP_RANK) | (u < NEG_SAMPLE)


def load_train_sample():
    """All positives + negatives ranked <= NEG_KEEP_RANK (by blocking score) + NEG_SAMPLE of the
    rest, weighted 1/NEG_SAMPLE so the model stays calibrated."""
    lf = pl.scan_parquet(os.path.join(FEAT_DIR, f"dev_train_{TAG}", "*.parquet"))
    df = lf.filter(_keep_expr()).collect(engine="streaming")
    w = pl.when((pl.col("y") == 0) & (pl.col("pre_rank_s1") > NEG_KEEP_RANK)).then(1.0 / NEG_SAMPLE).otherwise(1.0)
    df = df.with_columns(w.alias("w"))
    if os.environ.get("ER_S1_TESTW") == "1":
        # test-faithful weights (as stage-2 'w2'): records whose true S1 is a dropped (non-query) S1 of the training
        # universe are rare on test (US ~9%, India ~2% vs 19% simulated) -> weight at test density; records owned
        # outside the universe do not exist as orphans on test (their S1 is present) -> weight 0.05
        keep = {"US": 0.47, "India": 0.105}
        uni = universes()
        tr_ids = uni.filter(pl.col("universe") == "train").select(pl.col("entity_id").alias("owner"), "is_query")
        gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
        own = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                 .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
        c = load_prep("train_s1", ["entity_id", "country"]).rename({"entity_id": "owner"})
        f = (df.select("mid").unique().join(own, on="mid", how="inner").join(tr_ids, on="owner", how="left").join(c, on="owner", how="left")
               .with_columns(pl.when(pl.col("is_query").is_null()).then(0.05)
                               .when(~pl.col("is_query")).then(pl.col("country").replace_strict(keep, default=1.0, return_dtype=pl.Float64))
                               .otherwise(1.0).alias("f")).select("mid", "f"))
        df = df.join(f, on="mid", how="left").with_columns((pl.col("w") * pl.col("f").fill_null(1.0)).alias("w")).drop("f")
        log(f"test-faithful stage-1 weights: {(df['w'] < 1).sum():,} of {df.height:,} rows down-weighted")
    return df


def band_saver(kind):
    """If ER_SAVE_BAND=1, return fn(part_with_p) that writes full feature rows of the stage-2 band
    (0.02 <= p < 0.99) to features/{kind}_band_{TAG}/ ; else a no-op."""
    if os.environ.get("ER_SAVE_BAND") != "1":
        return lambda part: None
    d = os.path.join(FEAT_DIR, f"{kind}_band_{TAG}")
    os.makedirs(d, exist_ok=True)
    counter = [len(glob.glob(os.path.join(d, "*.parquet")))]

    def save(part):
        b = part.filter((pl.col("p") >= 0.02) & (pl.col("p") < 0.99))
        if b.height:
            b.write_parquet(os.path.join(d, f"part_{counter[0]:05d}.parquet"))
            counter[0] += 1
    return save


def score_val(m, feats):
    pos = gt_pairs().select("s1", "mid").with_columns(pl.lit(1, pl.Int8).alias("y"))
    save = band_saver("val")

    def score(part):
        part = part.with_columns(pl.Series("p", M.predict(m, part, feats)))
        save(part)
        return (part.select("s1", "mid", "hit", "p").join(pos, on=["s1", "mid"], how="left")
                    .with_columns(pl.col("y").fill_null(0)))
    log("[val] scoring")
    return _batched_features("val", score)


# Raw-count features whose scale depends on split size (train vs test differ in S1 count and
# pool density); v2a showed they shift test probabilities down. ER_DROP_COUNTS=1 removes them (v3).
COUNT_FEATURES = {"amb_country", "amb_region", "pool_cnt_c", "pool_cnt_s", "rare_tok_idf",
                  "nm_idf_total1", "nm_extra_tok_idf"}


def step_train():
    tr = load_train_sample()
    feats = [c for c in feature_columns(tr) if c != "w"]
    if os.environ.get("ER_DROP_COUNTS") == "1":
        feats = [c for c in feats if c not in COUNT_FEATURES]
    extra_drop = {c for c in os.environ.get("ER_DROP_FEATS", "").split(",") if c}
    feats = [c for c in feats if c not in extra_drop]
    log(f"training on {tr.height:,} pairs ({int(tr['y'].sum()):,} positive), {len(feats)} features")
    m = M.train(tr, feats)
    del tr
    gc.collect()
    m.save_model(os.path.join(ARTIFACT_DIR, f"lgbm_{TAG}.txt"))
    evaluate_model(m, feats)


def step_eval():
    import lightgbm as lgb
    m = lgb.Booster(model_file=os.path.join(ARTIFACT_DIR, f"lgbm_{TAG}.txt"))
    evaluate_model(m, m.feature_name())


def evaluate_model(m, feats):
    uni = universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    va = score_val(m, feats)
    try:
        va.write_parquet(os.path.join(FEAT_DIR, f"val_pred_{TAG}.parquet"))
    except Exception as e:  # disk full must not lose the metrics
        log(f"WARNING could not save val predictions: {e}")
    q_ids = uni.filter((pl.col("universe") == "val") & pl.col("is_query"))["entity_id"].to_list()
    truth = truth_map_from_gt(gt, q_ids)
    oracle, _ = macro_f05({s: truth[s] & set(g) for s, g in
                           va.filter(pl.col("y") == 1).group_by("s1").agg("mid").iter_rows()},
                          truth, q_ids)
    (thr, score), grid = M.tune_threshold(va, truth, q_ids)
    log(f"[{TAG}] VAL macro F0.5 (threshold) = {score:.4f} at thr={thr} | blocking-oracle = {oracle:.4f}")
    log("grid: " + ", ".join(f"{t}:{s:.4f}" for t, s in grid))
    ef_pred = M.expected_f_select(va.select("s1", "mid", "p"))
    ef_score, _ = macro_f05(ef_pred, truth, q_ids)
    log(f"[{TAG}] VAL macro F0.5 (expected-F) = {ef_score:.4f}")
    # ablation: drop candidates found ONLY by the exact-name key (E4 contribution)
    no8 = va.filter(pl.col("hit") != 8)
    s_no8, _ = macro_f05(M.assign(no8, thr), truth, q_ids)
    log(f"[{TAG}] ablation: without exact-name-key-only candidates = {s_no8:.4f}")
    decision = "ef" if ef_score > score else "thr"
    best_pred = ef_pred if decision == "ef" else M.assign(va, thr)
    _, sc = macro_f05(best_pred, truth, q_ids)
    ctry = load_prep("train_s1", ["entity_id", "country"], q_ids)
    bd = pl.DataFrame({"entity_id": q_ids, "f": sc}).join(ctry, on="entity_id")
    by_c = {r["country"]: round(r["f"], 4) for r in bd.group_by("country").agg(pl.col("f").mean()).to_dicts()}
    log(f"[{TAG}] by country: {by_c}")
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
    log("top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp[:30]))
    json.dump({"tag": TAG, "thr": thr, "val_f05_thr": score, "val_f05_ef": ef_score, "decision": decision,
               "oracle": oracle, "by_country": by_c, "ablation_no_exact_key": s_no8, "features": feats},
              open(os.path.join(ARTIFACT_DIR, f"result_{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["all"]
    if "all" in steps:
        steps = ["block", "features", "train"]
    for s in steps:
        {"block": step_block, "features": step_features, "train": step_train, "eval": step_eval}[s]()
