"""Stage-2 driver.

  python run_stage2.py dev  <tag>        # 2-fold region CV on the validation universe -> report
                                         # vs stage 1, then refit on all val -> artifacts/s2_<tag>.txt
  python run_stage2.py test <tag> <cfg.json> <out_dir>
                                         # re-score test band, decide, write matching_results.tsv
Stage-1 predictions: features/val_pred_<tag>.parquet (val) and features/test_pred_<tag>.parquet (test).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import polars as pl

import stage2 as S2
from config import ARTIFACT_DIR, FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from evaluate import macro_f05, truth_map_from_gt
from model import assign

T0 = time.time()
RICH = os.environ.get("ER_S2_RICH") == "1"   # use saved full stage-1 features of band pairs
CE_DIR = os.environ.get("ER_S2_CE")           # dir with {val,test}_ce.parquet (s1, mid, p_ce) from ce_train_infer.py


# band-only cross-encoders: ER_S2_CE_BAND="ce2=/path/band_{kind}_ce.parquet;ce3=/path/band_{kind}_ce3.parquet"
BAND_CES = [tuple(x.split("=", 1)) for x in os.environ.get("ER_S2_CE_BAND", "").split(";") if x]


def s2name(tag):
    return (f"s2_{tag}{'_rich' if RICH else ''}{'_ce' if CE_DIR else ''}{''.join('_' + n for n, _ in BAND_CES)}"
            f"{'_addr' if os.environ.get('ER_S2_ADDR') == '1' else ''}{'_rp' if os.environ.get('ER_S2_RP') else ''}")


def ce_scores(kind):
    return pl.read_parquet(os.path.join(CE_DIR, f"{kind}_ce.parquet")) if CE_DIR else None


def band_ce_scores(kind):
    return [(n, pl.read_parquet(t.format(kind=kind))) for n, t in BAND_CES]


def visible_s1(split):
    """ER_S2_ADDR=1: S1 records a test-like pipeline can see. Test: every test S1. Train: every train S1
    except the deliberately dropped (non-query) S1 of the held-out universes, as in the test simulation."""
    if os.environ.get("ER_S2_ADDR") != "1":
        return None
    cols = ["entity_id", "country", "n_core", "a_tokens", "a_house"]
    s = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=cols)
    if split == "train":
        import run_dev as R
        u = R.universes()
        drop = u.filter(pl.col("universe").is_in(["val", "wide"]) & ~pl.col("is_query")).select("entity_id")
        s = s.join(drop, on="entity_id", how="anti")
    return s


def stage1_band(kind, tag):
    if not RICH:
        return None
    d = os.path.join(FEAT_DIR, f"{kind}_band_{tag}")
    return pl.scan_parquet(os.path.join(d, "*.parquet")).collect(engine="streaming")


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def recs_for(split, ids):
    lf = pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f"{split}_s{i}.parquet")) for i in (1, 2, 3)])
    return lf.select(S2.REC).filter(pl.col("entity_id").is_in(ids.implode())).collect(engine="streaming")


def tune(pred, truth, q):  # kept for older callers
    best = (0.0, 0.0)
    for t in np.round(np.arange(0.5, 0.96, 0.025), 3):
        s, _ = macro_f05(assign(pred, float(t)), truth, q)
        best = max(best, (s, float(t)))
    return best


def universe_data(kind, tag, fold_offset):
    """Stage-1 predictions, truth and stage-2 band of one held-out universe ('val' or 'wide'),
    with 2 region folds numbered fold_offset, fold_offset + 1 (balanced greedily by size)."""
    import run_dev as R  # universes() honours ER_UNIVERSES=big (the 'wide' universe)
    uni = R.universes()
    q = uni.filter((pl.col("universe") == kind) & pl.col("is_query"))["entity_id"]
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    truth = truth_map_from_gt(gt, q.to_list())
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet")).select("s1", "mid", "p", "y")
    ids = pl.concat([pred["s1"], pred.filter(pl.col("p") >= S2.BAND[0])["mid"]]).unique()
    recs = recs_for("train", ids)
    band, feats = S2.build(pred, recs, stage1_band(kind, tag), ce_scores(kind), band_ce_scores(kind), visible_s1("train"))
    band = band.join(pred.select("s1", "mid", "y"), on=["s1", "mid"], how="left") if "y" not in band.columns else band
    log(f"[{kind}] pairs {pred.height:,}; S1 {q.len():,}; band {band.height:,} ({band.height / q.len():.2f}/S1), "
        f"positives {int(band['y'].sum()):,}; {len(feats)} features")
    reg = recs.filter(pl.col("entity_id").is_in(q.implode())).select(pl.col("entity_id").alias("s1"), "region")
    sizes = sorted(reg.group_by("region").len().iter_rows(), key=lambda r: -r[1])
    fold_of, load = {}, [0, 0]
    for r, n in sizes:
        k = int(load[1] < load[0])
        fold_of[r] = fold_offset + k
        load[k] += n
    band = band.join(reg, on="s1", how="left").with_columns(
        pl.col("region").replace_strict(fold_of, default=fold_offset, return_dtype=pl.Int8).alias("fold"))
    return {"q": q.to_list(), "truth": truth, "pred": pred, "band": band, "feats": feats}


def dev(tag):
    """Stage 2 trained on the held-out universes listed in ER_S2_TRAIN (default 'val'; v6: 'val,wide').
    Out-of-fold scores per universe; one threshold maximising the query-weighted score over all."""
    kinds = os.environ.get("ER_S2_TRAIN", "val,wide" if os.environ.get("ER_UNIVERSES") == "big" else "val").split(",")
    D = {k: universe_data(k, tag, 2 * i) for i, k in enumerate(kinds)}
    feats = D[kinds[0]]["feats"]
    assert all(D[k]["feats"] == feats for k in kinds), "feature lists differ between universes"
    band = pl.concat([D[k]["band"].select(["s1", "mid", "y", "fold"] + feats) for k in kinds])
    oof, iters = [], []
    for k in sorted(band["fold"].unique().to_list()):
        tr, te = band.filter(pl.col("fold") != k), band.filter(pl.col("fold") == k)
        es = tr["s1"].unique().sample(fraction=0.15, seed=3)
        m = S2.fit(tr.filter(~pl.col("s1").is_in(es.implode())), feats, valid=tr.filter(pl.col("s1").is_in(es.implode())))
        oof.append(te.select("s1", "mid").with_columns(pl.Series("p2", m.predict(te.select(feats).to_numpy().astype(np.float32)))))
        iters.append(m.best_iteration)
        log(f"fold {k}: trained on {tr.height:,}, predicted {te.height:,}, best_iter={m.best_iteration}")
    oof = pl.concat(oof)
    grid = [float(t) for t in np.round(np.arange(0.5, 0.96, 0.025), 3)]
    n_all = sum(len(D[k]["q"]) for k in kinds)
    curves = {}
    for k in kinds:
        d = D[k]
        d["new"] = d["pred"].join(oof, on=["s1", "mid"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        d["new"].write_parquet(os.path.join(FEAT_DIR, f"{k}_s2oof_{s2name(tag)}.parquet"))  # for error analysis
        for name, p_ in (("stage1", d["pred"]), ("stage2", d["new"])):
            curves[(k, name)] = [macro_f05(assign(p_, t), d["truth"], d["q"])[0] for t in grid]
    res = {}
    for name in ("stage1", "stage2"):
        w = [sum(curves[(k, name)][i] * len(D[k]["q"]) for k in kinds) / n_all for i in range(len(grid))]
        i = int(np.argmax(w))
        res[name] = {"thr": grid[i], "weighted": w[i], **{k: curves[(k, name)][i] for k in kinds},
                     **{f"{k}_best": max(curves[(k, name)]) for k in kinds}}
        log(f"{name.upper()} common thr {grid[i]}: weighted {w[i]:.4f} | " +
            " | ".join(f"{k} {curves[(k, name)][i]:.4f} (own best {max(curves[(k, name)]):.4f})" for k in kinds))
    c_map = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    for k in kinds:
        for name, p_ in (("stage1", D[k]["pred"]), ("stage2", D[k]["new"])):
            _, sc = macro_f05(assign(p_, res[name]["thr"]), D[k]["truth"], D[k]["q"])
            bd = pl.DataFrame({"entity_id": D[k]["q"], "f": sc}).join(c_map, on="entity_id")
            log(f"  [{k}] {name} by country: " + str({r['country']: round(r['f'], 4) for r in bd.group_by('country').agg(pl.col('f').mean()).to_dicts()}))
    final = S2.fit(band, feats, rounds=int(np.mean(iters) * 1.1))
    final.save_model(os.path.join(ARTIFACT_DIR, f"{s2name(tag)}.txt"))
    imp = sorted(zip(feats, final.feature_importance("gain")), key=lambda x: -x[1])
    log("s2 top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp[:20]))
    json.dump({"tag": tag, "trained_on": kinds, "stage1": res["stage1"], "stage2": res["stage2"],
               "stage1_val": res["stage1"].get("val"), "stage2_val": res["stage2"].get("val"),
               "thr": res["stage2"]["thr"], "features": feats, "band": S2.BAND, "conf": S2.CONF},
              open(os.path.join(ARTIFACT_DIR, f"{s2name(tag)}.json"), "w"), indent=1)


def test(tag, cfg_path, out_dir):
    import lightgbm as lgb
    cfg = json.load(open(cfg_path))
    s2cfg = json.load(open(os.path.join(ARTIFACT_DIR, f"{s2name(tag)}.json")))
    m = S2.load_bag(os.path.join(ARTIFACT_DIR, f"{s2name(tag)}.txt"))  # single model or seed bag
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{tag}.parquet")).select("s1", "mid", "p")
    ids = pl.concat([pred["s1"], pred["mid"]]).unique()
    recs = recs_for("test", ids)
    band, feats = S2.build(pred, recs, stage1_band("test", tag), ce_scores("test"), band_ce_scores("test"), visible_s1("test"))
    assert feats == s2cfg["features"], "feature mismatch between dev and test"
    new = S2.rescore(pred, band, m, feats)
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    new = new.join(s1.rename({"entity_id": "s1"}), on="s1", how="left")
    new.write_parquet(os.path.join(FEAT_DIR, f"test_pred_{s2name(tag)}.parquet"))
    thr = cfg.get("thr", s2cfg["thr"])
    matches = assign(new, thr, cfg.get("thr_by_country"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1["entity_id"].to_list():
            x = matches.get(s)
            f.write(f"{s}\t{','.join(sorted(x)) if x else ''}\n")
    # candidate_pairs.tsv = the exact set the final matcher runs inference over: blocking candidates
    # kept by the stage-1 pruning model (p >= BAND[0]); cross-encoder, stage 2 and the assignment only
    # ever see these pairs. Matches are a subset by construction.
    cand = dict(pred.select("s1", "mid").unique().group_by("s1").agg(pl.col("mid").sort().str.join(",")).iter_rows())
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in s1["entity_id"].to_list():
            f.write(f"{s}\t{cand.get(s, '')}\n")
    log(f"candidate_pairs.tsv: {sum(len(v.split(',')) for v in cand.values()) / s1.height:.2f} candidates/S1 "
        f"(stage-1 pruning p >= {S2.BAND[0]})")
    by = {}
    for s, c in s1.iter_rows():
        v = by.setdefault(c, [0, 0, 0])
        v[0] += 1; v[1] += bool(matches.get(s)); v[2] += len(matches.get(s, ()))
    for c, (n, ne, k) in sorted(by.items()):
        log(f"{c}: predicted singleton rate {1 - ne / n:.4f}, matches/S1 {k / n:.3f}")
    log(f"band pairs re-scored: {band.height:,}; thr={thr} thr_by_country={cfg.get('thr_by_country')}")


if __name__ == "__main__":
    {"dev": lambda: dev(sys.argv[2]), "test": lambda: test(*sys.argv[2:5])}[sys.argv[1]]()
