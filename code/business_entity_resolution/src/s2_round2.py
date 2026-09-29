"""Round-2 stage 2: re-decide the uncertain band from the round-1 (stage-2) probabilities q.

Round 1 computes its competition and count features from STAGE-1 probabilities. Round 2 recomputes
them from round-1 DECISIONS and adds collective evidence that no earlier model saw:
  * per-source counts of the S1's accepted records and of the best rival S1's accepted records, with a
    count prior learned from train (an S1 has 0..6 records per source: 13% / 36% / 30% / 15% / 5% ...),
    so an S1 that already shows its S2 copies is a less likely owner of one more S2 record;
  * twin consistency: identical records (same canonical name + address + house) belong to the same S1
    99.7% of the time (train), so where the record's twins were assigned is evidence for this record.

  python s2_round2.py dev  <s2name> <thr> [feature_set]    # 4 region folds over val + wide, out-of-fold
  python s2_round2.py test <s2name> <out_dir>               # apply to test round-1 predictions
"""
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
LO, HI = S2.BAND
BASE = ["r2_q", "r2_logit", "r2_rival_q", "r2_margin", "r2_rk", "r2_n_claim_other", "r2_s1_rank", "r2_s1_best_other",
        "r2_p1", "r2_is_s3", "r2_a_null"]
COUNTS = ["r2_n_own_same", "r2_n_own_othsrc", "r2_n_own_all", "r2_n_riv_same", "r2_n_riv_all", "r2_llr_own", "r2_llr_riv", "r2_llr_d"]
TWINS = ["r2_tw_n", "r2_tw_own", "r2_tw_oth", "r2_tw_qsum_own"]
SETS = {"base": BASE, "counts": BASE + COUNTS, "twins": BASE + TWINS, "all": BASE + COUNTS + TWINS}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def count_prior():
    """{source: log P(n+1)/P(n)} for the number of an S1's records in that source (train truth)."""
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id"])
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")).rename({"source1_entity_id": "entity_id"})
    g = s1.join(gt, on="entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    out = {}
    for src in ("S2", "S3"):
        p = np.bincount(g["matched_entity_ids"].str.count_matches(f"{src}-").to_numpy(), minlength=16).astype(float) + 1.0
        out[src] = np.log(p[1:] / p[:-1]).tolist()
    return out


def recs_of(split, mids):
    lf = pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f"{split}_s{i}.parquet")) for i in (2, 3)])
    return (lf.select("entity_id", "n_compact", "a_tokens", "a_house", "a_null").filter(pl.col("entity_id").is_in(mids.implode()))
              .collect(engine="streaming"))


def r2_features(q, p1, recs, thr, prior):
    """q: s1, mid, q for every candidate pair; p1: s1, mid, p1 (stage 1); recs: candidate records."""
    d = q.join(p1, on=["s1", "mid"], how="left").with_columns(pl.col("mid").str.slice(0, 2).alias("src"))
    d = d.with_columns(pl.col("q").rank("ordinal", descending=True).over("mid").alias("r2_rk"))
    top = d.filter(pl.col("r2_rk") <= 2).select("mid", "r2_rk", "s1", "q")
    t1 = top.filter(pl.col("r2_rk") == 1).select("mid", pl.col("s1").alias("top1"), pl.col("q").alias("q1"))
    t2 = top.filter(pl.col("r2_rk") == 2).select("mid", pl.col("s1").alias("top2"), pl.col("q").alias("q2"))
    d = d.join(t1, on="mid", how="left").join(t2, on="mid", how="left").with_columns(
        pl.when(pl.col("r2_rk") == 1).then(pl.col("top2")).otherwise(pl.col("top1")).alias("rival"),
        pl.when(pl.col("r2_rk") == 1).then(pl.col("q2")).otherwise(pl.col("q1")).fill_null(0.0).alias("r2_rival_q"),
        ((pl.col("r2_rk") == 1) & (pl.col("q") >= thr)).cast(pl.Int32).alias("a"),          # this pair accepted
        ((pl.col("r2_rk") != 1) & (pl.col("q1") >= thr)).cast(pl.Int32).alias("a_riv"))    # rival holds the record
    acc = d.filter(pl.col("a") == 1).select("s1", "mid", "src")
    c_src = acc.group_by("s1", "src").len("n_src")
    c_all = acc.group_by("s1").len("n_all")
    d = (d.join(c_src, on=["s1", "src"], how="left").join(c_all, on="s1", how="left")
          .join(c_src.rename({"s1": "rival", "n_src": "rn_src"}), on=["rival", "src"], how="left")
          .join(c_all.rename({"s1": "rival", "n_all": "rn_all"}), on="rival", how="left")
          .with_columns(pl.col(c).fill_null(0) for c in ("n_src", "n_all", "rn_src", "rn_all")))
    oth = pl.when(pl.col("src") == "S2").then(pl.lit("S3")).otherwise(pl.lit("S2"))
    d = (d.with_columns(oth.alias("osrc")).join(c_src.rename({"src": "osrc", "n_src": "n_osrc"}), on=["s1", "osrc"], how="left")
          .with_columns(pl.col("n_osrc").fill_null(0)))
    d = d.with_columns(
        (pl.col("n_src") - pl.col("a")).alias("r2_n_own_same"), pl.col("n_osrc").alias("r2_n_own_othsrc"),
        (pl.col("n_all") - pl.col("a")).alias("r2_n_own_all"),
        pl.when(pl.col("rival").is_null()).then(None).otherwise(pl.col("rn_src") - pl.col("a_riv")).alias("r2_n_riv_same"),
        pl.when(pl.col("rival").is_null()).then(None).otherwise(pl.col("rn_all") - pl.col("a_riv")).alias("r2_n_riv_all"))
    llr = lambda n: (pl.when(pl.col("src") == "S2").then(pl.col(n).clip(0, 14).replace_strict(list(range(15)), prior["S2"][:15], default=None))
                     .otherwise(pl.col(n).clip(0, 14).replace_strict(list(range(15)), prior["S3"][:15], default=None)))
    d = d.with_columns(llr("r2_n_own_same").alias("r2_llr_own"), llr("r2_n_riv_same").alias("r2_llr_riv"))
    d = d.with_columns((pl.col("r2_llr_own") - pl.col("r2_llr_riv")).alias("r2_llr_d"))
    # twins among candidate records: identical canonical name + address + house, address present
    tw = (recs.filter(pl.col("a_null").fill_null(1) == 0)
              .select(pl.col("entity_id").alias("mid"),
                      pl.concat_str([pl.col("n_compact").fill_null(""), pl.col("a_tokens").fill_null(""), pl.col("a_house").fill_null("")],
                                    separator="|").alias("key")))
    tw = tw.join(tw.group_by("key").len("tw_size"), on="key").filter(pl.col("tw_size") >= 2)
    d = d.join(tw, on="mid", how="left")
    acc_k = acc.join(tw, on="mid", how="inner")
    d = (d.join(acc_k.group_by("key", "s1").len("k_own"), on=["key", "s1"], how="left")
          .join(acc_k.group_by("key").len("k_all"), on="key", how="left")
          .with_columns(pl.col("k_own").fill_null(0), pl.col("k_all").fill_null(0), pl.col("tw_size").fill_null(1)))
    qk = d.filter(pl.col("key").is_not_null()).group_by("key", "s1").agg(pl.col("q").sum().alias("k_qsum"))
    d = d.join(qk, on=["key", "s1"], how="left").with_columns(pl.col("k_qsum").fill_null(pl.col("q")))
    a_any = (pl.col("a") + pl.col("a_riv")).clip(0, 1)
    d = d.with_columns(
        (pl.col("tw_size") - 1).alias("r2_tw_n"), (pl.col("k_own") - pl.col("a")).alias("r2_tw_own"),
        ((pl.col("k_all") - a_any) - (pl.col("k_own") - pl.col("a"))).alias("r2_tw_oth"),
        (pl.col("k_qsum") - pl.col("q")).alias("r2_tw_qsum_own"))
    conf = (pl.col("q") >= 0.5).cast(pl.Int32)
    d = d.with_columns(
        pl.col("q").alias("r2_q"), (pl.col("q").clip(1e-6, 1 - 1e-6) / (1 - pl.col("q").clip(1e-6, 1 - 1e-6))).log().alias("r2_logit"),
        (pl.col("q") - pl.col("r2_rival_q")).alias("r2_margin"), (conf.sum().over("mid") - conf).alias("r2_n_claim_other"),
        pl.col("q").rank("ordinal", descending=True).over("s1").alias("r2_s1_rank"),
        pl.when(pl.col("q") == pl.col("q").max().over("s1")).then(pl.col("q").sort(descending=True).slice(1, 1).first().over("s1").fill_null(0.0))
          .otherwise(pl.col("q").max().over("s1")).alias("r2_s1_best_other"),
        pl.col("p1").alias("r2_p1"), (pl.col("src") == "S3").cast(pl.Int8).alias("r2_is_s3"))
    d = d.join(recs.select(pl.col("entity_id").alias("mid"), pl.col("a_null").fill_null(1).cast(pl.Int8).alias("r2_a_null")), on="mid", how="left")
    return d.filter((pl.col("p1") >= LO) & (pl.col("p1") < HI)).select(["s1", "mid"] + SETS["all"])


def folds(kind, offset, s1_ids):
    """Same greedy region folds as run_stage2.universe_data (round-1 out-of-fold folds)."""
    reg = (pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "region"])
             .filter(pl.col("entity_id").is_in(s1_ids.implode())).select(pl.col("entity_id").alias("s1"), "region"))
    sizes = sorted(reg.group_by("region").len().iter_rows(), key=lambda r: -r[1])
    fold_of, load = {}, [0, 0]
    for r, n in sizes:
        k = int(load[1] < load[0])
        fold_of[r] = offset + k
        load[k] += n
    return reg.with_columns(pl.col("region").replace_strict(fold_of, default=offset, return_dtype=pl.Int8).alias("fold")).drop("region")


def dev(s2name, thr, which):
    import run_dev as R
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    prior = count_prior()
    log("count prior log P(n+1)/P(n):", {k: [round(x, 2) for x in v[:6]] for k, v in prior.items()})
    # ER_R2_ORPHAN_KEEP="US=0.47,India=0.105": keep only this share of orphan records (true S1 dropped from
    # the simulation), matching the orphan density measured on test (US ~9%, India ~2% vs 19% simulated)
    keep = {k: float(v) for k, v in (x.split("=") for x in os.environ.get("ER_R2_ORPHAN_KEEP", "").split(",") if x)}
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "owner"})
    D, bands = {}, []
    for i, kind in enumerate(("val", "wide")):
        qs = uni.filter((pl.col("universe") == kind) & pl.col("is_query"))["entity_id"]
        oof = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_s2oof_{s2name}.parquet"))
        if keep:
            o = (oof.select("mid").unique().join(owner, on="mid", how="inner").join(s1c, on="owner", how="left")
                    .filter(~pl.col("owner").is_in(qs.implode()))
                    .with_columns(pl.col("country").replace_strict(keep, default=1.0, return_dtype=pl.Float64).alias("k"),
                                  (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h")))
            drop = o.filter(pl.col("h") >= pl.col("k")).select("mid")
            log(f"[{kind}] orphan records {o.height:,}; dropping {drop.height:,} to match test density {keep}")
            oof = oof.join(drop, on="mid", how="anti")
        p1 = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"})
        recs = recs_of("train", oof["mid"].unique())
        f = r2_features(oof.select("s1", "mid", pl.col("p").alias("q")), p1, recs, thr, prior)
        f = f.join(oof.select("s1", "mid", "y"), on=["s1", "mid"], how="left").join(folds(kind, 2 * i, qs), on="s1", how="left")
        D[kind] = {"q": qs.to_list(), "truth": truth_map_from_gt(gt, qs.to_list()), "pred": oof.select("s1", "mid", "p")}
        bands.append(f)
        log(f"[{kind}] band {f.height:,} pairs, positives {int(f['y'].sum()):,}; twins in band {(f['r2_tw_n'] > 0).mean():.3f}")
    band = pl.concat(bands)
    n_all = sum(len(D[k]["q"]) for k in D)
    grid = [float(t) for t in np.round(np.arange(0.5, 0.96, 0.025), 3)]

    def score(new_p):
        curves = {}
        for k in D:
            p_ = D[k]["pred"] if new_p is None else D[k]["pred"].join(new_p, on=["s1", "mid"], how="left").with_columns(
                pl.coalesce("p2", "p").alias("p")).drop("p2")
            curves[k] = [macro_f05(assign(p_, t), D[k]["truth"], D[k]["q"])[0] for t in grid]
        w = [sum(curves[k][j] * len(D[k]["q"]) for k in D) / n_all for j in range(len(grid))]
        j = int(np.argmax(w))
        return grid[j], w[j], {k: round(curves[k][j], 5) for k in D}

    t, w, per = score(None)
    log(f"ROUND 1 (baseline): thr {t} weighted {w:.5f} {per}")
    results = {}
    for name in (which.split(",") if which else ["base", "counts", "twins", "all"]):
        feats = SETS[name]
        oof2, iters = [], []
        for k in sorted(band["fold"].unique().to_list()):
            tr, te = band.filter(pl.col("fold") != k), band.filter(pl.col("fold") == k)
            es = tr["s1"].unique().sample(fraction=0.15, seed=3)
            m = S2.fit(tr.filter(~pl.col("s1").is_in(es.implode())), feats, valid=tr.filter(pl.col("s1").is_in(es.implode())))
            oof2.append(te.select("s1", "mid").with_columns(pl.Series("p2", m.predict(te.select(feats).to_numpy().astype(np.float32)))))
            iters.append(m.best_iteration)
        t, w, per = score(pl.concat(oof2))
        results[name] = (t, w, iters)
        log(f"ROUND 2 [{name}] thr {t} weighted {w:.5f} {per} iters {iters}")
        if name == (which.split(",")[-1] if which else "all"):
            final = S2.fit(band, feats, rounds=int(np.mean(iters) * 1.1))
            tag = name + ("_orph" if keep else "")
            final.save_model(os.path.join(ARTIFACT_DIR, f"r2_{s2name}_{tag}.txt"))
            imp = sorted(zip(feats, final.feature_importance("gain")), key=lambda x: -x[1])
            log("top features: " + ", ".join(f"{a}={b:.0f}" for a, b in imp[:15]))
            json.dump({"s2name": s2name, "round1_thr": thr, "thr": t, "weighted": w, "features": feats, "prior": prior},
                      open(os.path.join(ARTIFACT_DIR, f"r2_{s2name}_{tag}.json"), "w"), indent=1)


def test(s2name, out_dir, name="all"):
    import lightgbm as lgb
    cfg = json.load(open(os.path.join(ARTIFACT_DIR, f"r2_{s2name}_{name}.json")))
    m = lgb.Booster(model_file=os.path.join(ARTIFACT_DIR, f"r2_{s2name}_{name}.txt"))
    q = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{s2name}.parquet"), columns=["s1", "mid", "p"])
    p1 = pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"})
    recs = recs_of("test", q["mid"].unique())
    f = r2_features(q.rename({"p": "q"}), p1, recs, cfg["round1_thr"], cfg["prior"])
    new = q.join(f.select("s1", "mid", pl.Series("p2", m.predict(f.select(cfg["features"]).to_numpy().astype(np.float32)))),
                 on=["s1", "mid"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    new.join(s1.rename({"entity_id": "s1"}), on="s1", how="left").write_parquet(os.path.join(FEAT_DIR, f"test_pred_r2_{s2name}_{name}.parquet"))
    matches = assign(new, cfg["thr"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fo:
        fo.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1["entity_id"].to_list():
            x = matches.get(s)
            fo.write(f"{s}\t{','.join(sorted(x)) if x else ''}\n")
    by = {}
    for s, c in s1.iter_rows():
        v = by.setdefault(c, [0, 0, 0])
        v[0] += 1; v[1] += bool(matches.get(s)); v[2] += len(matches.get(s, ()))
    for c, (n, ne, k) in sorted(by.items()):
        log(f"{c}: predicted singleton rate {1 - ne / n:.4f}, matches/S1 {k / n:.3f}")
    log(f"band pairs re-scored: {f.height:,}; thr={cfg['thr']}")


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        dev(sys.argv[2], float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        test(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
