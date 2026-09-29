"""Opt-in correction of frozen stage-1 claims using real cross-encoder evidence.

Commands: build (features), dev (record-group OOF and refit), apply (new artifacts).
Never changes the legacy stage-2 code or consumes its probabilities as model features.
See docs/05_CONFIDENT_PAIR_EXPERIMENT.md for the validation protocol and commands.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from evaluate import macro_f05, truth_map_from_gt

KEY = ["s1", "mid"]
REC = ["entity_id", "country", "region", "n_core", "n_compact", "a_tokens", "a_house", "a_null"]


def check_unique(df, keys, label):
    if df.select(pl.any_horizontal(pl.col(c).is_null() | (pl.col(c) == "") for c in keys).any()).item():
        raise ValueError(f"{label}: null or empty IDs")
    if df.unique(keys).height != df.height:
        raise ValueError(f"{label}: duplicate {keys}")


def check_probability(df, column, label):
    if df.filter(pl.col(column).is_null() | ~pl.col(column).is_finite() | ~pl.col(column).is_between(0, 1)).height:
        raise ValueError(f"{label}: invalid {column}")


def read_queries(path):
    q = pl.read_parquet(path)
    if "is_query" in q.columns:
        q = q.filter(pl.col("is_query"))
    if "s1" not in q.columns:
        q = q.rename({"entity_id": "s1"})
    check_unique(q, ["s1"], "queries")
    return q.select("s1")


def logit_expr(column):
    p = pl.col(column).clip(1e-7, 1 - 1e-7)
    return (p / (1 - p)).log()


def competition(df, column, prefix, absent=0.0):
    """Actual rival evidence. Equal best scores yield zero margins, never arbitrary ties."""
    agg = df.group_by("mid").agg(
        pl.col(column).max().alias("_best"),
        pl.col(column).sort(descending=True).slice(1, 1).first().fill_null(absent).alias("_second"),
    )
    d = df.join(agg, on="mid").with_columns(
        pl.when(pl.col(column) == pl.col("_best")).then(pl.col("_second"))
        .otherwise(pl.col("_best")).alias(prefix + "rival")
    )
    return d.with_columns((pl.col(column) - pl.col(prefix + "rival")).alias(prefix + "margin")).drop("_best", "_second")


def read_ce(specs, pairs):
    by_name = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        if not name.isidentifier():
            raise ValueError("CE names must be Python identifiers")
        d = pl.scan_parquet(path).join(pairs.lazy(), on=KEY, how="semi").collect(engine="streaming")
        if "ce_logit" not in d.columns:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("ce_logit"))
        by_name.setdefault(name, []).append(d.select(*KEY, pl.col("p_ce").cast(pl.Float64), pl.col("ce_logit").cast(pl.Float64)))
    result = {}
    for name, shards in sorted(by_name.items()):
        d = pl.concat(shards)
        check_unique(d, KEY, f"CE {name}; use disjoint old/new shards or merge first")
        check_probability(d, "p_ce", f"CE {name}")
        if d.filter(pl.col("ce_logit").is_not_null() & ~pl.col("ce_logit").is_finite()).height:
            raise ValueError(f"CE {name}: nonfinite raw logits")
        if d.height != pairs.height:
            raise ValueError(f"CE {name}: incomplete target-group coverage ({d.height}/{pairs.height})")
        result[name] = d
    return result


def build_features(stage1, q, s1_records, pool_records, ces, lower=.02, high=.99, country=None):
    """Select entire retained claimant groups whenever any member crosses high."""
    check_unique(stage1, KEY, "stage1")
    check_probability(stage1, "p", "stage1")
    for label, r in (("S1 records", s1_records), ("pool records", pool_records)):
        check_unique(r, ["entity_id"], label)
    if q.join(s1_records, left_on="s1", right_on="entity_id", how="anti").height:
        raise ValueError("Some query S1 records are missing")
    visible = s1_records.join(q, left_on="entity_id", right_on="s1", how="semi")
    d = stage1.join(q, on="s1", how="semi").filter(pl.col("p") >= lower).select(*KEY, pl.col("p").alias("x_p1"))
    d = d.join(s1_records.select(pl.col("entity_id").alias("s1"), "country", "region"), on="s1", how="left")
    mids = d.filter((pl.col("x_p1") >= high) & (pl.lit(True) if country is None else pl.col("country") == country)).select("mid").unique()
    d = d.join(mids, on="mid", how="semi")
    if d.height == 0:
        raise ValueError("No target groups satisfy the selection")
    if d.group_by("mid").agg(pl.col("country").n_unique().alias("n")).filter(pl.col("n") != 1).height:
        raise ValueError("A target record has claimants from multiple countries")
    if d.select("mid").unique().join(pool_records, left_on="mid", right_on="entity_id", how="anti").height:
        raise ValueError("Some target pool records are missing")
    a = visible.select(REC).rename({c: "a_" + c for c in REC})
    b = pool_records.select(REC).rename({c: "b_" + c for c in REC})
    d = d.join(a, left_on="s1", right_on="a_entity_id", how="left").join(b, left_on="mid", right_on="b_entity_id", how="left")
    if d.filter(pl.col("country") != pl.col("b_country")).height:
        raise ValueError("Cross-country candidate found")
    for c in ("a_n_core", "b_n_core", "a_n_compact", "b_n_compact", "a_a_tokens", "b_a_tokens", "a_a_house", "b_a_house"):
        d = d.with_columns(pl.col(c).fill_null(""))
    d = d.with_columns(
        logit_expr("x_p1").alias("x_p1_logit"),
        pl.len().over("mid").cast(pl.Float32).alias("x_claimants"),
        (pl.col("x_p1") >= high).cast(pl.Int8).alias("x_was_frozen"),
        ((pl.col("a_n_compact") == pl.col("b_n_compact")) & (pl.col("a_n_compact") != "")).cast(pl.Int8).alias("x_name_exact"),
        ((pl.col("a_a_tokens") == pl.col("b_a_tokens")) & (pl.col("a_a_tokens") != "")).cast(pl.Int8).alias("x_address_exact"),
        ((pl.col("a_a_house") == pl.col("b_a_house")) & (pl.col("a_a_house") != "")).cast(pl.Int8).alias("x_house_equal"),
        ((pl.col("a_a_house") != "") & (pl.col("b_a_house") != "")).cast(pl.Int8).alias("x_house_both"),
        pl.col("b_a_null").fill_null(1).cast(pl.Int8).alias("x_address_null"),
        (pl.col("a_region").fill_null("") == pl.col("b_region").fill_null("")).cast(pl.Int8).alias("x_region_equal"),
    )
    for left, right, scorer, name in (
        ("a_n_core", "b_n_core", fuzz.token_set_ratio, "x_name_tokens"),
        ("a_n_compact", "b_n_compact", fuzz.ratio, "x_name_ratio"),
        ("a_a_tokens", "b_a_tokens", fuzz.token_set_ratio, "x_address_tokens"),
    ):
        d = d.with_columns(pl.Series(name, cpdist(d[left].to_list(), d[right].to_list(), scorer=scorer, workers=1, dtype=np.float32)))
    addresses = visible.filter((pl.col("a_tokens").fill_null("") != "") & (pl.col("a_house").fill_null("") != ""))
    counts = addresses.group_by("country", "a_tokens", "a_house").len().rename({"country": "a_country", "a_tokens": "a_a_tokens", "a_house": "a_a_house", "len": "x_address_occupancy"})
    d = d.join(counts, on=["a_country", "a_a_tokens", "a_a_house"], how="left").with_columns(pl.col("x_address_occupancy").fill_null(0))
    d = competition(d, "x_p1", "x_p1_")
    for name, ce in sorted(ces.items()):
        ce = ce.join(d.select(KEY), on=KEY, how="semi")
        check_unique(ce, KEY, name)
        check_probability(ce, "p_ce", name)
        if ce.height != d.height:
            raise ValueError(f"{name}: missing target CE scores")
        pre = "x_" + name + "_"
        if "ce_logit" not in ce.columns:
            ce = ce.with_columns(pl.lit(None, dtype=pl.Float64).alias("ce_logit"))
        ce = ce.with_columns(
            pl.col("ce_logit").is_null().cast(pl.Int8).alias(pre + "logit_reconstructed"),
            pl.coalesce(pl.col("ce_logit"), logit_expr("p_ce")).alias(pre + "logit"),
        ).select(*KEY, pl.col("p_ce").alias(pre + "p"), pre + "logit", pre + "logit_reconstructed")
        d = d.join(ce, on=KEY)
        d = competition(d, pre + "p", pre, absent=0.)
        d = competition(d, pre + "logit", pre + "logit_", absent=-16.118095)
    d = d.with_columns((pl.col("country") + "|" + pl.col("region").fill_null("<unknown>")).alias("block"))
    return d.select(*KEY, "country", "block", *sorted(c for c in d.columns if c.startswith("x_")))


def build(args):
    q = read_queries(args.queries)
    s = pl.read_parquet(args.prepared_s1, columns=REC)
    pred = pl.scan_parquet(args.stage1).join(q.lazy(), on="s1", how="semi").select(*KEY, "p").collect(engine="streaming")
    # Select target IDs before loading canonical pool rows or cross-encoder exports.
    candidates = pred.filter(pl.col("p") >= args.lower).join(s.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
    target = candidates.filter((pl.col("p") >= args.high) & (pl.lit(True) if args.country is None else pl.col("country") == args.country)).select("mid").unique()
    pairs = candidates.join(target, on="mid", how="semi").select(KEY)
    pool = pl.concat([pl.scan_parquet(p).select(REC) for p in args.prepared_pool]).join(target.lazy(), left_on="entity_id", right_on="mid", how="semi").collect(engine="streaming")
    scores = read_ce(args.ce, pairs)
    d = build_features(pred, q, s, pool, scores, args.lower, args.high, args.country)
    out = Path(args.output)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(out)
    meta = {"selection": "all retained claimants of records with any p1 >= high", "lower": args.lower, "high": args.high,
            "country": args.country, "ce_names": sorted(scores), "pairs": d.height, "mids": d["mid"].n_unique(),
            "features": sorted(c for c in d.columns if c.startswith("x_")), "stage1": str(args.stage1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def predictions(df, threshold):
    check_unique(df, KEY, "prediction scores")
    check_probability(df, "p", "prediction scores")
    winner = df.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
    return {s: set(m) for s, m in winner.filter(pl.col("p") >= threshold).group_by("s1").agg("mid").iter_rows()}


def repair_baseline(base, corrected, target_mids):
    result = {s: m - target_mids for s, m in base.items()}
    for s, mids in corrected.items():
        result.setdefault(s, set()).update(mids)
    return result


def region_group_folds(d, k):
    """Every mid's rivals share a fold, anchored to its frozen stage-1 winner's region."""
    owner = d.sort(["x_p1", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True).select("mid", pl.col("block").alias("owner_block"))
    sizes = sorted(owner.group_by("owner_block").len().iter_rows(), key=lambda z: (-z[1], z[0]))
    if len(sizes) < k:
        raise ValueError(f"Need at least {k} owner regions; found {len(sizes)}")
    loads, mapping = [0] * k, {}
    for block, n in sizes:
        f = min(range(k), key=lambda i: loads[i])
        mapping[block] = f
        loads[f] += n
    return d.join(owner, on="mid").with_columns(pl.col("owner_block").replace_strict(mapping, return_dtype=pl.Int8).alias("fold")), mapping


def truth_pairs(path, q):
    gt = pl.read_parquet(path).select("source1_entity_id", "matched_entity_ids")
    if q.join(gt, left_on="s1", right_on="source1_entity_id", how="anti").height:
        raise ValueError("Ground truth is missing query rows")
    check_unique(gt, ["source1_entity_id"], "truth")
    truth = truth_map_from_gt(gt, q["s1"].to_list())
    positive = gt.rename({"source1_entity_id": "s1"}).join(q, on="s1", how="semi").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("mid")).explode("mid").filter(pl.col("mid") != "").select(KEY).unique()
    if positive.group_by("mid").len().filter(pl.col("len") > 1).height:
        raise ValueError("Ground truth violates one owner per pool record")
    return truth, positive


def load_feature_files(paths):
    frames = [pl.read_parquet(p) for p in paths]
    feats = sorted(c for c in frames[0].columns if c.startswith("x_"))
    if not feats or any(sorted(c for c in f.columns if c.startswith("x_")) != feats for f in frames):
        raise ValueError("Feature schemas differ or are empty")
    d = pl.concat(frames)
    check_unique(d, KEY, "features")
    # Competition was built inside each export; overlapping mids need joint feature construction.
    if sum(f["mid"].n_unique() for f in frames) != d["mid"].n_unique():
        raise ValueError("Feature files share mids: build their joint candidate universe before CV")
    return d, feats


def fit_model(d, feats, args):
    if d["y"].n_unique() < 2:
        raise ValueError("Training split lacks both classes")
    params = dict(objective="binary", learning_rate=.03, num_leaves=31, min_data_in_leaf=args.min_leaf,
                  lambda_l2=10., verbosity=-1, num_threads=args.threads, seed=17, deterministic=True,
                  force_col_wise=True)
    return lgb.train(params, lgb.Dataset(d.select(feats).to_numpy().astype(np.float32), label=d["y"].to_numpy(), feature_name=feats), num_boost_round=args.rounds)


def dev(args):
    if not args.upstream_oos:
        raise ValueError("Use --upstream-oos only after checking stage1 and CE training excluded these validation labels")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty output directory")
    q = pl.concat([read_queries(p) for p in args.queries])
    check_unique(q, ["s1"], "combined queries")
    d, feats = load_feature_files(args.features)
    if d.select("s1").unique().join(q, on="s1", how="anti").height:
        raise ValueError("Features include S1 outside query universe")
    truth, positives = truth_pairs(args.truth, q)
    d = d.join(positives.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    d, block_folds = region_group_folds(d, args.folds)
    rows, reports = [], []
    for fold in range(args.folds):
        te = d.filter(pl.col("fold") == fold)
        held_blocks = [b for b, f in block_folds.items() if f == fold]
        eligible = d.filter((pl.col("fold") != fold) & ~pl.col("block").is_in(held_blocks))
        tr = eligible.join(te.select("s1").unique(), on="s1", how="anti")
        if tr["mid"].is_in(te["mid"].unique().implode()).any():
            raise AssertionError("Pool record leaked across folds")
        model = fit_model(tr, feats, args)
        p = model.predict(te.select(feats).to_numpy().astype(np.float32), num_threads=args.threads)
        rows.append(te.select(*KEY, "fold").with_columns(pl.Series("p", p)))
        reports.append({"fold": fold, "train_pairs": tr.height, "test_pairs": te.height, "purged_shared_s1_pairs": eligible.height - tr.height, "train_negatives": tr.filter(pl.col("y") == 0).height})
        print(json.dumps(reports[-1]), flush=True)
    oof = pl.concat(rows)
    baseline = pl.concat([pl.read_parquet(p, columns=KEY + ["p"]) for p in args.baseline]).join(q, on="s1", how="semi")
    base_map = predictions(baseline, args.baseline_threshold)
    ids = q["s1"].to_list()
    score0, f0 = macro_f05(base_map, truth, ids)
    target = set(d["mid"].unique().to_list())
    curve = []
    best = None
    for t in args.thresholds:
        result = repair_baseline(base_map, predictions(oof.select(*KEY, "p"), t), target)
        score, scores = macro_f05(result, truth, ids)
        curve.append({"threshold": t, "macro_f05": score, "delta": score - score0})
        if best is None or score > best[0]:
            best = score, t, scores, result
    # Full per-query score files include unaffected queries and singleton entities.
    per = q.with_columns(pl.Series("baseline_f05", f0), pl.Series("corrected_f05", best[2])).with_columns((pl.col("corrected_f05") - pl.col("baseline_f05")).alias("delta"))
    changed = sum(base_map.get(s, set()) != best[3].get(s, set()) for s in ids)
    final = fit_model(d, feats, args)
    out.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out / "model.txt"))
    oof.write_parquet(out / "oof.parquet")
    per.write_parquet(out / "entity_scores.parquet")
    metadata = {"features": feats, "threshold": best[1], "baseline_threshold": args.baseline_threshold,
                "baseline_macro_f05": score0, "tuned_oof_macro_f05": best[0], "delta": best[0] - score0,
                "changed_entities": changed, "improved_entities": int((best[2] > f0).sum()), "harmed_entities": int((best[2] < f0).sum()),
                "target_mids": len(target), "folds": reports, "rounds": args.rounds,
                "validation": "record-group OOF, owner-region folds; shared S1 endpoints and heldout owner-region training rows purged; threshold tuned on OOF (not independent holdout)",
                "upstream_oos_asserted": True, "baseline_is_model_feature": False, "curve": curve}
    (out / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def decision_rescale(p, source_threshold, target_threshold):
    """Preserve rankings and threshold decisions; result is a decision score, not calibration."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    shift = np.log(target_threshold / (1 - target_threshold)) - np.log(source_threshold / (1 - source_threshold))
    z = np.log(p / (1 - p)) + shift
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def apply(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty output directory")
    cfg = json.loads((Path(args.model_dir) / "model.json").read_text(encoding="utf-8"))
    d, feats = load_feature_files(args.features)
    if feats != cfg["features"]:
        raise ValueError("Model/feature mismatch")
    q = read_queries(args.queries)
    if d.select("s1").unique().join(q, on="s1", how="anti").height:
        raise ValueError("Correction features contain non-query S1")
    base = pl.read_parquet(args.baseline, columns=KEY + ["p"]).join(q, on="s1", how="semi")
    check_unique(base, KEY, "baseline")
    check_probability(base, "p", "baseline")
    candidates = pl.read_parquet(args.candidates, columns=KEY)
    check_unique(candidates, KEY, "candidate manifest")
    if candidates.select("s1").unique().join(q, on="s1", how="anti").height:
        raise ValueError("Candidate manifest includes non-query S1")
    if d.select(KEY).join(candidates, on=KEY, how="anti").height:
        raise ValueError("Correction pairs missing from final candidate manifest")
    # This experiment does not expand retrieval. Require precisely the old final
    # matcher's candidate set, not merely its accepted pairs.
    if base.select(KEY).join(candidates, on=KEY, how="anti").height or candidates.join(base.select(KEY), on=KEY, how="anti").height:
        raise ValueError("Candidate manifest must equal baseline prediction keys exactly; provide the pruned final-matcher baseline")
    target = d.select("mid").unique()
    expected = candidates.join(target, on="mid", how="semi")
    if expected.join(d.select(KEY), on=KEY, how="anti").height:
        raise ValueError("Correction does not cover every candidate claimant of its target records")
    model = lgb.Booster(model_file=str(Path(args.model_dir) / "model.txt"))
    p = model.predict(d.select(feats).to_numpy().astype(np.float32), num_threads=args.threads)
    corrected = d.select(KEY).with_columns(pl.Series("p", decision_rescale(p, cfg["threshold"], cfg["baseline_threshold"])))
    merged = pl.concat([base.join(target, on="mid", how="anti"), corrected])
    matches = predictions(merged, cfg["baseline_threshold"])
    out.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out / "predictions.parquet")
    with (out / "matching_results.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in q["s1"]:
            f.write(s + "\t" + ",".join(sorted(matches.get(s, ()))) + "\n")
    c = dict(candidates.group_by("s1").agg(pl.col("mid").sort().str.join(",")).iter_rows())
    with (out / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in q["s1"]:
            f.write(s + "\t" + c.get(s, "") + "\n")
    report = {"query_count": q.height, "correction_pairs": d.height, "candidate_pairs": candidates.height,
              "decision_threshold": cfg["baseline_threshold"], "note": "Corrected p values are threshold-aligned decision scores, not calibrated probabilities."}
    (out / "application.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for flag in ("stage1", "queries", "prepared-s1", "output"):
        b.add_argument("--" + flag, required=True)
    b.add_argument("--prepared-pool", nargs="+", required=True)
    b.add_argument("--ce", action="append", required=True, help="NAME=PARQUET (repeat for disjoint shards)")
    b.add_argument("--country")
    b.add_argument("--lower", type=float, default=.02)
    b.add_argument("--high", type=float, default=.99)
    b.set_defaults(func=build)
    t = sub.add_parser("dev")
    for flag in ("features", "baseline", "queries"):
        t.add_argument("--" + flag, nargs="+", required=True)
    t.add_argument("--truth", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--upstream-oos", action="store_true")
    t.add_argument("--baseline-threshold", type=float, default=.725)
    t.add_argument("--thresholds", type=float, nargs="+", default=[.5, .6, .65, .7, .725, .75, .8, .85, .9, .95, .975, .99, .995])
    t.add_argument("--folds", type=int, default=4)
    t.add_argument("--rounds", type=int, default=600)
    t.add_argument("--min-leaf", type=int, default=200)
    t.add_argument("--threads", type=int, default=8)
    t.set_defaults(func=dev)
    a = sub.add_parser("apply")
    a.add_argument("--features", nargs="+", required=True)
    for flag in ("baseline", "queries", "candidates", "model-dir", "out-dir"):
        a.add_argument("--" + flag, required=True)
    a.add_argument("--threads", type=int, default=8)
    a.set_defaults(func=apply)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if hasattr(args, "thresholds") and any(not 0 < t < 1 for t in args.thresholds):
        raise ValueError("Thresholds must be strictly between zero and one")
    args.func(args)
