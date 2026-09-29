"""Read-only validation audit for the confident-pair correction experiment.

Example (paths are deliberately explicit; no training/model imports):
    python src/audit_confident_pairs.py --stage1 val_pred_v6.parquet \
        --final val_pred_s2_v6.parquet --truth train_gt.parquet \
        --queries val_queries.parquet --out audit_v6.json --threshold .725

Scored files require s1, mid, p. Queries require entity_id (or s1), with optional
country/region. When present, is_query automatically excludes dropped S1 rows.
If a query file has multiple universe values, select one with --universe.
Ground truth uses evaluate.py's source1_entity_id, matched_entity_ids format;
every query must have a row, including explicit empty/null singleton rows.

An oracle repairs WHOLE RECORDS, removing the old owner and adding the true owner
inside the query universe. It can recover a true pair absent from the input and
is therefore an optimistic scope ceiling, not an achievable model estimate.
Score-bin eligibility depends only on stage-1 candidates; bins' record sets can
overlap. Only the explicitly named missing-retrieval ceiling uses labels to
choose eligible records. Missing an exported row does not prove blocker failure.
"""
import argparse
import json
import math
from pathlib import Path

import polars as pl

KEYS = ["s1", "mid"]
BINS = ["below_0.02", "band_0.02_to_0.99", "confident_0.99_plus"]


def _require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _validate_ids(df, columns, name):
    if df.select(pl.any_horizontal(
            pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")
            for c in columns).any()).item():
        raise ValueError(f"{name} contains null/empty entity IDs")


def normalize_queries(queries, universe=None):
    if universe is not None:
        _require_columns(queries, ["universe"], "queries (--universe supplied)")
        queries = queries.filter(pl.col("universe").cast(pl.String) == universe)
    elif "universe" in queries.columns and queries["universe"].n_unique() > 1:
        raise ValueError("queries contains multiple universes; use --universe or prefilter the file")
    if "is_query" in queries.columns:
        # CSV/TSV readers intentionally load strings; accept the same flags there.
        flags = queries["is_query"].cast(pl.String).str.to_lowercase().replace_strict(
            {"true": True, "false": False, "1": True, "0": False},
            default=None, return_dtype=pl.Boolean)
        if flags.null_count():
            raise ValueError("queries contains null/invalid is_query flags")
        queries = queries.filter(flags)
    if "s1" not in queries.columns and "entity_id" in queries.columns:
        queries = queries.rename({"entity_id": "s1"})
    _require_columns(queries, ["s1"], "queries")
    columns = [c for c in ("s1", "country", "region") if c in queries.columns]
    queries = queries.select(pl.col(c).cast(pl.String) for c in columns)
    if not queries.height:
        raise ValueError("queries must contain at least one S1 entity")
    _validate_ids(queries, ["s1"], "queries")
    if queries["s1"].n_unique() != queries.height:
        raise ValueError("queries contains duplicate S1 IDs; filter to one query universe")
    return queries


def truth_pairs_from_gt(gt, queries):
    """Vectorized equivalent of evaluate.truth_map_from_gt for this universe."""
    _require_columns(gt, ["source1_entity_id", "matched_entity_ids"], "truth")
    rows = (gt.select(pl.col("source1_entity_id").cast(pl.String).alias("s1"),
                      pl.col("matched_entity_ids").cast(pl.String))
            .join(queries.select("s1"), on="s1", how="semi"))
    if rows["s1"].n_unique() != rows.height:
        raise ValueError("truth contains duplicate query S1 rows")
    missing = queries.select("s1").join(rows.select("s1"), on="s1", how="anti")
    if missing.height:
        raise ValueError(f"truth has no row for {missing.height} queries; "
                         "missing labels cannot be interpreted as singletons")
    pairs = (rows.filter(pl.col("matched_entity_ids").is_not_null()
                         & (pl.col("matched_entity_ids") != ""))
             .with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids", empty_as_null=True)
             .rename({"matched_entity_ids": "mid"}).unique())
    _validate_ids(pairs, KEYS, "truth")
    if pairs["mid"].n_unique() != pairs.height:
        raise ValueError("truth assigns a pool record to multiple query S1 entities")
    return pairs


def normalize_scored(df, queries, name):
    _require_columns(df, [*KEYS, "p"], name)
    # Preserve Float32 comparison semantics used by model.assign/stage2: casting
    # stored Float32(.02) to Float64 before comparison moves it below the floor.
    probability_dtype = df.schema["p"] if df.schema["p"] in (pl.Float32, pl.Float64) else pl.Float64
    df = df.select(*(pl.col(c).cast(pl.String) for c in KEYS),
                   pl.col("p").cast(probability_dtype))
    n_input = df.height
    df = df.join(queries.select("s1"), on="s1", how="semi")
    _validate_ids(df, KEYS, name)
    if df.select((pl.col("p").is_null() | ~pl.col("p").is_finite()
                  | ~pl.col("p").is_between(0.0, 1.0)).any()).item():
        raise ValueError(f"{name} contains null, non-finite, or out-of-range probabilities")
    if df.select(pl.struct(KEYS).n_unique()).item() != df.height:
        raise ValueError(f"{name} contains duplicate (s1, mid) pairs")
    return df, {"rows_input": n_input, "rows_in_query_universe": df.height,
                "rows_outside_query_universe": n_input - df.height}


def accepted_pairs(final, threshold):
    """Match model.assign, including its deterministic ownership tie break."""
    return (final.sort(["p", "s1"], descending=[True, False])
            .unique("mid", keep="first", maintain_order=True)
            .filter(pl.col("p") >= threshold).select(KEYS))


def _f05_expr(n_pred="n_pred", n_tp="n_tp"):
    return (pl.when(pl.col("n_true") == 0)
            .then((pl.col(n_pred) == 0).cast(pl.Float64))
            .otherwise(1.25 * pl.col(n_tp) /
                       (0.25 * pl.col("n_true") + pl.col(n_pred))))


def _counts_by_s1(df, name):
    return df.group_by("s1").agg(pl.len().cast(pl.Int64).alias(name))


def entity_metrics(queries, truth, accepted):
    true_accepted = accepted.join(truth, on=KEYS, how="semi")
    counts = queries
    for pairs, name in ((truth, "n_true"), (accepted, "n_pred"),
                        (true_accepted, "n_tp")):
        counts = counts.join(_counts_by_s1(pairs, name), on="s1", how="left")
    return (counts.with_columns(pl.col("n_true", "n_pred", "n_tp").fill_null(0))
            .with_columns(_f05_expr().alias("baseline_f05")))


def repair_oracle(metrics, accepted, truth, eligible_mids, name):
    """Repair every eligible record coherently, then update both affected owners.

    The count deltas avoid materializing another full prediction set and avoid
    Python dictionaries/sets for millions of pairs.
    """
    eligible = eligible_mids.select("mid").unique()
    removed = accepted.join(eligible, on="mid", how="semi")
    added = truth.join(eligible, on="mid", how="semi")
    removed_labeled = removed.join(
        truth.with_columns(pl.lit(True).alias("is_true")), on=KEYS, how="left")
    delta = pl.concat([
        removed_labeled.select("s1", pl.lit(-1, dtype=pl.Int64).alias("d_pred"),
                               (-pl.col("is_true").fill_null(False).cast(pl.Int64)).alias("d_tp")),
        added.select("s1", pl.lit(1, dtype=pl.Int64).alias("d_pred"),
                     pl.lit(1, dtype=pl.Int64).alias("d_tp")),
    ]).group_by("s1").agg(pl.col("d_pred").sum(), pl.col("d_tp").sum())
    scored = (metrics.join(delta, on="s1", how="left")
              .with_columns(pl.col("d_pred", "d_tp").fill_null(0))
              .with_columns((pl.col("n_pred") + pl.col("d_pred")).alias("new_n_pred"),
                            (pl.col("n_tp") + pl.col("d_tp")).alias("new_n_tp"))
              .with_columns(_f05_expr("new_n_pred", "new_n_tp").alias(name)))
    changed_records = pl.concat([
        removed.join(added, on=KEYS, how="anti").select("mid"),
        added.join(removed, on=KEYS, how="anti").select("mid"),
    ]).unique().height
    total = float(scored[name].mean())
    report = {"eligible_records": eligible.height,
              "eligible_true_records": added.height,
              "changed_records": changed_records,
              "affected_entities": scored.filter(
                  (pl.col("d_pred") != 0) | (pl.col("d_tp") != 0)).height,
              "improved_entities": scored.filter(pl.col(name) > pl.col("baseline_f05")).height,
              "macro_f05": total,
              "delta_macro_f05": total - float(metrics["baseline_f05"].mean())}
    return report, scored.select("s1", name)


def _group_scores(metrics, columns, oracle_names):
    if not all(c in metrics.columns for c in columns):
        return []
    grouped = metrics.group_by(columns).agg(
        pl.len().alias("queries"), pl.col("baseline_f05").mean().alias("macro_f05"),
        *((pl.col(c) - pl.col("baseline_f05")).mean().alias(c + "_delta")
          for c in oracle_names))
    return grouped.sort(columns, nulls_last=True).to_dicts()


def audit(stage1, final, gt, queries, threshold=0.725, universe=None):
    """Return a JSON-serializable audit; never alter model/prediction artifacts."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    query_selection = {"rows_input": queries.height,
                       "is_query_filter_applied": "is_query" in queries.columns,
                       "selected_universe": universe}
    queries = normalize_queries(queries, universe)
    query_selection["selected_queries"] = queries.height
    truth = truth_pairs_from_gt(gt, queries)
    stage1, stage1_coverage = normalize_scored(stage1, queries, "stage1")
    final, final_coverage = normalize_scored(final, queries, "final")
    accepted = accepted_pairs(final, threshold)
    metrics = entity_metrics(queries, truth, accepted)
    true_accepted = accepted.join(truth, on=KEYS, how="semi")
    missing_stage1 = truth.join(stage1.select(KEYS), on=KEYS, how="anti")
    missing_final = truth.join(final.select(KEYS), on=KEYS, how="anti")
    false_accepted = accepted.join(truth, on=KEYS, how="anti")
    stage1 = stage1.with_columns(
        pl.when(pl.col("p") < 0.02).then(pl.lit(BINS[0]))
        .when(pl.col("p") < 0.99).then(pl.lit(BINS[1]))
        .otherwise(pl.lit(BINS[2])).alias("bin"))
    rows = (stage1.join(final.rename({"p": "final_p"}), on=KEYS, how="left")
            .join(truth.with_columns(pl.lit(True).alias("is_true")), on=KEYS, how="left")
            .join(accepted.with_columns(pl.lit(True).alias("accepted")), on=KEYS, how="left")
            .with_columns(pl.col("is_true", "accepted").fill_null(False)))
    summaries = {}
    for bin_name in BINS:
        b = rows.filter(pl.col("bin") == bin_name)
        summaries[bin_name] = {
            "pairs": b.height, "records": b["mid"].n_unique(), "entities": b["s1"].n_unique(),
            "true_pairs": int(b["is_true"].sum()),
            "final_rows_present": b["final_p"].len() - b["final_p"].null_count(),
            "final_rows_missing": b["final_p"].null_count(),
            "final_above_threshold_before_ownership": b.filter(pl.col("final_p") >= threshold).height,
            "accepted_pairs": int(b["accepted"].sum()),
            "true_positives": b.filter(pl.col("is_true") & pl.col("accepted")).height,
            "false_positives": b.filter(~pl.col("is_true") & pl.col("accepted")).height,
            "false_negatives": b.filter(pl.col("is_true") & ~pl.col("accepted")).height,
        }
    del rows
    not_in_stage1 = final.join(stage1.select(KEYS), on=KEYS, how="anti")
    eligible_sets = [
        (name, stage1.filter(pl.col("bin") == name).select("mid")) for name in BINS
    ]
    eligible_sets.extend([
        ("retained_0.02_plus", stage1.filter(pl.col("p") >= 0.02).select("mid")),
        ("final_pairs_without_stage1_score", not_in_stage1.select("mid")),
        ("missing_stage1_true_pair_ceiling", missing_stage1.select("mid")),
    ])
    oracles = {}
    for name, eligible in eligible_sets:
        report, score = repair_oracle(metrics, accepted, truth, eligible, name)
        report["eligibility_uses_labels"] = name == "missing_stage1_true_pair_ceiling"
        report["eligible_true_pairs_missing_stage1"] = missing_stage1.join(
            eligible.unique(), on="mid", how="semi").height
        oracles[name] = report
        metrics = metrics.join(score, on="s1", how="left")
    coverage = {"stage1": stage1_coverage, "final": final_coverage,
                "stage1_queries_with_rows": stage1["s1"].n_unique(),
                "final_queries_with_rows": final["s1"].n_unique(),
                "stage1_pairs_missing_final": stage1.join(final.select(KEYS), on=KEYS, how="anti").height,
                "final_pairs_without_stage1_score": not_in_stage1.height,
                "true_pairs_missing_stage1": missing_stage1.height,
                "true_pairs_missing_final": missing_final.height,
                "stage1_true_pair_coverage": (1 - missing_stage1.height / truth.height) if truth.height else None,
                "final_true_pair_coverage": (1 - missing_final.height / truth.height) if truth.height else None}
    baseline = {"queries": queries.height, "true_pairs": truth.height,
                "singleton_queries": metrics.filter(pl.col("n_true") == 0).height,
                "accepted_pairs": accepted.height, "true_positives": true_accepted.height,
                "false_positives": false_accepted.height,
                "false_negatives": truth.height - true_accepted.height,
                "wrong_owner_records": false_accepted.join(truth.select("mid"), on="mid", how="semi").height,
                "accepted_records_without_query_owner": false_accepted.join(truth.select("mid"), on="mid", how="anti").height,
                "macro_f05": float(metrics["baseline_f05"].mean())}
    return {
        "schema_version": 1, "threshold": threshold,
        "definitions": {
            "assignment": "Highest final p per mid, lexical s1 tie-break, accepted at p >= threshold; query S1 only.",
            "score_bins": "Pair-disjoint bins, but record eligibility is ANY stage1 pair in the bin, so record sets overlap.",
            "false_negatives": "True pairs not assigned to their true owner, including ownership losses.",
            "oracles": "Optimistic per-record truth repairs inside this query universe; remove the old owner and add the true owner even if its pair was not exported. Gains can overlap and must not be added.",
            "missing_retrieval": "Missing from the supplied stage1 export, which can reflect blocking, pruning, or export filtering; this is not necessarily blocker recall.",
            "geography": "Scores and oracle deltas use supplied labeled queries only. No inference of unlabeled France or leaderboard scores.",
            "eligibility": "All oracle scopes are label-free except missing_stage1_true_pair_ceiling, whose label-defined scope is explicitly optimistic.",
            "truth_scope": "Every query requires an explicit GT row. Records with no true owner among query S1 are unmatched, including orphans whose owners lie outside the universe.",
        },
        "query_selection": query_selection,
        "baseline": baseline, "coverage": coverage, "stage1_bins": summaries,
        "oracles": oracles,
        "by_country": _group_scores(metrics, ["country"], oracles),
        "by_region": _group_scores(metrics, ["country", "region"] if "country" in queries.columns else ["region"], oracles),
    }


def _read_projected(path, columns):
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pl.scan_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        frame = pl.scan_csv(path, separator="\t" if suffix == ".tsv" else ",",
                            infer_schema=False)
    else:
        raise ValueError(f"Unsupported input format {suffix!r}: use Parquet, CSV, or TSV")
    schema = frame.collect_schema()
    return frame.select(c for c in columns if c in schema).collect(engine="streaming")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("stage1", "final", "truth", "queries", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threshold", type=float, default=0.725)
    parser.add_argument("--universe", help="Select one value from the query file's universe column")
    parser.add_argument("--label", help="Descriptive report label, e.g. 'local v2a diagnostic; not V6'")
    args = parser.parse_args(argv)
    inputs = {k: str(Path(getattr(args, k)).resolve()) for k in ("stage1", "final", "truth", "queries")}
    output = Path(args.out).resolve()
    if str(output).casefold() in {p.casefold() for p in inputs.values()}:
        parser.error("--out must not overwrite an input file")
    try:
        report = audit(
            _read_projected(args.stage1, [*KEYS, "p"]),
            _read_projected(args.final, [*KEYS, "p"]),
            _read_projected(args.truth, ["source1_entity_id", "matched_entity_ids"]),
            _read_projected(args.queries, ["s1", "entity_id", "country", "region", "is_query", "universe"]),
            args.threshold, args.universe)
    except (ValueError, pl.exceptions.PolarsError) as exc:
        parser.error(str(exc))
    report["inputs"] = inputs
    if args.label:
        report["label"] = args.label
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Baseline macro F0.5: {report['baseline']['macro_f05']:.8f}; report: {output}")
    for name, result in report["oracles"].items():
        print(f"  {name}: optimistic delta {result['delta_macro_f05']:+.8f}; "
              f"{result['changed_records']:,} changed records")
    print("Oracle gains can overlap. Missing exported rows do not establish blocker failure.")


if __name__ == "__main__":
    main()
