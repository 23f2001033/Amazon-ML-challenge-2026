"""Opt-in raw-text CE export, including the stage-1 p >= .99 population.

Run ``export --help`` and ``merge --help`` for explicit-path commands. An export
directory contains the complete ``manifest.parquet``, ``pairs_00000.parquet``
inference shards, ``reused_ce.parquet``, and ``metadata.json``. The manifest
retains stage-1 ``p``; no production candidates or scores are modified.

Reuse only scores from the SAME checkpoint and raw-text serialization. Old
probabilities can be reused, but their unavailable raw logits remain null.
Omit --existing-ce to obtain actual logits for every retained pair.
"""
import argparse
import json
from pathlib import Path

import polars as pl

KEYS = ["s1", "mid"]


def validate_keys(frame, label):
    if any(frame[c].null_count() for c in KEYS):
        raise ValueError(f"{label}: null pair IDs")
    if any((frame[c].cast(pl.String).str.strip_chars() == "").any() for c in KEYS):
        raise ValueError(f"{label}: empty pair IDs")
    if frame.select(KEYS).is_duplicated().any():
        raise ValueError(f"{label}: duplicate pair IDs")


def read_scores(path):
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    required = {*KEYS, "p_ce"}
    if not required <= set(schema):
        raise ValueError("CE scores require s1, mid, p_ce columns")
    raw_logit = pl.col("ce_logit") if "ce_logit" in schema else pl.lit(None)
    frame = scan.select(
        *[pl.col(c).cast(pl.String) for c in KEYS],
        pl.col("p_ce").cast(pl.Float64),
        raw_logit.cast(pl.Float32).alias("ce_logit"),
    ).collect(engine="streaming")
    validate_keys(frame, "CE scores")
    if frame.filter(
        pl.col("p_ce").is_null()
        | ~pl.col("p_ce").is_finite()
        | ~pl.col("p_ce").is_between(0, 1)
        | (pl.col("ce_logit").is_not_null() & ~pl.col("ce_logit").is_finite())
    ).height:
        raise ValueError("CE scores contain invalid probabilities or logits")
    # A small tolerance permits saved/half-precision rounding while rejecting
    # mixed checkpoints or accidentally paired probability/logit columns.
    expected_p = 1 / (1 + (-pl.col("ce_logit").cast(pl.Float64)).exp())
    if frame.filter(pl.col("ce_logit").is_not_null()
                    & ((expected_p - pl.col("p_ce")).abs() > 1e-3)).height:
        raise ValueError("CE probabilities disagree with sigmoid(raw logits)")
    return frame


def _raw_records(paths, ids, key, with_country=False):
    scans = []
    for path in paths:
        scan = pl.scan_parquet(path)
        schema = scan.collect_schema()
        required = {"entity_id", "business_name", "business_address"}
        if not required <= set(schema):
            raise ValueError("Raw records require entity_id, business_name, business_address")
        columns = [pl.col("entity_id").cast(pl.String).alias(key)]
        columns.append(
            (
                pl.col("business_name").cast(pl.String).fill_null("")
                + " | "
                + pl.col("business_address").cast(pl.String).fill_null("")
            ).alias("text")
        )
        if with_country:
            columns.append(
                (pl.col("country") if "country" in schema else pl.lit(None))
                .cast(pl.String).alias("country")
            )
        scans.append(scan.select(columns).join(ids.lazy(), on=key, how="semi"))
    records = pl.concat(scans).collect(engine="streaming")
    if records[key].null_count() or records[key].is_duplicated().any():
        raise ValueError(f"Raw {key} records have null or duplicate selected IDs")
    missing = ids.join(records.select(key), on=key, how="anti").height
    if missing:
        raise ValueError(f"Missing raw {key} records for {missing:,} selected IDs")
    return records


def export_pairs(stage1, s1, s2, s3, out_dir, country=None, existing_ce=(),
                 min_p=0.02, shard_size=250_000):
    if not 0 <= min_p <= 1 or shard_size <= 0:
        raise ValueError("min_p must be in [0, 1]; shard_size must be positive")
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Export directory must be absent or empty")
    # Validate the whole stage-1 input before retaining pairs. Never deduplicate
    # silently: repeated IDs can conceal inconsistent scores or source joins.
    stage = pl.scan_parquet(stage1).select(
        *[pl.col(c).cast(pl.String) for c in KEYS], pl.col("p")
    ).collect(engine="streaming")
    if not stage.schema["p"].is_numeric():
        raise ValueError("Stage-1 probabilities must have a numeric dtype")
    validate_keys(stage, "Stage-1 predictions")
    if stage.filter(pl.col("p").is_null() | ~pl.col("p").is_finite()
                    | ~pl.col("p").is_between(0, 1)).height:
        raise ValueError("Stage-1 predictions contain invalid probabilities")
    # Keep original precision: float32(.02), promoted to float64, is slightly
    # below .02. Promotion before comparison would disagree with stage 1.
    wanted = stage.filter(pl.col("p") >= min_p)
    del stage
    left = _raw_records([s1], wanted.select("s1").unique(), "s1", with_country=True)
    if country is not None:
        left = left.filter(pl.col("country").str.strip_chars().str.to_lowercase()
                           == country.strip().lower())
        if not left.height:
            raise ValueError("Country selection has no retained S1 records; check raw country values")
    wanted = wanted.join(left.select("s1", "country"), on="s1", how="inner")
    # Validate raw IDs even for reused pairs, so the manifest is a verifiable
    # target universe rather than a list whose missing records were hidden.
    right = _raw_records([s2, s3], wanted.select("mid").unique(), "mid")
    wanted = wanted.sort(KEYS)
    reused = pl.DataFrame(schema={"s1": pl.String, "mid": pl.String,
                                  "p_ce": pl.Float64, "ce_logit": pl.Float32})
    if existing_ce:
        reused = pl.concat([read_scores(path) for path in existing_ce])
        validate_keys(reused, "Combined existing CE scores")
        reused = reused.join(wanted.select(KEYS), on=KEYS, how="semi")
    missing = wanted.join(reused.select(KEYS), on=KEYS, how="anti")
    pairs = (missing.select(KEYS)
             .join(left.select("s1", pl.col("text").alias("text_a")), on="s1", how="left")
             .join(right.select("mid", pl.col("text").alias("text_b")), on="mid", how="left")
             .sort(KEYS))
    if pairs["text_a"].null_count() or pairs["text_b"].null_count():
        raise ValueError("Text attachment unexpectedly produced null text")
    metadata = {
        "serialization": "raw business_name | business_address; null values become empty strings",
        "stage1": str(Path(stage1).resolve()), "country": country,
        "min_p": min_p, "retained_pairs": wanted.height,
        "confident_pairs": wanted.filter(pl.col("p") >= 0.99).height,
        "reused_pairs": reused.height, "inference_pairs": pairs.height,
        "shard_size": shard_size,
        "shards": [f"pairs_{i:05d}.parquet" for i in range((pairs.height + shard_size - 1) // shard_size)],
        "existing_ce": [str(Path(p).resolve()) for p in existing_ce],
        "reuse_assumption": "Existing scores use the same checkpoint and serialization; not automatically verifiable",
    }
    out.mkdir(parents=True, exist_ok=True)
    wanted.write_parquet(out / "manifest.parquet")
    reused.write_parquet(out / "reused_ce.parquet")
    for i, name in enumerate(metadata["shards"]):
        pairs.slice(i * shard_size, shard_size).write_parquet(out / name)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Retained {wanted.height:,}; confident {metadata['confident_pairs']:,}; "
          f"reused {reused.height:,}; inference {pairs.height:,}; shards {len(metadata['shards'])}")
    return metadata


def merge_scores(manifest, scores, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("CE output already exists")
    wanted = pl.read_parquet(manifest, columns=KEYS).with_columns(
        *[pl.col(c).cast(pl.String) for c in KEYS])
    validate_keys(wanted, "Manifest")
    if not scores:
        raise ValueError("Supply at least one score file (reused_ce.parquet may be empty)")
    merged = pl.concat([read_scores(path) for path in scores])
    validate_keys(merged, "Merged CE scores")
    absent = wanted.join(merged.select(KEYS), on=KEYS, how="anti").height
    extra = merged.select(KEYS).join(wanted, on=KEYS, how="anti").height
    if absent or extra:
        raise ValueError(f"CE coverage mismatch: {absent:,} missing and {extra:,} extra pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.sort(KEYS).write_parquet(output)
    print(f"Merged {merged.height:,} pairs with exact manifest coverage; "
          f"{merged['ce_logit'].null_count():,} unavailable raw logits")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export", help="Export every retained raw-text pair, including p >= .99")
    for name in ("stage1", "s1", "s2", "s3", "out-dir"):
        exp.add_argument(f"--{name}", required=True)
    exp.add_argument("--country", help="Case-insensitive exact raw S1 country value, e.g. France")
    exp.add_argument("--existing-ce", nargs="+", default=[], help="Same-model existing score files")
    exp.add_argument("--min-p", type=float, default=0.02)
    exp.add_argument("--shard-size", type=int, default=250_000)
    merge = commands.add_parser("merge", help="Validate exact manifest coverage and merge scores")
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--scores", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (export_pairs if command == "export" else merge_scores)(**args)


if __name__ == "__main__":
    main()
