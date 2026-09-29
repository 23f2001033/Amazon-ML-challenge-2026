"""Shared loaders for EDA scripts."""
import os
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "data", "parquet")
FIG = os.path.join(ROOT, "eda", "figures")
OUT = os.path.join(ROOT, "eda", "outputs")


def load(name, lazy=False):
    p = os.path.join(PQ, f"{name}.parquet")
    return pl.scan_parquet(p) if lazy else pl.read_parquet(p)


def gt_pairs():
    """Ground truth exploded to one row per (s1, matched id), with source tag."""
    gt = load("train_gt")
    return (gt.drop_nulls("matched_entity_ids")
              .with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"})
              .with_columns(pl.col("mid").str.slice(0, 2).alias("src")))
