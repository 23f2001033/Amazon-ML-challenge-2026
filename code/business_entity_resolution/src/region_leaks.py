"""Learn region-leak pairs for blocking (v6, blocking.py pass F) from TRAIN labels only.

A pair (country, R, R') is kept when the true matches of S1 in region R land in pool records
tagged R' at rate >= MIN_RATE with >= MIN_N pairs, or >= BIG_N pairs at rate >= BIG_RATE.
Writes artifacts/region_leaks.json.   python region_leaks.py
"""
import json
import os

import polars as pl

from config import ARTIFACT_DIR, PARQUET_DIR, PREPARED_DIR

MIN_RATE, MIN_N, BIG_RATE, BIG_N = 0.02, 30, 0.005, 500


def main():
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country", "region"])
    pool = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"train_{x}.parquet"), columns=["entity_id", "region"])
                      for x in ("s2", "s3")]).rename({"entity_id": "mid", "region": "m_region"})
    gt = (pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")).drop_nulls("matched_entity_ids")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .rename({"source1_entity_id": "entity_id", "matched_entity_ids": "mid"}))
    g = gt.join(s1, on="entity_id").join(pool, on="mid").filter(pl.col("region") != "")
    tot = g.group_by("country", "region").len().rename({"len": "tot"})
    x = (g.filter((pl.col("m_region") != pl.col("region")) & (pl.col("m_region") != ""))
           .group_by("country", "region", "m_region").len().join(tot, on=["country", "region"])
           .with_columns((pl.col("len") / pl.col("tot")).alias("rate"))
           .filter(((pl.col("rate") >= MIN_RATE) & (pl.col("len") >= MIN_N)) |
                   ((pl.col("rate") >= BIG_RATE) & (pl.col("len") >= BIG_N)))
           .sort("len", descending=True))
    print(x)
    json.dump({"rule": dict(MIN_RATE=MIN_RATE, MIN_N=MIN_N, BIG_RATE=BIG_RATE, BIG_N=BIG_N),
               "leaks": [[c, r, m] for c, r, m in x.select("country", "region", "m_region").iter_rows()],
               "evidence": x.rows()},
              open(os.path.join(ARTIFACT_DIR, "region_leaks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
