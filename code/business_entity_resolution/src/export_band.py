"""Export uncertain-band pairs (stage-1 p in [0.02, 0.99)) with raw + canonical text for training an
external cross-encoder (S2-XENC). Usage: python export_band.py <val|test> <tag> <out.parquet>"""
import os
import sys

import polars as pl

from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR

COLS = ["entity_id", "country", "name", "address", "n_canon", "a_tokens"]


def main(kind, tag, out):
    split = "train" if kind == "val" else "test"
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_{tag}.parquet"))
    band = pred.filter((pl.col("p") >= 0.02) & (pl.col("p") < 0.99))
    keep = ["s1", "mid", "p"] + (["y"] if "y" in band.columns else [])
    band = band.select(keep).rename({"p": "p_stage1"})
    ids = pl.concat([band["s1"], band["mid"]]).unique()
    recs = (pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f"{split}_s{i}.parquet")) for i in (1, 2, 3)])
              .select(COLS).filter(pl.col("entity_id").is_in(ids.implode())).collect(engine="streaming"))
    a = recs.rename({"entity_id": "s1", "name": "s1_name", "address": "s1_address", "n_canon": "s1_name_canon",
                     "a_tokens": "s1_addr_canon"})
    b = recs.drop("country").rename({"entity_id": "mid", "name": "c_name", "address": "c_address",
                                     "n_canon": "c_name_canon", "a_tokens": "c_addr_canon"})
    band = band.join(a, on="s1", how="left").join(b, on="mid", how="left")
    band.write_parquet(out, compression="zstd")
    print(kind, band.height, "pairs", band.columns)
    if "y" in band.columns:
        print("positives:", int(band["y"].sum()), "by country:", band.group_by("country").len().to_dicts())


if __name__ == "__main__":
    main(*sys.argv[1:])
