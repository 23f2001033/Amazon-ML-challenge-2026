"""Convert raw challenge TSVs to Parquet once, so every EDA/modeling step loads fast.

Raw TSVs are read with quoting disabled: business names contain quotes/apostrophes
that must be preserved verbatim, and the only delimiter is TAB.
"""
import csv
import os
import time

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
OUT = os.path.join(ROOT, "data", "parquet")

FILES = {
    "train_s1": "train/train_source1.tsv",
    "train_s2": "train/train_source2.tsv",
    "train_s3": "train/train_source3.tsv",
    "train_gt": "train/train_ground_truth.tsv",
    "test_s1": "test/test_source1.tsv",
    "test_s2": "test/test_source2.tsv",
    "test_s3": "test/test_source3.tsv",
}


def main():
    for name, rel in FILES.items():
        t = time.time()
        path = os.path.join(RAW, rel)
        # Count physical lines to verify no rows are merged/split by parsing.
        with open(path, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f) - 1
        df = pl.read_csv(
            path, separator="\t", quote_char=None, infer_schema=False,
            empty_string_is_null=True, encoding="utf8",
        )
        df.write_parquet(os.path.join(OUT, f"{name}.parquet"), compression="zstd")
        print(f"{name}: rows={df.height} lines={n_lines} cols={df.columns} "
              f"match={df.height == n_lines} ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
