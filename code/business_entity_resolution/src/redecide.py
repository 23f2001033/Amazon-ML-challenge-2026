"""Re-apply the decision rule to saved test predictions (no re-blocking, no re-scoring).

Usage: python redecide.py <config.json> <out_dir> [pred.parquet] [candidate_pairs.tsv]
  config keys: thr (global threshold), optional thr_by_country {country: thr}, decision ("thr"/"ef")
Reads data/features/test_pred.parquet (written by run_test.py; all pairs with p >= 0.02) and writes
<out_dir>/matching_results.tsv. The candidate set is unchanged, so <out_dir>/candidate_pairs.tsv is
a hard link to output/candidate_pairs.tsv.
"""
import json
import os
import sys

import polars as pl

from config import FEAT_DIR, OUTPUT_DIR, PREPARED_DIR
from model import assign, expected_f_select


def main(cfg_path, out_dir, pred_path=None, cand_path=None):
    cfg = json.load(open(cfg_path))
    os.makedirs(out_dir, exist_ok=True)
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    pred = (pl.read_parquet(pred_path or os.path.join(FEAT_DIR, "test_pred.parquet"))
              .join(s1.rename({"entity_id": "s1"}), on="s1", how="left"))
    if cfg.get("decision") == "ef":
        matches = expected_f_select(pred.select("s1", "mid", "p"))
    else:
        matches = assign(pred, cfg["thr"], cfg.get("thr_by_country"))
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1["entity_id"].to_list():
            ids = matches.get(s)
            f.write(f"{s}\t{','.join(sorted(ids)) if ids else ''}\n")
    link = os.path.join(out_dir, "candidate_pairs.tsv")
    if not os.path.exists(link):
        os.link(cand_path or os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), link)
    by = {}
    for s, c in s1.iter_rows():
        by.setdefault(c, [0, 0, 0])
        by[c][0] += 1
        by[c][1] += bool(matches.get(s))
        by[c][2] += len(matches.get(s, ()))
    for c, (n, ne, k) in sorted(by.items()):
        print(f"{c}: {n:,} S1, predicted singleton rate {1 - ne / n:.4f}, matches/S1 {k / n:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
