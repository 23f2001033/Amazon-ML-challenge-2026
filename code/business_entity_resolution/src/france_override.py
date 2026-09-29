"""France-only override on top of a stage-2 test prediction: where stage 2 ACCEPTS a French band pair
but BOTH French-specialist cross-encoders (CE 3-FR, CE 4-FR; trained with guaranteed-label French pairs)
say it is a different business, reject it. US / India are untouched.

  python france_override.py <s2name> <thr> <ce_max> <out_dir> [<cand_src_dir>]
  e.g. python france_override.py s2_v6_rich_ce_ce2_ce3fr_ce4fr 0.70 0.05 ~/er/sub_v6fr3_ovr ~/er/sub_v6fr3
"""
import os
import sys

import polars as pl

from config import FEAT_DIR, PREPARED_DIR
from model import assign

C = "/home/ubuntu/er/ce"


def main(s2name, thr, ce_max, out_dir, cand_src=None):
    thr, ce_max = float(thr), float(ce_max)
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{s2name}.parquet")).select("s1", "mid", "p", "country")
    a = pl.read_parquet(f"{C}/ce3frv6/band_test_ce3fr.parquet").rename({"p_ce": "ce3fr"})
    b = pl.read_parquet(f"{C}/ce4frv6/band_test_ce4fr.parquet").rename({"p_ce": "ce4fr"})
    pred = pred.join(a, on=["s1", "mid"], how="left").join(b, on=["s1", "mid"], how="left")
    veto = (pl.col("country") == "France") & (pl.col("p") >= thr) & (pl.col("p") < 0.99) & (pl.col("ce3fr") < ce_max) & (pl.col("ce4fr") < ce_max)
    n_veto = pred.filter(veto).height
    pred = pred.with_columns(pl.when(veto).then(0.0).otherwise(pl.col("p")).alias("p"))
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    m = assign(pred.select("s1", "mid", "p"), thr)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1["entity_id"].to_list():
            x = m.get(s)
            f.write(f"{s}\t{','.join(sorted(x)) if x else ''}\n")
    if cand_src:
        src, dst = os.path.join(cand_src, "candidate_pairs.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    fr = s1.filter(pl.col("country") == "France")["entity_id"].to_list()
    print(f"France vetoed pairs: {n_veto:,} ({n_veto / len(fr):.4f}/S1) | France matches/S1 {sum(len(m.get(s, ())) for s in fr) / len(fr):.3f}, "
          f"empty {sum(1 for s in fr if not m.get(s)) / len(fr):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
