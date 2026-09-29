"""France-only additions on top of a stage-2 submission: add French band pairs that BOTH French specialists
(CE 3-FR, CE 4-FR) strongly accept, when the record has a real address, the stack rejected the pair, and the
record is not already assigned to another S1. Everything else (and US / India) is left exactly as the stack decided.
Rationale: hand review showed the specialists recover typos, acronyms and suffix variants the stack misses,
while their own blind spots (empty-address / pseudo-name records, absent from their synthetic training) only
affect REJECTIONS, which this rule never applies.

  python france_add.py <s2name> <stack_thr> <ce_min> <out_dir> <cand_src_dir> [show=0]
"""
import os
import sys

import polars as pl

from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from model import assign

C = "/home/ubuntu/er/ce"


def main(s2name, thr, ce_min, out_dir, cand_src, show="0"):
    thr, ce_min = float(thr), float(ce_min)
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{s2name}.parquet")).select("s1", "mid", "p", "country")
    m = assign(pred.select("s1", "mid", "p"), thr)                      # the stack's own decision
    taken = {x for ms in m.values() for x in ms}
    a = pl.read_parquet(f"{C}/ce3frv6/band_test_ce3fr.parquet").rename({"p_ce": "a"})
    b = pl.read_parquet(f"{C}/ce4frv6/band_test_ce4fr.parquet").rename({"p_ce": "b"})
    null_addr = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"test_s{i}.parquet"), columns=["entity_id", "a_null"]) for i in (2, 3)]).rename(
        {"entity_id": "mid"})
    cand = (pred.filter((pl.col("country") == "France") & (pl.col("p") < thr)).join(a, on=["s1", "mid"]).join(b, on=["s1", "mid"])
                .join(null_addr, on="mid", how="left")
                .filter((pl.col("a") >= ce_min) & (pl.col("b") >= ce_min) & (pl.col("a_null").fill_null(0) != 1))
                .filter(~pl.col("mid").is_in(list(taken)))
                .with_columns(((pl.col("a") + pl.col("b")) / 2).alias("q"))
                .sort(["q", "s1"], descending=[True, False]).unique("mid", keep="first"))
    for s, x in cand.select("s1", "mid").iter_rows():
        m.setdefault(s, set()).add(x)
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    fr = s1.filter(pl.col("country") == "France")["entity_id"].to_list()
    print(f"ce_min {ce_min}: France additions {cand.height:,} ({cand.height / len(fr):.4f}/S1) | France matches/S1 "
          f"{sum(len(m.get(s, ())) for s in fr) / len(fr):.3f}, empty {sum(1 for s in fr if not m.get(s)) / len(fr):.4f}")
    if int(show):
        raw = pl.concat([pl.read_parquet(os.path.join(PARQUET_DIR, f"test_s{i}.parquet"), columns=["entity_id", "business_name", "business_address"]) for i in (1, 2, 3)])
        ex = cand.sample(min(int(show), cand.height), seed=4).join(raw.rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1"}), on="s1").join(
            raw.rename({"entity_id": "mid", "business_name": "n2", "business_address": "a2"}), on="mid")
        for r in ex.select("p", "a", "b", "n1", "a1", "n2", "a2").rows():
            print(f"  {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} | {r[3]} | {(r[4] or '')[:44]}\n                 -> {r[5]} | {(r[6] or '')[:44]}")
    if out_dir != "-":
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s in s1["entity_id"].to_list():
                x = m.get(s)
                f.write(f"{s}\t{','.join(sorted(x)) if x else ''}\n")
        dst = os.path.join(out_dir, "candidate_pairs.tsv")
        if os.path.exists(dst):
            os.remove(dst)
        os.link(os.path.join(cand_src, "candidate_pairs.tsv"), dst)


if __name__ == "__main__":
    main(*sys.argv[1:7])
