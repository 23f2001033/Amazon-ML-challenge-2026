"""Inference on the TEST set -> output/matching_results.tsv + output/candidate_pairs.tsv.

Usage: python run_test.py <model.txt> <config.json> [out_dir] [tag]
  e.g. python run_test.py ../models/lgbm_v1.txt ../models/v1_config.json            # shipped v1 model
       python run_test.py ../../../data/artifacts/lgbm_v2a.txt ../../../data/artifacts/result_v2a.json                           ../../../output_v2a v2a
Memory-bounded: S1 queries are processed in region batches (<= BATCH_S1 queries). For each batch:
blocking -> candidates streamed into candidate_pairs.tsv -> features -> LightGBM scores. Only
scores >= P_FLOOR are kept for the final (global) decision step, which applies the partition
constraint and the decision rule chosen on validation.
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import polars as pl

from blocking import generate
from config import ARTIFACT_DIR, FEAT_DIR, OUTPUT_DIR, PREPARED_DIR
from features import REC_COLS, build_chunked
from model import assign, expected_f_select
from run_dev import BLOCK_COLS, band_saver, load_prep, log, split_stats

BATCH_S1 = 100_000
P_FLOOR = 0.02


def region_batches(q, batch_s1=None):
    """Group regions (largest first) into batches of at most batch_s1 queries."""
    batch_s1 = batch_s1 or BATCH_S1
    sizes = q.group_by("region").len().sort("len", descending=True).iter_rows()
    batches, cur, n = [], [], 0
    for reg, size in sizes:
        if cur and n + size > batch_s1:
            batches.append(cur)
            cur, n = [], 0
        cur.append(reg)
        n += size
    if cur:
        batches.append(cur)
    return batches


def main(model_path=None, cfg_path=None, out_dir=None, tag="v1"):
    cfg = json.load(open(cfg_path or os.path.join(ARTIFACT_DIR, "dev_result.json")))
    model_path = model_path or os.path.join(ARTIFACT_DIR, "lgbm_dev.txt")
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    stats = split_stats("test")
    feats = cfg["features"]
    m = lgb.Booster(model_file=model_path)
    s1_blk = load_prep("test_s1", BLOCK_COLS)
    s1_rec = load_prep("test_s1", REC_COLS)
    all_ids = s1_blk["entity_id"].to_list()
    save = band_saver("test")

    def score(part):
        part = part.with_columns(pl.Series("p", m.predict(part.select(feats).to_numpy(), num_threads=0)))
        save(part)
        return part.select("s1", "mid", "p").filter(pl.col("p") >= P_FLOOR)
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    seen = set()
    preds = []
    with open(cand_path, "w", encoding="utf-8", newline="\n") as cf:
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in sorted(s1_blk["country"].unique().to_list()):
            qc = s1_blk.filter(pl.col("country") == country)
            # one resident copy of this country's pool, used for blocking AND features
            pool = (pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f"{n}.parquet"))
                               for n in ("test_s2", "test_s3")])
                      .select(REC_COLS).filter(pl.col("country") == country).collect(engine="streaming"))
            for bi, regs in enumerate(region_batches(qc)):
                t = time.time()
                q = qc.filter(pl.col("region").is_in(regs))
                cand = generate(q, pool, log=log, workers=int(os.environ.get("ER_WORKERS", "2")))
                for s, mids in cand.group_by("s1").agg("mid").iter_rows():
                    cf.write(f"{s}\t{','.join(sorted(set(mids)))}\n")
                    seen.add(s)
                pool_rec = pool.filter(pl.col("entity_id").is_in(cand["mid"].unique().implode()))
                preds.append(build_chunked(cand, s1_rec, pool_rec, s1_all=s1_rec, log=lambda *a: None,
                                           extra=score, stats=stats))
                log(f"[{country} batch {bi}] regions={len(regs)} queries={q.height:,} "
                    f"candidates={cand.height:,} ({cand.height / max(1, q.height):.1f}/S1) "
                    f"in {time.time() - t:.0f}s")
                del cand, pool_rec
                gc.collect()
            del pool
        for s in all_ids:  # S1 entities with no candidate at all still need a row
            if s not in seen:
                cf.write(f"{s}\t\n")
    pred = pl.concat(preds)
    pred.write_parquet(os.path.join(FEAT_DIR, "test_pred.parquet" if tag == "v1" else f"test_pred_{tag}.parquet"))
    matches = expected_f_select(pred) if cfg.get("decision") == "ef" else assign(pred, cfg["thr"])
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in all_ids:
            ids = matches.get(s)
            f.write(f"{s}\t{','.join(sorted(ids)) if ids else ''}\n")
    n_nonempty = sum(1 for s in all_ids if matches.get(s))
    n_pairs = sum(len(v) for v in matches.values())
    log(f"written {len(all_ids):,} S1 rows; {n_nonempty:,} with matches "
        f"(predicted singleton rate {1 - n_nonempty / len(all_ids):.4f}); "
        f"{n_pairs:,} matched pairs ({n_pairs / len(all_ids):.2f}/S1); decision={cfg.get('decision')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
