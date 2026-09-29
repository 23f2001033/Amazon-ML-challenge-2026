"""Official-metric reimplementation and validation universes.

Metric: F_beta (beta=0.5) per Source-1 entity, macro-averaged over ALL S1 entities in the
evaluation set. Singleton (no true match): 1.0 iff prediction empty, else 0.0.

Universes (D-020): train S1 entities are grouped by (country, region) blocks; whole blocks are
assigned to VAL or TRAIN so geography never leaks between them. Inside a universe we keep
KEEP_FRAC of S1 as queries; the dropped S1s' S2/S3 records stay in the pool as orphans, which
raises distractor density from ~1.2 to ~2.3 per S1 like the test set (D-010).
"""
import hashlib

import numpy as np
import polars as pl

BETA2 = 0.25
KEEP_FRAC = 0.81


def f05(pred, truth):
    """Per-entity F0.5 for two python sets."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred_map, truth_map, s1_ids):
    """pred_map/truth_map: {s1: set(ids)}; s1_ids: every S1 in the evaluation set."""
    scores = np.array([f05(pred_map.get(s, set()), truth_map.get(s, set())) for s in s1_ids])
    return float(scores.mean()), scores


def truth_map_from_gt(gt, s1_ids=None):
    g = gt if s1_ids is None else gt.filter(pl.col("source1_entity_id").is_in(list(s1_ids)))
    out = {}
    for s, m in g.iter_rows():
        out[s] = set(m.split(",")) if m else set()
    return out


def _h(x, seed):
    return int(hashlib.md5(f"{seed}:{x}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def make_universes(s1_prep, val_frac=0.08, train_frac=0.10, seed=42):
    """Assign S1 entities to 'val' / 'train' / 'unused' by (country, region) block, per country,
    then keep KEEP_FRAC of each universe as queries. A block joins a universe only if the
    universe stays within 1.5x its target share of that country (no single giant state).

    Returns DataFrame(entity_id, universe, is_query).
    """
    blocks = s1_prep.group_by(["country", "region"]).len()
    assign = []
    for country in sorted(blocks["country"].unique().to_list()):
        b = blocks.filter(pl.col("country") == country)
        tot = b["len"].sum()
        order = sorted(b.iter_rows(), key=lambda r: _h(f"{r[0]}|{r[1]}", seed))
        filled = {"val": 0, "train": 0}
        for _, region, n in order:
            uni = "unused"
            for u, frac in (("val", val_frac), ("train", train_frac)):
                if filled[u] < frac * tot and filled[u] + n <= 1.5 * frac * tot:
                    uni = u
                    filled[u] += n
                    break
            assign.append((country, region, uni))
    amap = pl.DataFrame(assign, schema=["country", "region", "universe"], orient="row")
    out = s1_prep.join(amap, on=["country", "region"], how="left").select("entity_id", "universe")
    out = out.with_columns(pl.col("entity_id").map_elements(lambda e: _h(e, seed + 1) < KEEP_FRAC,
                                                            return_dtype=pl.Boolean).alias("is_query"))
    return out
