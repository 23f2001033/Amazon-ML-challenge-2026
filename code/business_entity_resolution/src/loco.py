"""Leave-one-country-out stage 2 (D-014 / E5): the France (zero-shot) proxy at stage-2 level.
Stage 2 config <cfg> (default w, the model France uses) is trained on the US band only and scores India, and trained
on India only and scores US. It is compared with the usual mixed-country out-of-fold model on the same test-like
evaluation (variant B), per country and over a wide threshold grid: how much F0.5 is lost zero-shot, and where the
best threshold moves when a country is unseen.
  ER_EVAL_B=1 python loco.py [cfg]      (same environment as s2_tune.py)"""
import json
import os
import sys

import numpy as np
import polars as pl

import s2_tune as S
from config import ARTIFACT_DIR
from evaluate import macro_f05
from model import assign

GRID = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def curve(D, oof):
    acc = {}
    for k in D:
        d = D[k]
        new = d["pred"].join(oof, on=["s1", "mid"], how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
        drop = d["ok"].filter(pl.col("h") >= pl.col("wk")).select("mid")
        drop = pl.concat([drop, d["ok"].filter(pl.col("outside")).select("mid")]).unique()
        new = new.join(drop, on="mid", how="anti")
        ctry = dict(d["country"].iter_rows())
        qs = d["q"]
        msk = {c: np.array([ctry[s] == c for s in qs]) for c in ("US", "India")}
        for t in GRID:
            sc = np.asarray(macro_f05(assign(new, t), d["truth"], qs)[1])
            for c in ("US", "India"):
                acc.setdefault((c, t), []).append((int(msk[c].sum()), float(sc[msk[c]].mean())))
    return {(c, t): sum(a * b for a, b in v) / sum(a for a, _ in v) for (c, t), v in acc.items()}


def main(cfg="w"):
    D, feats = S.load()
    params, weighted = S.CONFIGS[cfg]
    iters = json.load(open(os.path.join(ARTIFACT_DIR, f"s2tune_{cfg}.json")))["iters"]
    rounds = int(np.mean(iters) * 1.1)
    parts = []
    for k in D:
        parts.append(D[k]["band"].select(["s1", "mid", "y", "fold", "wk"] + feats).join(D[k]["country"], on="s1"))
    band = pl.concat(parts)
    S.log(f"band {band.height:,} pairs: {band.group_by('country').len().rows()}; rounds {rounds}")
    X = lambda df: df.select(feats).to_numpy().astype(np.float32)
    # mixed-country out-of-fold (region folds), as in s2_tune dev
    mixed = []
    for f in sorted(band["fold"].unique().to_list()):
        tr, te = band.filter(pl.col("fold") != f), band.filter(pl.col("fold") == f)
        m = S.train(tr, feats, params, weighted, rounds=rounds)
        mixed.append(te.select("s1", "mid").with_columns(pl.Series("p2", m.predict(X(te)))))
    S.log("mixed-country OOF done")
    # zero-shot: train on one country, score the other
    zs = []
    for src, dst in (("US", "India"), ("India", "US")):
        m = S.train(band.filter(pl.col("country") == src), feats, params, weighted, rounds=rounds)
        te = band.filter(pl.col("country") == dst)
        zs.append(te.select("s1", "mid").with_columns(pl.Series("p2", m.predict(X(te)))))
        S.log(f"zero-shot {src} -> {dst} done")
    a, b = curve(D, pl.concat(mixed)), curve(D, pl.concat(zs))
    for c in ("US", "India"):
        print(f"\n{c}: threshold | mixed-country F0.5 | zero-shot F0.5", flush=True)
        for t in GRID:
            print(f"   {t:.2f}   {a[(c, t)]:.5f}   {b[(c, t)]:.5f}", flush=True)
        bm, bz = max(GRID, key=lambda t: a[(c, t)]), max(GRID, key=lambda t: b[(c, t)])
        print(f"   best: mixed {bm} ({a[(c, bm)]:.5f}) | zero-shot {bz} ({b[(c, bz)]:.5f}) | zero-shot loss {a[(c, bm)] - b[(c, bz)]:.5f}", flush=True)
    print("LOCO_DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
