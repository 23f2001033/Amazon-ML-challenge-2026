"""Typo'd category swaps: at an exact address, the record lacks an S1 word and adds a word that is NOT a real
vocabulary word but is a misspelling of a DIFFERENT real vocabulary word (not of the removed S1 word),
e.g. "Tourcoing Union SAS" -> "SAS Tourcoing Cultuere".  Labelled precision on wide, counts on test."""
import os
import sys
import polars as pl
from rapidfuzz import fuzz, process
import sizebias_scan as SB
from config import FEAT_DIR, PREPARED_DIR
from model import assign

pl.Config.set_tbl_rows(-1)


def vocab(split):
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=["country", "n_core"])
    t = s1.select("country", pl.col("n_core").fill_null("").str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t") != "").group_by("country", "t").len()
    return {c: [w for w in t.filter((pl.col("country") == c) & (pl.col("len") >= 20))["t"].to_list() if len(w) >= 4] for c in t["country"].unique().to_list()}


def flag(d, voc):
    vs = {c: set(v) for c, v in voc.items()}
    out = []
    for c, a, b in d.select("country", "c1", "c2").iter_rows():
        s, r = set((a or "").split()) - {""}, set((b or "").split()) - {""}
        add, rem = [w for w in r - s if w not in SB.NOISE and len(w) >= 4], [w for w in s - r if len(w) >= 3]
        hit = False
        if rem and (s & r):
            for w in add:
                if w in vs.get(c, ()):
                    continue                                   # real word: handled by the exact swap veto
                if any(fuzz.ratio(w, x) >= 70 for x in rem):
                    continue                                   # typo of the removed S1 word -> noise
                m = process.extractOne(w, voc.get(c, []), scorer=fuzz.ratio, score_cutoff=80)
                if m and m[0] not in s:
                    hit = True
                    break
        out.append(hit)
    return pl.Series("tswap", out)


def accepted(d):
    m = assign(d, 0.725, SB.THR)
    return pl.DataFrame([(s, x) for s, xs in m.items() for x in xs], schema=["s1", "mid"], orient="row")


if sys.argv[1] == "wide":
    w = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "y"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "wide_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    w = w.join(pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
    allp = SB.describe(SB.frame("train", w.filter(pl.col("p1") >= 0.02).drop("country"))).filter(pl.col("arel") == "exact")
    allp = allp.with_columns(flag(allp, vocab("train"))).filter(pl.col("tswap"))
    acc = accepted(w).with_columns(pl.lit(1).alias("acc"))
    allp = allp.join(acc, on=["s1", "mid"], how="left").with_columns(pl.col("acc").fill_null(0))
    print("WIDE labelled typo'd swaps at exact address:")
    print(allp.group_by("country", "acc").agg(pl.len(), pl.col("y").mean().round(3).alias("P_true")).sort("country", "acc"))
else:
    t = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "country"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
    t = t.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
    acc = accepted(t)
    a = SB.describe(SB.frame("test", acc.join(t.drop("country"), on=["s1", "mid"]))).filter(pl.col("arel") == "exact")
    a = a.with_columns(flag(a, vocab("test"))).filter(pl.col("tswap"))
    n1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country"]).group_by("country").len("S1")
    print(a.group_by("country").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("S1") * 1000).round(2).alias("per1k")))
    a.filter(pl.col("country") == "France").select("s1", "mid").write_parquet("/home/ubuntu/er/fr_typo_swap.parquet")
    for r in a.filter(pl.col("country") == "France").sample(20, seed=1).select("c1", "c2").rows():
        print(f"  {r[0]:<36} -> {r[1]}")
