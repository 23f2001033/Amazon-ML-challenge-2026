"""Label-free false-merge estimator ("size-bias test") and a pattern scan over France.

A record that is a TRUE copy of its S1 lands on S1s in proportion to how many copies they have (size-biased
per-source count); a planted DISTRACTOR lands on S1s independently of that (count = normal count + 1).
For a group G of (S1, record) pairs, the per-source count distribution of the S1s (counting the G record)
is a mixture  f * D + (1 - f) * T  ->  f = estimated distractor share of G.

  python sizebias_scan.py validate   # wide universe (US/India): estimated vs true false share per group
  python sizebias_scan.py france     # France test: accepted groups (veto candidates) and rejected groups (add candidates)
"""
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from model import assign

S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr"
THR = {"US": 0.6, "India": 0.5, "France": 0.725}
NOISE = {"groupe", "developpement", "holding", "participations", "distribution", "international", "associes", "services",
         "fils", "cie", "france", "et", "center", "service", "holdings", "group", "enterprises", "partners", "co", "company",
         "llc", "inc", "corp", "ltd", "limited", "pvt", "private"}
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)


def est_f(counts_with_record, P):
    """counts_with_record: per-S1 count in the record's source INCLUDING the G record (k >= 1); P: count distribution of all S1s."""
    K = 9
    P = np.asarray(P[:K], float); P = P / P.sum()
    k = np.arange(K)
    T = k * P; T = T / T.sum()
    D = np.r_[0.0, P[:-1]]; D = D / D.sum()
    O = np.bincount(np.clip(counts_with_record, 0, K - 1), minlength=K)[:K].astype(float); O = O / max(O.sum(), 1)
    d = D - T
    f = float(np.dot(O - T, d) / np.dot(d, d))
    return f


def name_rel(c1, c2):
    s, r = set((c1 or "").split()) - {""}, set((c2 or "").split()) - {""}
    if not r:
        return "rec_name_empty"
    if s == r:
        return "same_tokens"
    add, rem = r - s, s - r
    add_n = {w for w in add if w in NOISE}
    add_o = add - add_n
    if not add_o:
        return "drop_only" if not add_n else "drop+noise"
    if not (s & r):
        return "no_shared_token"
    if all(any(fuzz.ratio(w, x) >= 75 for x in rem) for w in add_o):
        return "typo"
    return "added_word"


def describe(d):
    d = d.with_columns(pl.col(c).fill_null("") for c in ("c1", "c2", "a1", "a2", "h1", "h2"))
    d = d.with_columns(pl.Series("ad", cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    d = d.with_columns(pl.Series("nrel", [name_rel(a, b) for a, b in d.select("c1", "c2").iter_rows()]))
    return d.with_columns(
        pl.when(pl.col("null2").fill_null(1) == 1).then(pl.lit("no_addr"))
          .when((pl.col("h1") == pl.col("h2")) & (pl.col("h1") != "") & (pl.col("ad") >= 90)).then(pl.lit("exact"))
          .when((pl.col("h1") != pl.col("h2")) & (pl.col("h1") != "") & (pl.col("h2") != "") & (pl.col("ad") >= 80)).then(pl.lit("other_house"))
          .when(pl.col("ad") >= 70).then(pl.lit("close")).otherwise(pl.lit("far")).alias("arel"),
        pl.when(pl.col("p1") >= 0.99).then(pl.lit("p1>=.99")).otherwise(pl.lit("band")).alias("conf"))


def frame(split, pred):
    cols = ["entity_id", "country", "n_core", "a_tokens", "a_house", "a_null"]
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=cols).select(
        pl.col("entity_id").alias("s1"), "country", pl.col("n_core").alias("c1"), pl.col("a_tokens").alias("a1"), pl.col("a_house").alias("h1"))
    rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s{i}.parquet"), columns=cols) for i in (2, 3)]).select(
        pl.col("entity_id").alias("mid"), pl.col("n_core").alias("c2"), pl.col("a_tokens").alias("a2"), pl.col("a_house").alias("h2"), pl.col("a_null").alias("null2"))
    return pred.join(s1, on="s1").join(rec, on="mid")


def scan(best, acc, label=None, min_n=400):
    """best: one row per record = its best S1 (s1, mid, p, p1, country, describe() columns); acc: accepted (s1, mid)."""
    acc_src = acc.with_columns(pl.col("mid").str.slice(0, 2).alias("src"))
    out = []
    for c in best["country"].unique().to_list():
        cnt = acc_src.join(best.select("s1", "country").unique(), on="s1").filter(pl.col("country") == c).group_by("s1", "src").len("n")
        s1s = best.filter(pl.col("country") == c).select("s1").unique()
        P = {}
        for src in ("S2", "S3"):
            n_all = s1s.join(cnt.filter(pl.col("src") == src), on="s1", how="left").with_columns(pl.col("n").fill_null(0))["n"].to_numpy()
            P[src] = np.bincount(n_all, minlength=12)
        b = best.filter(pl.col("country") == c).with_columns(pl.col("mid").str.slice(0, 2).alias("src"))
        b = b.join(acc.select("s1", "mid", pl.lit(1).alias("is_acc")), on=["s1", "mid"], how="left").with_columns(pl.col("is_acc").fill_null(0))
        b = b.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n").fill_null(0))
        b = b.with_columns((pl.col("n") + (1 - pl.col("is_acc"))).alias("k"))   # count including this record
        for (arel, nrel, conf, isacc), g in b.group_by(["arel", "nrel", "conf", "is_acc"]):
            one = g.filter(pl.len().over("s1", "src") == 1)
            if one.height < min_n:
                continue
            fs = [est_f(one.filter(pl.col("src") == s)["k"].to_numpy(), P[s]) for s in ("S2", "S3") if one.filter(pl.col("src") == s).height >= min_n // 4]
            if not fs:
                continue
            row = {"country": c, "accepted": int(isacc), "arel": arel, "nrel": nrel, "conf": conf, "pairs": g.height, "f_est": round(float(np.mean(fs)), 3)}
            if label is not None:
                row["f_true"] = round(1 - float(g[label].mean()), 3)
            out.append(row)
    return pl.DataFrame(out).sort("country", "accepted", "pairs", descending=[False, True, True])


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        d = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{S2N}.parquet"), columns=["s1", "mid", "p", "y"]).join(
            pl.read_parquet(os.path.join(FEAT_DIR, "wide_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
        d = d.join(pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
        acc_map = assign(d, 0.725, THR)
        acc = pl.DataFrame([(s, m) for s, ms in acc_map.items() for m in ms], schema=["s1", "mid"], orient="row")
        best = d.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first").filter(pl.col("p") >= 0.05)
        best = describe(frame("train", best.drop("country")))
        print(scan(best, acc, label="y"))
    else:
        t = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S2N}.parquet"), columns=["s1", "mid", "p", "country"]).join(
            pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
        veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
        t = t.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
        acc_map = assign(t, 0.725, THR)
        acc = pl.DataFrame([(s, m) for s, ms in acc_map.items() for m in ms], schema=["s1", "mid"], orient="row")
        best = t.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first").filter(pl.col("p") >= 0.05)
        best = describe(frame("test", best.drop("country")))
        r = scan(best, acc)
        r.write_parquet("/home/ubuntu/er/sizebias_scan_test.parquet")
        print(r)
