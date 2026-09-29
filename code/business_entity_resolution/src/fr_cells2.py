"""Finer France-vs-US/India pattern cells on the FINAL v6fr3x decisions (Sol's items 1 and 3).

Test: accepted pairs and rejected best-candidate pairs per 1k S1, by country.
Labelled US/India (val + wide, stage-2 w2 out-of-fold, test-like variant B thinning, thr 0.70): the true rate of the
same cell. A France-heavy accepted cell with a low US/India true rate -> veto candidate; a France-heavy rejected
cell with a high true rate -> add candidate.
Also: residual category swaps (the swap veto re-applied to the final decisions).
  python fr_cells2.py"""
import os

import polars as pl
from rapidfuzz import fuzz

import final_decide as FD
import run_dev as R
from config import FEAT_DIR, PARQUET_DIR, PREPARED_DIR
from model import assign
from sizebias_scan import describe, frame

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(260)
S2N = "s2_v6_rich_ce_ce2_ce3fr_ce4fr_w2us_wfr"
KEEP = {"US": 0.47, "India": 0.105}
NOISE, REAL = FD.vocab()


def subtype(country, c1, c2):
    s, r = set((c1 or "").split()) - {""}, set((c2 or "").split()) - {""}
    if not r:
        return "rec_empty"
    if s == r:
        return "same"
    add, rem = r - s, s - r
    nz, rl = NOISE.get(country, set()) | FD.NOISE_WORDS, REAL.get(country, set())
    typo = {w for w in add if any(fuzz.ratio(w, x) >= 75 or FD._subseq(w, x) or FD._subseq(x, w) for x in rem)}
    real = {w for w in add - typo if w not in nz and w in rl and len(w) >= 4}
    unk = add - typo - real - nz
    if real:
        if not rem:
            return "add_real_only"
        return "swap_real_1" if len(real) == 1 and len(rem) == 1 else "swap_real_multi"
    if unk:
        return "swap_unknown" if rem else "add_unknown_only"
    if typo:
        return "typo"
    if add:
        return "drop+noise" if rem else "noise_only"
    return "drop_only"


def s1_context(split, ids=None):
    s = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=["entity_id", "country", "n_core", "a_tokens", "a_house"])
    if ids is not None:
        s = s.filter(pl.col("entity_id").is_in(ids.implode()))
    s = s.with_columns(pl.col("n_core").fill_null(""), (pl.col("a_tokens").fill_null("") + "|" + pl.col("a_house").fill_null("")).alias("akey"))
    nn = s.group_by("country", "n_core").len("n_name")
    na = s.filter(pl.col("a_house").fill_null("") != "").group_by("country", "akey").len("n_addr")
    return (s.join(nn, on=["country", "n_core"]).join(na, on=["country", "akey"], how="left")
             .select(pl.col("entity_id").alias("s1"), "n_name", pl.col("n_addr").fill_null(1)))


def cells(d, ctx):
    d = describe(d).join(ctx, on="s1", how="left")
    d = d.with_columns(pl.Series("sub", [subtype(c, a, b) for c, a, b in d.select("country", "c1", "c2").iter_rows()]))
    return d.with_columns(pl.when(pl.col("n_addr") >= 2).then(pl.lit("hub")).otherwise(pl.lit("-")).alias("hub"),
                          pl.when(pl.col("n_name") == 1).then(pl.lit("uniq")).otherwise(pl.lit("shared")).alias("name"),
                          pl.when(pl.col("p") >= 0.3).then(pl.lit(".3+")).otherwise(pl.lit("<.3")).alias("pb"))


def labelled():
    uni = R.universes()
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
    owner = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("matched_entity_ids").alias("mid"), pl.col("source1_entity_id").alias("owner")))
    s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
    oof = pl.read_parquet(os.path.join(FEAT_DIR, "s2tune_oof_w2.parquet"))
    acc_all, rej_all, nq = [], [], {}
    for kind in ("val", "wide"):
        u = uni.filter(pl.col("universe") == kind)
        qs = u.filter(pl.col("is_query"))["entity_id"]
        d = pl.read_parquet(os.path.join(FEAT_DIR, f"{kind}_pred_v6.parquet"), columns=["s1", "mid", "p", "y"]).rename({"p": "p1"}).filter(pl.col("p1") >= 0.02)
        d = d.join(oof, on=["s1", "mid"], how="left").with_columns(pl.coalesce("p2", "p1").alias("p")).drop("p2")
        m = d.select("mid").unique().join(owner, on="mid", how="left").join(s1c.rename({"entity_id": "owner"}), on="owner", how="left")
        outside = m.filter(pl.col("owner").is_not_null() & ~pl.col("owner").is_in(u["entity_id"].implode())).select("mid")
        dropped = m.filter(pl.col("owner").is_in(u["entity_id"].implode()) & ~pl.col("owner").is_in(qs.implode())).with_columns(
            pl.col("country").replace_strict(KEEP, default=1.0, return_dtype=pl.Float64).alias("k"), (pl.col("mid").hash(seed=11) % 10000 / 10000).alias("h"))
        drop = pl.concat([outside, dropped.filter(pl.col("h") >= pl.col("k")).select("mid")]).unique()
        d = d.join(drop, on="mid", how="anti").join(s1c.rename({"entity_id": "s1"}), on="s1")
        acc = assign(d, 0.7)
        a = pl.DataFrame([(s, x) for s, xs in acc.items() for x in xs], schema=["s1", "mid"], orient="row")
        taken = a.select("mid")
        best = d.filter(pl.col("p") >= 0.05).sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
        ctx = s1_context("train", qs)
        acc_all.append(cells(frame("train", d.join(a, on=["s1", "mid"]).drop("country")), ctx))
        rej_all.append(cells(frame("train", best.join(taken, on="mid", how="anti").drop("country")), ctx))
        for c, n in s1c.filter(pl.col("entity_id").is_in(qs.implode())).group_by("country").len().iter_rows():
            nq[c] = nq.get(c, 0) + n
        print(f"labelled {kind}: accepted {a.height:,}", flush=True)
    return pl.concat(acc_all), pl.concat(rej_all), nq


def main():
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{S2N}.parquet"), columns=["s1", "mid", "p", "country"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    dec = (pl.read_csv("/home/ubuntu/er/sub_v6fr3x/matching_results.tsv", separator="\t").drop_nulls("matched_entity_ids")
             .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("mid")))
    # residual swaps: the veto re-applied to the final decisions
    res = FD.swap_pairs(dec, ["France"])
    print(f"residual France category swaps in the final decisions: {res.height:,} pairs, {res['s1'].n_unique():,} S1", flush=True)
    res.write_parquet("/home/ubuntu/er/fr_residual_swaps.parquet")
    base = assign(pred, 0.725)
    b = pl.DataFrame([(s, x) for s, xs in base.items() for x in xs], schema=["s1", "mid"], orient="row")
    veto = FD.swap_pairs(b, ["France"])
    pv = pred.join(veto.with_columns(pl.lit(1).alias("v")), on=["s1", "mid"], how="left").with_columns(
        pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
    best = pv.filter(pl.col("p") >= 0.05).sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
    ctx = s1_context("test")
    ta = cells(frame("test", pv.join(dec, on=["s1", "mid"]).drop("country")), ctx)
    tr = cells(frame("test", best.join(dec.select("mid"), on="mid", how="anti").drop("country")), ctx)
    n1 = dict(pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country"]).group_by("country").len().iter_rows())
    la, lr, nq = labelled()
    for name, t, lab, keys in (("ACCEPTED (veto candidates: France excess high, US/India P_true low)", ta, la, ["arel", "sub", "hub"]),
                               ("REJECTED best candidates (add candidates: France excess high, US/India P_true high)", tr, lr, ["arel", "sub", "name", "pb"])):
        r = t.group_by(["country"] + keys).len()
        r = r.with_columns((pl.col("len") / pl.col("country").replace_strict(n1, return_dtype=pl.Float64) * 1000).round(2).alias("k"))
        w = r.pivot(on="country", index=keys, values="k").fill_null(0.0)
        L = lab.group_by(keys).agg(pl.len().alias("lab_n"), pl.col("y").mean().round(3).alias("P_true"),
                                   (pl.len() / sum(nq.values()) * 1000).round(2).alias("lab_1k"))
        w = w.join(L, on=keys, how="left").with_columns(((pl.col("US") + pl.col("India")) / 2).round(2).alias("USIN"))
        w = w.with_columns((pl.col("France") - pl.col("USIN")).round(2).alias("FR_excess"), (pl.col("France") * n1["France"] / 1000).round(0).alias("FR_pairs"))
        print(f"\n=== {name} per 1k S1 ===")
        print(w.select(keys + ["France", "US", "India", "FR_excess", "FR_pairs", "P_true", "lab_n", "lab_1k"]).filter(pl.col("France") >= 0.3)
               .sort("FR_excess", descending=True).head(30))
    ta.filter(pl.col("country") == "France").select("s1", "mid", "p", "p1", "arel", "sub", "hub", "name", "c1", "c2", "a1", "a2").write_parquet("/home/ubuntu/er/fr_cells_acc.parquet")
    tr.filter(pl.col("country") == "France").select("s1", "mid", "p", "p1", "arel", "sub", "hub", "name", "c1", "c2", "a1", "a2").write_parquet("/home/ubuntu/er/fr_cells_rej.parquet")
    la.select("s1", "mid", "p", "y", "country", "arel", "sub", "hub", "name", "c1", "c2").write_parquet("/home/ubuntu/er/lab_cells_acc.parquet")
    lr.select("s1", "mid", "p", "y", "country", "arel", "sub", "hub", "name", "pb", "c1", "c2").write_parquet("/home/ubuntu/er/lab_cells_rej.parquet")
    print("CELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
