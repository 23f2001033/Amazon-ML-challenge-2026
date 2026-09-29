"""Where does France behave unlike US/India?  For every pattern cell (address relation x name relation x confidence):
  * labelled truth rate in US/India held-out data (wide universe) among ALL best-candidate pairs of the cell,
    and among the ACCEPTED ones;
  * accepted pairs per S1 on test: US, India, France (v6fr3s decisions);
  * best-candidate pairs REJECTED per S1 on test.
Cells France accepts far more often than US/India while US/India says they are mostly false -> veto candidates;
cells France rejects far more often while US/India says they are mostly true -> add candidates."""
import os
import numpy as np
import polars as pl
from rapidfuzz import fuzz
import sizebias_scan as SB
from config import FEAT_DIR, PREPARED_DIR
from model import assign

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(240)


def vocab(split):
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=["country", "n_core"])
    t = s1.select("country", pl.col("n_core").fill_null("").str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t") != "").group_by("country", "t").len()
    return {c: set(t.filter((pl.col("country") == c) & (pl.col("len") >= 20))["t"].to_list()) for c in t["country"].unique().to_list()}


def refine(d, voc):
    out = []
    for c, a, b, n in d.select("country", "c1", "c2", "nrel").iter_rows():
        if n != "added_word":
            out.append(n); continue
        s, r = set((a or "").split()), set((b or "").split())
        add = [w for w in r - s if w not in SB.NOISE]
        out.append("added_real" if any(w in voc.get(c, set()) and len(w) >= 4 for w in add) else "added_unknown")
    return d.with_columns(pl.Series("nrel", out))


def decisions(p_df):
    m = assign(p_df, 0.725, SB.THR)
    return pl.DataFrame([(s, x) for s, xs in m.items() for x in xs], schema=["s1", "mid"], orient="row").with_columns(pl.lit(1).alias("acc"))


# ---- labelled wide
w = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "y"]).join(
    pl.read_parquet(os.path.join(FEAT_DIR, "wide_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
w = w.join(pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
acc = decisions(w)
wb = w.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first").filter(pl.col("p") >= 0.05)
wb = refine(SB.describe(SB.frame("train", wb.drop("country"))), vocab("train")).join(acc, on=["s1", "mid"], how="left").with_columns(pl.col("acc").fill_null(0))
lab = wb.group_by("arel", "nrel", "conf").agg(pl.len().alias("wide_n"), pl.col("y").mean().round(3).alias("P_true_all"),
                                              pl.col("y").filter(pl.col("acc") == 1).mean().round(3).alias("P_true_acc"))
# ---- test
t = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "country"]).join(
    pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
t = t.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
acc_t = decisions(t)
tb = t.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first").filter(pl.col("p") >= 0.05)
tb = refine(SB.describe(SB.frame("test", tb.drop("country"))), vocab("test")).join(acc_t, on=["s1", "mid"], how="left").with_columns(pl.col("acc").fill_null(0))
tb.select("s1", "mid", "country", "p", "p1", "arel", "nrel", "conf", "acc").write_parquet("/home/ubuntu/er/cells_test.parquet")
n1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country"]).group_by("country").len("S1")
r = tb.group_by("country", "arel", "nrel", "conf").agg(pl.col("acc").sum().alias("acc"), (1 - pl.col("acc")).sum().alias("rej")).join(n1, on="country")
r = r.with_columns((pl.col("acc") / pl.col("S1") * 1000).round(2).alias("acc1k"), (pl.col("rej") / pl.col("S1") * 1000).round(2).alias("rej1k"))
wide = r.pivot(on="country", index=["arel", "nrel", "conf"], values=["acc1k", "rej1k"]).fill_null(0.0)
wide = wide.join(lab, on=["arel", "nrel", "conf"], how="left")
wide = wide.with_columns(((pl.col("acc1k_US") + pl.col("acc1k_India")) / 2).alias("acc1k_USIN"),
                         ((pl.col("rej1k_US") + pl.col("rej1k_India")) / 2).alias("rej1k_USIN"))
wide = wide.with_columns((pl.col("acc1k_France") - pl.col("acc1k_USIN")).round(2).alias("FR_excess_acc"),
                         (pl.col("rej1k_France") - pl.col("rej1k_USIN")).round(2).alias("FR_excess_rej"))
cols = ["arel", "nrel", "conf", "P_true_all", "P_true_acc", "wide_n", "acc1k_US", "acc1k_India", "acc1k_France", "FR_excess_acc", "rej1k_USIN", "rej1k_France", "FR_excess_rej"]
print("=== sorted by France excess ACCEPTED per 1k S1 (veto candidates if P_true low) ===")
print(wide.select(cols).sort("FR_excess_acc", descending=True).head(22))
print("=== sorted by France excess REJECTED per 1k S1 (add candidates if P_true high) ===")
print(wide.select(cols).sort("FR_excess_rej", descending=True).head(16))
