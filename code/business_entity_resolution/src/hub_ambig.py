"""Hub ambiguity: an accepted exact-address pair where the record's name cannot tell which of the S1s at that
address it belongs to. compat = number of S1s at the address whose name tokens contain all of the record's
non-noise tokens (a renamed record with no shared token is compatible with every S1 at the hub).
  python hub_ambig.py wide   -> labelled precision by (name relation, compat) on US/India held-out data
  python hub_ambig.py test   -> counts by country + France veto candidates"""
import os
import sys
import polars as pl
import sizebias_scan as SB
from config import FEAT_DIR, PREPARED_DIR
from model import assign

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
split = "train" if sys.argv[1] == "wide" else "test"
if split == "train":
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"wide_s2oof_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "y"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "wide_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    d = d.join(pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
else:
    d = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{SB.S2N}.parquet"), columns=["s1", "mid", "p", "country"]).join(
        pl.read_parquet(os.path.join(FEAT_DIR, "test_pred_v6.parquet"), columns=["s1", "mid", "p"]).rename({"p": "p1"}), on=["s1", "mid"])
    veto = pl.read_parquet("/home/ubuntu/er/sub_v6fr3s/france_veto_pairs.parquet").with_columns(pl.lit(1).alias("v"))
    d = d.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
m = assign(d, 0.725, SB.THR)
acc = pl.DataFrame([(s, x) for s, xs in m.items() for x in xs], schema=["s1", "mid"], orient="row")
a = SB.describe(SB.frame(split, acc.join(d.drop("country"), on=["s1", "mid"])))
a = a.filter(pl.col("arel") == "exact")
s1 = pl.read_parquet(os.path.join(PREPARED_DIR, f"{split}_s1.parquet"), columns=["entity_id", "country", "n_core", "a_tokens", "a_house"])
s1 = s1.with_columns(pl.concat_str([pl.col("a_tokens").fill_null(""), pl.col("a_house").fill_null("")], separator="|").alias("k"))
hubs = s1.group_by("country", "k").agg(pl.col("n_core").fill_null("").alias("names"), pl.len().alias("hub_n")).filter(pl.col("hub_n") >= 2)
a = a.join(s1.select(pl.col("entity_id").alias("s1"), "k"), on="s1").join(hubs, on=["country", "k"], how="inner")

def compat(rn, names):
    r = {w for w in (rn or "").split() if w not in SB.NOISE}
    if not r:
        return len(names)
    shared_any = any(r & set(n.split()) for n in names)
    if not shared_any:
        return len(names)                       # renamed: compatible with every S1 at the hub
    return sum(1 for n in names if r <= set(n.split()) or len(r & set(n.split())) >= max(1, len(r) - 1))

a = a.with_columns(pl.Series("compat", [compat(r, n) for r, n in a.select("c2", "names").iter_rows()]))
a = a.with_columns(pl.col("compat").cut([1, 2, 4], labels=["1", "2", "3-4", "5+"]).alias("amb"))
if split == "train":
    print("WIDE labelled: accepted exact-address pairs at hub addresses")
    print(a.group_by("country", "nrel", "amb").agg(pl.len(), pl.col("y").mean().round(3).alias("P_true")).filter(pl.col("len") >= 30).sort("country", "nrel", "amb"))
    print("pooled by ambiguity:", a.group_by("amb").agg(pl.len(), pl.col("y").mean().round(3)).sort("amb").rows())
else:
    n1 = s1.group_by("country").len("S1")
    r = a.group_by("country", "amb").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("S1") * 1000).round(2).alias("per1k"))
    print(r.pivot(on="country", index="amb", values="per1k").sort("amb"))
    a.filter((pl.col("country") == "France") & (pl.col("compat") >= 2)).select("s1", "mid", "nrel", "compat", "p", "p1").write_parquet("/home/ubuntu/er/fr_hub_ambig.parquet")
    print(a.filter((pl.col("country") == "France") & (pl.col("compat") >= 2)).group_by("nrel").len().sort("len", descending=True))
