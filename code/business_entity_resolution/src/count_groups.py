"""Label-free validation of candidate France additions with the count model (count_fit.py):
for each group of currently REJECTED best-candidate French pairs, add the group to the v6fr3s decision and
refit (r, extra). True copies raise r and leave 'extra' flat; distractors raise 'extra'.
Groups come from /home/ubuntu/er/cells_test.parquet (best candidate per record, p >= 0.05, after the veto)."""
import os
import numpy as np
import polars as pl
import count_fit as CF
from config import PREPARED_DIR

gen = CF.generator_pmf()
s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country", "n_compact"])
fr = s1c.filter(pl.col("country") == "France")
base = CF.counts_from_tsv("/home/ubuntu/er/sub_v6fr3s/matching_results.tsv", s1c.select("entity_id", "country")).filter(pl.col("country") == "France")
cells = pl.read_parquet("/home/ubuntu/er/cells_test.parquet").filter((pl.col("country") == "France") & (pl.col("acc") == 0))
rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"test_s{i}.parquet"), columns=["entity_id", "n_compact"]) for i in (2, 3)]).rename({"entity_id": "mid"})
kn = fr.group_by("n_compact").len("k")
cells = cells.join(rec, on="mid", how="left").join(kn, on="n_compact", how="left").with_columns(pl.col("k").fill_null(0))
cells = cells.with_columns(pl.when(pl.col("arel") == "no_addr").then(pl.concat_str([pl.lit("no_addr k="), pl.col("k").cut([1, 3], labels=["1", "2-3", "4+"]).cast(pl.String)]))
                             .otherwise(pl.col("arel")).alias("a2"),
                           pl.col("p").cut([0.2, 0.5], labels=["<.2", ".2-.5", ">=.5"]).alias("pb"))


def fit_counts(extra_pairs):
    b = base
    if extra_pairs is not None and extra_pairs.height:
        add = (extra_pairs.with_columns(pl.col("mid").str.slice(0, 2).alias("src")).group_by("s1", "src").len("n")
                          .pivot(on="src", index="s1", values="n").fill_null(0))
        for c in ("S2", "S3"):
            if c not in add.columns:
                add = add.with_columns(pl.lit(0).alias(c))
        b = b.join(add.rename({"s1": "entity_id", "S2": "a2", "S3": "a3"}), on="entity_id", how="left").with_columns(
            (pl.col("S2") + pl.col("a2").fill_null(0)).alias("S2"), (pl.col("S3") + pl.col("a3").fill_null(0)).alias("S3"))
    e = {s: CF.fit(b[s].to_numpy(), gen[s]) for s in ("S2", "S3")}
    return (e["S2"][0] + e["S3"][0]) / 2, e["S2"][1] + e["S3"][1]


r0, x0 = fit_counts(None)
print(f"France base v6fr3s: r {r0:.4f}  extra/S1 {x0:.4f}", flush=True)
groups = cells.group_by("a2", "nrel", "pb").len().filter(pl.col("len") >= 800).sort("len", descending=True)
for a2, nrel, pb, n in groups.iter_rows():
    g = cells.filter((pl.col("a2") == a2) & (pl.col("nrel") == nrel) & (pl.col("pb") == pb)).select("s1", "mid")
    r, x = fit_counts(g)
    per = n / fr.height
    # if all true: dr ~ per / 3.46 and dx ~ 0 ; if all false: dx ~ per
    share_true = 1 - min(max((x - x0) / per, 0), 1)
    print(f"  {a2:<14} {nrel:<16} p{pb:<6} n={n:>6} ({per:.4f}/S1): r {r:.4f} ({r - r0:+.4f})  extra {x:.4f} ({x - x0:+.4f})  -> est. true share {share_true:.2f}", flush=True)
