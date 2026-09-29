"""EDA 8: (a) US city agreement in true pairs; (b) do singletons have look-alike decoys?"""
import polars as pl
from common import load, OUT
from norm import basic, us_state

pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
F = pl.read_parquet(f"{OUT}/04_pair_features.parquet")
ids = pl.concat([F["s1"], F["mid"]]).unique()
recs = pl.concat([load("train_s1"), load("train_s2"), load("train_s3")]).filter(pl.col("entity_id").is_in(ids.implode()))
rd = {r[0]: r[1:3] for r in recs.iter_rows()}


def us_city(addr):
    """City = the non-numeric, non-state part (US S1 format: 'num street, city, ST' in any order)."""
    if not addr:
        return None
    parts = [basic(p) for p in addr.split(",")]
    st = us_state(addr)
    cands = [p for p in parts if p and not any(ch.isdigit() for ch in p) and p != st
             and p not in ("null", "n a", "none") and not p.startswith(("unit", "apt", "po box"))]
    from norm import US_STATES
    cands = [p for p in cands if p not in US_STATES]
    return cands[-1] if cands else None


pos = F.filter((pl.col("y") == 1) & (pl.col("country") == "US"))
eq = tot = fuzzy = 0
ex = []
from rapidfuzz import fuzz
for a, b in pos.select("s1", "mid").iter_rows():
    c1, c2 = us_city(rd[a][1]), us_city(rd[b][1])
    if c1 and c2:
        tot += 1
        eq += c1 == c2
        f = fuzz.ratio(c1.replace("city of ", "").replace(" cdp", "").replace(" township", ""),
                       c2.replace("city of ", "").replace(" cdp", "").replace(" township", ""))
        fuzzy += f >= 80
        if f < 80 and len(ex) < 15:
            ex.append((c1, c2))
print(f"(a) US positive pairs with city parsed on both sides: {tot}; exact city eq={eq/tot:.3f}; fuzzy(>=80, prefix-stripped)={fuzzy/tot:.3f}")
print("    examples of city disagreement:", ex)

# (b) singletons: their top-10 name neighbours (all non-matches by definition)
gt = load("train_gt")
sing = set(gt.filter(pl.col("matched_entity_ids").is_null())["source1_entity_id"].to_list())
neg = F.filter(pl.col("y") == 0).with_columns(pl.col("s1").is_in(list(sing)).alias("singleton"))
best = neg.sort("name_tset", descending=True).group_by(["s1", "singleton"]).first()
print("\n(b) best hard-negative per S1 (by name token-set) — singleton vs non-singleton S1:")
print(best.group_by("singleton").agg(
    pl.len(), pl.col("core_exact").mean().round(3), (pl.col("name_tset") >= 95).mean().round(3).alias("name_tset>=95"),
    pl.col("addr_tset").filter(pl.col("addr_tset") >= 0).mean().round(1).alias("addr_tset"),
    pl.col("first_num_eq").mean().round(3), pl.col("num_jacc").filter(pl.col("num_jacc") >= 0).mean().round(3)))
# decoys: same core name AND same first number -> most dangerous negatives
d = neg.filter((pl.col("core_exact") == 1) & (pl.col("first_num_eq") == 1))
print(f"\nnegatives with identical core name AND same house number: {d.height} "
      f"({d.height / neg.height:.4f} of hard negs); on singletons: {d.filter(pl.col('singleton')).height}")
for a, b in d.sample(min(10, d.height), seed=0).select("s1", "mid").iter_rows():
    print(f"   S1 | {rd[a][0]!s:<40} | {rd[a][1]}\n   {b[:2]} | {rd[b][0]!s:<40} | {rd[b][1]}\n")
