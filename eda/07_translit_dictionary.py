"""EDA 7: can a native-script -> English token dictionary be learned from train pairs?

For positive pairs whose S2/S3 name is fully native-script and has the same token count as
the S1 name, align tokens by position and count (native, english) co-occurrences. A clean
dictionary = each native token maps overwhelmingly to one English token.
"""
import re
from collections import Counter, defaultdict

import polars as pl
from common import load, gt_pairs, OUT

INDIC = re.compile(r"[\u0900-\u0D7F]")
pos = gt_pairs()
s1 = load("train_s1").filter(pl.col("country") == "India").select("entity_id", "business_name")
m = pl.concat([load("train_s2"), load("train_s3")]).filter(
    (pl.col("country") == "India") & pl.col("business_name").str.contains(r"[\u0900-\u0D7F]"))
pairs = (pos.join(m.select(pl.col("entity_id").alias("mid"), pl.col("business_name").alias("n2")), on="mid")
            .join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1")), on="s1"))
print("native-name positive pairs:", pairs.height)
co = defaultdict(Counter)
aligned = 0
for n1, n2 in pairs.select("n1", "n2").iter_rows():
    t1, t2 = n1.split(), n2.split()
    if len(t1) == len(t2) and all(INDIC.search(t) for t in t2):
        aligned += 1
        for a, b in zip(t2, t1):
            co[a][b.lower()] += 1
print("same-length fully-native pairs aligned:", aligned, f"({aligned / pairs.height:.3f})")
purity = []
for tok, c in co.items():
    tot = sum(c.values()); top, cnt = c.most_common(1)[0]
    purity.append((tok, top, cnt / tot, tot))
P = pl.DataFrame(purity, schema=["native", "english", "purity", "count"], orient="row").sort("count", descending=True)
print("tokens learned:", P.height, " weighted purity:", (P["purity"] * P["count"]).sum() / P["count"].sum())
print("tokens with purity>=0.9:", P.filter(pl.col("purity") >= 0.9).height)
pl.Config.set_tbl_rows(40)
print(P.head(30))
print(P.sort("purity").head(10))
P.write_parquet(f"{OUT}/07_translit_dict_prototype.parquet")
