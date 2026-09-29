"""EDA 2: print full clusters (S1 record + all its S2/S3 matches) for qualitative review."""
import sys
import polars as pl
from common import load, gt_pairs

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
country = sys.argv[2] if len(sys.argv) > 2 else None
k = int(sys.argv[3]) if len(sys.argv) > 3 else 12

s1 = load("train_s1"); gt = load("train_gt")
recs = pl.concat([load("train_s2"), load("train_s3")])
if country:
    s1 = s1.filter(pl.col("country") == country)
samp = s1.sample(k, seed=seed)
p = gt_pairs().filter(pl.col("s1").is_in(samp["entity_id"].implode()))
m = p.join(recs, left_on="mid", right_on="entity_id")
for r in samp.iter_rows(named=True):
    print(f"\n[{r['entity_id']}] {r['country']}")
    print(f"   S1 | {r['business_name']!s:<55} | {r['business_address']}")
    sub = m.filter(pl.col("s1") == r["entity_id"]).sort("src")
    if sub.height == 0:
        print("   (singleton)")
    for x in sub.iter_rows(named=True):
        print(f"   {x['src']} | {x['business_name']!s:<55} | {x['business_address']}")
