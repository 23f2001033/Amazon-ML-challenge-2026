"""EDA 4: blocking recall (TF-IDF kNN on name / address) and positive-vs-hard-negative similarity.

For a random sample of train S1 entities we search the FULL train S2+S3 pool of the same
country (so distractor density is realistic) and measure:
  * recall@k of char-3gram TF-IDF on normalised name, on normalised address, and their union
  * similarity-feature distributions of true pairs vs hard negatives (top name neighbours
    that are not matches) -> tells us which features separate the classes
Outputs: outputs/04_*.txt / .parquet
"""
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import load, gt_pairs, OUT
from norm import norm_name, core_name, norm_addr, addr_numbers, us_state

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
K = 50
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


s1 = load("train_s1").sample(N_SAMPLE, seed=42)
pool = pl.concat([load("train_s2"), load("train_s3")])
pos = gt_pairs().filter(pl.col("s1").is_in(s1["entity_id"].implode()))
truth = {}
for a, b in pos.select("s1", "mid").iter_rows():
    truth.setdefault(a, set()).add(b)
log("sample", s1.height, "positives", pos.height)

results, neigh_rows = [], []
for country in s1["country"].unique().to_list():
    q = s1.filter(pl.col("country") == country)
    p = pool.filter(pl.col("country") == country)
    log(country, "queries", q.height, "pool", p.height)
    pids = p["entity_id"].to_numpy()
    for field, fn in [("name", norm_name), ("addr", norm_addr)]:
        col = "business_name" if field == "name" else "business_address"
        ptxt = [fn(x) for x in p[col].to_list()]
        qtxt = [fn(x) for x in q[col].to_list()]
        log(" normalised", field)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32,
                              sublinear_tf=True)
        P = vec.fit_transform(ptxt)
        del ptxt
        Q = vec.transform(qtxt)
        log(" tfidf", field, P.shape, P.nnz)
        PT = P.T.tocsr()
        del P
        R = sp_matmul_topn(Q, PT, top_n=K, threshold=0.0, n_threads=12).tocsr()
        P = PT
        log(" knn done", field)
        for i, sid in enumerate(q["entity_id"].to_list()):
            lo, hi = R.indptr[i], R.indptr[i + 1]
            idx, sc = R.indices[lo:hi], R.data[lo:hi]
            order = np.argsort(-sc)
            for rank, j in enumerate(order):
                neigh_rows.append((sid, pids[idx[j]], field, rank, float(sc[j])))
        del P, Q, R

nb = pl.DataFrame(neigh_rows, schema=["s1", "mid", "field", "rank", "score"], orient="row")
nb = nb.with_columns(pl.struct(["s1", "mid"]).map_elements(
    lambda r: r["mid"] in truth.get(r["s1"], ()), return_dtype=pl.Boolean).alias("is_match"))
nb.write_parquet(f"{OUT}/04_neighbours.parquet")
log("neighbours saved", nb.height)

# ---- recall@k -------------------------------------------------------------------------
tot = pos.height
cc = s1.select(pl.col("entity_id").alias("s1"), "country")
lines = ["## Blocking recall (fraction of true pairs retrieved) by k\n"]
for k in [1, 3, 5, 10, 20, 30, 50]:
    sub = nb.filter((pl.col("rank") < k) & pl.col("is_match"))
    r_name = sub.filter(pl.col("field") == "name").unique(["s1", "mid"]).height / tot
    r_addr = sub.filter(pl.col("field") == "addr").unique(["s1", "mid"]).height / tot
    r_union = sub.unique(["s1", "mid"]).height / tot
    lines.append(f"k={k:>3}: name={r_name:.4f} addr={r_addr:.4f} union={r_union:.4f}")
lines.append("\nper country union recall@k (k=10,50):")
posc = pos.join(cc, on="s1")
for c in cc["country"].unique().to_list():
    tc = posc.filter(pl.col("country") == c).height
    for k in [10, 50]:
        sub = nb.join(cc, on="s1").filter((pl.col("country") == c) & (pl.col("rank") < k) & pl.col("is_match"))
        lines.append(f"  {c} k={k}: name={sub.filter(pl.col('field') == 'name').unique(['s1', 'mid']).height / tc:.4f}"
                     f" addr={sub.filter(pl.col('field') == 'addr').unique(['s1', 'mid']).height / tc:.4f}"
                     f" union={sub.unique(['s1', 'mid']).height / tc:.4f}")
# Singletons: how close is their best neighbour vs non-singletons' best true match?
lines.append("\nTop-1 name score: singletons vs entities with matches")
top1 = nb.filter((pl.col("field") == "name") & (pl.col("rank") == 0))
sing = set(s1["entity_id"].to_list()) - set(truth)
top1 = top1.with_columns(pl.col("s1").is_in(list(sing)).alias("singleton"))
lines.append(str(top1.group_by("singleton").agg(pl.col("score").quantile(q).alias(f"q{int(q*100)}") for q in [.1, .25, .5, .75, .9])))
open(f"{OUT}/04_blocking_recall.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# ---- positive vs hard-negative feature distributions -----------------------------------
recs = pl.concat([s1, pool]).select("entity_id", "business_name", "business_address", "country")
rd = {r[0]: r[1:] for r in recs.filter(pl.col("entity_id").is_in(
    pl.concat([nb["s1"], nb["mid"], pos["mid"]]).unique().implode())).iter_rows()}
hn = nb.filter(~pl.col("is_match") & (pl.col("rank") < 10)).unique(["s1", "mid"]).select("s1", "mid").with_columns(pl.lit(0).alias("y"))
pp = pos.select("s1", "mid").with_columns(pl.lit(1).alias("y"))
pairs = pl.concat([pp, hn])
feats = []
for a, b, y in pairs.iter_rows():
    n1, a1, c1 = rd[a]; n2, a2, _ = rd[b]
    nn1, nn2 = norm_name(n1), norm_name(n2)
    cn1, cn2 = core_name(n1), core_name(n2)
    na1, na2 = norm_addr(a1), norm_addr(a2)
    t1, t2 = set(nn1.split()), set(nn2.split())
    num1, num2 = addr_numbers(a1), addr_numbers(a2)
    feats.append(dict(
        s1=a, mid=b, y=y, country=c1, src=b[:2],
        name_ratio=fuzz.ratio(nn1, nn2), name_tset=fuzz.token_set_ratio(nn1, nn2),
        name_tsort=fuzz.token_sort_ratio(nn1, nn2), name_partial=fuzz.partial_ratio(nn1, nn2),
        core_ratio=fuzz.ratio(cn1, cn2), core_tset=fuzz.token_set_ratio(cn1, cn2),
        name_jacc=len(t1 & t2) / max(1, len(t1 | t2)), name_exact=int(nn1 == nn2), core_exact=int(cn1 == cn2),
        name_lev=Levenshtein.distance(cn1, cn2),
        addr_ratio=fuzz.ratio(na1, na2) if na2 else -1, addr_tset=fuzz.token_set_ratio(na1, na2) if na2 else -1,
        addr_null=int(not na2),
        num_jacc=len(num1 & num2) / max(1, len(num1 | num2)) if num2 else -1,
        first_num_eq=int(bool(num1 and num2) and sorted(num1)[0] in num2),
        state_eq=(int(us_state(a1) == us_state(a2)) if c1 == "US" and us_state(a2) else -1),
        name_nonascii=int(any(ord(ch) > 127 for ch in n2)),
    ))
F = pl.DataFrame(feats)
F.write_parquet(f"{OUT}/04_pair_features.parquet")
num_cols = [c for c in F.columns if c not in ("s1", "mid", "y", "country", "src")]
summ = F.group_by(["country", "y"]).agg([pl.col(c).filter(pl.col(c) >= 0).mean().round(3).alias(c) for c in num_cols] + [pl.len()]).sort(["country", "y"])
pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(400)
out = str(summ)
open(f"{OUT}/04_pos_vs_hardneg.txt", "w", encoding="utf-8").write(out)
print(out)
log("done")
