"""Stage 2: candidate generation (blocking).

For each S1 query, candidates come from the union of (D-011):
  A. name pass    : char 3-gram TF-IDF on the compact core name, top-K_NAME, within (country, region)
  B. address pass : word/number TF-IDF on canonical address tokens, top-K_ADDR, within (country, region)
  C. fallback     : name TF-IDF against S2/S3 records with NO detectable region (null or
                    region-less addresses), top-K_FALL, within country.
S1 queries whose region is unknown search their whole country (A + B).

Work is done block by block with sparse top-k products (sparse_dot_topn), so memory is bounded
by the largest block. For every candidate we also compute BOTH cosines (name and address) with
the block's own TF-IDF, which the matcher later uses as features.

  D. (E4) exact compact-name key within country across regions, for specific names only.

  E. (v6, ER_COMB=1) combined pass: TF-IDF name and address vectors concatenated, top-K_COMB.
     Breaks ties among identical names by address (Maharashtra: half of the missed true pairs had
     the exact S1 name, shared by ~35 pool records, and a truncated address).
  G. (v6n, ER_CONT=K) record-centric address containment: each pool record takes the K S1s of its block
     that contain the largest share of the RECORD's address weight (IDF^2 of its tokens, no penalty for a
     long S1 address). Targets truncated S2/S3 addresses ("Office No. 815, Thane, MH" vs the S1's full
     address), which cosine top-K misses when thousands of S1s share the city tokens (India).
  F. (v6, ER_LEAKS=1) region-leak expansion: queries of region R also search the pool of the
     regions their true matches leak into, learned from train labels (artifacts/region_leaks.json;
     e.g. Telangana S1 vs S2/S3 written with the old state name "Andhra Pradesh": 18.5% of pairs).

Output columns: s1, mid, name_cos, addr_cos, hit
(bitmask 1=name, 2=addr, 4=fallback, 8=exact name, 16=combined, 32=leak-region expansion, 64=containment)
"""
import json
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS

K_NAME, K_ADDR, K_FALL = 25, 25, 20   # v1 used K_FALL=10; E4: 20 (+0.0016 recall, +10 cand/S1)
K_COMB = 25 if os.environ.get("ER_COMB") == "1" else 0
# v7: wider search (ER_K="name,addr,comb,fall"), e.g. "40,40,40,30"
if os.environ.get("ER_K"):
    K_NAME, K_ADDR, K_COMB, K_FALL = (int(x) for x in os.environ["ER_K"].split(","))
USE_LEAKS = os.environ.get("ER_LEAKS") == "1"
K_CONT = int(os.environ.get("ER_CONT", "0"))          # v6n: record-centric address containment pass
CONT_MIN = float(os.environ.get("ER_CONT_MIN", "0.4"))
K_REV = int(os.environ.get("ER_FALLREV", "0"))       # record-centric fallback: each region-less record -> its top-K S1 names
EXACT_MAX_POOL, EXACT_MAX_S1 = 30, 10  # E4 exact-name key: skip generic names
Q_CHUNK = 20000
PAIR_CHUNK = 2_000_000


def _vectorizers(n_docs):
    # Prune very frequent features only in big blocks (in small blocks every n-gram is useful).
    max_df = 0.05 if n_docs > 20000 else 1.0
    vn = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True,
                         max_df=max_df)
    va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", dtype=np.float32, sublinear_tf=True,
                         max_df=max_df)
    return vn, va


def _fit(vec, q_txt, p_txt):
    try:
        vec.fit(p_txt + q_txt)
    except ValueError:  # empty vocabulary
        return None, None
    return vec.transform(q_txt), vec.transform(p_txt)


def _topk(Q, P, k, threshold=0.05):
    PT = P.T.tocsr()
    rows, cols = [], []
    for s in range(0, Q.shape[0], Q_CHUNK):
        R = sp_matmul_topn(Q[s:s + Q_CHUNK], PT, top_n=k, threshold=threshold, n_threads=N_JOBS).tocoo()
        rows.append(R.row.astype(np.int64) + s); cols.append(R.col.astype(np.int64))
    return np.concatenate(rows), np.concatenate(cols)


def _rowdot(Q, P, qi, pj):
    """cosine for aligned index pairs (rows are L2-normalised by TfidfVectorizer)."""
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), PAIR_CHUNK):
        a, b = Q[qi[s:s + PAIR_CHUNK]], P[pj[s:s + PAIR_CHUNK]]
        out[s:s + PAIR_CHUNK] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _combined(N, A):
    """Row-wise [name | address] concatenation, each half unit-norm: dot = (name_cos + addr_cos) / 2."""
    from scipy.sparse import hstack
    return (hstack([N, A]).tocsr() * np.float32(0.5 ** 0.5)).astype(np.float32)


def _containment(Qa, Pa, k):
    """Record-centric: for each pool row, the k query rows maximising sum over shared tokens of the pool row's
    squared TF-IDF weights (= share of the record's address weight found in the S1 address)."""
    Psq = Pa.multiply(Pa).tocsr().astype(np.float32)
    Qb = Qa.copy().astype(np.float32)
    Qb.data[:] = 1.0
    r, c = _topk(Psq, Qb, k, threshold=CONT_MIN)
    return c, r                                  # (query index, pool index)


def _block(q, p, k_name, k_addr, hit_name, hit_addr, k_comb=0, hit_comb=16, k_cont=0, hit_cont=64, k_rev=0):
    """Run name/address(/combined) kNN of queries q against pool p; return candidate frame or None."""
    if q.height == 0 or p.height == 0:
        return None
    vn, va = _vectorizers(q.height + p.height)
    Qn, Pn = _fit(vn, q["n_compact"].to_list(), p["n_compact"].to_list())
    Qa, Pa = _fit(va, q["a_tokens"].to_list(), p["a_tokens"].to_list())
    parts = []
    if Qn is not None and k_name:
        r, c = _topk(Qn, Pn, k_name)
        parts.append((r, c, np.full(len(r), hit_name, np.int8)))
    if Qa is not None and k_addr:
        r, c = _topk(Qa, Pa, k_addr)
        parts.append((r, c, np.full(len(r), hit_addr, np.int8)))
    if k_comb and Qn is not None and Qa is not None:
        r, c = _topk(_combined(Qn, Qa), _combined(Pn, Pa), k_comb)
        parts.append((r, c, np.full(len(r), hit_comb, np.int8)))
    if k_cont and Qa is not None:
        r, c = _containment(Qa, Pa, k_cont)
        parts.append((r, c, np.full(len(r), hit_cont, np.int8)))
    if k_rev and Qn is not None:
        pr, qc_ = _topk(Pn, Qn, k_rev)                  # pool row -> its top-k S1 names
        parts.append((qc_, pr, np.full(len(pr), 64 | hit_name, np.int8)))
    if not parts:
        return None
    r = np.concatenate([x[0] for x in parts]); c = np.concatenate([x[1] for x in parts])
    h = np.concatenate([x[2] for x in parts])
    key = r * p.height + c
    uniq, inv = np.unique(key, return_inverse=True)
    hits = np.zeros(len(uniq), np.int8)
    np.bitwise_or.at(hits, inv, h)
    qi, pj = uniq // p.height, uniq % p.height
    name_cos = _rowdot(Qn, Pn, qi, pj) if Qn is not None else np.zeros(len(qi), np.float32)
    addr_cos = _rowdot(Qa, Pa, qi, pj) if Qa is not None else np.zeros(len(qi), np.float32)
    return pl.DataFrame({"s1": q["entity_id"].to_numpy()[qi], "mid": p["entity_id"].to_numpy()[pj],
                         "name_cos": name_cos, "addr_cos": addr_cos, "hit": hits})


def _block_job(args):
    return _block(*args)


def _pair_cos(q_txt, p_txt, analyzer):
    """Cosine of aligned (query, pool) text pairs with a vectorizer fit on just these texts."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=(3, 3) if analyzer == "char" else (1, 1),
                          token_pattern=r"\S+", dtype=np.float32, sublinear_tf=True)
    try:
        vec.fit(q_txt + p_txt)
    except ValueError:
        return np.zeros(len(q_txt), np.float32)
    Q, P = vec.transform(q_txt), vec.transform(p_txt)
    return np.asarray(Q.multiply(P).sum(axis=1)).ravel().astype(np.float32)


def _exact_name_pairs(qc, pc, max_pool=EXACT_MAX_POOL, max_s1=EXACT_MAX_S1):
    """E4: exact compact-name key across regions within a country (catches region mismatches and
    region-less records). Only reasonably specific names (bounded block sizes)."""
    qn = qc.filter(pl.col("n_compact").str.len_chars() >= 4)
    pn = pc.filter(pl.col("n_compact").str.len_chars() >= 4)
    pc_cnt = pn.group_by("n_compact").len().filter(pl.col("len") <= max_pool).select("n_compact")
    qc_cnt = qn.group_by("n_compact").len().filter(pl.col("len") <= max_s1).select("n_compact")
    keys = pc_cnt.join(qc_cnt, on="n_compact")
    pairs = (qn.join(keys, on="n_compact").select(pl.col("entity_id").alias("s1"), "n_compact",
                                                  pl.col("a_tokens").alias("qa"))
               .join(pn.select(pl.col("entity_id").alias("mid"), "n_compact", pl.col("a_tokens").alias("pa")),
                     on="n_compact"))
    if pairs.height == 0:
        return None
    addr_cos = _pair_cos(pairs["qa"].to_list(), pairs["pa"].to_list(), "word")
    return pairs.select("s1", "mid").with_columns(
        pl.lit(1.0, pl.Float32).alias("name_cos"), pl.Series("addr_cos", addr_cos),
        pl.lit(8, pl.Int8).alias("hit"))


def load_leaks():
    from config import ARTIFACT_DIR
    out = {}
    for country, reg, leak in json.load(open(os.path.join(ARTIFACT_DIR, "region_leaks.json")))["leaks"]:
        out.setdefault((country, reg), []).append(leak)
    return out


def generate(s1, pool, log=print, workers=4, exact_key=False):  # exact key: +0.0001 recall, off
    """s1, pool: prepared frames (columns entity_id, country, region, n_compact, a_tokens).

    Region blocks are independent; with workers>1 they run in parallel processes (largest first).
    """
    t0 = time.time()
    cols = ["entity_id", "country", "region", "n_compact", "a_tokens"]
    s1, pool = s1.select(cols), pool.select(cols)
    leaks = load_leaks() if USE_LEAKS else {}
    if leaks:
        log(f"  region-leak expansion on: {sum(len(v) for v in leaks.values())} (region -> region) pairs")
    jobs, exact_parts = [], []
    for country in sorted(s1["country"].unique().to_list()):
        qc = s1.filter(pl.col("country") == country)
        pc = pool.filter(pl.col("country") == country)
        regions = sorted(qc.filter(pl.col("region") != "")["region"].unique().to_list())
        for reg in regions:
            jobs.append((qc.filter(pl.col("region") == reg), pc.filter(pl.col("region") == reg),
                         K_NAME, K_ADDR, 1, 2, K_COMB, 16, K_CONT, 64))
            for leak in leaks.get((country, reg), []):
                jobs.append((qc.filter(pl.col("region") == reg), pc.filter(pl.col("region") == leak),
                             K_NAME, K_ADDR, 1 | 32, 2 | 32, K_COMB, 16 | 32))
        jobs.append((qc, pc.filter(pl.col("region") == ""), K_FALL, 0, 4, 4, 0, 16, 0, 64, K_REV))
        q_noreg = qc.filter(pl.col("region") == "")
        if q_noreg.height:
            jobs.append((q_noreg, pc, K_NAME, K_ADDR, 1, 2, K_COMB, 16))
        if exact_key:
            ex = _exact_name_pairs(qc, pc)
            if ex is not None:
                exact_parts.append(ex)
        log(f"  blocking {country}: queries={qc.height:,} pool={pc.height:,} regions={len(regions)} "
            f"no-region queries={q_noreg.height:,}")
    del s1, pool
    jobs.sort(key=lambda j: -(j[0].height + j[1].height))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_block_job, jobs))
    else:
        outs = [_block(*j) for j in jobs]
    log(f"  blocking done: {len(jobs)} blocks in {time.time() - t0:.0f}s")
    outs = [o for o in outs if o is not None] + exact_parts
    cand = pl.concat(outs)
    return (cand.group_by(["s1", "mid"])
                .agg(pl.col("name_cos").max(), pl.col("addr_cos").max(), pl.col("hit").bitwise_or()))
