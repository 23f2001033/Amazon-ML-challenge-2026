"""Decompose each country's errors from predicted per-source match counts, without labels.
True per-source counts follow the generator's distribution (train ground truth: joint (n2, n3), singletons
included). Observed = Binomial(true, r) + Poisson(lam) per source: r = copy recall, lam = extra (false)
records per S1 per source. Fit r, lam per country and source by maximum likelihood on the predicted
(n2, n3) of the final submission; validate on labelled US/India held-out predictions."""
import os
import sys
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import binom, poisson
from config import PARQUET_DIR, PREPARED_DIR

K = 12


def generator_pmf():
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id"])
    gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")).rename({"source1_entity_id": "entity_id"})
    g = s1.join(gt, on="entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    out = {}
    for src in ("S2", "S3"):
        n = g["matched_entity_ids"].str.count_matches(f"{src}-").to_numpy()
        out[src] = np.bincount(n, minlength=K)[:K] / len(n)
    return out


def observed_pmf_model(true_pmf, r, lam):
    obs = np.zeros(K)
    for n, pn in enumerate(true_pmf):
        if pn == 0:
            continue
        thin = binom.pmf(np.arange(K), n, r)
        obs += pn * np.convolve(thin, poisson.pmf(np.arange(K), lam))[:K]
    return obs / obs.sum()


def fit(counts, true_pmf):
    h = np.bincount(np.clip(counts, 0, K - 1), minlength=K)[:K]
    def nll(x):
        r, lam = 1 / (1 + np.exp(-x[0])), np.exp(x[1])
        return -np.sum(h * np.log(observed_pmf_model(true_pmf, r, lam) + 1e-12))
    res = minimize(nll, x0=[3.0, -4.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 2000})
    return 1 / (1 + np.exp(-res.x[0])), float(np.exp(res.x[1]))


def counts_from_tsv(path, s1_countries):
    m = pl.read_csv(path, separator="\t", schema_overrides={"matched_entity_ids": pl.String}).with_columns(pl.col("matched_entity_ids").fill_null(""))
    m = m.rename({"source1_entity_id": "entity_id"}).join(s1_countries, on="entity_id")
    return m.with_columns(pl.col("matched_entity_ids").str.count_matches("S2-").alias("S2"), pl.col("matched_entity_ids").str.count_matches("S3-").alias("S3"))


if __name__ == "__main__":
    gen = generator_pmf()
    print("generator mean per source:", {s: round(float(np.dot(np.arange(K), p)), 3) for s, p in gen.items()})
    if sys.argv[1] == "validate":
        # labelled check: wide universe, v6fr3t decision; true r and lam are computable
        import run_dev as R
        from model import assign
        from evaluate import truth_map_from_gt
        uni = R.universes()
        qs = uni.filter((pl.col("universe") == "wide") & pl.col("is_query"))["entity_id"]
        d = pl.read_parquet(os.path.join(os.environ.get("FEAT", "/home/ubuntu/er/data/features"), "wide_s2oof_s2_v6_rich_ce_ce2_ce3fr_ce4fr.parquet"), columns=["s1", "mid", "p"])
        c = pl.read_parquet(os.path.join(PREPARED_DIR, "train_s1.parquet"), columns=["entity_id", "country"])
        d = d.join(c.rename({"entity_id": "s1"}), on="s1")
        m = assign(d, 0.725, {"US": 0.6, "India": 0.5})
        truth = truth_map_from_gt(pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet")), qs.to_list())
        rows = []
        for s in qs.to_list():
            pr, tr = m.get(s, set()), truth.get(s, set())
            rows.append((s, sum(x.startswith("S2") for x in pr), sum(x.startswith("S3") for x in pr),
                         len(pr & tr), len(tr), len(pr - tr)))
        f = pl.DataFrame(rows, schema=["entity_id", "S2", "S3", "tp", "t", "fp"], orient="row").join(c, on="entity_id")
        for ctry in ("US", "India"):
            g = f.filter(pl.col("country") == ctry)
            est = {src: fit(g[src].to_numpy(), gen[src]) for src in ("S2", "S3")}
            print(f"[wide {ctry}] TRUE recall {g['tp'].sum() / g['t'].sum():.4f}, TRUE false/S1 {g['fp'].sum() / g.height:.4f} | "
                  f"FIT r S2 {est['S2'][0]:.4f} S3 {est['S3'][0]:.4f}, extra/S1 {est['S2'][1] + est['S3'][1]:.4f}")
    else:
        s1c = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
        for name in sys.argv[2:]:
            f = counts_from_tsv(f"/home/ubuntu/er/sub_{name}/matching_results.tsv", s1c)
            for ctry in ("US", "India", "France"):
                g = f.filter(pl.col("country") == ctry)
                est = {src: fit(g[src].to_numpy(), gen[src]) for src in ("S2", "S3")}
                print(f"[{name} {ctry}] FIT r S2 {est['S2'][0]:.4f} S3 {est['S3'][0]:.4f} | extra records/S1 {est['S2'][1] + est['S3'][1]:.4f}", flush=True)
