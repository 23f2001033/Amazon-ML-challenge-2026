"""Stage 4-5: LightGBM pair classifier (MIT license) and the match decision rule."""
import numpy as np
import polars as pl
import lightgbm as lgb

from evaluate import macro_f05

PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "verbose": -1, "num_threads": 0, "seed": 42,
}


def train(df, feats, holdout_frac=0.1, rounds=3000):
    """Train with an internal S1-level holdout (early stopping only)."""
    s1 = df["s1"].unique()
    hold = set(s1.sample(fraction=holdout_frac, seed=7).to_list())
    is_hold = df["s1"].is_in(list(hold)).to_numpy()
    X, y = df.select(feats).to_numpy().astype(np.float32), df["y"].to_numpy()
    w = df["w"].to_numpy() if "w" in df.columns else None
    dtr = lgb.Dataset(X[~is_hold], y[~is_hold], weight=None if w is None else w[~is_hold],
                      feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X[is_hold], y[is_hold], weight=None if w is None else w[is_hold], reference=dtr)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    return m


def predict(m, df, feats):
    return m.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=0)


def assign(df, thr, thr_by_country=None):
    """Partition constraint (D-007) + threshold: every S2/S3 record goes to at most one S1,
    the one with the highest probability, and only if p >= thr. Returns {s1: set(mid)}.

    thr_by_country: optional {country: threshold} overriding `thr` (df must have `country`)."""
    best = (df.sort(["p", "s1"], descending=[True, False])  # deterministic tie-break
              .unique("mid", keep="first", maintain_order=True))
    if thr_by_country:
        t = pl.col("country").replace_strict(thr_by_country, default=thr, return_dtype=pl.Float64)
        sel = best.filter(pl.col("p") >= t)
    else:
        sel = best.filter(pl.col("p") >= thr)
    out = {}
    for s, m in sel.select("s1", "mid").iter_rows():
        out.setdefault(s, set()).add(m)
    return out


def expected_f_select(df, n_samples=128, max_c=40, chunk=20000, floor=0.02, seed=0):
    """Per-entity expected-F0.5 subset selection (D-015).

    After the partition constraint, for each S1 sort candidates by p and choose k in 0..n that
    maximises E[F0.5(top-k)], labels ~ independent Bernoulli(p). k=0 scores P(no true match).
    Monte-Carlo in vectorised numpy chunks. Returns {s1: set(mid)}.
    """
    best = (df.sort(["p", "s1"], descending=[True, False])  # deterministic tie-break
              .unique("mid", keep="first", maintain_order=True))
    best = best.filter(pl.col("p") >= floor).sort(["s1", "p"], descending=[False, True])
    g = best.group_by("s1", maintain_order=True).agg(pl.col("mid").head(max_c), pl.col("p").head(max_c))
    rng = np.random.default_rng(seed)
    out = {}
    s1s, mids, ps = g["s1"].to_list(), g["mid"].to_list(), g["p"].to_list()
    for s in range(0, len(s1s), chunk):
        P = np.zeros((min(chunk, len(s1s) - s), max_c), np.float32)
        for i, pl_ in enumerate(ps[s:s + chunk]):
            P[i, :len(pl_)] = pl_
        X = rng.random((n_samples,) + P.shape, dtype=np.float32) < P          # [S, n, c]
        T = X.sum(-1, dtype=np.float32)                                        # [S, n]
        TP = np.cumsum(X, axis=-1, dtype=np.float32)                           # [S, n, c]
        k = np.arange(1, max_c + 1, dtype=np.float32)
        Fk = (1.25 * TP / (0.25 * T[..., None] + k)).mean(0)                   # [n, c]
        F0 = (T == 0).mean(0)                                                  # [n]
        valid = P > 0
        Fk = np.where(valid, Fk, -1)
        kbest = np.where(F0 >= Fk.max(1), 0, Fk.argmax(1) + 1)
        for i, kb in enumerate(kbest):
            if kb:
                out[s1s[s + i]] = set(mids[s + i][:kb])
    return out


def tune_threshold(df, truth, s1_ids, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.2, 0.96, 0.05), 2)
    res = []
    for t in grid:
        score, _ = macro_f05(assign(df, t), truth, s1_ids)
        res.append((float(t), score))
    best = max(res, key=lambda x: x[1])
    return best, res
