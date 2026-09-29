"""Stage 2 (E1): specialist re-scoring of the uncertain band.

Stage 1 decides ~99% of candidate pairs with near-certainty. On v2a validation, perfect decisions
for the few pairs with p in [0.02, 0.99) would lift macro F0.5 from 0.981 to 0.994 while touching
~1 pair per S1 (docs/04_EXPERIMENTS.md). Stage 2 re-scores only that band, adding evidence that a
pairwise model cannot see:

  * p-space competition: best rival S1 probability for the same record, rank among claimants,
    confident-match counts of the S1 (overall and per source);
  * sibling support (H-3): agreement of the candidate with the S1's other confident candidates
    (shared house number, name / address similarity to confident siblings);
  * a few direct pair checks recomputed from the canonical records.

Trained on stage-1 predictions for the VALIDATION universe (honest: stage 1 never saw it),
evaluated by 2-fold cross-validation over regions, then refit on all of it for test.
"""
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

BAND = (0.02, 0.99)      # stage-2 re-scores pairs with p in [lo, hi)
CONF = 0.75              # "confident sibling" threshold on stage-1 p
S2_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
             "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
             "verbose": -1, "num_threads": 0, "seed": 7}
REC = ["entity_id", "region", "n_core", "n_compact", "a_tokens", "a_house", "a_nums", "a_null"]


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def group_features(pred):
    """p-space features over ALL candidates with p >= BAND[0] (pred: s1, mid, p)."""
    pred = pred.filter(pl.col("p") >= BAND[0]).with_columns(
        (pl.col("mid").str.slice(0, 2) == "S3").cast(pl.Int8).alias("is_s3"))
    conf = (pl.col("p") >= CONF).cast(pl.Int32)
    return pred.with_columns(
        # within the S1's candidate set
        (conf.sum().over("s1") - conf).alias("g_n_conf_other"),
        ((conf * pl.col("is_s3")).sum().over("s1") - conf * pl.col("is_s3")).alias("g_n_conf_s3_other"),
        ((conf * (1 - pl.col("is_s3"))).sum().over("s1") - conf * (1 - pl.col("is_s3"))).alias("g_n_conf_s2_other"),
        (pl.col("p").sum().over("s1") - pl.col("p")).alias("g_sum_p_other"),
        pl.col("p").rank("ordinal", descending=True).over("s1").alias("g_rank_in_s1"),
        pl.len().over("s1").alias("g_n_cand"),
        # competition for the same record across S1s
        pl.col("p").rank("ordinal", descending=True).over("mid").alias("g_rank_in_mid"),
        pl.len().over("mid").alias("g_n_claimants"),
        (pl.col("p").max().over("mid")).alias("g_mid_max_p"),
    ).with_columns(
        # best rival = best other claimant (if I am the max, the 2nd best)
        pl.when(pl.col("g_rank_in_mid") == 1)
          .then(pl.col("p").sort(descending=True).slice(1, 1).first().over("mid").fill_null(0.0))
          .otherwise(pl.col("g_mid_max_p")).alias("g_rival_p"),
    ).with_columns((pl.col("p") - pl.col("g_rival_p")).alias("g_margin_vs_rival"))


def sibling_features(band, grp, recs):
    """Support from the S1's confident siblings for each band pair.

    band: (s1, mid) pairs to score; grp: all candidates with p (for siblings); recs: REC columns
    for the S1s and all involved S2/S3 records.
    """
    sib = grp.filter(pl.col("p") >= CONF).select("s1", pl.col("mid").alias("sib"))
    pairs = band.select("s1", "mid").join(sib, on="s1", how="inner").filter(pl.col("mid") != pl.col("sib"))
    r = recs.select("entity_id", "n_core", "a_tokens", "a_house")
    pairs = (pairs.join(r.rename({"entity_id": "mid", "n_core": "c_name", "a_tokens": "c_addr", "a_house": "c_house"}), on="mid", how="left")
                  .join(r.rename({"entity_id": "sib", "n_core": "s_name", "a_tokens": "s_addr", "a_house": "s_house"}), on="sib", how="left")
                  .with_columns(pl.col(c).fill_null("") for c in ("c_name", "c_addr", "c_house", "s_name", "s_addr", "s_house")))
    if pairs.height:
        pairs = pairs.with_columns(
            pl.Series("sn", _cp(pairs["c_name"].to_list(), pairs["s_name"].to_list(), fuzz.token_set_ratio)),
            pl.Series("sa", _cp(pairs["c_addr"].to_list(), pairs["s_addr"].to_list(), fuzz.token_set_ratio)),
            ((pl.col("c_house") == pl.col("s_house")) & (pl.col("c_house") != "")).cast(pl.Float32).alias("sh"))
    agg = pairs.group_by(["s1", "mid"]).agg(
        pl.len().alias("sib_n"), pl.col("sn").max().alias("sib_name_max"), pl.col("sn").mean().alias("sib_name_mean"),
        pl.col("sa").max().alias("sib_addr_max"), pl.col("sa").mean().alias("sib_addr_mean"),
        pl.col("sh").sum().alias("sib_same_house"), pl.col("sh").mean().alias("sib_same_house_frac"))
    return band.join(agg, on=["s1", "mid"], how="left").with_columns(
        pl.col("sib_n").fill_null(0), *(pl.col(c).fill_null(-1.0) for c in
        ("sib_name_max", "sib_name_mean", "sib_addr_max", "sib_addr_mean", "sib_same_house", "sib_same_house_frac")))


def pair_features(band, recs):
    """A few direct checks between S1 and candidate (canonical fields)."""
    a = recs.rename({c: f"{c}_1" for c in REC if c != "entity_id"})
    b = recs.rename({c: f"{c}_2" for c in REC if c != "entity_id"})
    d = band.join(a, left_on="s1", right_on="entity_id", how="left").join(b, left_on="mid", right_on="entity_id", how="left")
    d = d.with_columns(pl.col(c).fill_null("") for c in d.columns if d[c].dtype == pl.String and c not in ("s1", "mid"))
    d = d.with_columns(
        pl.Series("pf_name_tset", _cp(d["n_core_1"].to_list(), d["n_core_2"].to_list(), fuzz.token_set_ratio)),
        pl.Series("pf_name_ratio", _cp(d["n_compact_1"].to_list(), d["n_compact_2"].to_list(), fuzz.ratio)),
        pl.Series("pf_addr_tset", _cp(d["a_tokens_1"].to_list(), d["a_tokens_2"].to_list(), fuzz.token_set_ratio)),
        pl.Series("pf_addr_partial", _cp(d["a_tokens_1"].to_list(), d["a_tokens_2"].to_list(), fuzz.partial_ratio)),
        ((pl.col("a_house_1") == pl.col("a_house_2")) & (pl.col("a_house_1") != "")).cast(pl.Int8).alias("pf_house_eq"),
        ((pl.col("a_house_1") != "") & (pl.col("a_house_2") != "")).cast(pl.Int8).alias("pf_house_both"),
        pl.col("a_null_2").fill_null(1).cast(pl.Int8).alias("pf_c_addr_null"),
        (pl.col("region_1") == pl.col("region_2")).cast(pl.Int8).alias("pf_region_eq"),
        (pl.col("n_compact_1") == pl.col("n_compact_2")).cast(pl.Int8).alias("pf_name_exact"),
    )
    keep = [c for c in d.columns if c.startswith("pf_")]
    return d.select(["s1", "mid"] + keep)


def ce_features(ce):
    """Cross-encoder evidence (ce: s1, mid, p_ce for all scored candidates of a query set):
    the CE probability itself plus CE-space competition (best other claimant of the record,
    best other candidate of the S1)."""
    ce = ce.select("s1", "mid", pl.col("p_ce").cast(pl.Float32)).unique(["s1", "mid"])
    top2 = lambda k: pl.col("p_ce").sort(descending=True).slice(1, 1).first().over(k).fill_null(0.0)
    conf = (pl.col("p_ce") >= 0.5).cast(pl.Int32)
    return ce.with_columns(
        (pl.col("p_ce").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p_ce").clip(1e-6, 1 - 1e-6))).log().alias("ce_logit"),
        pl.when(pl.col("p_ce") == pl.col("p_ce").max().over("mid")).then(top2("mid"))
          .otherwise(pl.col("p_ce").max().over("mid")).alias("ce_rival"),
        pl.when(pl.col("p_ce") == pl.col("p_ce").max().over("s1")).then(top2("s1"))
          .otherwise(pl.col("p_ce").max().over("s1")).alias("ce_s1_best_other"),
        (conf.sum().over("mid") - conf).alias("ce_mid_n_conf_other"),   # other S1s the CE thinks claim this record
        (conf.sum().over("s1") - conf).alias("ce_s1_n_conf_other"),     # the S1's other CE-confident records
        pl.col("p_ce").rank("ordinal", descending=True).over("s1").alias("ce_rank_in_s1"),
    ).with_columns((pl.col("p_ce") - pl.col("ce_rival")).alias("ce_margin_vs_rival"))


def band_ce_features(grp, sc, name):
    """Features of a cross-encoder that scored only the uncertain band: confident stage-1 pairs
    (p >= BAND[1]) count as p = 1.0 in the competition features (they are near-certain matches)."""
    full = (grp.select("s1", "mid", "p").join(sc.select("s1", "mid", "p_ce").unique(["s1", "mid"]), on=["s1", "mid"], how="left")
               .with_columns(pl.when(pl.col("p") >= BAND[1]).then(1.0).otherwise(pl.col("p_ce")).alias("p_ce"))
               .drop_nulls("p_ce"))
    f = ce_features(full.select("s1", "mid", "p_ce"))
    return f.rename({c: (f"p_{name}" if c == "p_ce" else c.replace("ce_", f"{name}_", 1)) for c in f.columns if c not in ("s1", "mid")})


def addr_neighbor_features(band, recs, s1_vis):
    """Shared-address evidence (France shares S1 addresses 3x more than US/India):
    nb_n          other visible S1s at this S1's exact canonical address (tokens + house)
    nb_best_name  best name similarity between the record and any of those neighbours
    nb_margin     name similarity to this S1 minus nb_best_name (negative -> a neighbour fits better)
    nm_nocity     name similarity after removing the S1's own address tokens (city/locality) from both names
    s1_vis: visible S1 records of the split (entity_id, country, n_core, a_tokens, a_house)."""
    key = s1_vis.filter((pl.col("a_tokens").fill_null("") != "") & (pl.col("a_house").fill_null("") != "")).with_columns(
        (pl.col("country") + "|" + pl.col("a_tokens") + "|" + pl.col("a_house")).alias("akey"))
    grp = key.group_by("akey").agg(pl.col("entity_id").alias("members"), pl.col("n_core").alias("names")).filter(pl.col("members").list.len() > 1)
    s1k = key.select(pl.col("entity_id").alias("s1"), "akey").join(grp, on="akey")
    b = band.select("s1", "mid")
    r = recs.select("entity_id", "n_core", "a_tokens")
    b = (b.join(r.rename({"entity_id": "s1", "n_core": "s_name", "a_tokens": "s_addr"}), on="s1", how="left")
          .join(r.select(pl.col("entity_id").alias("mid"), pl.col("n_core").alias("m_name")), on="mid", how="left")
          .with_columns(pl.col(c).fill_null("") for c in ("s_name", "s_addr", "m_name")))
    # name similarity without the S1's own address tokens (city / locality words)
    def strip(n, addr_toks):
        keep = [t for t in n.split() if t not in addr_toks]
        return " ".join(keep) if keep else n
    at = [set(a.split()) for a in b["s_addr"].to_list()]
    sn = [strip(n, a) for n, a in zip(b["s_name"].to_list(), at)]
    mn = [strip(n, a) for n, a in zip(b["m_name"].to_list(), at)]
    b = b.with_columns(pl.Series("nm_nocity", _cp(sn, mn, fuzz.token_set_ratio)),
                       pl.Series("nm_self", _cp(b["s_name"].to_list(), b["m_name"].to_list(), fuzz.token_set_ratio)))
    # neighbours at the S1's address
    nb = b.join(s1k, on="s1", how="inner").explode(["members", "names"]).filter(pl.col("members") != pl.col("s1"))
    if nb.height:
        nb = nb.with_columns(pl.Series("nsim", _cp(nb["m_name"].to_list(), nb["names"].fill_null("").to_list(), fuzz.token_set_ratio)))
        agg = nb.group_by("s1", "mid").agg(pl.len().alias("nb_n"), pl.col("nsim").max().alias("nb_best_name"))
    else:
        agg = pl.DataFrame(schema={"s1": pl.Utf8, "mid": pl.Utf8, "nb_n": pl.UInt32, "nb_best_name": pl.Float32})
    b = b.join(agg, on=["s1", "mid"], how="left").with_columns(pl.col("nb_n").fill_null(0), pl.col("nb_best_name").fill_null(0.0))
    b = b.with_columns((pl.col("nm_self") - pl.col("nb_best_name")).alias("nb_margin"))
    return b.select("s1", "mid", "nb_n", "nb_best_name", "nb_margin", "nm_nocity")


def build(pred, recs, stage1_band=None, ce=None, band_ces=(), s1_vis=None):
    """pred: (s1, mid, p[, y]) for ALL scored candidates of a query set.
    stage1_band: optional frame of full stage-1 feature rows for band pairs (saved with
    ER_SAVE_BAND=1); its numeric columns are joined with an f1_ prefix ("rich" stage 2).
    Returns (band frame with stage-2 features, feature list)."""
    grp = group_features(pred)
    band = grp.filter((pl.col("p") >= BAND[0]) & (pl.col("p") < BAND[1]))
    if stage1_band is not None:
        keep = [c for c in stage1_band.columns if c not in ("s1", "mid", "p", "y", "hit") and stage1_band[c].dtype.is_numeric()]
        import os
        drop = {c for c in os.environ.get("ER_DROP_FEATS", "").split(",") if c}
        if os.environ.get("ER_DROP_COUNTS") == "1":  # keep stage 2 consistent with a count-free stage 1
            drop |= {"amb_country", "amb_region", "pool_cnt_c", "pool_cnt_s", "rare_tok_idf", "nm_idf_total1", "nm_extra_tok_idf"}
        keep = [c for c in keep if c not in drop]
        band = band.join(stage1_band.select(["s1", "mid"] + keep).rename({c: f"f1_{c}" for c in keep}),
                         on=["s1", "mid"], how="left")
    if ce is not None:  # missing CE score (pair outside the exported set) stays null; LightGBM handles it
        band = band.join(ce_features(ce), on=["s1", "mid"], how="left")
    for name, sc in band_ces:  # band-only cross-encoders (CE v2, CE 3)
        band = band.join(band_ce_features(grp, sc, name), on=["s1", "mid"], how="left")
    if s1_vis is not None:
        band = band.join(addr_neighbor_features(band, recs, s1_vis), on=["s1", "mid"], how="left")
    import os
    if os.environ.get("ER_S2_RP"):  # record prior: P(record is a true record | record alone), see record_prior.py
        rp = pl.read_parquet(os.environ["ER_S2_RP"])
        band = band.join(rp, on="mid", how="left")
    band = sibling_features(band, grp, recs)
    band = band.join(pair_features(band.select("s1", "mid"), recs), on=["s1", "mid"], how="left")
    band = band.with_columns(pl.col("p").clip(1e-6, 1 - 1e-6).log().alias("logp"),
                             (1 - pl.col("p").clip(1e-6, 1 - 1e-6)).log().alias("log1mp"))
    feats = [c for c in band.columns if c not in ("s1", "mid", "y", "fold", "hit", "country", "region") and band[c].dtype.is_numeric()]
    import os
    if os.environ.get("ER_S2_SAFE") == "1":  # v4: drop count/rank competition features (shift on test)
        unsafe = {"g_rank_in_mid", "g_n_claimants", "g_n_cand", "g_rank_in_s1"}
        feats = [c for c in feats if c not in unsafe]
    return band, feats


S2_ROUNDS = int(__import__("os").environ.get("ER_S2_ROUNDS", "2000"))  # early-stopping cap (v6 folds hit 2000)


S2_BAGS = int(__import__("os").environ.get("ER_S2_BAGS", "1"))  # seed-bagged stage 2 (averaged)


class Bag:
    """Average of several LightGBM boosters trained with different seeds (same interface subset)."""
    def __init__(self, models):
        self.models = models
        self.best_iteration = int(np.mean([m.best_iteration or m.current_iteration() for m in models]))

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def feature_importance(self, kind):
        return np.sum([m.feature_importance(kind) for m in self.models], axis=0)

    def save_model(self, path):
        for i, m in enumerate(self.models):
            m.save_model(path if i == 0 else f"{path}.bag{i}")


def load_bag(path):
    import os
    models, i = [lgb.Booster(model_file=path)], 1
    while os.path.exists(f"{path}.bag{i}"):
        models.append(lgb.Booster(model_file=f"{path}.bag{i}"))
        i += 1
    return Bag(models)


def _fit1(band, feats, rounds, valid, seed):
    params = dict(S2_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    dtr = lgb.Dataset(band.select(feats).to_numpy().astype(np.float32), band["y"].to_numpy(), feature_name=feats)
    if valid is None:
        return lgb.train(params, dtr, rounds)
    dva = lgb.Dataset(valid.select(feats).to_numpy().astype(np.float32), valid["y"].to_numpy(), reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def fit(band, feats, rounds=None, valid=None):
    rounds = rounds or S2_ROUNDS
    ms = [_fit1(band, feats, rounds, valid, S2_PARAMS["seed"] + 10 * b) for b in range(S2_BAGS)]
    return ms[0] if S2_BAGS == 1 else Bag(ms)


def rescore(pred, band, model, feats):
    """Replace stage-1 p by stage-2 p for band pairs; others keep stage-1 p."""
    s2 = band.select("s1", "mid").with_columns(
        pl.Series("p2", model.predict(band.select(feats).to_numpy().astype(np.float32))))
    return (pred.join(s2, on=["s1", "mid"], how="left")
                .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))
