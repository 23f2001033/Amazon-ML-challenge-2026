"""Stage 3: pair features for (S1, candidate) pairs.

All features are country-agnostic (D-014): similarities, containments, number relations,
ambiguity counts and within-query / within-candidate rank context. Country is never a feature.

String similarities use rapidfuzz.process.cpdist (vectorised C++, multi-threaded); set features
use Polars list ops (Rust). No Python per-row loops.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

REC_COLS = ["entity_id", "country", "region", "region_src", "n_canon", "n_core", "n_compact",
            "n_legal", "n_native", "n_alias", "n_web", "n_upper", "a_tokens", "a_nums", "a_house",
            "a_null"]


def compute_stats(s1_df, pool_df):
    """Per-split statistics for E2 features (no labels used).
    s1_df: (country, n_core) of all S1; pool_df: (country, n_compact) of all S2+S3.
    Returns dict(idf=(country, tok, idf), pool_cnt=(country, n_compact, pool_cnt))."""
    n = s1_df.group_by("country").len().rename({"len": "N"})
    idf = (s1_df.with_columns(pl.col("n_core").str.split(" ").list.unique().alias("tok"))
                .explode("tok").filter(pl.col("tok") != "")
                .group_by(["country", "tok"]).len()
                .join(n, on="country")
                .with_columns((pl.col("N") / pl.col("len")).log().cast(pl.Float32).alias("idf"))
                .select("country", "tok", "idf"))
    pool_cnt = (pool_df.filter(pl.col("n_compact") != "").group_by(["country", "n_compact"]).len()
                       .rename({"len": "pool_cnt"}))
    max_idf = n.with_columns(pl.col("N").log().cast(pl.Float32).alias("max_idf")).select("country", "max_idf")
    return {"idf": idf, "pool_cnt": pool_cnt, "max_idf": max_idf}


def _idf_feats(df, stats):
    """IDF-weighted name containment both ways + rarest-S1-token match (E2/E6)."""
    idf, max_idf = stats["idf"], stats["max_idf"]
    base = df.select(pl.int_range(pl.len()).alias("_rid"), pl.col("country_1").alias("country"),
                     pl.col("n_core_1").str.split(" ").list.unique().alias("ta"),
                     pl.col("n_core_2").str.split(" ").list.unique().alias("tb"),
                     pl.col("n_compact_2").alias("cb"))
    ea = (base.select("_rid", "country", pl.col("ta").alias("tok"), "cb").explode("tok")
              .filter(pl.col("tok") != "")
              .join(idf, on=["country", "tok"], how="left")
              .join(max_idf, on="country", how="left")
              .with_columns(pl.col("idf").fill_null(pl.col("max_idf"))))
    eb = base.select("_rid", pl.col("tb").alias("tok")).explode("tok").filter(pl.col("tok") != "").unique()
    ea = ea.join(eb.with_columns(pl.lit(1.0).alias("hit")), on=["_rid", "tok"], how="left")            .with_columns(pl.col("hit").fill_null(0.0),
                         pl.col("cb").str.contains(pl.col("tok"), literal=True).cast(pl.Float32).alias("sub"))
    aggs = ea.group_by("_rid").agg(
        ((pl.col("idf") * pl.col("hit")).sum() / pl.col("idf").sum().clip(lower_bound=1e-6)).alias("nm_idf_s_in_c"),
        pl.col("hit").sort_by("idf").last().alias("rare_tok_hit"),
        pl.col("sub").sort_by("idf").last().alias("rare_tok_sub"),
        pl.col("idf").max().alias("rare_tok_idf"),
        pl.col("idf").sum().alias("nm_idf_total1"))
    ec = (base.select("_rid", "country", pl.col("tb").alias("tok")).explode("tok").filter(pl.col("tok") != "")
              .join(idf, on=["country", "tok"], how="left").join(max_idf, on="country", how="left")
              .with_columns(pl.col("idf").fill_null(pl.col("max_idf"))))
    ea_tok = base.select("_rid", pl.col("ta").alias("tok")).explode("tok").filter(pl.col("tok") != "").unique()
    ec = ec.join(ea_tok.with_columns(pl.lit(1.0).alias("hit")), on=["_rid", "tok"], how="left")            .with_columns(pl.col("hit").fill_null(0.0))
    aggc = ec.group_by("_rid").agg(
        ((pl.col("idf") * pl.col("hit")).sum() / pl.col("idf").sum().clip(lower_bound=1e-6)).alias("nm_idf_c_in_s"),
        ((1 - pl.col("hit")) * pl.col("idf")).max().alias("nm_extra_tok_idf"))
    out = (base.select("_rid").join(aggs, on="_rid", how="left").join(aggc, on="_rid", how="left")
               .sort("_rid").drop("_rid").fill_null(0.0))
    return pl.concat([df, out], how="horizontal")


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _set_feats(df, c1, c2, prefix):
    """Jaccard and both containments for space-separated token strings."""
    l1, l2 = pl.col(c1).str.split(" "), pl.col(c2).str.split(" ")
    df = df.with_columns(l1.list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("_a"),
                         l2.list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("_b"))
    df = df.with_columns(pl.col("_a").list.set_intersection("_b").list.len().alias("_i"),
                         pl.col("_a").list.len().alias("_na"), pl.col("_b").list.len().alias("_nb"))
    return df.with_columns(
        (pl.col("_i") / (pl.col("_na") + pl.col("_nb") - pl.col("_i")).clip(lower_bound=1)).alias(f"{prefix}_jacc"),
        (pl.col("_i") / pl.col("_nb").clip(lower_bound=1)).alias(f"{prefix}_c_in_s"),   # cand tokens in S1
        (pl.col("_i") / pl.col("_na").clip(lower_bound=1)).alias(f"{prefix}_s_in_c"),   # S1 tokens in cand
        pl.col("_na").alias(f"{prefix}_n1"), pl.col("_nb").alias(f"{prefix}_n2"),
    ).drop(["_a", "_b", "_i", "_na", "_nb"])


def _digits_sorted(col):
    return pl.col(col).str.split("").list.sort().list.join("")


def build(cand, s1_prep, pool_prep, s1_all=None, stats=None):
    """cand: (s1, mid, name_cos, addr_cos, hit). Returns feature frame (keeps s1, mid).
    stats: output of compute_stats() for this split; enables the v2 (E2) features."""
    a = s1_prep.select(REC_COLS).rename({c: f"{c}_1" for c in REC_COLS if c != "entity_id"})
    b = pool_prep.select(REC_COLS).rename({c: f"{c}_2" for c in REC_COLS if c != "entity_id"})
    df = (cand.join(a, left_on="s1", right_on="entity_id", how="left")
              .join(b, left_on="mid", right_on="entity_id", how="left"))

    # ---- name similarities --------------------------------------------------------------
    c1, c2 = df["n_core_1"].to_list(), df["n_core_2"].to_list()
    k1, k2 = df["n_compact_1"].to_list(), df["n_compact_2"].to_list()
    f = {
        "nm_ratio": _cp(c1, c2, fuzz.ratio), "nm_tset": _cp(c1, c2, fuzz.token_set_ratio),
        "nm_tsort": _cp(c1, c2, fuzz.token_sort_ratio), "nm_partial": _cp(c1, c2, fuzz.partial_ratio),
        "nm_jw": _cp(k1, k2, JaroWinkler.normalized_similarity),
        "nm_lev": _cp(k1, k2, Levenshtein.normalized_similarity),
        "nm_cpartial": _cp(k1, k2, fuzz.partial_ratio),
        "nm_canon_ratio": _cp(df["n_canon_1"].to_list(), df["n_canon_2"].to_list(), fuzz.ratio),
    }
    del c1, c2, k1, k2
    # ---- address similarities -----------------------------------------------------------
    t1, t2 = df["a_tokens_1"].to_list(), df["a_tokens_2"].to_list()
    f.update({"ad_ratio": _cp(t1, t2, fuzz.ratio), "ad_tset": _cp(t1, t2, fuzz.token_set_ratio),
              "ad_partial": _cp(t1, t2, fuzz.partial_ratio),
              "ad_tsort": _cp(t1, t2, fuzz.token_sort_ratio)})
    del t1, t2
    df = df.with_columns([pl.Series(k, v) for k, v in f.items()])
    del f

    df = _set_feats(df, "n_core_1", "n_core_2", "nt")
    df = _set_feats(df, "a_tokens_1", "a_tokens_2", "at")
    df = _set_feats(df, "a_nums_1", "a_nums_2", "num")

    # ---- exact / structural flags --------------------------------------------------------
    h1, h2 = pl.col("a_house_1"), pl.col("a_house_2")
    both_h = (h1 != "") & (h2 != "")
    df = df.with_columns(
        (pl.col("n_compact_1") == pl.col("n_compact_2")).cast(pl.Int8).alias("nm_compact_eq"),
        (pl.col("n_canon_1") == pl.col("n_canon_2")).cast(pl.Int8).alias("nm_canon_eq"),
        (pl.col("n_legal_1") == pl.col("n_legal_2")).cast(pl.Int8).alias("legal_eq"),
        ((pl.col("n_legal_1") != "") & (pl.col("n_legal_2") != "")).cast(pl.Int8).alias("legal_both"),
        pl.col("n_compact_2").str.contains(pl.col("n_compact_1"), literal=True).cast(pl.Int8).alias("nm_s_sub_c"),
        pl.col("n_compact_1").str.contains(pl.col("n_compact_2"), literal=True).cast(pl.Int8).alias("nm_c_sub_s"),
        (pl.col("region_1") == pl.col("region_2")).cast(pl.Int8).alias("region_eq"),
        pl.col("region_src_2").alias("region_src_c"),
        both_h.cast(pl.Int8).alias("house_both"),
        (both_h & (h1 == h2)).cast(pl.Int8).alias("house_eq"),
        (both_h & (h1.str.len_chars() == h2.str.len_chars())).cast(pl.Int8).alias("house_len_eq"),
        (both_h & (_digits_sorted("a_house_1") == _digits_sorted("a_house_2"))).cast(pl.Int8).alias("house_perm"),
        (both_h & (h1.str.contains(h2, literal=True) | h2.str.contains(h1, literal=True))).cast(pl.Int8).alias("house_sub"),
        (both_h & (h1.str.slice(-1) == h2.str.slice(-1))).cast(pl.Int8).alias("house_last_eq"),
        pl.when(both_h).then((h1.str.slice(0, 9).cast(pl.Int64, strict=False) -
                              h2.str.slice(0, 9).cast(pl.Int64, strict=False)).abs().cast(pl.Float32).log1p())
          .otherwise(-1.0).alias("house_gap_log"),
        pl.col("a_nums_1").str.split(" ").list.contains(pl.col("a_house_2")).cast(pl.Int8).alias("house_c_in_snums"),
        pl.col("a_nums_2").str.split(" ").list.contains(pl.col("a_house_1")).cast(pl.Int8).alias("house_s_in_cnums"),
        pl.col("n_native_2").alias("c_native"), pl.col("n_alias_2").alias("c_alias"),
        pl.col("n_web_2").alias("c_web"), pl.col("n_upper_2").alias("c_upper"),
        pl.col("a_null_2").alias("c_addr_null"),
        (pl.col("mid").str.slice(0, 2) == "S3").cast(pl.Int8).alias("c_is_s3"),
        pl.col("n_compact_1").str.len_chars().alias("nm_len1"),
        pl.col("n_compact_2").str.len_chars().alias("nm_len2"),
    )

    # ---- ambiguity: how many S1 entities share this compact name (country / region) -------
    s1_all = s1_prep if s1_all is None else s1_all
    amb_c = s1_all.group_by(["country", "n_compact"]).len().rename({"len": "amb_country"})
    amb_r = s1_all.group_by(["country", "region", "n_compact"]).len().rename({"len": "amb_region"})
    df = (df.join(amb_c, left_on=["country_1", "n_compact_1"], right_on=["country", "n_compact"], how="left")
            .join(amb_r, left_on=["country_1", "region_1", "n_compact_1"],
                  right_on=["country", "region", "n_compact"], how="left")
            .with_columns(pl.col("amb_country").fill_null(0), pl.col("amb_region").fill_null(0)))

    if stats is not None:  # ---- v2 / E2: pool name frequency + IDF-weighted name evidence ----
        pc = stats["pool_cnt"]
        df = (df.join(pc.rename({"pool_cnt": "pool_cnt_c"}), left_on=["country_2", "n_compact_2"],
                      right_on=["country", "n_compact"], how="left")
                .join(pc.rename({"pool_cnt": "pool_cnt_s"}), left_on=["country_1", "n_compact_1"],
                      right_on=["country", "n_compact"], how="left")
                .with_columns(pl.col("pool_cnt_c").fill_null(0), pl.col("pool_cnt_s").fill_null(0)))
        df = _idf_feats(df, stats)

    drop = [c for c in df.columns if c.endswith("_1") or c.endswith("_2")]
    return df.drop(drop)


def build_chunked(cand, s1_prep, pool_prep, s1_all=None, chunk_s1=10000, log=print,
                  out_dir=None, extra=None, stats=None, part_prefix="part"):
    """Add global context features on the thin candidate frame, then build the per-pair
    features in S1-chunks (bounded memory). Context must see ALL candidates of a query set.

    out_dir: if given, each chunk is written to out_dir/part_XXXX.parquet (streaming) and None is
             returned; otherwise the concatenated frame is returned.
    extra:   optional callable(frame) -> frame applied to each chunk before writing (e.g. labels,
             or scoring + column pruning at inference time).
    """
    import os
    cand = add_context(cand)
    s1_prep = s1_prep.filter(pl.col("entity_id").is_in(cand["s1"].unique().implode()))
    ids = cand["s1"].unique().sort()
    parts = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for n, i in enumerate(range(0, ids.len(), chunk_s1)):
        sub = cand.filter(pl.col("s1").is_in(ids.slice(i, chunk_s1).implode()))
        part = build(sub, s1_prep, pool_prep, s1_all, stats)
        part = part.with_columns([pl.col(c).cast(pl.Float32) for c in part.columns
                                  if c not in ("s1", "mid", "hit") and part[c].dtype.is_numeric()])
        if extra is not None:
            part = extra(part)
        if out_dir:
            part.write_parquet(os.path.join(out_dir, f"{part_prefix}_{n:04d}.parquet"))
        else:
            parts.append(part)
        if n % 10 == 0:
            log(f"  features: {min(i + chunk_s1, ids.len()):,}/{ids.len():,} S1")
    return None if out_dir else pl.concat(parts)


def add_context(df, score="pre"):
    """Rank/competition features within each S1 query and within each candidate record."""
    if score == "pre":
        df = df.with_columns((pl.col("name_cos") + pl.col("addr_cos")).alias("pre"))
    s = pl.col(score)
    df = df.with_columns(
        s.rank("ordinal", descending=True).over("s1").alias(f"{score}_rank_s1"),
        pl.col("name_cos").rank("ordinal", descending=True).over("s1").alias("namecos_rank_s1"),
        pl.col("addr_cos").rank("ordinal", descending=True).over("s1").alias("addrcos_rank_s1"),
        (s.max().over("s1") - s).alias(f"{score}_gap_s1"),
        pl.len().over("s1").alias("n_cand_s1"),
        s.rank("ordinal", descending=True).over("mid").alias(f"{score}_rank_mid"),
        (s.max().over("mid") - s).alias(f"{score}_gap_mid"),
        pl.len().over("mid").alias("n_s1_for_mid"),
    )
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in df.columns
                            if c not in ("s1", "mid", "hit") and df[c].dtype.is_numeric()])


def feature_columns(df):
    return [c for c in df.columns if c not in ("s1", "mid", "y", "hit", "country") and
            df[c].dtype.is_numeric()]
