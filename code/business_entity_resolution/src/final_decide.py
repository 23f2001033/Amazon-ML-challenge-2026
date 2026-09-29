"""Final decision layer on top of the stage-2 probabilities (docs/04_EXPERIMENTS.md, 27 Sep).

  python final_decide.py <config.json> <out_dir>

config keys:
  s2name          stage-2 artifact whose test predictions (features/test_pred_<s2name>.parquet) are decided
  thr, thr_by_country
                  partition + per-country thresholds. US 0.60 / India 0.50 were tuned on the held-out
                  universes after thinning orphan-like records to the density measured on test.
  swap_veto       list of countries. Removes accepted pairs at an EXACT address where the record swaps in a
                  different real business-name word: "Pornic Club SAS" -> "Pornic Sportive SAS". The pattern
                  is the generator's same-address sister-business distractor: 2-4% true in labelled US/India,
                  but accepted ~100x more often in France, where names are City/Person + category word.
  add_drop_noise  list of countries. Accepts rejected best-candidate pairs at an exact/close address whose
                  record only drops S1 words and adds generator noise words (98-99% true in US/India).
  swap_veto_final list of countries. The swap veto re-applied (up to 3 rounds) to the final decisions, catching swaps
                  accepted below the base threshold or promoted after a veto.
  add_acronym     list of countries. Accepts rejected best-candidate pairs whose record name is the acronym of the S1
                  name at an exact/close address (US/India labelled: 100% true; France rejects ~950).
  thr_by_bucket   {country: {bucket: thr}}: thresholds per country and per S1 candidate-count bucket (1, 2-3, 4-6, 7-10,
                  11+), cross-fitted val<->wide in bucket_thr.py (+0.00005 test-like, positive in all 4 cells).
  empty_rescue    threshold. An S1 left empty takes its best still-unassigned candidate if p >= it.
Writes <out_dir>/matching_results.tsv, plus <out_dir>/candidate_pairs.tsv: the pruned candidate set
(stage-1 p >= 0.02) that stage 2 and this layer run on. Every match is inside it.
"""
import json
import os
import sys

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import FEAT_DIR, PREPARED_DIR
from model import assign

COLS = ["entity_id", "country", "n_core", "a_tokens", "a_house", "a_null"]
NOISE_RATIO, NOISE_MIN_FREQ, REAL_MIN_DF = 1.1, 0.5, 20
# generator noise / legal words for the drop+noise rule (the words the noise step appends in US/India/France)
NOISE_WORDS = {"groupe", "developpement", "holding", "participations", "distribution", "international", "associes", "services",
               "fils", "cie", "france", "et", "center", "service", "holdings", "group", "enterprises", "partners", "co", "company",
               "llc", "inc", "corp", "ltd", "limited", "pvt", "private"}


def vocab():
    """Per country: generator noise words (over-represented in S2/S3 names vs S1 names) and real name words."""
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["country", "n_core"])
    pool = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"test_s{i}.parquet"), columns=["country", "n_core"]) for i in (2, 3)])
    tk = lambda d: (d.select("country", pl.col("n_core").fill_null("").str.split(" ").list.unique().alias("t")).explode("t")
                     .filter(pl.col("t") != "").group_by("country", "t").len())
    a = tk(s1).join(s1.group_by("country").len("N1"), on="country").with_columns((pl.col("len") / pl.col("N1") * 1000).alias("f1")).select("country", "t", pl.col("len").alias("df1"), "f1")
    b = tk(pool).join(pool.group_by("country").len("N2"), on="country").with_columns((pl.col("len") / pl.col("N2") * 1000).alias("f2")).select("country", "t", "f2")
    j = a.join(b, on=["country", "t"], how="full", coalesce=True).with_columns(pl.col("f1").fill_null(0.0), pl.col("f2").fill_null(0.0), pl.col("df1").fill_null(0))
    cs = j["country"].unique().to_list()
    noise = {c: set(j.filter((pl.col("country") == c) & (pl.col("f2") >= NOISE_MIN_FREQ) & (pl.col("f2") / (pl.col("f1") + 0.05) >= NOISE_RATIO))["t"].to_list()) for c in cs}
    real = {c: set(j.filter((pl.col("country") == c) & (pl.col("df1") >= REAL_MIN_DF))["t"].to_list()) for c in cs}
    return noise, real


def _subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def attach(pairs):
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=COLS).select(
        pl.col("entity_id").alias("s1"), pl.col("country").alias("ctry"), pl.col("n_core").alias("c1"), pl.col("a_tokens").alias("a1"), pl.col("a_house").alias("h1"))
    rec = pl.concat([pl.read_parquet(os.path.join(PREPARED_DIR, f"test_s{i}.parquet"), columns=COLS) for i in (2, 3)]).select(
        pl.col("entity_id").alias("mid"), pl.col("n_core").alias("c2"), pl.col("a_tokens").alias("a2"), pl.col("a_house").alias("h2"), pl.col("a_null").alias("null2"))
    d = pairs.join(s1, on="s1").join(rec, on="mid").with_columns(pl.col(c).fill_null("") for c in ("c1", "c2", "a1", "a2", "h1", "h2"))
    return d.with_columns(pl.Series("ad", cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))


def swap_pairs(accepted, countries):
    """Accepted pairs at an exact address whose record swaps in a real, non-noise, non-typo word."""
    noise, real = vocab()
    d = attach(accepted).filter(pl.col("ctry").is_in(countries))
    d = d.filter((pl.col("h1") == pl.col("h2")) & (pl.col("h1") != "") & (pl.col("ad") >= 90) & (pl.col("null2").fill_null(1) == 0))
    keep = []
    for c, a, b in d.select("ctry", "c1", "c2").iter_rows():
        s, r = set(a.split()) - {""}, set(b.split()) - {""}
        add, rem = r - s, s - r
        nz, rl = noise.get(c, set()), real.get(c, set())
        hit = [w for w in add if w not in nz and w in rl and len(w) >= 4
               and not any(fuzz.ratio(w, x) >= 75 or _subseq(w, x) or _subseq(x, w) for x in rem)]
        keep.append(bool(hit) and bool(rem))
    return d.filter(pl.Series(keep)).select("s1", "mid")


def drop_noise_pairs(best_rejected, countries):
    d = attach(best_rejected).filter(pl.col("ctry").is_in(countries) & (pl.col("null2").fill_null(1) == 0) & (pl.col("ad") >= 70))
    keep = []
    for c, a, b, h1, h2, ad in d.select("ctry", "c1", "c2", "h1", "h2", "ad").iter_rows():
        s, r = set(a.split()) - {""}, set(b.split()) - {""}
        add = r - s
        exact = h1 == h2 and h1 != "" and ad >= 90
        close = ad >= 70 and not (h1 != h2 and h1 != "" and h2 != "" and ad >= 80)
        keep.append(bool(r) and r != s and bool(add) and add <= NOISE_WORDS and (exact or close))
    return d.filter(pl.Series(keep)).select("s1", "mid")


ACRO_STOP = {"&", "et", "de", "du", "des", "la", "le", "les", "of", "and", "the"}


def _acronym(c1, c2):
    """record core name = initials of the S1 core name ("manuel benkadi amicale" -> "mba")."""
    words = [w for w in (c1 or "").split() if w]
    r = (c2 or "").replace(" ", "").replace(".", "")
    if len(r) < 2 or len(words) < 2:
        return False
    return r in ("".join(w[0] for w in words if w not in ACRO_STOP), "".join(w[0] for w in words))


def acronym_pairs(best_rejected, countries):
    """Rejected best-candidate pairs whose record is the acronym of the S1 name at an exact/close address.
    Labelled US/India: 4,853/4,853 accepted and 30/30 rejected such pairs are true (fr_cells2 cells, test-like);
    France rejects ~950 of them (the French cross-encoders under-score initials)."""
    d = attach(best_rejected).filter(pl.col("ctry").is_in(countries) & (pl.col("null2").fill_null(1) == 0) & (pl.col("ad") >= 70))
    d = d.filter(~((pl.col("h1") != pl.col("h2")) & (pl.col("h1") != "") & (pl.col("h2") != "") & (pl.col("ad") >= 80)))
    keep = [_acronym(a, b) for a, b in d.select("c1", "c2").iter_rows()]
    return d.filter(pl.Series(keep, dtype=pl.Boolean)).select("s1", "mid")


def main(cfg_path, out_dir):
    cfg = json.load(open(cfg_path))
    pred = pl.read_parquet(os.path.join(FEAT_DIR, f"test_pred_{cfg['s2name']}.parquet"))   # s1, mid, p, country
    thr, by = cfg["thr"], cfg.get("thr_by_country")
    bt = cfg.get("thr_by_bucket")   # {country: {bucket: thr}}; bucket = number of pruned candidates of the S1 (bucket_thr.py)
    if bt:
        edges = [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 10, "7-10"), (11, 10 ** 9, "11+")]
        pred = pred.join(pred.group_by("s1").len("nc"), on="s1")
        b = pl.lit("11+")
        for lo, hi, name in reversed(edges):
            b = pl.when((pl.col("nc") >= lo) & (pl.col("nc") <= hi)).then(pl.lit(name)).otherwise(b)
        keymap = {f"{c}|{k}": float(v) for c, d in bt.items() for k, v in d.items()}
        pred = pred.with_columns(pl.coalesce(
            (pl.col("country") + "|" + b).replace_strict(keymap, default=None, return_dtype=pl.Float64),
            pl.col("country").replace_strict(by or {}, default=thr, return_dtype=pl.Float64)).alias("t_row")).drop("nc")

    def decide(df):
        if not bt:
            return assign(df, thr, by)
        best = df.sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
        out = {}
        for s, x in best.filter(pl.col("p") >= pl.col("t_row")).select("s1", "mid").iter_rows():
            out.setdefault(s, set()).add(x)
        return out

    if cfg.get("swap_veto"):
        base = assign(pred, thr, {c: thr for c in by} if by else None) if cfg.get("veto_at_base_thr", True) else assign(pred, thr, by)
        acc = pl.DataFrame([(s, x) for s, xs in base.items() for x in xs], schema=["s1", "mid"], orient="row")
        veto = swap_pairs(acc, cfg["swap_veto"]).with_columns(pl.lit(1).alias("v"))
        print(f"swap veto: {veto.height:,} pairs", flush=True)
        pred = pred.join(veto, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
    if cfg.get("llm_rules"):
        # LLM judge (Qwen2.5-7B-Instruct, Apache-2.0, zero-shot P(Yes)) on the uncertain pairs of the listed countries:
        #   accept: stage-2 p >= accept_p and P(Yes) >= accept_llm  -> p = 0.99
        #   veto:   stage-2 p <  veto_p_max and P(Yes) <= veto_llm  -> p = 0
        lr = cfg["llm_rules"]
        sc = pl.concat([pl.read_parquet(f) for f in lr["files"]]).unique(["s1", "mid"], keep="first")
        pred = pred.join(sc, on=["s1", "mid"], how="left")
        inside = pl.col("country").is_in(lr["countries"]) & pl.col("p_llm").is_not_null()
        acc_rule = inside & (pl.col("p") >= lr.get("accept_p", 2.0)) & (pl.col("p_llm") >= lr.get("accept_llm", 2.0))
        veto_rule = inside & (pl.col("p") < lr.get("veto_p_max", -1.0)) & (pl.col("p_llm") <= lr.get("veto_llm", -1.0))
        print(f"LLM rules: accept {pred.filter(acc_rule).height:,} pairs, veto {pred.filter(veto_rule).height:,} pairs", flush=True)
        pred = pred.with_columns(pl.when(veto_rule).then(0.0).when(acc_rule).then(pl.max_horizontal("p", pl.lit(0.99))).otherwise(pl.col("p")).alias("p")).drop("p_llm")
    if cfg.get("add_drop_noise"):
        m = decide(pred)
        taken = {x for xs in m.values() for x in xs}
        best = pred.filter(pl.col("p") >= 0.05).sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
        rej = best.filter(~pl.col("mid").is_in(list(taken))).select("s1", "mid")
        add = drop_noise_pairs(rej, cfg["add_drop_noise"]).with_columns(pl.lit(1).alias("r"))
        print(f"drop+noise additions: {add.height:,} pairs", flush=True)
        pred = pred.join(add, on=["s1", "mid"], how="left").with_columns(
            pl.when(pl.col("r") == 1).then(pl.max_horizontal("p", pl.lit(0.99))).otherwise(pl.col("p")).alias("p")).drop("r")
    if cfg.get("add_acronym"):
        m = decide(pred)
        taken = {x for xs in m.values() for x in xs}
        best = pred.filter(pl.col("p") >= 0.05).sort(["p", "s1"], descending=[True, False]).unique("mid", keep="first", maintain_order=True)
        rej = best.filter(~pl.col("mid").is_in(list(taken))).select("s1", "mid")
        add = acronym_pairs(rej, cfg["add_acronym"]).with_columns(pl.lit(1).alias("r"))
        print(f"acronym additions: {add.height:,} pairs", flush=True)
        pred = pred.join(add, on=["s1", "mid"], how="left").with_columns(
            pl.when(pl.col("r") == 1).then(pl.max_horizontal("p", pl.lit(0.99))).otherwise(pl.col("p")).alias("p")).drop("r")
    if cfg.get("swap_veto_final"):
        # the swap veto re-applied to the final decisions: pairs accepted below the base threshold, or records whose
        # next-best S1 took over after a veto, can be the same category swap (fr_cells2.py: 1,033 on v6fr3x)
        for _ in range(3):
            m = decide(pred)
            acc = pl.DataFrame([(s, x) for s, xs in m.items() for x in xs], schema=["s1", "mid"], orient="row")
            v2 = swap_pairs(acc, cfg["swap_veto_final"]).with_columns(pl.lit(1).alias("v"))
            print(f"final swap veto: {v2.height:,} pairs", flush=True)
            if v2.height == 0:
                break
            pred = pred.join(v2, on=["s1", "mid"], how="left").with_columns(pl.when(pl.col("v") == 1).then(0.0).otherwise(pl.col("p")).alias("p")).drop("v")
    m = decide(pred)
    if cfg.get("empty_rescue"):
        taken = {x for xs in m.values() for x in xs}
        extra = (pred.filter((pl.col("p") >= cfg["empty_rescue"]) & ~pl.col("s1").is_in(list(m.keys())) & ~pl.col("mid").is_in(list(taken)))
                     .sort(["p", "s1", "mid"], descending=[True, False, False]).unique("mid", keep="first", maintain_order=True)
                     .unique("s1", keep="first", maintain_order=True))
        print(f"empty-S1 rescues: {extra.height:,}", flush=True)
        for s, x in extra.select("s1", "mid").iter_rows():
            m[s] = {x}
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1["entity_id"].to_list():
            x = m.get(s)
            f.write(f"{s}\t{','.join(sorted(x)) if x else ''}\n")
    cand = dict(pred.select("s1", "mid").unique().group_by("s1").agg(pl.col("mid").sort().str.join(",")).iter_rows())
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in s1["entity_id"].to_list():
            f.write(f"{s}\t{cand.get(s, '')}\n")
    n = {c: [0, 0] for c in s1["country"].unique().to_list()}
    for s, c in s1.iter_rows():
        n[c][0] += 1; n[c][1] += len(m.get(s, ()))
    print("matches per S1:", {c: round(k / t, 4) for c, (t, k) in sorted(n.items())}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
