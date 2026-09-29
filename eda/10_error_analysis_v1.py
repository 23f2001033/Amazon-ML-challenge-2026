"""EDA 10: where does v1 lose macro-F0.5 on validation? (drives the v2 plan)

Loss = mean(1 - F_entity). Decomposed by error type (FP / FN), pipeline stage (blocking miss,
below threshold, lost competition), and record type (native script, null address, alias,
website, house number relation, name ambiguity).
"""
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
from config import ARTIFACT_DIR, FEAT_DIR, PARQUET_DIR, PREPARED_DIR  # noqa: E402
from evaluate import f05  # noqa: E402
from model import assign  # noqa: E402

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)
cfg = json.load(open(os.path.join(ARTIFACT_DIR, "dev_result.json")))
THR = cfg["thr"]
va = pl.read_parquet(os.path.join(FEAT_DIR, "dev_val_pred.parquet"))
uni = pl.read_parquet(os.path.join(ARTIFACT_DIR, "universes.parquet"))
q_ids = uni.filter((pl.col("universe") == "val") & pl.col("is_query"))["entity_id"]
gt = pl.read_parquet(os.path.join(PARQUET_DIR, "train_gt.parquet"))
pos = (gt.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"}))
owner = pos.rename({"s1": "owner"})
truth = {}
for s, m in pos.filter(pl.col("s1").is_in(q_ids.implode())).iter_rows():
    truth.setdefault(s, set()).add(m)
pred = assign(va, THR)

# ---- entity-level decomposition ----------------------------------------------------------
rows = []
for s in q_ids.to_list():
    T, P = truth.get(s, set()), pred.get(s, set())
    rows.append((s, len(T), len(P), len(T & P), len(P - T), len(T - P), f05(P, T)))
E = pl.DataFrame(rows, schema=["s1", "n_true", "n_pred", "tp", "fp", "fn", "f"], orient="row")
N = E.height
E = E.with_columns((1 - pl.col("f")).alias("loss"))
tot = E["loss"].sum()
print(f"val entities={N:,}  macro F0.5={E['f'].mean():.4f}  total loss={tot / N:.4f}")
cat = (pl.when((pl.col("n_true") == 0) & (pl.col("fp") > 0)).then(pl.lit("singleton given a false match"))
         .when((pl.col("n_true") > 0) & (pl.col("n_pred") == 0)).then(pl.lit("has matches, predicted empty"))
         .when((pl.col("fp") > 0) & (pl.col("fn") > 0)).then(pl.lit("both FP and FN"))
         .when(pl.col("fp") > 0).then(pl.lit("extra false match(es) only"))
         .when(pl.col("fn") > 0).then(pl.lit("missed match(es) only"))
         .otherwise(pl.lit("perfect")))
print(E.with_columns(cat.alias("type")).group_by("type").agg(
    pl.len().alias("entities"), (pl.col("loss").sum() / N).round(5).alias("loss_pts"),
    (pl.col("loss").sum() / tot).round(3).alias("share_of_loss")).sort("loss_pts", descending=True))

# ---- pair-level: FN by stage, FP by kind --------------------------------------------------
va_q = va.filter(pl.col("s1").is_in(q_ids.implode()))
P = pl.DataFrame([(s, m) for s, ms in pred.items() for m in ms], schema=["s1", "mid"], orient="row")
Tq = pos.filter(pl.col("s1").is_in(q_ids.implode()))
fn = Tq.join(P, on=["s1", "mid"], how="anti").join(va_q.select("s1", "mid", "p"), on=["s1", "mid"], how="left")
winner = P.rename({"s1": "assigned_to"})
fn = fn.join(winner, on="mid", how="left").with_columns(
    pl.when(pl.col("p").is_null()).then(pl.lit("blocking miss (not a candidate)"))
      .when(pl.col("assigned_to").is_not_null()).then(pl.lit("lost competition to another S1"))
      .otherwise(pl.lit("scored below threshold")).alias("stage"))
print(f"\nFalse negatives: {fn.height:,} pairs")
print(fn.group_by("stage").len().sort("len", descending=True))
print("p distribution of below-threshold FNs:",
      fn.filter(pl.col("stage") == "scored below threshold")["p"].describe().to_dicts())
fp = P.join(Tq, on=["s1", "mid"], how="anti").join(owner, on="mid", how="left").with_columns(
    pl.when(pl.col("owner").is_null()).then(pl.lit("record matches no S1 (true distractor)"))
      .when(pl.col("owner").is_in(q_ids.implode())).then(pl.lit("record belongs to another queried S1"))
      .otherwise(pl.lit("record belongs to a dropped S1 (orphan)")).alias("kind"))
fp = fp.join(va_q.select("s1", "mid", "p"), on=["s1", "mid"], how="left")
print(f"\nFalse positives: {fp.height:,} pairs")
print(fp.group_by("kind").len().sort("len", descending=True))

# ---- record-type profile of errors ----------------------------------------------------------
cols = ["entity_id", "n_core", "n_compact", "n_native", "n_alias", "n_web", "a_null", "a_house", "a_nums", "region", "name", "address"]
recs = pl.concat([pl.scan_parquet(os.path.join(PREPARED_DIR, f)) for f in ("train_s1.parquet", "train_s2.parquet", "train_s3.parquet")]) \
         .select(cols).filter(pl.col("entity_id").is_in(pl.concat([fn["s1"], fn["mid"], fp["s1"], fp["mid"], Tq["mid"]]).unique().implode())) \
         .collect(engine="streaming")
a = recs.rename({c: c + "_1" for c in cols if c != "entity_id"})
b = recs.rename({c: c + "_2" for c in cols if c != "entity_id"})


def profile(df, label):
    d = df.join(a, left_on="s1", right_on="entity_id").join(b, left_on="mid", right_on="entity_id")
    return d.select(
        pl.lit(label).alias("set"), pl.len().alias("pairs"),
        pl.col("n_native_2").mean().round(3).alias("native"), pl.col("a_null_2").mean().round(3).alias("addr_null"),
        pl.col("n_alias_2").mean().round(3).alias("alias"), pl.col("n_web_2").mean().round(3).alias("web"),
        (pl.col("n_compact_1") == pl.col("n_compact_2")).mean().round(3).alias("core_eq"),
        ((pl.col("a_house_1") == pl.col("a_house_2")) & (pl.col("a_house_1") != "")).mean().round(3).alias("house_eq"),
        (pl.col("region_1") == pl.col("region_2")).mean().round(3).alias("region_eq"),
    ), d


prof_all, _ = profile(Tq, "all true pairs")
prof_fn, dfn = profile(fn, "false negatives")
prof_fp, dfp = profile(fp, "false positives")
print("\nRecord-type profile:")
print(pl.concat([prof_all, prof_fn, prof_fp]))
for label, d in (("FALSE NEGATIVES (missed)", dfn), ("FALSE POSITIVES (wrong merges)", dfp)):
    print(f"\n{label}: examples")
    for r in d.sample(min(12, d.height), seed=1).iter_rows(named=True):
        extra = r.get("stage") or r.get("kind")
        print(f"  [{extra}; p={r['p'] if r['p'] is not None else 'n/a'}]")
        print(f"     S1 | {r['name_1']!s:<42} | {r['address_1']}")
        print(f"     {r['mid'][:2]} | {r['name_2']!s:<42} | {r['address_2']}")
