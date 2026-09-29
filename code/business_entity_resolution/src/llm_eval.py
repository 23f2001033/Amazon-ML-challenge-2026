"""Evaluate the LLM judge (Qwen2.5-7B-Instruct zero-shot, P(Yes)) on labelled US/India uncertain pairs, then
size its effect on France.  Files in /home/ubuntu/er/llm: llm_val.parquet (+y), llm_val_scores.parquet,
llm_fr.parquet, llm_fr_scores.parquet (may be partial)."""
import os
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
L = "/home/ubuntu/er/llm"
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
v = pl.read_parquet(f"{L}/llm_val.parquet").join(pl.read_parquet(f"{L}/llm_val_scores.parquet"), on=["s1", "mid"])
print(f"validation pairs scored: {v.height:,} (true share {v['y'].mean():.3f})")
y = v["y"].to_numpy(); p = v["p"].to_numpy(); q = v["p_llm"].to_numpy()
X = np.c_[lg(p), lg(q), lg(p) * lg(q)]
comb = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
print(f"AUC  stage-2 {roc_auc_score(y, p):.4f} | LLM {roc_auc_score(y, q):.4f} | combined (5-fold) {roc_auc_score(y, comb):.4f}")
for c in ("US", "India"):
    m = (v["country"] == c).to_numpy()
    print(f"   {c}: stage-2 {roc_auc_score(y[m], p[m]):.4f} LLM {roc_auc_score(y[m], q[m]):.4f} combined {roc_auc_score(y[m], comb[m]):.4f}")
v = v.with_columns(pl.col("p").cut([0.3, 0.5, 0.725, 0.9], labels=["p<.3", "p.3-.5", "p.5-.725", "p.725-.9", "p>=.9"]).alias("pb"),
                   pl.col("p_llm").cut([0.1, 0.5, 0.9, 0.98], labels=["L<.1", "L.1-.5", "L.5-.9", "L.9-.98", "L>=.98"]).alias("lb"))
print("P(true) by stage-2 band x LLM band (n):")
t = v.group_by("pb", "lb").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("P"))
print(t.with_columns(pl.concat_str([pl.col("P").cast(pl.String), pl.lit(" ("), pl.col("n").cast(pl.String), pl.lit(")")]).alias("cell"))
       .pivot(on="lb", index="pb", values="cell").sort("pb"))
if os.path.exists(f"{L}/llm_fr_scores.parquet"):
    f = pl.read_parquet(f"{L}/llm_fr.parquet").join(pl.read_parquet(f"{L}/llm_fr_scores.parquet"), on=["s1", "mid"])
    print(f"\nFrance pairs scored: {f.height:,}")
    print("sanity: vetoed category swaps, LLM P(Yes) quantiles:", f.filter(pl.col("vetoed") == 1)["p_llm"].quantile(0.25), f.filter(pl.col("vetoed") == 1)["p_llm"].median(),
          "| share < 0.5:", round((f.filter(pl.col("vetoed") == 1)["p_llm"] < 0.5).mean(), 3))
    fv = f.filter(pl.col("vetoed") == 0).with_columns(pl.col("p").cut([0.3, 0.5, 0.725, 0.9], labels=["p<.3", "p.3-.5", "p.5-.725", "p.725-.9", "p>=.9"]).alias("pb"),
                                                     pl.col("p_llm").cut([0.1, 0.5, 0.9, 0.98], labels=["L<.1", "L.1-.5", "L.5-.9", "L.9-.98", "L>=.98"]).alias("lb"))
    print("France counts by stage-2 band x LLM band:")
    print(fv.group_by("pb", "lb").len().pivot(on="lb", index="pb", values="len").sort("pb"))
    raw = fv.filter((pl.col("p") < 0.725) & (pl.col("p_llm") >= 0.98)).sample(min(12, fv.height), seed=1)
    print("examples: rejected by stage 2, LLM >= 0.98:")
    for r in raw.select("p", "p_llm", "name1", "addr1", "name2", "addr2").rows():
        print(f"  p={r[0]:.2f} L={r[1]:.3f} {r[2][:30]:<30} | {r[3][:30]:<30} -> {r[4][:30]:<30} | {r[5][:30]}")
    raw = fv.filter((pl.col("p") >= 0.725) & (pl.col("p_llm") < 0.1)).sample(min(12, fv.filter((pl.col("p") >= 0.725) & (pl.col("p_llm") < 0.1)).height), seed=1)
    print("examples: accepted by stage 2, LLM < 0.1:")
    for r in raw.select("p", "p_llm", "name1", "addr1", "name2", "addr2").rows():
        print(f"  p={r[0]:.2f} L={r[1]:.3f} {r[2][:30]:<30} | {r[3][:30]:<30} -> {r[4][:30]:<30} | {r[5][:30]}")
