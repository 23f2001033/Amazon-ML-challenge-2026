"""EDA 9: key figures for the EDA report (static PNGs, light surface)."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from common import load, OUT, FIG

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=150)
    plt.close(fig)


# 1. matches per S1 -----------------------------------------------------------------------
g = pl.read_parquet(f"{OUT}/gt_cardinality.parquet")["n"].value_counts().sort("n")
fig, ax = plt.subplots(figsize=(7, 3.4))
tot = g["count"].sum()
ax.bar(g["n"], g["count"] / tot * 100, width=0.6, color=BLUE)
ax.bar([0], [g.filter(pl.col("n") == 0)["count"][0] / tot * 100], width=0.6, color=ORANGE)
ax.annotate("singletons 5.6%", (0, 5.6), xytext=(-0.4, 20), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_title("Matches per S1 entity (train)")
ax.set_xlabel("number of matched S2 + S3 records"); ax.set_ylabel("% of S1 entities")
ax.set_xticks(range(0, 12)); ax.grid(axis="x", visible=False)
save(fig, "01_matches_per_s1")

# 2. blocking recall@k ---------------------------------------------------------------------
rows = []
for line in open(f"{OUT}/04_blocking_recall.txt", encoding="utf-8"):
    m = re.match(r"k=\s*(\d+): name=([\d.]+) addr=([\d.]+) union=([\d.]+)", line)
    if m:
        rows.append([float(x) for x in m.groups()])
k, rn, ra, ru = zip(*rows)
fig, ax = plt.subplots(figsize=(7, 3.6))
for y, c, lab in [(ru, AQUA, "name ∪ address"), (ra, ORANGE, "address TF-IDF"), (rn, BLUE, "name TF-IDF")]:
    ax.plot(k, [v * 100 for v in y], color=c, lw=2, marker="o", ms=4)
    ax.annotate(f"{lab}  {y[-1]*100:.1f}%", (k[-1], y[-1] * 100), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
ax.set_title("Blocking recall@k (char-3gram TF-IDF kNN, same country)")
ax.set_xlabel("k nearest neighbours per S1"); ax.set_ylabel("% of true pairs retrieved")
ax.set_ylim(0, 100); ax.set_xlim(0, 75)
save(fig, "02_blocking_recall_at_k")

# 3. positive vs hard negative ------------------------------------------------------------
F = pl.read_parquet(f"{OUT}/04_pair_features.parquet")
feats = [("core_exact", "core name identical", 1), ("name_jacc", "name token Jaccard", 1),
         ("name_tset", "name token-set ratio", 100), ("addr_tset", "address token-set ratio", 100),
         ("num_jacc", "address-number Jaccard", 1), ("first_num_eq", "house number equal", 1)]
agg = F.group_by("y").agg([pl.col(f).filter(pl.col(f) >= 0).mean().alias(f) for f, _, _ in feats]).sort("y")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ys = range(len(feats))
neg = [agg.filter(pl.col("y") == 0)[f][0] / s * 100 for f, _, s in feats]
pos = [agg.filter(pl.col("y") == 1)[f][0] / s * 100 for f, _, s in feats]
h = 0.36
ax.barh([y + h / 2 + 0.02 for y in ys], pos, height=h, color=BLUE, label="true match")
ax.barh([y - h / 2 - 0.02 for y in ys], neg, height=h, color=ORANGE, label="hard negative (top-10 name neighbour)")
ax.set_yticks(list(ys)); ax.set_yticklabels([l for _, l, _ in feats]); ax.invert_yaxis()
ax.set_xlabel("mean value (scaled to 0–100)"); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.set_title("Address numbers separate matches from hard negatives")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=2, fontsize=9)
save(fig, "03_pos_vs_hardneg")

# 4. country mix train vs test -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))
data = {}
for sp in ["train", "test"]:
    c = load(f"{sp}_s1")["country"].value_counts()
    data[sp] = {r["country"]: r["count"] / c["count"].sum() * 100 for r in c.iter_rows(named=True)}
countries = ["US", "India", "France"]
cols = [BLUE, ORANGE, AQUA]
for i, sp in enumerate(["train", "test"]):
    left = 0
    for ctry, col in zip(countries, cols):
        v = data[sp].get(ctry, 0)
        ax.barh(i, v, left=left, color=col, height=0.5, edgecolor=SURFACE, linewidth=2,
                label=ctry if i == 1 else None)
        if v > 4:
            ax.text(left + v / 2, i, f"{ctry} {v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels(["train S1", "test S1"]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("% of S1 entities"); ax.grid(axis="y", visible=False)
ax.set_title("Country mix shifts: test is India-heavy and adds unseen France")
h_, l_ = ax.get_legend_handles_labels()
ax.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=9)
save(fig, "04_country_mix")
print("figures written to", FIG)
