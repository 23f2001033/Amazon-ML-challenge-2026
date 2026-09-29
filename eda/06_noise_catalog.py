"""EDA 6: catalogue of the transformations that turn an S1 record into its S2/S3 matches.

For a sample of positive pairs we tag each (S1, match) pair with the noise operations that
are visible, per source and country. Also measures native-script (Indic) token vocabulary:
how many distinct native tokens exist and how much of the TEST native vocabulary is covered
by TRAIN (-> can a transliteration dictionary be learned from train alone?).
"""
import re
from collections import Counter

import polars as pl
from common import load, gt_pairs, OUT
from norm import norm_name, core_name, norm_addr, addr_numbers, us_state, basic, LEGAL

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
N = 150_000
INDIC = re.compile(r"[ऀ-ൿ]")

pos = gt_pairs().sample(N, seed=7)
recs = pl.concat([load("train_s1"), load("train_s2"), load("train_s3")]).filter(
    pl.col("entity_id").is_in(pl.concat([pos["s1"], pos["mid"]]).implode()))
rd = {r[0]: r[1:] for r in recs.iter_rows()}


def legal_set(n):
    return {t for t in norm_name(n).split() if t in LEGAL.values()}


rows = []
for s1, mid, src in pos.iter_rows():
    n1, a1, c = rd[s1]
    n2, a2, _ = rd[mid]
    nn1, nn2 = norm_name(n1), norm_name(n2)
    cn1, cn2 = core_name(n1), core_name(n2)
    t1, t2 = cn1.split(), cn2.split()
    p1 = [basic(x) for x in (a1 or "").split(",") if basic(x)]
    p2 = [basic(x) for x in (a2 or "").split(",") if basic(x)]
    num1, num2 = addr_numbers(a1), addr_numbers(a2)
    rows.append(dict(
        src=src, country=c,
        name_verbatim=n1 == n2,
        name_equal_after_norm=nn1 == nn2,
        core_equal=cn1 == cn2,
        core_same_tokens_diff_order=(sorted(t1) == sorted(t2)) and t1 != t2,
        core_token_dropped=set(t2) < set(t1),
        core_token_added=set(t1) < set(t2),
        native_script_name=bool(INDIC.search(n2)),
        website_name=bool(re.search(r"(?i)(\.com|\.in|\.fr|www\.)", n2)),
        formerly=bool(re.search(r"(?i)formerly", n2)),
        dba=bool(re.search(r"(?i)\b(dba|d/b/a|t/a|aka)\b", n2)),
        legal_changed=legal_set(n1) != legal_set(n2),
        upper_name=n2.isupper(), lower_name=n2.islower(),
        accent_added=bool(re.search(r"[À-ÿ]", n2)) and not re.search(r"[À-ÿ]", n1),
        leet=bool(re.search(r"[A-Za-z][0-9][A-Za-z]", n2)) and not re.search(r"[A-Za-z][0-9][A-Za-z]", n1),
        addr_null=a2 is None or not norm_addr(a2),
        addr_verbatim=a1 == a2,
        addr_same_parts_reordered=sorted(p1) == sorted(p2) and p1 != p2,
        addr_first_part_moved=bool(p1 and p2) and p1[0] != p2[0] and p1[0] in p2,
        addr_parts_dropped=len(p2) < len(p1),
        addr_parts_added=len(p2) > len(p1),
        addr_native_script=bool(a2 and INDIC.search(a2)),
        addr_numbers_equal=num1 == num2,
        addr_numbers_subset=bool(num2) and num2 <= num1,
        addr_number_changed=bool(num1 and num2) and not (num1 & num2),
        addr_has_null_token=bool(a2 and re.search(r"(?i)(<null>|\bnull\b|\bn/a\b)", a2)),
        us_state_eq=(us_state(a1) == us_state(a2)) if c == "US" else None,
        us_state_missing=(us_state(a2) is None) if c == "US" else None,
    ))
D = pl.DataFrame(rows)
cols = [c for c in D.columns if c not in ("src", "country")]
summ = D.group_by(["country", "src"]).agg([pl.col(c).cast(pl.Float64).mean().round(3).alias(c) for c in cols]).sort(["country", "src"])
t = summ.with_columns((pl.col("country") + "|" + pl.col("src")).alias("k")).drop(["country", "src"])
tt = t.drop("k").transpose(include_header=True, header_name="operation", column_names=t["k"].to_list())
out = ["## Noise-operation frequency among positive pairs (fraction of pairs)", str(tt)]

# ---- native script vocabulary (India) ------------------------------------------------
def native_tokens(df):
    c = Counter()
    for n in df.filter(pl.col("country") == "India")["business_name"].to_list():
        for tok in n.split():
            if INDIC.search(tok):
                c[tok] += 1
    return c

tr = native_tokens(pl.concat([load("train_s2"), load("train_s3")]))
te = native_tokens(pl.concat([load("test_s2"), load("test_s3")]))
cov_types = len(set(te) & set(tr)) / len(te)
cov_tokens = sum(v for k, v in te.items() if k in tr) / sum(te.values())
out += ["\n## Native-script name tokens (India)",
        f"train distinct native tokens: {len(tr):,}  occurrences: {sum(tr.values()):,}",
        f"test  distinct native tokens: {len(te):,}  occurrences: {sum(te.values()):,}",
        f"test token TYPES seen in train: {cov_types:.3f}; test token OCCURRENCES covered: {cov_tokens:.4f}",
        "top native tokens: " + ", ".join(f"{k}({v})" for k, v in tr.most_common(30))]
txt = "\n".join(out)
open(f"{OUT}/06_noise_catalog.txt", "w", encoding="utf-8").write(txt)
print(txt)
