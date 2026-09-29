"""French training pairs with GUARANTEED labels, built only from the test France S1 records
(no ground truth, no external data): two different S1 entities are different businesses by
definition, and an S1 record paired with a noisy copy of itself is a true match.

  python make_fr_synth.py <train3.parquet> <out_dir> [n_us_in=450000]
  -> out_dir/train_fr.parquet  (text_a, text_b, y, kind)   French synthetic + US/India raw pairs
     out_dir/holdout_fr.parquet                             5% of French S1 groups, for a sanity AUC

Positives: S1 vs noisy(S1). The noise imitates what French S2/S3 records show: street-type
abbreviations (R., AV, BD, IMP, PL, RTE, CHE, Q., CRS), 'N°'/'#'/'(n)' house prefixes, leading zeros,
region dropped or replaced by the department that S2/S3 use for that city (learned from the test
S2/S3 addresses), component reordering, upper case, accent stripping, legal-form swaps/drops,
added generic words, small typos.
Hard negatives (label 0, always different entities):
  addr   - another S1 at the SAME address (13.9% of French S1 share an address)
  name   - another S1 with the same core name elsewhere ("Bordeaux Club" x 530)
  word   - another S1 whose name shares the first word (city) but differs in the rest
  decoy  - noisy copy of the S1 with a shifted house number (planted-decoy pattern)
"""
import os
import random
import re
import sys
import unicodedata

import polars as pl

from config import PARQUET_DIR, PREPARED_DIR

LEGAL = ["SAS", "SARL", "EURL", "SA", "SASU", "EI", "SCI"]
STREET = {"rue": ["R", "R.", "RUE"], "avenue": ["AV", "Av.", "Ave", "AVE"], "boulevard": ["BD", "Bd", "Bld"],
          "allée": ["All", "ALL.", "Allee"], "allee": ["All", "ALL."], "impasse": ["Imp", "IMP"], "place": ["PL", "Pl."],
          "route": ["RTE", "Rte."], "chemin": ["CHE", "Chem."], "quai": ["Q.", "QUAI"], "cours": ["CRS", "Crs"],
          "cité": ["Cite", "CITE"], "résidence": ["Res", "RES"], "residence": ["Res", "RES"]}
EXTRA = ["Groupe", "France", "(France)", "Associés", "Services", "Centre", "Club"]
rng = random.Random(7)


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def typo(s):
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.random()
    if op < 0.4:
        return s[:i] + s[i + 1:]
    if op < 0.7:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + rng.choice("aeiourstln") + s[i + 1:]


def noisy_name(n):
    toks = n.split()
    legal = [t for t in toks if t.upper().strip(".") in LEGAL]
    core = [t for t in toks if t.upper().strip(".") not in LEGAL]
    r = rng.random()
    if legal and r < 0.30:
        legal = [rng.choice([l for l in LEGAL if l != legal[0].upper()])]
    elif legal and r < 0.50:
        legal = []
    elif not legal and r < 0.10:
        legal = [rng.choice(LEGAL)]
    if len(core) >= 3 and rng.random() < 0.08:
        core.pop(rng.randrange(1, len(core)))
    if rng.random() < 0.10:
        core.insert(rng.randrange(1, len(core) + 1), rng.choice(EXTRA))
    out = core + legal
    if legal and rng.random() < 0.25:
        out = legal + core
    s = " ".join(out) if out else n
    if rng.random() < 0.02:
        return re.sub(r"[^a-z0-9]", "", strip_acc(s).lower()) + rng.choice([".com", ".fr"])
    if rng.random() < 0.30:
        s = strip_acc(s)
    r = rng.random()
    s = s.upper() if r < 0.25 else s.lower() if r < 0.35 else s
    return typo(s) if rng.random() < 0.05 else s


def noisy_addr(a, dept_of, shift_house=False):
    parts = [p.strip() for p in a.split(",") if p.strip()]
    region = parts[-1] if len(parts) >= 3 else None
    out = []
    for p in parts:
        m = re.match(r"^(\d+)(\s*(?:bis|ter|b|a)?)\s+(.*)$", p, flags=re.I)
        if m:
            num, suf, rest = m.groups()
            if shift_house:
                num = str(int(num) + rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, -1, -2, -3, -4]))
                num = num if int(num) > 0 else str(int(num) + 20)
            words = rest.split()
            if words and words[0].lower() in STREET and rng.random() < 0.40:
                words[0] = rng.choice(STREET[words[0].lower()])
            r = rng.random()
            pre = "N°" if r < 0.10 else "No " if r < 0.15 else "#" if r < 0.18 else ""
            if rng.random() < 0.03:
                num = "0" + num
            if rng.random() < 0.03:
                num = f"({num})"
            p = f"{pre}{num}{suf} {' '.join(words)}".strip()
            if rng.random() < 0.03:
                p = " ".join(words)
        out.append(p)
    if region and region in out:
        r = rng.random()
        city = next((q for q in parts if q not in (region,) and not re.match(r"^\d", q)), None)
        if r < 0.40:
            out.remove(region)
        elif r < 0.70 and city and dept_of.get(city.lower()):
            out[out.index(region)] = dept_of[city.lower()]
    if len(out) > 1 and rng.random() < 0.25:
        k = rng.randrange(1, len(out))
        out = out[k:] + out[:k]
    s = ", ".join(out)
    if rng.random() < 0.30:
        s = s.upper()
    if rng.random() < 0.30:
        s = strip_acc(s)
    return typo(s) if rng.random() < 0.05 else s


def learn_departments(regions):
    """city -> the most frequent non-region last component of French S2/S3 addresses mentioning it."""
    pool = pl.concat([pl.read_parquet(os.path.join(PARQUET_DIR, f"test_s{i}.parquet"), columns=["business_address", "country"]) for i in (2, 3)])
    pool = pool.filter(pl.col("country") == "France").drop_nulls("business_address")
    comps = pool.select(pl.col("business_address").str.split(",").list.eval(pl.element().str.strip_chars())).to_series().to_list()
    from collections import Counter, defaultdict
    last = Counter(c[-1] for c in comps if len(c) >= 2)
    region_l = {r.lower() for r in regions}
    s1a = pl.read_parquet(os.path.join(PARQUET_DIR, "test_s1.parquet"), columns=["business_address", "country"]).filter(
        pl.col("country") == "France").drop_nulls("business_address")
    cities = {x.lower() for c in s1a.select(pl.col("business_address").str.split(",").list.eval(pl.element().str.strip_chars())).to_series().to_list()
              for x in c if not re.search(r"\d", x) and x.lower() not in region_l}
    depts = {k for k, v in last.items() if v >= 2000 and k.lower() not in region_l and k.lower() not in cities and not re.search(r"\d", k)}
    co = defaultdict(Counter)
    for c in comps:
        d = [x for x in c if x in depts]
        if d:
            for x in c:
                if x not in depts and not re.search(r"\d", x) and x.lower() not in region_l:
                    co[x.lower()][d[0]] += 1
    return {city: cnt.most_common(1)[0][0] for city, cnt in co.items() if sum(cnt.values()) >= 50}, depts


def main(train3, out, n_us_in="450000"):
    os.makedirs(out, exist_ok=True)
    raw = pl.read_parquet(os.path.join(PARQUET_DIR, "test_s1.parquet"), columns=["entity_id", "business_name", "business_address", "country"])
    prep = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "n_compact", "a_tokens", "a_house", "region"])
    fr = raw.filter(pl.col("country") == "France").join(prep, on="entity_id").with_columns(
        pl.col("business_address").fill_null(""), (pl.col("a_tokens").fill_null("") + "#" + pl.col("a_house").fill_null("")).alias("akey"),
        pl.col("business_name").str.split(" ").list.first().str.to_lowercase().alias("w0"))
    dept_of, depts = learn_departments(fr["region"].drop_nulls().unique().to_list() + ["Nouvelle-Aquitaine", "Hauts-de-France", "Pays de la Loire"])
    print(f"France S1 {fr.height:,}; departments learned: {sorted(depts)}; cities mapped: {len(dept_of)}")
    recs = {r[0]: r for r in fr.select("entity_id", "business_name", "business_address", "akey", "n_compact", "w0").iter_rows()}
    by_addr = fr.filter(pl.col("akey") != "#").group_by("akey").agg("entity_id").filter(pl.col("entity_id").list.len() > 1)
    by_addr = {k: v for k, v in by_addr.iter_rows()}
    by_name = {k: v for k, v in fr.group_by("n_compact").agg("entity_id").filter(pl.col("entity_id").list.len() > 1).iter_rows()}
    by_w0 = {k: v for k, v in fr.group_by("w0").agg("entity_id").filter(pl.col("entity_id").list.len() > 1).iter_rows()}
    text = lambda n, a: f"{n} | {a}"
    rows = []
    for eid, (_, n, a, akey, nc, w0) in recs.items():
        grp = "hold" if hash(nc) % 20 == 0 else "train"
        rows.append((text(n, a), text(noisy_name(n), noisy_addr(a, dept_of)), 1, "pos", grp))
        if rng.random() < 0.3:
            rows.append((text(n, a), text(noisy_name(n), noisy_addr(a, dept_of)), 1, "pos", grp))
        for other in rng.sample([x for x in by_addr.get(akey, []) if x != eid], k=min(2, max(0, len(by_addr.get(akey, [])) - 1))):
            o = recs[other]
            rows.append((text(n, a), text(noisy_name(o[1]), noisy_addr(o[2], dept_of)), 0, "addr", grp))
        cands = [x for x in by_name.get(nc, []) if x != eid]
        if cands:
            o = recs[rng.choice(cands)]
            rows.append((text(n, a), text(noisy_name(o[1]), noisy_addr(o[2], dept_of)), 0, "name", grp))
        cands = by_w0.get(w0, [])
        if len(cands) > 1:
            o = recs[rng.choice(cands)]
            if o[0] != eid and o[4] != nc:
                rows.append((text(n, a), text(noisy_name(o[1]), noisy_addr(o[2], dept_of)), 0, "word", grp))
        if rng.random() < 0.5 and re.search(r"\d", a):
            rows.append((text(n, a), text(noisy_name(n), noisy_addr(a, dept_of, shift_house=True)), 0, "decoy", grp))
    d = pl.DataFrame(rows, schema=["text_a", "text_b", "y", "kind", "grp"], orient="row")
    print(d.group_by("grp", "kind").len().sort("grp", "kind").rows())
    us = pl.read_parquet(train3).select("text_a", "text_b", pl.col("y").cast(pl.Int64)).head(int(n_us_in)).with_columns(pl.lit("us_in").alias("kind"))
    d = d.with_columns(pl.col("y").cast(pl.Int64))
    tr = pl.concat([d.filter(pl.col("grp") == "train").drop("grp"), us]).sample(fraction=1.0, seed=11, shuffle=True)
    tr.write_parquet(os.path.join(out, "train_fr.parquet"))
    d.filter(pl.col("grp") == "hold").drop("grp").write_parquet(os.path.join(out, "holdout_fr.parquet"))
    print(f"train_fr: {tr.height:,} pairs ({int(tr['y'].sum()):,} positive) | holdout {d.filter(pl.col('grp') == 'hold').height:,}")
    for r in d.filter(pl.col("kind").is_in(["pos", "addr", "word"])).sample(9, seed=3).select("kind", "text_a", "text_b").rows():
        print(f"  [{r[0]}] {r[1][:70]}\n        -> {r[2][:70]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
