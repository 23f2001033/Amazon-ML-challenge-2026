"""Learn normalisation resources from the data (no external sources).

* native_name : Indic-script name token -> English token. Learned by position alignment of
                TRAIN positive pairs (S1 English name vs S2/S3 fully-native name of equal length).
* variants    : English spelling-variant groups (laxmi/lakshmi, jai/jay, shree/sri/shri) that
                share one native spelling -> canonical token.
* native_addr : Indic-script address part (e.g. 'महाराष्ट्र') -> canonical region name, learned
                from TRAIN positive pairs.
* city_region : {country: {address part -> region}} learned from S1 records (train + test S1
                inputs, no labels) where the region is explicit; used when S2/S3 omit the region.
"""
import json
import os
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from config import ARTIFACT_DIR, PARQUET_DIR
from textnorm import INDIC, STATIC_REGIONS, basic, place_key

RES_PATH = os.path.join(ARTIFACT_DIR, "resources.json")


def _load(name):
    return pl.read_parquet(os.path.join(PARQUET_DIR, f"{name}.parquet"))


def _gt_pairs():
    gt = _load("train_gt").drop_nulls("matched_entity_ids")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"}))


def _explicit_region(addr, country):
    static = STATIC_REGIONS.get(country, {})
    for part in (addr or "").split(","):
        p = basic(part)
        if p in static:
            return static[p]
    return None


def learn_native_name(pairs):
    """pairs: iterable of (s1_name, native_name). Returns (dict, variants)."""
    co = defaultdict(Counter)
    for n1, n2 in pairs:
        t1, t2 = n1.split(), n2.split()
        if len(t1) != len(t2) or not all(INDIC.search(t) for t in t2):
            continue
        for a, b in zip(t2, t1):
            e = basic(b)
            if e:
                co[a][e.replace(" ", "")] += 1
    # spelling-variant groups: English tokens sharing a native spelling and looking alike
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    freq = Counter()
    for tok, c in co.items():
        tot = sum(c.values())
        cands = [(e, n) for e, n in c.most_common(5) if n / tot >= 0.15 and n >= 20]
        for e, n in cands:
            freq[e] += n
        for i in range(1, len(cands)):
            a, b = cands[0][0], cands[i][0]
            similar = fuzz.ratio(a, b) >= 60 or (a[0] == b[0] and len(a) <= 6 and len(b) <= 6)
            if similar:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    variants = {}
    # manual group for the honorific-like prefix the noise injects / spells variously
    for m in ("sri", "shree", "sree", "shri"):
        freq[m] += 1
        if find(m) != find("shri"):
            parent[find(m)] = find("shri")
    groups = defaultdict(list)
    for e in list(freq):
        groups[find(e)].append(e)
    for members in groups.values():
        members[:] = [m for m in members if len(m) >= 3 and m.isalpha()]
        if len(members) > 1:
            canon = max(members, key=lambda m: freq[m])
            for m in members:
                if m != canon:
                    variants[m] = canon
    native = {}
    for tok, c in co.items():
        e = c.most_common(1)[0][0]
        native[tok] = variants.get(e, e)
        native.setdefault(tok.rstrip("."), native[tok])
    return native, variants


def learn_native_addr(pairs):
    """pairs: iterable of (s1_addr, s2/s3_addr) for India. Native part -> S1 explicit region."""
    co = defaultdict(Counter)
    for a1, a2 in pairs:
        r = _explicit_region(a1, "India")
        if not r or not a2:
            continue
        for part in a2.split(","):
            p = part.strip()
            if p and INDIC.search(p):
                co[p][r] += 1
    out = {}
    for p, c in co.items():
        r, n = c.most_common(1)[0]
        if n >= 10 and n / sum(c.values()) >= 0.9:
            out[p] = r
    return out


def learn_city_region(s1_frames):
    """Address parts of S1 records -> region, when the S1 record states its region explicitly."""
    co = defaultdict(lambda: defaultdict(Counter))
    for df in s1_frames:
        for addr, country in df.select("business_address", "country").iter_rows():
            r = _explicit_region(addr, country)
            if not r:
                continue
            static = STATIC_REGIONS.get(country, {})
            for part in addr.split(","):
                p = basic(part)
                if p and p not in static and not any(ch.isdigit() for ch in p):
                    co[country][place_key(p)][r] += 1
    out = {}
    for country, parts in co.items():
        m = {}
        for p, c in parts.items():
            r, n = c.most_common(1)[0]
            if n >= 3 and n / sum(c.values()) >= 0.9:
                m[p] = r
        out[country] = m
    return out


def build_resources():
    pos = _gt_pairs()
    s1 = _load("train_s1")
    s23 = pl.concat([_load("train_s2"), _load("train_s3")])
    india = (pos.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                                pl.col("business_address").alias("a1"), "country"), on="s1")
                .filter(pl.col("country") == "India")
                .join(s23.select(pl.col("entity_id").alias("mid"), pl.col("business_name").alias("n2"),
                                 pl.col("business_address").alias("a2")), on="mid"))
    nat = india.filter(pl.col("n2").str.contains(r"[ऀ-ൿ]"))
    native_name, variants = learn_native_name(nat.select("n1", "n2").iter_rows())
    native_addr = learn_native_addr(
        india.filter(pl.col("a2").str.contains(r"[ऀ-ൿ]")).select("a1", "a2").iter_rows())
    city_region = learn_city_region([s1, _load("test_s1")])
    res = {"native_name": native_name, "variants": variants, "native_addr": native_addr,
           "city_region": city_region}
    with open(RES_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"native_name={len(native_name)} variants={len(variants)} native_addr={len(native_addr)} "
          f"city_region={ {k: len(v) for k, v in city_region.items()} }")
    return res


def load_resources():
    with open(RES_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    build_resources()
