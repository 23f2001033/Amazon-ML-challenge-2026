"""Record canonicalisation: invert the noise operations observed in EDA (docs/03_EDA_REPORT.md).

All rules are generic language knowledge (abbreviations, legal forms, state/region names) or
dictionaries learned from the TRAINING data (see resources.py). No external lookups.

Main entry points:
    canon_name(raw, res)      -> dict of name fields
    canon_address(raw, country, res) -> dict of address fields
"""
import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------------------
# Static tables
# --------------------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": ["ap"], "arunachal pradesh": ["ar"], "assam": ["as"], "bihar": ["br"],
    "chhattisgarh": ["cg", "ct"], "goa": ["ga"], "gujarat": ["gj"], "haryana": ["hr"],
    "himachal pradesh": ["hp"], "jharkhand": ["jh"], "karnataka": ["ka"], "kerala": ["kl"],
    "madhya pradesh": ["mp"], "maharashtra": ["mh"], "manipur": ["mn"], "meghalaya": ["ml"],
    "mizoram": ["mz"], "nagaland": ["nl"], "odisha": ["od", "or", "orissa"], "punjab": ["pb"],
    "rajasthan": ["rj"], "sikkim": ["sk"], "tamil nadu": ["tn"], "telangana": ["tg", "ts"],
    "tripura": ["tr"], "uttar pradesh": ["up"], "uttarakhand": ["uk", "ut", "uttaranchal"],
    "west bengal": ["wb"], "andaman and nicobar islands": ["an"], "chandigarh": ["ch"],
    "dadra and nagar haveli and daman and diu": ["dn", "dd"], "delhi": ["dl"],
    "jammu and kashmir": ["jk"], "ladakh": ["la"], "lakshadweep": ["ld"], "puducherry": ["py", "pondicherry"],
}
FR_REGIONS = {
    "hauts de france": ["nord", "pas de calais", "aisne", "oise", "somme"],
    "nouvelle aquitaine": ["gironde", "landes", "dordogne", "lot et garonne", "pyrenees atlantiques",
                           "charente", "charente maritime", "vienne", "haute vienne", "correze",
                           "creuse", "deux sevres"],
    "pays de la loire": ["loire atlantique", "vendee", "maine et loire", "sarthe", "mayenne"],
    "ile de france": ["paris", "seine saint denis", "hauts de seine", "val de marne"],
    "bretagne": ["finistere", "morbihan", "ille et vilaine", "cotes d armor"],
    "occitanie": ["haute garonne", "herault", "gard"],
    "auvergne rhone alpes": ["rhone", "isere", "haute savoie", "savoie"],
    "provence alpes cote d azur": ["bouches du rhone", "alpes maritimes", "var"],
    "grand est": ["bas rhin", "haut rhin", "moselle"],
    "normandie": ["seine maritime", "calvados", "manche", "eure", "orne"],
    "centre val de loire": ["loiret", "indre et loire"],
    "bourgogne franche comte": ["cote d or", "doubs"],
}


def _static_region_table():
    """{country: {normalised alias: canonical region}} for explicit region mentions."""
    us = {}
    for name, code in US_STATES.items():
        us[name] = code
        us[code] = code
    ind = {}
    for name, aliases in IN_STATES.items():
        ind[name] = name
        for a in aliases:
            ind[a] = name
    ind["new delhi"] = "delhi"  # frequently used as a state label in S2/S3
    fr = {}
    for reg, deps in FR_REGIONS.items():
        fr[reg] = reg
        for d in deps:
            fr[d] = reg
    return {"US": us, "India": ind, "France": fr}


STATIC_REGIONS = _static_region_table()

LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "comp": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "pc": "pc", "plc": "plc", "public": "public", "opc": "opc", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "snc": "snc", "ei": "ei", "selarl": "selarl", "sca": "sca", "scop": "scop",
}
LEGAL_CANON = set(LEGAL.values())
HONORIFIC = {"the", "ms", "m", "dr", "smt", "mr", "mrs", "and", "of", "et", "le", "la", "les", "de", "du", "des"}

STREET = {
    "street": "st", "st": "st", "str": "st", "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave",
    "av": "ave", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bvd": "blvd", "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl", "trail": "trl", "trl": "trl", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "terrace": "ter", "ter": "ter", "square": "sq", "sq": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "rue": "rue", "r": "rue", "route": "rte", "rte": "rte",
    "chemin": "chem", "ch": "chem", "chem": "chem", "impasse": "imp", "imp": "imp", "allee": "allee",
    "all": "allee", "quai": "quai", "cours": "crs", "crs": "crs", "residence": "res", "res": "res",
    "saint": "st", "sainte": "ste", "floor": "fl", "flr": "fl", "fl": "fl", "sector": "sec",
    "sec": "sec", "nagar": "nagar", "ngr": "nagar", "building": "bldg", "bldg": "bldg",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy", "center": "ctr",
    "centre": "ctr", "ctr": "ctr", "heights": "hts", "hts": "hts", "junction": "jct", "jct": "jct",
}
# Tokens that noise injects around numbers / units without adding identity.
ADDR_STOP = {"no", "number", "h", "hno", "door", "unit", "apt", "apartment", "ste", "suite", "cdp",
             "township", "twp", "borough", "null", "none", "na", "n", "a", "and", "of", "the", "de",
             "du", "des", "la", "le", "les", "d", "l", "city"}
# Noise-only tokens in US names/addresses that S1 never contains.
PO_BOX = re.compile(r"(?i)\b(?:p\.?\s?o\.?\s?box|pmb)\s*#?\s*\d+")
NULL_TOKENS = re.compile(r"(?i)(<null>|\bnull\b|\bn/a\b|\bnone\b|\bnan\b)")

INDIC = re.compile(r"[ऀ-ൿ]")
ALIAS = re.compile(r"(?i)^(.+?)\s+(?:formerly|f/k/a|fka|d/b/a|dba|aka|a/k/a|t/a|trading\s+as)\b\s*:?\s*(.+)$")
URL_NAME = re.compile(r"(?i)^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-\.]*?)\.(?:co\.in|com|in|fr|net|org|biz|info|io|co)/?$")
HANDLE = re.compile(r"^[#@]([A-Za-z0-9_\-\.]+)$")
NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
SPACES = re.compile(r"\s+")
DIGIT_RUN = re.compile(r"\d+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"})


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def fold(s):
    """ASCII-fold Latin text (accents) and lowercase."""
    if not s.isascii():
        s = unidecode(s)
    return s.lower()


def basic(s):
    """fold + '&'->' and ' + punctuation->space + collapse spaces."""
    if not s:
        return ""
    s = fold(s).replace("&", " and ")
    s = NON_ALNUM.sub(" ", s)
    return SPACES.sub(" ", s).strip()


def place_key(p):
    """Key for place names: basic() + street/saint abbreviations (ST.-NAZAIRE == Saint-Nazaire)."""
    return " ".join(STREET.get(t, t) for t in basic(p).split() if t not in ("city", "of", "cdp", "township"))


def collapse_initials(tokens):
    """Join runs of >=2 single-letter tokens: ['l','l','c'] -> ['llc'], ['m','d'] -> ['md']."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def fix_leet(tok):
    """Map digits inside alphabetic tokens to letters (Ca1lahan -> callahan)."""
    if tok.isdigit() or tok.isalpha():
        return tok
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    if letters >= 2 and digits <= 2:
        return tok.translate(LEET)
    return tok


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
def canon_name(raw, res):
    """Canonicalise a business name.

    res: resources dict with 'native_name' (native token -> english token) and
         'variants' (english spelling variant -> canonical).
    Returns dict(n_canon, n_core, n_compact, n_legal, n_native, n_alias, n_web).
    """
    s = (raw or "").strip()
    native = alias = web = 0
    if "|" in s:  # "NAME | www.site.com" -> keep the non-URL part
        parts = [p.strip() for p in s.split("|") if p.strip()]
        keep = [p for p in parts if not URL_NAME.match(p) and "www." not in p.lower()]
        s = keep[0] if keep else parts[0] if parts else ""
    m = ALIAS.match(s)
    if m:  # "Xylo formerly: Real Name LLC" / "Quonex d/b/a Real Name" -> real name
        s, alias = m.group(2), 1
    if INDIC.search(s):
        native = 1
        dic = res["native_name"]
        s = " ".join(dic.get(t, dic.get(t.rstrip("."), t)) for t in s.split())
    m = URL_NAME.match(s) or HANDLE.match(s)
    if m:
        s, web = m.group(1).replace("-", "").replace(".", " "), 1
    s = basic(s)
    toks = collapse_initials([fix_leet(t) for t in s.split()])
    var = res["variants"]
    toks = [var.get(t, t) for t in toks]
    toks = [LEGAL.get(t, t) for t in toks]
    legal = sorted({t for t in toks if t in LEGAL_CANON})
    core = [t for t in toks if t not in LEGAL_CANON and t not in HONORIFIC]
    if not core:  # name made only of legal/honorific words: keep everything
        core = toks
    return {
        "n_canon": " ".join(toks),
        "n_core": " ".join(core),
        "n_compact": "".join(core),
        "n_legal": " ".join(legal),
        "n_native": native, "n_alias": alias, "n_web": web,
    }


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
def _norm_part(p, res_native_addr):
    p = p.strip()
    if not p:
        return ""
    if INDIC.search(p):
        hit = res_native_addr.get(p)
        if hit is not None:
            return hit
        return basic(p)
    return basic(p)


def canon_address(raw, country, res):
    """Canonicalise an address and detect its region (state / region).

    Returns dict(a_tokens, a_nums, a_house, region, region_src, a_null).
      a_tokens : space-joined canonical tokens, explicit region part removed
      a_nums   : space-joined digit runs (leading zeros stripped), in order
      a_house  : first digit run of the first part that contains a digit ('' if none)
      region   : canonical region label ('' if unknown); region_src: 2 explicit, 1 inferred, 0 none
    """
    if raw is None:
        return {"a_tokens": "", "a_nums": "", "a_house": "", "region": "", "region_src": 0, "a_null": 1}
    s = NULL_TOKENS.sub(" ", raw)
    s = PO_BOX.sub(" ", s)
    static = STATIC_REGIONS.get(country, {})
    city_map = res["city_region"].get(country, {})
    native_addr = res["native_addr"]
    region, src = "", 0
    kept_parts = []
    for part in s.split(","):
        np_ = _norm_part(part, native_addr)
        if not np_:
            continue
        if np_ in static:
            if not region or src < 2:
                region, src = static[np_], 2
            continue  # explicit region mention is compared via `region`, not tokens
        kept_parts.append((part, np_))
    if not region:
        for _, np_ in kept_parts:
            r = city_map.get(place_key(np_))
            if r:
                region, src = r, 1
                break
    toks, nums, house = [], [], ""
    for part, np_ in kept_parts:
        runs = [d.lstrip("0") or "0" for d in DIGIT_RUN.findall(part)]
        if runs and not house:
            house = runs[0]
        nums.extend(runs)
        for t in np_.split():
            t = STREET.get(t, t)
            if t in ADDR_STOP:
                continue
            toks.append(t)
    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in toks]
    return {
        "a_tokens": " ".join(toks),
        "a_nums": " ".join(nums),
        "a_house": house,
        "region": region,
        "region_src": src,
        "a_null": int(not toks),
    }
