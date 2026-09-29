"""Text normalisation used by the EDA (prototype of the pipeline normaliser).

Everything here is rule-based and self-contained: no external lookups. The only
"knowledge" embedded is generic language knowledge (US state codes, common
street/legal abbreviations), which is allowed under the fair-play rules.
"""
import re
from unidecode import unidecode

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
US_CODES = set(US_STATES.values())

LEGAL = {
    "private": "pvt", "pvt": "pvt", "limited": "ltd", "ltd": "ltd", "corporation": "corp",
    "corp": "corp", "incorporated": "inc", "inc": "inc", "company": "co", "co": "co", "llc": "llc",
    "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "snc": "snc", "ei": "ei", "opc": "opc",
}
STREET = {
    "street": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl", "trail": "trl", "trl": "trl", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "terrace": "ter", "ter": "ter", "way": "way", "square": "sq",
    "sq": "sq", "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e",
    "w": "w", "rue": "rue", "r": "rue", "floor": "flr", "flr": "flr", "apartment": "apt",
    "apt": "apt", "suite": "ste", "ste": "ste", "unit": "unit", "number": "no", "no": "no",
    "near": "near", "nr": "near", "opp": "opp", "opposite": "opp",
}
NULL_TOKENS = re.compile(r"(?i)(<null>|\bnull\b|\bn/a\b|\bnone\b)")
FORMERLY = re.compile(r"(?i)^(.*?)\s*formerly:?\s*(.*)$")
NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
SPACES = re.compile(r"\s+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def basic(s):
    """Lowercase ASCII-folded string with punctuation removed and spaces collapsed."""
    if s is None:
        return ""
    s = unidecode(s).lower().replace("&", " and ")
    s = NON_ALNUM.sub(" ", s)
    return SPACES.sub(" ", s).strip()


def norm_name(s):
    """Canonical name: ASCII fold, legal-suffix canonicalisation, drop 'formerly' prefix."""
    if s is None:
        return ""
    m = FORMERLY.match(s)
    if m:  # "Xylo formerly: Real Name LLC" -> keep the part after 'formerly'
        s = m.group(2)
    s = re.sub(r"(?i)\bl\.l\.c\.?", "llc", s)
    s = re.sub(r"(?i)\bpvt\.", "pvt ", s)
    s = re.sub(r"(?i)\bltd\.", "ltd ", s)
    s = re.sub(r"(?i)(www\.|\.com\b|\.in\b|\.fr\b|\.net\b|\.org\b)", " ", s)
    toks = basic(s).split()
    return " ".join(LEGAL.get(t, t) for t in toks)


def core_name(s):
    """Name without legal/generic tokens (the distinctive part)."""
    return " ".join(t for t in norm_name(s).split() if t not in LEGAL.values() and t not in {"the", "and", "m", "s", "dr"})


def norm_addr(s):
    if s is None:
        return ""
    s = NULL_TOKENS.sub(" ", s)
    toks = basic(s).split()
    return " ".join(STREET.get(t, t) for t in toks)


def addr_numbers(s):
    """Set of numeric tokens (leading zeros stripped) in an address."""
    if s is None:
        return set()
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", s)}


def us_state(addr):
    """Extract US state code from an address in any of the observed formats."""
    if addr is None:
        return None
    parts = [basic(p) for p in addr.split(",")]
    for p in parts:
        if p in US_CODES:
            return p
        if p in US_STATES:
            return US_STATES[p]
    return None
