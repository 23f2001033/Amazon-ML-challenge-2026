"""EDA 3: text-level profile of names and addresses per split x source x country.

Measures the frequency of each noise pattern so we know which normalisations matter.
"""
import polars as pl
from common import load, OUT

pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(40)
N = 300_000

SCRIPTS = {
    "devanagari": r"[\u0900-\u097F]", "bengali": r"[\u0980-\u09FF]", "gurmukhi": r"[\u0A00-\u0A7F]",
    "gujarati": r"[\u0A80-\u0AFF]", "odia": r"[\u0B00-\u0B7F]", "tamil": r"[\u0B80-\u0BFF]",
    "telugu": r"[\u0C00-\u0C7F]", "kannada": r"[\u0C80-\u0CFF]", "malayalam": r"[\u0D00-\u0D7F]",
}
NAME_PATTERNS = {
    "n_formerly": r"(?i)formerly", "n_dba": r"(?i)\b(dba|d/b/a|t/a|trading as|aka)\b",
    "n_website": r"(?i)(\.com|\.in|\.fr|\.net|\.org|www\.)", "n_pipe": r"\|",
    "n_lead_junk": r"^[^\p{L}\p{N}]", "n_trail_junk": r"[^\p{L}\p{N}.)\]]$",
    "n_brackets": r"[\(\)\[\]]", "n_amp": r"&", "n_and": r"(?i)\band\b",
    "n_digit": r"\d", "n_leet": r"\p{L}\d\p{L}|\p{L}\d\b|\b\d\p{L}{2,}",
    "n_double_space": r"  ", "n_accent": r"[À-ÿ]", "n_ms": r"(?i)^m/?s\.?\b",
    "n_the": r"(?i)^the\b", "n_dr": r"(?i)^dr\.?\b", "n_all_upper": r"^[^a-z]*[A-Z][^a-z]*$",
    "n_all_lower": r"^[^A-Z]*[a-z][^A-Z]*$",
    "n_legal_us": r"(?i)\b(inc|llc|l\.l\.c|corp|corporation|co|ltd|lp|llp|pllc|pc|company|incorporated)\b\.?",
    "n_legal_in": r"(?i)\b(pvt|private|limited|ltd|llp|opc)\b",
    "n_legal_fr": r"(?i)\b(sarl|sas|sasu|sa|sci|eurl|snc|ei)\b",
}
ADDR_PATTERNS = {
    "a_null_token": r"(?i)(\bnull\b|<null>|\bn/a\b|\bnone\b)", "a_zip5": r"\b\d{5}(-\d{4})?\b",
    "a_pin6": r"\b\d{3}\s?\d{3}\b", "a_pobox": r"(?i)p\.?\s?o\.?\s?box", "a_unit": r"(?i)(\bunit\b|#|\bapt\b|\bsuite\b|\bste\b)",
    "a_near": r"(?i)\b(near|opp|opposite|behind|beside|nr)\b", "a_co": r"(?i)c/o",
    "a_all_upper": r"^[^a-z]*$", "a_starts_alpha": r"^\p{L}", "a_starts_digit": r"^\d",
    "a_lead_zero_num": r"(^|\s)0\d+", "a_accent": r"[À-ÿ]", "a_bis": r"(?i)\b\d+\s?(bis|ter)\b",
    "a_double_space": r"  ", "a_hash": r"#",
}


def profile(df):
    exprs = [pl.len().alias("n"),
             pl.col("business_name").str.len_chars().mean().alias("name_len"),
             pl.col("business_name").str.split(" ").list.len().mean().alias("name_tokens"),
             pl.col("business_address").str.len_chars().mean().alias("addr_len"),
             pl.col("business_address").is_null().mean().alias("addr_null"),
             (pl.col("business_address").str.count_matches(",") + 1).mean().alias("addr_parts"),
             pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean().alias("n_nonascii"),
             pl.col("business_address").str.contains(r"[^\x00-\x7F]").mean().alias("a_nonascii")]
    for k, r in SCRIPTS.items():
        exprs.append(pl.col("business_name").str.contains(r).mean().alias(f"n_{k}"))
        exprs.append(pl.col("business_address").str.contains(r).mean().alias(f"a_{k}"))
    exprs += [pl.col("business_name").str.contains(r).mean().alias(k) for k, r in NAME_PATTERNS.items()]
    exprs += [pl.col("business_address").str.contains(r).mean().alias(k) for k, r in ADDR_PATTERNS.items()]
    return df.group_by("country").agg(exprs)


rows = []
for split in ["train", "test"]:
    for s in ["s1", "s2", "s3"]:
        df = load(f"{split}_{s}")
        df = df.sample(min(N, df.height), seed=0)
        rows.append(profile(df).with_columns(pl.lit(f"{split}_{s}").alias("file")))
res = pl.concat(rows).sort(["country", "file"])
res.write_csv(f"{OUT}/03_text_profile.csv")
# transpose for readability: rows = metrics, cols = file|country
t = res.with_columns((pl.col("file") + "|" + pl.col("country")).alias("key")).drop(["file", "country"])
tt = t.drop("key").transpose(include_header=True, header_name="metric", column_names=t["key"].to_list())
tt = tt.with_columns([pl.col(c).round(3) for c in tt.columns if c != "metric"])
for c in ["India", "US", "France"]:
    cols = ["metric"] + [x for x in tt.columns if x.endswith("|" + c)]
    print(f"\n### {c}")
    print(tt.select(cols))
