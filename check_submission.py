"""Pre-upload checks beyond the official validator (run before every leaderboard submission).

  python check_submission.py [output_dir] [zip_path] [--zip-only]

TSV checks (both files): exact header, UTF-8 without BOM, ASCII, LF endings, one TAB per line,
no quotes/spaces/empty ids, exactly one row per test S1 (France included), ids match S2-/S3-
and exist in their own test source file, no duplicates, matches subset of candidates,
same-country matches, one-S1-per-record partition.
Zip checks: CRC, required layout, python files compile, data files exactly as the spec
requires (full package: the two output TSVs, byte-identical to the validated ones), team filled in.
Exit code 0 = everything passed.
"""
import hashlib
import os
import re
import sys
import zipfile

import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST = os.environ.get("ER_TEST_DIR", os.path.join(ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "test"))
IDRE = re.compile(r"^S[23]-\d+$")
FAIL = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def md5_stream(f):
    h = hashlib.md5()
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
    return h.hexdigest()


def raw_format(path, header):
    name = os.path.basename(path)
    with open(path, "rb") as f:
        head = f.read(4096)
    check(not head.startswith(b"\xef\xbb\xbf"), f"{name}: no UTF-8 BOM")
    check(head.split(b"\n", 1)[0] == header.encode(), f"{name}: exact header {header!r}")
    cr = quotes = spaces = tabs = nonascii = commas = 0
    last = b""
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            cr += b"\r" in line
            quotes += b'"' in line
            spaces += b" " in line
            tabs += line.count(b"\t") != 1
            nonascii += not line.isascii()
            if i:
                body = line.rstrip(b"\n")
                commas += body.endswith(b",") or b"\t," in body or b",," in body
            last = line
    check(cr == 0, f"{name}: LF line endings, no CR ({cr})")
    check(quotes == 0, f"{name}: no quote characters ({quotes})")
    check(spaces == 0, f"{name}: no spaces ({spaces})")
    check(tabs == 0, f"{name}: exactly one TAB per line ({tabs} bad)")
    check(nonascii == 0, f"{name}: pure ASCII ({nonascii} lines)")
    check(commas == 0, f"{name}: no empty ids / stray commas ({commas} rows)")
    check(last.endswith(b"\n"), f"{name}: ends with a newline")


def tsv_checks(out_dir):
    mpath = os.path.join(out_dir, "matching_results.tsv")
    cpath = os.path.join(out_dir, "candidate_pairs.tsv")
    print("== raw format")
    raw_format(mpath, "source1_entity_id\tmatched_entity_ids")
    raw_format(cpath, "source1_entity_id\tcandidate_entity_ids")

    print("== coverage and ids")
    read = lambda f, cols: pl.read_csv(os.path.join(TEST, f), separator="\t", quote_char=None,
                                       infer_schema=False, columns=cols)
    s1 = read("test_source1.tsv", ["entity_id", "country"])
    s2 = read("test_source2.tsv", ["entity_id", "country"])
    s3 = read("test_source3.tsv", ["entity_id", "country"])
    ids2, ids3 = set(s2["entity_id"].to_list()), set(s3["entity_id"].to_list())
    s1set = set(s1["entity_id"].to_list())
    matched = {}
    for path, col in ((mpath, "matched_entity_ids"), (cpath, "candidate_entity_ids")):
        name = os.path.basename(path)
        seen = set()
        dup_rows = not_s1 = n_ids = bad_pat = dup_list = unknown = orphan = 0
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                s, _, rest = line.rstrip("\n").partition("\t")
                dup_rows += s in seen
                seen.add(s)
                not_s1 += s not in s1set
                ids = rest.split(",") if rest else []
                n_ids += len(ids)
                dup_list += len(ids) != len(set(ids))
                for i in ids:
                    bad_pat += not IDRE.match(i)
                    unknown += not ((i.startswith("S2-") and i in ids2) or (i.startswith("S3-") and i in ids3))
                if col == "matched_entity_ids":
                    if ids:
                        matched[s] = set(ids)
                elif s in matched:
                    orphan += len(matched[s] - set(ids))
        check(len(seen) == len(s1set) and not_s1 == 0 and not (s1set - seen),
              f"{name}: exactly the {len(s1set):,} test S1 ids (France included), no extras")
        check(dup_rows == 0, f"{name}: no duplicate S1 rows ({dup_rows})")
        check(bad_pat == 0, f"{name}: all {n_ids:,} listed ids match ^S[23]-digits$ ({bad_pat} bad)")
        check(dup_list == 0, f"{name}: no duplicate id inside a list ({dup_list} rows)")
        check(unknown == 0, f"{name}: every id exists in its own test source file ({unknown} unknown)")
        if col == "candidate_entity_ids":
            check(orphan == 0, f"{name}: every matched id is among that S1's candidates ({orphan} missing)")
        else:
            m = pl.DataFrame([(a, b) for a, x in matched.items() for b in x], schema=["s1", "mid"], orient="row")
            tgt = pl.concat([s2, s3]).rename({"country": "c2"})
            cross = (m.join(s1, left_on="s1", right_on="entity_id")
                      .join(tgt, left_on="mid", right_on="entity_id")
                      .filter(pl.col("country") != pl.col("c2")).height)
            check(cross == 0, f"{name}: matches share the S1's country ({cross} cross-country)")
            multi = m.group_by("mid").len().filter(pl.col("len") > 1).height
            check(multi == 0, f"{name}: each S2/S3 id assigned to at most one S1 ({multi})")


def zip_checks(out_dir, zip_path):
    print("== zip:", os.path.basename(zip_path))
    full = os.path.basename(zip_path).endswith("_submission.zip")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        check(z.testzip() is None, "archive CRC ok")
        for req in ("code/business_entity_resolution/README.md",
                    "code/business_entity_resolution/requirements.txt",
                    "Documentation_template.md"):
            check(req in names, f"contains {req}")
        src = [n for n in names if n.startswith("code/business_entity_resolution/src/") and n.endswith(".py")]
        check(len(src) >= 8, f"contains src/ with {len(src)} python files")
        bad_py = []
        for n in src:
            try:
                compile(z.read(n).decode("utf-8"), n, "exec")
            except SyntaxError:
                bad_py.append(n)
        check(not bad_py, f"all src files compile ({bad_py})")
        check(any(n.startswith("code/business_entity_resolution/models/") for n in names),
              "ships the trained model (models/) for exact reproduction")
        tops = sorted({n.split("/")[0] for n in names})
        want = sorted(["Documentation_template.md", "code"] + (["output"] if full else []))
        check(tops == want, f"top level is exactly {want} (found {tops})")
        data = sorted(n for n in names if n.endswith((".tsv", ".parquet", ".pyc", ".csv")))
        expected = ["output/candidate_pairs.tsv", "output/matching_results.tsv"] if full else []
        check(data == expected, f"data files are exactly {expected or 'none'} (found {data})")
        if full:
            for n in expected:
                with z.open(n) as a, open(os.path.join(out_dir, os.path.basename(n)), "rb") as b:
                    check(md5_stream(a) == md5_stream(b), f"{n} is byte-identical to the validated file")
        doc = z.read("Documentation_template.md").decode("utf-8")
        check("[Your Team Name]" not in doc and "[List all team members]" not in doc and "Embedding" in doc,
              "Documentation_template.md: team name and members filled in")


def main(out_dir=None, zip_path=None, *flags):
    out_dir = out_dir or os.path.join(ROOT, "output")
    if "--zip-only" not in flags:
        tsv_checks(out_dir)
    if zip_path:
        zip_checks(out_dir, zip_path)
    print("RESULT:", "ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
