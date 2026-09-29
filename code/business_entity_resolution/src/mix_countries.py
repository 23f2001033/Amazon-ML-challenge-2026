"""Build a submission whose rows come from different versions per country.

  python mix_countries.py <out_dir> <country>=<sub_dir> [<country>=<sub_dir> ...] [default=<sub_dir>]

Each <sub_dir> holds matching_results.tsv and candidate_pairs.tsv of one version. For each test S1,
both its matches and its candidates are taken from the version assigned to its country, so matches
remain a subset of that version's candidates. Every version's candidates come from the same blocker.
Example:  python mix_countries.py ~/er/sub_mix US=~/er/sub_v4bs India=~/er/sub_v4bs France=~/er/sub_v2c
"""
import os
import sys

import polars as pl

from config import PREPARED_DIR


def read_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, rest = line.rstrip("\n").partition("\t")
            rows[s] = rest
    return rows


def main(out_dir, *assign):
    amap = dict(a.split("=", 1) for a in assign)
    default = amap.pop("default", None)
    s1 = pl.read_parquet(os.path.join(PREPARED_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    srcs = set(amap.values()) | ({default} if default else set())
    cache = {d: {k: read_rows(os.path.join(d, f"{k}.tsv")) for k in ("matching_results", "candidate_pairs")} for d in srcs}
    os.makedirs(out_dir, exist_ok=True)
    headers = {"matching_results": "source1_entity_id\tmatched_entity_ids\n",
               "candidate_pairs": "source1_entity_id\tcandidate_entity_ids\n"}
    used = {}
    for kind, header in headers.items():
        with open(os.path.join(out_dir, f"{kind}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            for sid, country in s1.iter_rows():
                d = amap.get(country, default)
                if d is None:
                    raise SystemExit(f"no source for country {country}")
                used[country] = d
                f.write(f"{sid}\t{cache[d][kind][sid]}\n")
    print("country -> source:", used)


if __name__ == "__main__":
    main(*sys.argv[1:])
