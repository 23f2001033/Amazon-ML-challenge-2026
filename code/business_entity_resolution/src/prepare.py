"""Stage 1: convert raw TSVs to Parquet and canonicalise every record (parallel).

Output: data/prepared/{split}_{src}.parquet with columns
  entity_id, country, name, address (raw), n_canon, n_core, n_compact, n_legal,
  n_native, n_alias, n_web, n_upper, a_tokens, a_nums, a_house, region, region_src, a_null
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from config import N_JOBS, PARQUET_DIR, PREPARED_DIR, RAW_DIR, RAW_FILES
from resources import build_resources, load_resources, RES_PATH
from textnorm import canon_address, canon_name

_RES = None


def _init(res):
    global _RES
    _RES = res


def _work(chunk):
    """chunk: polars frame slice -> polars frame of canonical fields (compact Arrow transfer)."""
    cols = None
    for name, addr, country in chunk.iter_rows():
        d = canon_name(name, _RES)
        d.update(canon_address(addr, country, _RES))
        d["n_upper"] = int(bool(name) and name.isupper())
        if cols is None:
            cols = {k: [] for k in d}
        for k, v in d.items():
            cols[k].append(v)
    return pl.DataFrame(cols)


def raw_to_parquet():
    """Read each TSV exactly (TAB, no quoting) and cache as Parquet."""
    for name, rel in RAW_FILES.items():
        dst = os.path.join(PARQUET_DIR, f"{name}.parquet")
        if os.path.exists(dst):
            continue
        df = pl.read_csv(os.path.join(RAW_DIR, rel), separator="\t", quote_char=None,
                         infer_schema=False, empty_string_is_null=True)
        df.write_parquet(dst, compression="zstd")
        print(f"parquet {name}: {df.height:,} rows")


def prepare_file(name, res, chunk=20000):
    t = time.time()
    df = pl.read_parquet(os.path.join(PARQUET_DIR, f"{name}.parquet"))
    src = df.select("business_name", "business_address", "country")
    chunks = [src.slice(i, chunk) for i in range(0, src.height, chunk)]
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=_init, initargs=(res,)) as ex:
        feats = pl.concat(list(ex.map(_work, chunks)))
    out = pl.concat([df.rename({"business_name": "name", "business_address": "address"}), feats],
                    how="horizontal")
    out.write_parquet(os.path.join(PREPARED_DIR, f"{name}.parquet"), compression="zstd")
    print(f"prepared {name}: {out.height:,} rows in {time.time() - t:.0f}s; "
          f"region coverage={(out['region'] != '').mean():.4f}", flush=True)


def main(names=None):
    raw_to_parquet()
    res = load_resources() if os.path.exists(RES_PATH) else build_resources()
    for n in names or ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        if os.path.exists(os.path.join(PREPARED_DIR, f"{n}.parquet")):
            continue
        prepare_file(n, res)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
