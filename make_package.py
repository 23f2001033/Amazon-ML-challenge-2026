"""Build submission zips.

  python make_package.py TEAM            -> dist/TEAM_code.zip        (portal "code file" upload)
  python make_package.py TEAM --full     -> dist/TEAM_submission.zip  (final package incl. output/)

Code zip layout:  code/business_entity_resolution/{src/, README.md, requirements.txt}
                  Documentation_template.md
Full zip adds:    output/matching_results.tsv, output/candidate_pairs.tsv
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code", "business_entity_resolution")


# model binaries the final submission uses (v6fr3zF2); older experiment binaries stay local, their small configs ship
KEEP_MODELS = {"lgbm_v6.txt", "lgbm_v6t.txt", "s2_v6_rich_ce_ce2_ce3fr_ce4fr_w.txt", "s2_v6_rich_ce_ce2_ce3fr_ce4fr_w2.txt",
               "s2_v6t_rich_ce_ce2_ce3fr_ce4fr_w2.txt"}


def add_code(z):
    for base, dirs, files in os.walk(CODE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith((".pyc", ".log")):
                continue
            if os.path.basename(base) == "models" and fn.endswith(".txt") and fn not in KEEP_MODELS:
                continue
            full = os.path.join(base, fn)
            z.write(full, os.path.relpath(full, ROOT))
    z.write(os.path.join(ROOT, "Documentation_template.md"), "Documentation_template.md")


def main():
    team = sys.argv[1] if len(sys.argv) > 1 else "team"
    full = "--full" in sys.argv
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    name = f"{team}_submission.zip" if full else f"{team}_code.zip"
    path = os.path.join(ROOT, "dist", name)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        add_code(z)
        if full:
            for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
                z.write(os.path.join(ROOT, "output", fn), f"output/{fn}")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB, {len(names)} files")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
