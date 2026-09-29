"""Paths and global settings for the entity-resolution pipeline."""
import os

# Repository layout: <root>/code/business_entity_resolution/src/config.py
ROOT = os.environ.get(
    "ER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)


def _find_raw_dir():
    """Dataset folder containing train/ and test/: $ER_RAW_DIR, else common layouts under ROOT."""
    if os.environ.get("ER_RAW_DIR"):
        return os.environ["ER_RAW_DIR"]
    for cand in (os.path.join(ROOT, "dataset"),
                 os.path.join(ROOT, "student_resource", "dataset"),
                 os.path.join(ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")):
        if os.path.isdir(os.path.join(cand, "test")):
            return cand
    return os.path.join(ROOT, "dataset")


RAW_DIR = _find_raw_dir()
DATA_DIR = os.path.join(ROOT, "data")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")      # raw TSV -> parquet copies
PREPARED_DIR = os.path.join(DATA_DIR, "prepared")    # canonicalised records
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")   # learned dictionaries, models
CAND_DIR = os.path.join(DATA_DIR, "candidates")      # blocking output
FEAT_DIR = os.path.join(DATA_DIR, "features")        # pair features
OUTPUT_DIR = os.path.join(ROOT, "output")            # submission files

RAW_FILES = {
    "train_s1": "train/train_source1.tsv",
    "train_s2": "train/train_source2.tsv",
    "train_s3": "train/train_source3.tsv",
    "train_gt": "train/train_ground_truth.tsv",
    "test_s1": "test/test_source1.tsv",
    "test_s2": "test/test_source2.tsv",
    "test_s3": "test/test_source3.tsv",
}

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 2)

for _d in (PARQUET_DIR, PREPARED_DIR, ARTIFACT_DIR, CAND_DIR, FEAT_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)
