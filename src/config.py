"""
Central configuration for the Business Entity Resolution pipeline.

All paths and tunable constants live here so that every script/notebook uses
the same values and nothing is hard-coded deep inside the pipeline. Paths are
resolved relative to this file so the project works from any working
directory ("python -m src.xxx" or from a notebook under notebooks/).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
# This file lives at business_entity_resolution/src/config.py
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent                       # business_entity_resolution/
# REPO_DIR = PROJECT_DIR.parent                      # AmazonML/

# Raw challenge data, as provided by the organisers. We read directly from
# here instead of copying ~1.3GB/1.2GB of TSVs into the project tree.
# RAW_DATA_DIR = REPO_DIR / "student_resource" / "dataset"

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = REPO_DIR / "data"

RAW_TRAIN_DIR = RAW_DATA_DIR / "train"
RAW_TEST_DIR = RAW_DATA_DIR / "test"

# Derived / cached artefacts (parquet caches, audit reports, sampled splits).
DATA_DIR = PROJECT_DIR / "data"
TRAIN_CACHE_DIR = DATA_DIR / "train"
TEST_CACHE_DIR = DATA_DIR / "test"
CACHE_DIR = PROJECT_DIR / "cache"

EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
MODELS_DIR = PROJECT_DIR / "models"

# Final deliverables are written at the repo root, mirroring the official
# submission zip layout (`output/` sits next to `code/`).
OUTPUT_DIR = REPO_DIR / "output"

# Raw file paths -------------------------------------------------------------
TRAIN_SOURCE1 = RAW_TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = RAW_TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = RAW_TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = RAW_TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = RAW_TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = RAW_TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = RAW_TEST_DIR / "test_source3.tsv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Scale of the raw data (measured, Phase 1 audit) -- kept here for reference
# and for sizing decisions elsewhere in the pipeline. Do not rely on these
# numbers for control flow; always compute counts from the data itself.
# ---------------------------------------------------------------------------
MEASURED_ROW_COUNTS = {
    "train_source1": 2_206_821,
    "train_source2": 5_034_616,
    "train_source3": 5_285_603,
    "train_ground_truth": 2_206_821,
    "test_source1": 1_732_544,
    "test_source2": 4_887_273,
    "test_source3": 5_082_316,
}

# ---------------------------------------------------------------------------
# Experiment sampling
# ---------------------------------------------------------------------------
# The full training set has ~2.2M Source-1 entities. Given the CPU-only,
# ~12GB-RAM development machine used to build this pipeline, exhaustive
# feature engineering / model selection experiments are run on a fixed-seed
# random sample of Source-1 TRAIN entities rather than all 2.2M. The blocking
# and candidate-generation code path is identical for the sample and for the
# full test run -- only the number of S1 "query" entities differs, since S2
# and S3 are always loaded and blocked against in full.
EXPERIMENT_SAMPLE_SIZE = 40_000   # number of train S1 entities sampled for dev/validation
SECOND_SPLIT_SAMPLE_SIZE = 15_000  # independent confirmation split

# Train / validation split (by Source-1 entity, no leakage across pairs).
VALIDATION_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Candidate generation / blocking
# ---------------------------------------------------------------------------
TFIDF_NAME_NGRAM_RANGE = (3, 5)
TFIDF_ADDRESS_NGRAM_RANGE = (3, 5)
TFIDF_TOP_K = 20          # neighbours retrieved per S1 entity per source, per field
# Exact TF-IDF candidate retrieval rescans the target corpus per S1 batch.
# Keep it opt-in for test inference until a reusable exact index is available.
ENABLE_TFIDF_RETRIEVAL = False
MAX_BLOCK_SIZE = 400      # skip (or subsample) hash-blocks bigger than this - avoids
                          # combinatorial blow-up from extremely common keys (e.g. empty
                          # names) while barely affecting recall (see audit notebook).

# ---------------------------------------------------------------------------
# Chunking for memory-bounded, streaming processing of the full-size data
# ---------------------------------------------------------------------------
S1_CHUNK_SIZE = 25_000    # number of Source-1 entities processed per streaming batch

# ---------------------------------------------------------------------------
# Legal / company suffix vocabulary used by normalization.py. Deliberately
# broad and multi-country (not US/India specific) since suffixes recur across
# many jurisdictions; this is just a normalization aid, not a hard filter.
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = [
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc",
    "llp", "lp", "ltd", "limited", "pvt", "private", "plc", "gmbh", "sa",
    "sas", "sarl", "bv", "nv", "ag", "kg", "oy", "ab", "spa", "srl", "pllc",
    "pc", "assoc", "associates", "group", "holdings", "enterprises", "ent",
    "intl", "international", "natl", "national",
]

RANDOM_STATE = RANDOM_SEED
