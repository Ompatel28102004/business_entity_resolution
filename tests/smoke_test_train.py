"""Smoke-test src.train.run_training end-to-end on the tiny synthetic dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, train as train_module
import tests.smoke_test as st  # noqa: F401 (builds st.s1_df/s2_df/s3_df/s1_norm/...)

# Monkeypatch every real-data loader train.run_training touches.
_norm_map = {
    ("train", "source1"): st.s1_norm,
    ("train", "source2"): st.s2_norm,
    ("train", "source3"): st.s3_norm,
}
train_module.load_normalized_source = lambda split, source, n_jobs=1: _norm_map[(split, source)]

gt_rows = [{"source1_entity_id": s1, "matched_entity_ids": ",".join(sorted(m))} for s1, m in st.gt.items()]
gt_wide = pd.DataFrame(gt_rows)
gt_exploded_rows = [{"source1_entity_id": s1, "matched_entity_id": m} for s1, matches in st.gt.items() for m in matches]
gt_exploded = pd.DataFrame(gt_exploded_rows, columns=["source1_entity_id", "matched_entity_id"])

train_module.load_ground_truth = lambda: gt_wide
train_module.load_ground_truth_exploded = lambda: gt_exploded

# Redirect experiments/models output to a scratch dir so this doesn't clobber real artifacts.
scratch = Path(__file__).resolve().parent / "_smoke_train_output"
config.EXPERIMENTS_DIR = scratch / "experiments"
config.MODELS_DIR = scratch / "models"
train_module.config.EXPERIMENTS_DIR = config.EXPERIMENTS_DIR
train_module.config.MODELS_DIR = config.MODELS_DIR

summary = train_module.run_training(
    sample_size=0,  # use all 10 synthetic S1 ids
    val_fraction=0.4,
    seed=42,
    chunk_size=4,
    models_to_try=("rule_based", "logistic_regression", "random_forest"),
)
print(summary["results"].to_string(index=False))
print("BEST:", summary["best"])
assert (config.MODELS_DIR / "threshold.json").exists()
assert (config.MODELS_DIR / "idf_tables.joblib").exists()
print("TRAIN SMOKE TEST PASSED")
