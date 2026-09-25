"""Smoke-test src.inference.run_pipeline end-to-end (chunked orchestration) plus
the official validator, on the same tiny synthetic dataset as smoke_test.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import features, inference, model as model_module
from src.normalization import add_all_normalizations
import tests.smoke_test as st  # reuse the synthetic data built there (module executes on import)

out_dir = Path(__file__).resolve().parent / "_smoke_pipeline_output"
out_dir.mkdir(exist_ok=True)

# Monkeypatch the data loader used inside run_pipeline to serve our synthetic tables.
_norm_map = {
    ("test", "source1"): st.s1_norm,
    ("test", "source2"): st.s2_norm,
    ("test", "source3"): st.s3_norm,
}
inference.load_normalized_source = lambda split, source, n_jobs=1: _norm_map[(split, source)]

wrapper = st.wrapper  # trained LogisticRegression wrapper from smoke_test.py
idf_tables = {"name_idf": st.name_idf, "addr_idf": st.addr_idf}

result = inference.run_pipeline(
    split="test",
    model_wrapper=wrapper,
    name_idf=idf_tables["name_idf"],
    addr_idf=idf_tables["addr_idf"],
    threshold=st.search.best_threshold,
    output_dir=out_dir,
    chunk_size=4,  # force multiple chunks with only 10 S1 rows
    write_outputs=True,
)
print("run_pipeline result:", result)

match_df = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", keep_default_na=False)
cand_df = pd.read_csv(out_dir / "candidate_pairs.tsv", sep="\t", keep_default_na=False)
print(match_df)
print(cand_df.head())
assert len(match_df) == 10, f"expected 10 rows, got {len(match_df)}"
assert list(match_df.columns) == ["source1_entity_id", "matched_entity_ids"]
assert list(cand_df.columns) == ["source1_entity_id", "candidate_entity_ids"]

# Build a tiny test_dir and run the OFFICIAL validator against these outputs.
test_dir = out_dir / "test_dir"
test_dir.mkdir(exist_ok=True)
st.s1_df.rename(columns={}).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
st.s2_df.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
st.s3_df.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

import subprocess, sys as _sys
validator = Path(__file__).resolve().parents[2] / "student_resource" / "utils" / "validate_submission.py"
proc = subprocess.run(
    [_sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
     "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir), "--check-ids"],
    capture_output=True, text=True,
)
print(proc.stdout)
print(proc.stderr)
assert proc.returncode == 0, "official validator FAILED on smoke-test output"
print("PIPELINE + OFFICIAL VALIDATOR SMOKE TEST PASSED")
