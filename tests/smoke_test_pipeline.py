"""Smoke-test src.inference.run_pipeline end-to-end (chunked orchestration) plus
the official validator, on the same tiny synthetic dataset as smoke_test.py.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from unittest.mock import patch

from src import blocking, config, features, inference, model as model_module
from src.blocking_index import RULE_NAMES, SQLiteBlockingIndex
from src.normalization import add_all_normalizations
import tests.smoke_test as st  # reuse the synthetic data built there (module executes on import)

out_dir = Path(__file__).resolve().parent / "_smoke_pipeline_output"
out_dir.mkdir(exist_ok=True)
index_scratch = tempfile.TemporaryDirectory(prefix="ber-blocking-index-smoke-")
scratch_path = Path(index_scratch.name)
source2_path = scratch_path / "test_source2.tsv"
source3_path = scratch_path / "test_source3.tsv"
st.s2_df.to_csv(source2_path, sep="\t", index=False)
st.s3_df.to_csv(source3_path, sep="\t", index=False)
config.TEST_SOURCE2 = source2_path
config.TEST_SOURCE3 = source3_path
config.CACHE_DIR = scratch_path / "cache"
built_index = SQLiteBlockingIndex.ensure_test_index(max_block_size=400)
index_path = built_index.path
built_index.close()
blocking_index = SQLiteBlockingIndex.ensure_test_index(max_block_size=400)
assert blocking_index.path == index_path, "manifest-matched index was not reused"

# Query one persistent index from multiple S1 chunks and compare exact rule/pair output.
indexed_frames = []
for start in range(0, len(st.s1_norm), 4):
    s1_part = st.s1_norm.iloc[start : start + 4]
    indexed_frames.extend((
        blocking_index.candidates_for_chunk(s1_part, "source2"),
        blocking_index.candidates_for_chunk(s1_part, "source3"),
    ))
indexed_pairs = pd.concat(indexed_frames, ignore_index=True)
expected_pairs = pd.concat((
    blocking.generate_hash_candidates(st.s1_norm, st.s2_norm, max_block_size=400),
    blocking.generate_hash_candidates(st.s1_norm, st.s3_norm, max_block_size=400),
), ignore_index=True)
pair_columns = ["entity_id_s1", "entity_id_other", "rule"]
actual_set = set(map(tuple, indexed_pairs[pair_columns].to_numpy()))
expected_set = set(map(tuple, expected_pairs[pair_columns].to_numpy()))
assert actual_set == expected_set, "SQLite blocking output differs from legacy blocking"
assert {rule for _, _, rule in actual_set} == set(RULE_NAMES), "not all hash rules produced candidates"
true_pairs = {(s1_id, other_id) for s1_id, matches in st.gt.items() for other_id in matches}
assert true_pairs <= {(s1_id, other_id) for s1_id, other_id, _ in actual_set}

# Monkeypatch the data loader used inside run_pipeline to serve our synthetic tables.
_norm_map = {
    ("test", "source1"): st.s1_norm,
    ("test", "source2"): st.s2_norm,
    ("test", "source3"): st.s3_norm,
}
loaded_sources = []

def _load_synthetic_source(split, source, n_jobs=1, columns=None):
    loaded_sources.append(source)
    return _norm_map[(split, source)]

inference.load_normalized_source = _load_synthetic_source

wrapper = st.wrapper  # trained LogisticRegression wrapper from smoke_test.py
idf_tables = {"name_idf": st.name_idf, "addr_idf": st.addr_idf}

with patch.object(inference.retrieval, "fit_field_vectorizer", side_effect=AssertionError("TF-IDF fit must be skipped")):
    with patch.object(inference.retrieval, "top_k_candidates_chunked", side_effect=AssertionError("TF-IDF corpus scan must be skipped")):
        result = inference.run_pipeline(
            split="test",
            model_wrapper=wrapper,
            name_idf=idf_tables["name_idf"],
            addr_idf=idf_tables["addr_idf"],
            threshold=st.search.best_threshold,
            output_dir=out_dir,
            chunk_size=4,  # force multiple chunks with only 10 S1 rows
            tfidf_other_chunk_size=11,
            blocking_index=blocking_index,
            write_outputs=True,
        )
assert loaded_sources == ["source1"], f"test fallback loaded full other sources: {loaded_sources}"
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

validator = Path(__file__).resolve().parents[1] / "utils" / "validate_submission.py"
# validator = Path(__file__).resolve().parents[2] / "student_resource" / "utils" / "validate_submission.py"
proc = subprocess.run(
    [_sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
     "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir), "--check-ids"],
    capture_output=True, text=True,
)
print(proc.stdout)
print(proc.stderr)
assert proc.returncode == 0, "official validator FAILED on smoke-test output"
print("PIPELINE + OFFICIAL VALIDATOR SMOKE TEST PASSED")
blocking_index.close()
index_scratch.cleanup()
