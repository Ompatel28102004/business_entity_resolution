"""
End-to-end candidate generation -> feature scoring -> decision pipeline.

Usable both as:
  * a library (``run_pipeline(...)``) from notebooks / validation scripts, and
  * a CLI:  ``python -m src.inference --split test --output-dir ../output``

Source-1 and Source-2/3 are processed in bounded chunks. Sparse TF-IDF
similarities are tiled, and a running global top-K is retained for each
Source-1 batch rather than materializing full Source-2/3 matrices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import time

import numpy as np
import pandas as pd

from . import blocking, config, features, output, retrieval
from .blocking_index import SQLiteBlockingIndex, ensure_test_blocking_index
from .data_loader import load_normalized_source
from .model import SklearnModelWrapper
from .thresholding import apply_decision_policy
from .utils import log, log_mem, timer

HASH_RULE_NAMES = [fn.__name__.replace("block_", "") for fn in blocking.ALL_HASH_RULES] + ["rare_name_token"]
ALL_RULE_NAMES = HASH_RULE_NAMES + ["tfidf_name", "tfidf_address"]
INFERENCE_COLUMNS = [
    "entity_id", "country_norm", "name_norm", "name_alnum", "name_core",
    "name_compact", "name_tokens", "name_sorted", "name_numbers",
    "name_first_token", "address_norm", "address_alnum", "address_tokens",
    "address_numbers", "address_postal", "address_first_number",
]


def _pivot_rule_flags(long_pairs: pd.DataFrame) -> pd.DataFrame:
    """Long (entity_id_s1, entity_id_other, rule) -> one row per pair with found_by_* booleans."""
    pairs = long_pairs[["entity_id_s1", "entity_id_other"]].drop_duplicates().reset_index(drop=True)
    out = pairs.copy()
    for rule in ALL_RULE_NAMES:
        rule_pairs = long_pairs.loc[long_pairs["rule"] == rule, ["entity_id_s1", "entity_id_other"]].copy()
        rule_pairs[f"found_by_{rule}"] = True
        out = out.merge(rule_pairs, on=["entity_id_s1", "entity_id_other"], how="left")
    flag_cols = [f"found_by_{r}" for r in ALL_RULE_NAMES]
    out[flag_cols] = out[flag_cols].fillna(False)
    return out


class VectorizedOtherSide:
    """Holds one normalized other-side table and optional training TF-IDF matrices."""

    def __init__(self, norm_df: pd.DataFrame, name_vec=None, addr_vec=None, source_name: str = "other"):
        self.norm_df = norm_df
        self.ids = self.norm_df["entity_id"].to_numpy()
        self.source_name = source_name
        self.index_source = source_name.lower().replace("-", "")
        self.name_matrix = name_vec.transform(self.norm_df["name_alnum"]) if name_vec is not None else None
        self.addr_matrix = addr_vec.transform(self.norm_df["address_alnum"]) if addr_vec is not None else None


def candidates_for_chunk_and_source(
    s1_chunk: pd.DataFrame,
    other: VectorizedOtherSide,
    name_vec,
    addr_vec,
    max_block_size: int,
    tfidf_k: int,
    tfidf_chunk_size: int = 256,
    tfidf_other_chunk_size: int = 10_000,
    blocking_index: Optional[SQLiteBlockingIndex] = None,
    enable_tfidf_retrieval: bool = True,
) -> pd.DataFrame:
    """Union hash-blocking + TF-IDF retrieval candidates for one (s1_chunk, other-source) pair.

    Returns a long frame: entity_id_s1, entity_id_other, rule, plus
    tfidf_score_name / tfidf_score_address columns (NaN where not
    applicable) so scores survive into feature computation without a
    second retrieval pass.
    """
    if blocking_index is None:
        hash_long = blocking.generate_hash_candidates(s1_chunk, other.norm_df, max_block_size)
    else:
        hash_long = blocking_index.candidates_for_chunk(s1_chunk, other.index_source)

    if enable_tfidf_retrieval:
        s1_name_matrix = name_vec.transform(s1_chunk["name_alnum"])
        s1_addr_matrix = addr_vec.transform(s1_chunk["address_alnum"])
        tfidf_name = retrieval.top_k_candidates_chunked(
            s1_name_matrix, other.norm_df["name_alnum"], name_vec,
            s1_chunk["entity_id"].to_numpy(), other.ids,
            k=tfidf_k, other_chunk_size=tfidf_other_chunk_size,
            s1_chunk_size=tfidf_chunk_size, label=f"{other.source_name} name",
        )
        tfidf_addr = retrieval.top_k_candidates_chunked(
            s1_addr_matrix, other.norm_df["address_alnum"], addr_vec,
            s1_chunk["entity_id"].to_numpy(), other.ids,
            k=tfidf_k, other_chunk_size=tfidf_other_chunk_size,
            s1_chunk_size=tfidf_chunk_size, label=f"{other.source_name} address",
        )
    else:
        empty_tfidf = pd.DataFrame(columns=["entity_id_s1", "entity_id_other", "tfidf_score"])
        tfidf_name = empty_tfidf
        tfidf_addr = empty_tfidf

    frames = [hash_long[["entity_id_s1", "entity_id_other", "rule"]]]
    if not tfidf_name.empty:
        frames.append(tfidf_name[["entity_id_s1", "entity_id_other"]].assign(rule="tfidf_name"))
    if not tfidf_addr.empty:
        frames.append(tfidf_addr[["entity_id_s1", "entity_id_other"]].assign(rule="tfidf_address"))
    long_pairs = pd.concat(frames, axis=0, ignore_index=True) if frames else hash_long.iloc[0:0]

    wide = _pivot_rule_flags(long_pairs)
    if not tfidf_name.empty:
        wide = wide.merge(
            tfidf_name.rename(columns={"tfidf_score": "tfidf_score_name"})[["entity_id_s1", "entity_id_other", "tfidf_score_name"]],
            on=["entity_id_s1", "entity_id_other"], how="left",
        )
    else:
        wide["tfidf_score_name"] = np.nan
    if not tfidf_addr.empty:
        wide = wide.merge(
            tfidf_addr.rename(columns={"tfidf_score": "tfidf_score_address"})[["entity_id_s1", "entity_id_other", "tfidf_score_address"]],
            on=["entity_id_s1", "entity_id_other"], how="left",
        )
    else:
        wide["tfidf_score_address"] = np.nan

    flag_cols = [f"found_by_{r}" for r in ALL_RULE_NAMES]
    wide["n_blocking_rules"] = wide[flag_cols].sum(axis=1)
    return wide


def score_chunk(
    s1_chunk: pd.DataFrame,
    s2_side: VectorizedOtherSide,
    s3_side: VectorizedOtherSide,
    name_vec,
    addr_vec,
    model_wrapper: SklearnModelWrapper,
    name_idf: dict,
    addr_idf: dict,
    max_block_size: int = config.MAX_BLOCK_SIZE,
    tfidf_k: int = config.TFIDF_TOP_K,
    tfidf_other_chunk_size: int = 10_000,
    blocking_index: Optional[SQLiteBlockingIndex] = None,
    enable_tfidf_retrieval: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate candidates, compute features, and score them for one Source-1 chunk.

    Returns (scored_pairs, candidate_pairs) where ``candidate_pairs`` is
    exactly the union set fed to the model (what goes into
    ``candidate_pairs.tsv``) and ``scored_pairs`` additionally has a
    ``score`` column (what the decision policy filters).
    """
    cand_s2 = candidates_for_chunk_and_source(
        s1_chunk, s2_side, name_vec, addr_vec, max_block_size, tfidf_k,
        tfidf_other_chunk_size=tfidf_other_chunk_size,
        blocking_index=blocking_index,
        enable_tfidf_retrieval=enable_tfidf_retrieval,
    )
    cand_s3 = candidates_for_chunk_and_source(
        s1_chunk, s3_side, name_vec, addr_vec, max_block_size, tfidf_k,
        tfidf_other_chunk_size=tfidf_other_chunk_size,
        blocking_index=blocking_index,
        enable_tfidf_retrieval=enable_tfidf_retrieval,
    )
    all_cand = pd.concat([cand_s2, cand_s3], axis=0, ignore_index=True)
    if all_cand.empty:
        return all_cand.assign(score=[]), all_cand

    rule_flag_cols = [c for c in all_cand.columns if c.startswith("found_by_")] + ["n_blocking_rules"]

    s2_records = (
        blocking_index.fetch_records("source2", cand_s2["entity_id_other"])
        if blocking_index is not None else s2_side.norm_df
    )
    s3_records = (
        blocking_index.fetch_records("source3", cand_s3["entity_id_other"])
        if blocking_index is not None else s3_side.norm_df
    )
    merged_s2 = features.merge_pair_fields(cand_s2[["entity_id_s1", "entity_id_other"]], s1_chunk, s2_records)
    merged_s2 = merged_s2.assign(
        tfidf_score_name=cand_s2["tfidf_score_name"].to_numpy(),
        tfidf_score_address=cand_s2["tfidf_score_address"].to_numpy(),
    )
    feat_s2 = features.compute_pair_features(merged_s2, name_idf, addr_idf, rule_flags=cand_s2[rule_flag_cols])

    merged_s3 = features.merge_pair_fields(cand_s3[["entity_id_s1", "entity_id_other"]], s1_chunk, s3_records)
    merged_s3 = merged_s3.assign(
        tfidf_score_name=cand_s3["tfidf_score_name"].to_numpy(),
        tfidf_score_address=cand_s3["tfidf_score_address"].to_numpy(),
    )
    feat_s3 = features.compute_pair_features(merged_s3, name_idf, addr_idf, rule_flags=cand_s3[rule_flag_cols])

    feat_all = pd.concat([feat_s2, feat_s3], axis=0, ignore_index=True)
    scores = model_wrapper.predict_proba_pair(feat_all)
    scored = feat_all[["entity_id_s1", "entity_id_other"]].copy()
    scored["score"] = scores
    return scored, all_cand[["entity_id_s1", "entity_id_other"]].drop_duplicates()


def run_pipeline(
    split: str,
    model_wrapper: SklearnModelWrapper,
    name_idf: dict,
    addr_idf: dict,
    threshold: float,
    output_dir: Path,
    s1_ids: Optional[Iterable[str]] = None,
    chunk_size: int = config.S1_CHUNK_SIZE,
    apply_consistency_rule: bool = False,
    min_score_gap: Optional[float] = None,
    n_jobs_normalize: int = 1,
    write_outputs: bool = True,
    test_sample_size: Optional[int] = None,
    tfidf_other_chunk_size: int = 10_000,
    blocking_index: Optional[SQLiteBlockingIndex] = None,
    enable_tfidf_retrieval: Optional[bool] = None,
) -> Dict[str, object]:
    """Run candidate generation + scoring + decision end to end for one data split.

    If ``s1_ids`` is given, only those Source-1 entities are processed
    (used for validation on a sampled subset); otherwise every entity in
    the split's Source-1 table is processed (the full-scale test run).
    """
    tfidf_enabled = (
        config.ENABLE_TFIDF_RETRIEVAL if enable_tfidf_retrieval is None else enable_tfidf_retrieval
    ) if split == "test" else True

    if split == "test" and blocking_index is None:
        blocking_index = ensure_test_blocking_index()

    with timer(f"load normalized tables ({split})"):
        s1_norm = load_normalized_source(split, "source1", n_jobs=n_jobs_normalize, columns=INFERENCE_COLUMNS)
        if split == "test" and not tfidf_enabled:
            other_columns = ["entity_id", "name_alnum", "address_alnum"]
            s2_norm = pd.DataFrame(columns=other_columns)
            s3_norm = pd.DataFrame(columns=other_columns)
        else:
            other_columns = ["entity_id", "name_alnum", "address_alnum"] if split == "test" else INFERENCE_COLUMNS
            s2_norm = load_normalized_source(split, "source2", n_jobs=n_jobs_normalize, columns=other_columns)
            s3_norm = load_normalized_source(split, "source3", n_jobs=n_jobs_normalize, columns=other_columns)
    log_mem("after loading normalized tables")

    if test_sample_size is not None:
        if split != "test":
            raise ValueError("--test-sample-size can only be used with --split test")
        if test_sample_size < 1 or test_sample_size > len(s1_norm):
            raise ValueError(f"test sample size must be between 1 and {len(s1_norm)}")
        s1_norm = s1_norm.sample(n=test_sample_size, random_state=config.RANDOM_SEED).reset_index(drop=True)
        log(f"Using deterministic test sample: {len(s1_norm):,} Source-1 rows")

    if s1_ids is not None:
        s1_norm = s1_norm[s1_norm["entity_id"].isin(set(s1_ids))].reset_index(drop=True)

    if tfidf_enabled:
        with timer("fit TF-IDF vectorizers"):
            name_vec = retrieval.fit_field_vectorizer(
                [s1_norm["name_alnum"], s2_norm["name_alnum"], s3_norm["name_alnum"]],
                ngram_range=config.TFIDF_NAME_NGRAM_RANGE,
            )
            addr_vec = retrieval.fit_field_vectorizer(
                [s1_norm["address_alnum"], s2_norm["address_alnum"], s3_norm["address_alnum"]],
                ngram_range=config.TFIDF_ADDRESS_NGRAM_RANGE,
            )
    else:
        name_vec = addr_vec = None
        log("TF-IDF candidate retrieval disabled for test; using indexed hash candidates only")

    s2_side = VectorizedOtherSide(s2_norm, source_name="Source-2")
    s3_side = VectorizedOtherSide(s3_norm, source_name="Source-3")
    del s2_norm, s3_norm

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    all_scored_frames = []
    all_candidate_frames = []
    n_chunks = -(-len(s1_norm) // chunk_size)
    for i, start in enumerate(range(0, len(s1_norm), chunk_size)):
        chunk = s1_norm.iloc[start : start + chunk_size]
        chunk_started = time.perf_counter()
        with timer(f"chunk {i + 1}/{n_chunks} ({len(chunk)} S1 entities)"):
            scored, cand = score_chunk(
                chunk, s2_side, s3_side, name_vec, addr_vec, model_wrapper, name_idf, addr_idf,
                tfidf_other_chunk_size=tfidf_other_chunk_size,
                blocking_index=blocking_index if split == "test" else None,
                enable_tfidf_retrieval=tfidf_enabled,
            )
            preds = apply_decision_policy(
                scored, threshold, min_score_gap=min_score_gap, apply_consistency_rule=apply_consistency_rule,
            )
            cand_map = cand.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()
            if write_outputs:
                output.append_output_chunk(
                    preds, chunk["entity_id"].tolist(), matching_path, "source1_entity_id", "matched_entity_ids",
                    first_chunk=(i == 0),
                )
                output.append_output_chunk(
                    cand_map, chunk["entity_id"].tolist(), candidate_path, "source1_entity_id", "candidate_entity_ids",
                    first_chunk=(i == 0),
                )
            else:
                all_scored_frames.append(scored.assign(**{"score": scored["score"]}))
                all_candidate_frames.append(cand)
        log_mem(
            f"Source-1 chunk {i + 1}/{n_chunks}: rows processed={min(start + len(chunk), len(s1_norm))}/{len(s1_norm)}, "
            f"candidate count={len(cand)}, elapsed={time.perf_counter() - chunk_started:.1f}s"
        )

    result = {"n_s1_processed": len(s1_norm)}
    if not write_outputs:
        result["scored"] = pd.concat(all_scored_frames, ignore_index=True) if all_scored_frames else pd.DataFrame()
        result["candidates"] = pd.concat(all_candidate_frames, ignore_index=True) if all_candidate_frames else pd.DataFrame()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Business Entity Resolution inference.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--model-path", default=str(config.MODELS_DIR / "final_model.joblib"))
    parser.add_argument("--idf-path", default=str(config.MODELS_DIR / "idf_tables.joblib"))
    parser.add_argument("--threshold", type=float, default=None, help="Overrides the threshold stored in models/threshold.json if given.")
    parser.add_argument("--chunk-size", type=int, default=config.S1_CHUNK_SIZE)
    parser.add_argument("--tfidf-other-chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--tfidf-retrieval", action=argparse.BooleanOptionalAction,
        default=None, help="Override test TF-IDF candidate retrieval (legacy exact corpus sweeps).",
    )
    parser.add_argument("--test-sample-size", type=int, default=None, help="Run inference on a deterministic sample of test Source-1 rows.")
    parser.add_argument("--n-jobs-normalize", type=int, default=1)
    parser.add_argument("--apply-consistency-rule", action="store_true")
    parser.add_argument("--min-score-gap", type=float, default=None)
    args = parser.parse_args()

    import json
    import joblib

    model_wrapper = SklearnModelWrapper.load(Path(args.model_path))
    idf_tables = joblib.load(Path(args.idf_path))
    threshold = args.threshold
    threshold_path = config.MODELS_DIR / "threshold.json"
    if threshold is None and threshold_path.exists():
        threshold = json.loads(threshold_path.read_text())["threshold"]
    if threshold is None:
        raise SystemExit("No threshold given and models/threshold.json not found. Pass --threshold.")

    run_pipeline(
        split=args.split,
        model_wrapper=model_wrapper,
        name_idf=idf_tables["name_idf"],
        addr_idf=idf_tables["addr_idf"],
        threshold=threshold,
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        apply_consistency_rule=args.apply_consistency_rule,
        min_score_gap=args.min_score_gap,
        n_jobs_normalize=args.n_jobs_normalize,
        write_outputs=True,
        test_sample_size=args.test_sample_size,
        tfidf_other_chunk_size=args.tfidf_other_chunk_size,
        enable_tfidf_retrieval=args.tfidf_retrieval,
    )


if __name__ == "__main__":
    main()
