"""
End-to-end candidate generation -> feature scoring -> decision pipeline.

Usable both as:
  * a library (``run_pipeline(...)``) from notebooks / validation scripts, and
  * a CLI:  ``python -m src.inference --split test --output-dir ../output``

Designed to run in bounded memory regardless of dataset size: Source-1
entities are processed in chunks (``config.S1_CHUNK_SIZE``) while the
Source-2/3 tables are loaded, normalized and vectorized ONCE up front and
then only read (never copied/grown) across chunks. This is the piece of the
pipeline meant to run on a SageMaker instance sized per
``aws/aws_config.yaml`` -- see ``aws/README_AWS.md`` -- not on a laptop.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import blocking, config, features, output, retrieval
from .data_loader import load_normalized_source
from .model import SklearnModelWrapper
from .thresholding import apply_decision_policy
from .utils import chunked, log, log_mem, timer

HASH_RULE_NAMES = [fn.__name__.replace("block_", "") for fn in blocking.ALL_HASH_RULES] + ["rare_name_token"]
ALL_RULE_NAMES = HASH_RULE_NAMES + ["tfidf_name", "tfidf_address"]


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
    """Holds a normalized Source-2 or Source-3 table plus its TF-IDF matrices.

    Built once per split and reused read-only across every Source-1 chunk,
    which is what keeps candidate generation from re-transforming millions
    of rows once per chunk.
    """

    def __init__(self, norm_df: pd.DataFrame, name_vec, addr_vec):
        self.norm_df = norm_df.reset_index(drop=True)
        self.ids = self.norm_df["entity_id"].to_numpy()
        self.name_matrix = name_vec.transform(self.norm_df["name_alnum"])
        self.addr_matrix = addr_vec.transform(self.norm_df["address_alnum"])


def candidates_for_chunk_and_source(
    s1_chunk: pd.DataFrame,
    other: VectorizedOtherSide,
    name_vec,
    addr_vec,
    max_block_size: int,
    tfidf_k: int,
    tfidf_chunk_size: int = 2000,
) -> pd.DataFrame:
    """Union hash-blocking + TF-IDF retrieval candidates for one (s1_chunk, other-source) pair.

    Returns a long frame: entity_id_s1, entity_id_other, rule, plus
    tfidf_score_name / tfidf_score_address columns (NaN where not
    applicable) so scores survive into feature computation without a
    second retrieval pass.
    """
    hash_long = blocking.generate_hash_candidates(s1_chunk, other.norm_df, max_block_size)

    s1_name_matrix = name_vec.transform(s1_chunk["name_alnum"])
    s1_addr_matrix = addr_vec.transform(s1_chunk["address_alnum"])
    tfidf_name = retrieval.top_k_candidates(
        s1_name_matrix, other.name_matrix, s1_chunk["entity_id"].to_numpy(), other.ids,
        k=tfidf_k, chunk_size=tfidf_chunk_size,
    )
    tfidf_addr = retrieval.top_k_candidates(
        s1_addr_matrix, other.addr_matrix, s1_chunk["entity_id"].to_numpy(), other.ids,
        k=tfidf_k, chunk_size=tfidf_chunk_size,
    )

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate candidates, compute features, and score them for one Source-1 chunk.

    Returns (scored_pairs, candidate_pairs) where ``candidate_pairs`` is
    exactly the union set fed to the model (what goes into
    ``candidate_pairs.tsv``) and ``scored_pairs`` additionally has a
    ``score`` column (what the decision policy filters).
    """
    cand_s2 = candidates_for_chunk_and_source(s1_chunk, s2_side, name_vec, addr_vec, max_block_size, tfidf_k)
    cand_s3 = candidates_for_chunk_and_source(s1_chunk, s3_side, name_vec, addr_vec, max_block_size, tfidf_k)
    all_cand = pd.concat([cand_s2, cand_s3], axis=0, ignore_index=True)
    if all_cand.empty:
        return all_cand.assign(score=[]), all_cand

    rule_flag_cols = [c for c in all_cand.columns if c.startswith("found_by_")] + ["n_blocking_rules"]

    merged_s2 = features.merge_pair_fields(cand_s2[["entity_id_s1", "entity_id_other"]], s1_chunk, s2_side.norm_df)
    merged_s2 = merged_s2.assign(
        tfidf_score_name=cand_s2["tfidf_score_name"].to_numpy(),
        tfidf_score_address=cand_s2["tfidf_score_address"].to_numpy(),
    )
    feat_s2 = features.compute_pair_features(merged_s2, name_idf, addr_idf, rule_flags=cand_s2[rule_flag_cols])

    merged_s3 = features.merge_pair_fields(cand_s3[["entity_id_s1", "entity_id_other"]], s1_chunk, s3_side.norm_df)
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
) -> Dict[str, object]:
    """Run candidate generation + scoring + decision end to end for one data split.

    If ``s1_ids`` is given, only those Source-1 entities are processed
    (used for validation on a sampled subset); otherwise every entity in
    the split's Source-1 table is processed (the full-scale test run).
    """
    with timer(f"load normalized tables ({split})"):
        s1_norm = load_normalized_source(split, "source1", n_jobs=n_jobs_normalize)
        s2_norm = load_normalized_source(split, "source2", n_jobs=n_jobs_normalize)
        s3_norm = load_normalized_source(split, "source3", n_jobs=n_jobs_normalize)
    log_mem("after loading normalized tables")

    if s1_ids is not None:
        s1_norm = s1_norm[s1_norm["entity_id"].isin(set(s1_ids))].reset_index(drop=True)

    with timer("fit TF-IDF vectorizers"):
        name_vec = retrieval.fit_field_vectorizer(
            [s1_norm["name_alnum"], s2_norm["name_alnum"], s3_norm["name_alnum"]],
            ngram_range=config.TFIDF_NAME_NGRAM_RANGE,
        )
        addr_vec = retrieval.fit_field_vectorizer(
            [s1_norm["address_alnum"], s2_norm["address_alnum"], s3_norm["address_alnum"]],
            ngram_range=config.TFIDF_ADDRESS_NGRAM_RANGE,
        )

    with timer("vectorize Source-2 / Source-3"):
        s2_side = VectorizedOtherSide(s2_norm, name_vec, addr_vec)
        s3_side = VectorizedOtherSide(s3_norm, name_vec, addr_vec)
    log_mem("after vectorizing S2/S3")

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    all_scored_frames = []
    all_candidate_frames = []
    n_chunks = -(-len(s1_norm) // chunk_size)
    for i, start in enumerate(range(0, len(s1_norm), chunk_size)):
        chunk = s1_norm.iloc[start : start + chunk_size]
        with timer(f"chunk {i + 1}/{n_chunks} ({len(chunk)} S1 entities)"):
            scored, cand = score_chunk(chunk, s2_side, s3_side, name_vec, addr_vec, model_wrapper, name_idf, addr_idf)
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
        log_mem(f"after chunk {i + 1}/{n_chunks}")

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
    )


if __name__ == "__main__":
    main()
