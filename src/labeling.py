"""
Label generation and candidate-recall measurement.

A candidate pair is labelled 1 iff it is a true match in
``train_ground_truth.tsv``, else 0. Because every negative candidate pair
here has already survived at least one blocking rule (exact key match or a
top-K TF-IDF hit), negatives generated this way are automatically "hard
negatives" in the sense the challenge asks for: they are, by construction,
textually or structurally similar to the Source-1 query and yet wrong. No
separate random-negative-sampling step is needed (and random pairs would
mostly be trivially easy negatives that add little training signal).
"""
from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def truth_dict_from_exploded(exploded: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build {source1_entity_id: {matched_entity_id, ...}} from the exploded ground truth."""
    return exploded.groupby("source1_entity_id")["matched_entity_id"].apply(set).to_dict()


def truth_dict_from_wide(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build the same mapping directly from the raw (wide) ground-truth table.

    Useful when you need every Source-1 id represented (including those with
    an empty match list), which a purely exploded representation would drop.
    """
    out: Dict[str, Set[str]] = {}
    for s1, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = set(matches.split(",")) if matches else set()
    return out


def label_candidates(candidates: pd.DataFrame, truth: Dict[str, Set[str]]) -> pd.Series:
    """Return a 0/1 label Series aligned to ``candidates`` (entity_id_s1, entity_id_other)."""
    def _is_match(row) -> int:
        return int(row.entity_id_other in truth.get(row.entity_id_s1, ()))

    return candidates.apply(_is_match, axis=1)


def label_candidates_fast(candidates: pd.DataFrame, exploded: pd.DataFrame) -> pd.Series:
    """Vectorized labelling via a merge indicator (much faster than a Python-level apply).

    ``exploded`` is the long-format ground truth (source1_entity_id,
    matched_entity_id). A left-merge with an indicator flags exactly the
    candidate rows that also appear in the true-pairs table.
    """
    key_cols = ["entity_id_s1", "entity_id_other"]
    truth_pairs = exploded.rename(
        columns={"source1_entity_id": "entity_id_s1", "matched_entity_id": "entity_id_other"}
    )[key_cols].drop_duplicates()
    truth_pairs["_label"] = 1
    merged = candidates[key_cols].merge(truth_pairs, on=key_cols, how="left")
    return merged["_label"].fillna(0).astype(int)


def candidate_recall(
    candidate_pairs: pd.DataFrame, truth: Dict[str, Set[str]], s1_ids: pd.Index | None = None
) -> Dict[str, float]:
    """Measure candidate recall: fraction of TRUE matched pairs present in the candidate set.

    Only Source-1 ids in ``s1_ids`` (if given) are considered, so recall can
    be measured on a validation subsample without loading the full
    candidate universe.
    """
    cand_set = set(zip(candidate_pairs["entity_id_s1"], candidate_pairs["entity_id_other"]))
    total_true = 0
    found_true = 0
    keys = truth.keys() if s1_ids is None else (s1_ids if s1_ids is not None else truth.keys())
    for s1 in keys:
        true_matches = truth.get(s1, set())
        total_true += len(true_matches)
        for m in true_matches:
            if (s1, m) in cand_set:
                found_true += 1
    recall = found_true / total_true if total_true else float("nan")
    return {"n_true_pairs": total_true, "n_found_pairs": found_true, "candidate_recall": recall}
