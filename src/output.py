"""
Writing the two required submission files, with the format rules from the
challenge spec enforced *before* anything is written to disk:

  * exactly one row per required Source-1 id
  * matched/candidate ids only from Source-2/Source-3
  * no Source-1 ids inside a match/candidate list (no self-matches)
  * no duplicate ids within a single list
  * every id in matching_results.tsv must also appear in candidate_pairs.tsv
    for the same Source-1 entity
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd


class OutputFormatError(ValueError):
    """Raised when a prediction/candidate mapping violates a submission rule."""


def _check_ids(id_set: Set[str], label: str, s1: str) -> None:
    for mid in id_set:
        if mid.startswith("S1-"):
            raise OutputFormatError(f"{label} for {s1} contains a Source-1 id (self-match): {mid}")
        if not (mid.startswith("S2-") or mid.startswith("S3-")):
            raise OutputFormatError(f"{label} for {s1} contains an id without S2-/S3- prefix: {mid}")


def build_output_frame(
    predictions: Dict[str, Set[str]], required_s1_ids: Iterable[str], id_col: str, list_col: str
) -> pd.DataFrame:
    """Build a validated, ready-to-write DataFrame with exactly one row per required id.

    Raises ``OutputFormatError`` on any rule violation (fail fast, before
    writing anything -- much cheaper than discovering a bad submission
    after uploading it).
    """
    required_s1_ids = list(required_s1_ids)
    rows = []
    seen = set()
    for s1 in required_s1_ids:
        if s1 in seen:
            raise OutputFormatError(f"duplicate required id passed in: {s1}")
        seen.add(s1)
        id_set = predictions.get(s1, set())
        _check_ids(id_set, list_col, s1)
        if len(id_set) != len(set(id_set)):
            raise OutputFormatError(f"duplicate ids within list for {s1}")
        rows.append({id_col: s1, list_col: ",".join(sorted(id_set))})
    return pd.DataFrame(rows, columns=[id_col, list_col])


def write_matching_results(
    predictions: Dict[str, Set[str]], required_s1_ids: Iterable[str], path: Path
) -> pd.DataFrame:
    """Write ``matching_results.tsv``. Returns the DataFrame that was written."""
    df = build_output_frame(predictions, required_s1_ids, "source1_entity_id", "matched_entity_ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    return df


def write_candidate_pairs(
    candidates: Dict[str, Set[str]], required_s1_ids: Iterable[str], path: Path
) -> pd.DataFrame:
    """Write ``candidate_pairs.tsv``. Returns the DataFrame that was written."""
    df = build_output_frame(candidates, required_s1_ids, "source1_entity_id", "candidate_entity_ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    return df


def append_output_chunk(
    predictions: Dict[str, Set[str]],
    s1_ids_chunk: Iterable[str],
    path: Path,
    id_col: str,
    list_col: str,
    first_chunk: bool,
) -> None:
    """Append one streaming chunk of validated output rows to ``path``.

    Used by ``inference.py`` to keep peak memory bounded by
    ``config.S1_CHUNK_SIZE`` regardless of the total number of Source-1
    entities (1.7M+ at full test scale): each chunk is validated with the
    same fail-fast rules as ``build_output_frame`` and written immediately,
    instead of accumulating one giant in-memory mapping for the whole run.
    """
    df = build_output_frame(predictions, s1_ids_chunk, id_col, list_col)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
        mode="w" if first_chunk else "a",
        header=first_chunk,
    )


def check_matches_subset_of_candidates(
    predictions: Dict[str, Set[str]], candidates: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """Return {s1: extra_ids} for any Source-1 entity whose matches are not a subset of its candidates.

    An empty dict means every final match came from the candidate set, as
    required by the spec ("Every final match must appear inside
    candidate_entity_ids").
    """
    offenders = {}
    for s1, matched in predictions.items():
        extra = matched - candidates.get(s1, set())
        if extra:
            offenders[s1] = extra
    return offenders
