"""
Entity-level F0.5 threshold search and final match-set decision logic.

The pair classifier produces a score in [0, 1] for every candidate pair.
Turning scores into final match sets is *not* a plain ``score >= 0.5`` cut:
this module searches, on a labelled validation split, for the decision
policy that maximizes the exact challenge metric
(``metrics.evaluate_entity_level_f05``), and implements the "no-match"
(singleton) and multi-match decision logic the methodology calls for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import numpy as np
import pandas as pd

from .metrics import evaluate_entity_level_f05, pair_level_precision_recall


def predictions_at_threshold(
    scores: pd.DataFrame, threshold: float, score_col: str = "score"
) -> Dict[str, Set[str]]:
    """Accept every candidate pair with score >= threshold, grouped by Source-1 id.

    ``scores`` must have columns entity_id_s1, entity_id_other, <score_col>.
    Source-1 ids with no accepted candidate are simply absent from the
    returned mapping (callers should treat missing keys as empty sets, which
    is exactly what ``metrics.evaluate_entity_level_f05`` does).
    """
    accepted = scores[scores[score_col] >= threshold]
    return accepted.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()


def resolve_one_parent_per_match(
    scores: pd.DataFrame, threshold: float, score_col: str = "score"
) -> Dict[str, Set[str]]:
    """Like ``predictions_at_threshold``, plus a consistency rule.

    Training data shows every true matched Source-2/3 id belongs to exactly
    one Source-1 entity (measured: 0 out of 7,638,365 true pairs violate
    this). When the same candidate id is accepted for more than one
    Source-1 query, keep only the highest-scoring assignment and drop the
    rest -- trading a small amount of recall for higher precision on
    genuinely ambiguous, shared candidates. Only used if validation shows it
    helps (see experiments/results.csv); otherwise the plain,
    independent-per-pair decision above is used.
    """
    accepted = scores[scores[score_col] >= threshold].copy()
    accepted = accepted.sort_values(score_col, ascending=False)
    accepted = accepted.drop_duplicates(subset=["entity_id_other"], keep="first")
    return accepted.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()


@dataclass
class ThresholdSearchResult:
    best_threshold: float
    table: pd.DataFrame


def search_best_threshold(
    scores: pd.DataFrame,
    truth: Dict[str, Set[str]],
    all_s1_ids,
    thresholds=None,
    score_col: str = "score",
    apply_consistency_rule: bool = False,
) -> ThresholdSearchResult:
    """Grid-search decision thresholds, scoring each with the entity-level F0.5 metric.

    Returns the threshold with the best macro F0.5 plus a full metrics table
    (one row per threshold) for inspection/plotting.
    """
    if thresholds is None:
        coarse = np.round(np.arange(0.30, 0.96, 0.05), 2)
        thresholds = coarse

    rows = []
    predictor = resolve_one_parent_per_match if apply_consistency_rule else predictions_at_threshold
    for t in thresholds:
        preds = predictor(scores, t, score_col=score_col)
        result = evaluate_entity_level_f05(preds, truth, all_s1_ids)
        pair_metrics = pair_level_precision_recall(preds, truth)
        n_pred_total = sum(len(v) for v in preds.values())
        rows.append(
            {
                "threshold": t,
                "macro_f05": result["macro_f05"],
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "singleton_accuracy": result["singleton_accuracy"],
                "pair_precision": pair_metrics["pair_precision"],
                "pair_recall": pair_metrics["pair_recall"],
                "avg_predicted_matches": n_pred_total / len(list(all_s1_ids)),
            }
        )
    table = pd.DataFrame(rows)
    best_row = table.loc[table["macro_f05"].idxmax()]
    return ThresholdSearchResult(best_threshold=float(best_row["threshold"]), table=table)


def refine_threshold_search(
    scores: pd.DataFrame,
    truth: Dict[str, Set[str]],
    all_s1_ids,
    coarse_best: float,
    span: float = 0.05,
    step: float = 0.01,
    score_col: str = "score",
    apply_consistency_rule: bool = False,
) -> ThresholdSearchResult:
    """Fine-grained search in a narrow band around a coarse best threshold."""
    lo, hi = max(0.0, coarse_best - span), min(1.0, coarse_best + span)
    fine = np.round(np.arange(lo, hi + step / 2, step), 3)
    return search_best_threshold(
        scores, truth, all_s1_ids, thresholds=fine, score_col=score_col,
        apply_consistency_rule=apply_consistency_rule,
    )


def search_source_specific_thresholds(
    scores: pd.DataFrame,
    truth: Dict[str, Set[str]],
    all_s1_ids,
    global_threshold: float,
    min_support: int = 500,
    thresholds=None,
    score_col: str = "score",
) -> Dict[str, float]:
    """Optionally tune separate thresholds for Source-1->Source-2 vs ->Source-3 pairs.

    Falls back to ``global_threshold`` for a source when it has too little
    validation support (< ``min_support`` candidate pairs) to avoid
    overfitting a subgroup threshold, per the methodology's explicit
    instruction not to overfit tiny subgroups.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.96, 0.02), 2)
    result = {}
    for source_tag, prefix in (("source2", "S2-"), ("source3", "S3-")):
        subset = scores[scores["entity_id_other"].str.startswith(prefix)]
        if len(subset) < min_support:
            result[source_tag] = global_threshold
            continue
        search = search_best_threshold(subset, truth, all_s1_ids, thresholds=thresholds, score_col=score_col)
        result[source_tag] = search.best_threshold
    return result


def apply_decision_policy(
    scores: pd.DataFrame,
    threshold: float,
    min_score_gap: float | None = None,
    score_col: str = "score",
    apply_consistency_rule: bool = False,
) -> Dict[str, Set[str]]:
    """Apply the final, frozen decision policy used at inference time.

    Optionally requires the accepted score to be at least ``min_score_gap``
    above the runner-up candidate for the same Source-1 entity when there
    are >= 2 candidates -- an ambiguity guard: if the top two candidates are
    nearly tied, evidence is weaker that either one specifically is correct.
    Only enabled if validation shows an improvement (see experiments log);
    disabled (``None``) reduces to the plain threshold rule.
    """
    df = scores.copy()
    if min_score_gap is not None:
        df = df.sort_values(score_col, ascending=False)
        df["_rank"] = df.groupby("entity_id_s1").cumcount()
        top1 = df[df["_rank"] == 0][["entity_id_s1", score_col]].rename(columns={score_col: "_top1"})
        top2 = df[df["_rank"] == 1][["entity_id_s1", score_col]].rename(columns={score_col: "_top2"})
        gaps = top1.merge(top2, on="entity_id_s1", how="left")
        gaps["_top2"] = gaps["_top2"].fillna(-1.0)
        gaps["_gap_ok"] = (gaps["_top1"] - gaps["_top2"]) >= min_score_gap
        ok_s1 = set(gaps.loc[gaps["_gap_ok"], "entity_id_s1"])
        # Only the (unique) top-1 pair is affected by the gap rule; all other
        # accepted candidates for the same entity are unaffected.
        is_top1 = df["_rank"] == 0
        drop_mask = is_top1 & (~df["entity_id_s1"].isin(ok_s1))
        df = df[~drop_mask]

    if apply_consistency_rule:
        return resolve_one_parent_per_match(df, threshold, score_col=score_col)
    return predictions_at_threshold(df, threshold, score_col=score_col)
