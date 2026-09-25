"""
The official challenge metric: macro-averaged, entity-level F-beta (beta=0.5).

F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed per Source-1 entity and averaged over ALL Source-1 entities in the
evaluation set (singletons included). This module is deliberately
self-contained (no pandas required for the core computation) so it can be
unit-tested trivially and trusted as the single source of truth for every
threshold-selection / model-comparison decision made elsewhere in the
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Set

import numpy as np


@dataclass
class EntityScore:
    """Per-entity scoring breakdown, useful for error analysis."""

    source1_id: str
    precision: float
    recall: float
    f05: float
    n_true: int
    n_pred: int
    n_tp: int


def entity_prf(pred_set: Set[str], true_set: Set[str]) -> tuple[float, float, float]:
    """Compute (precision, recall, F0.5) for one Source-1 entity.

    Edge cases (matching the official spec exactly):
      * true empty, pred empty   -> (1.0, 1.0, 1.0)   -- singleton correctly kept empty
      * true empty, pred non-empty -> precision = 0    -> F0.5 = 0.0 (false merge)
      * true non-empty, pred empty -> recall = 0        -> F0.5 = 0.0 (missed everything)
      * otherwise standard precision/recall/F-beta with beta=0.5
    """
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        # true_set is non-empty here (the all-empty case was handled above)
        return 0.0, 0.0, 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0
    if precision == 0.0 and recall == 0.0:
        return precision, recall, 0.0
    beta2 = 0.25  # beta^2 for beta=0.5
    f05 = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    return precision, recall, f05


def evaluate_entity_level_f05(
    predictions: Mapping[str, Set[str]],
    truth: Mapping[str, Set[str]],
    all_source1_ids: Iterable[str] | None = None,
) -> Dict[str, object]:
    """Compute the macro-averaged entity-level F0.5 metric over a set of S1 entities.

    Parameters
    ----------
    predictions : mapping source1_entity_id -> set of predicted matched ids.
        Entities absent from this mapping are treated as an empty prediction.
    truth : mapping source1_entity_id -> set of true matched ids.
        Entities absent from this mapping are treated as true empty (no match).
    all_source1_ids : the full universe of Source-1 ids to average over. If
        None, uses the union of keys seen in ``predictions`` and ``truth``.

    Returns
    -------
    A dict with:
        macro_f05, macro_precision, macro_recall,
        singleton_accuracy (fraction of true-empty entities correctly predicted empty),
        n_entities, per_entity (list[EntityScore])
    """
    if all_source1_ids is None:
        all_source1_ids = set(predictions.keys()) | set(truth.keys())
    all_source1_ids = list(all_source1_ids)

    per_entity = []
    precisions = np.empty(len(all_source1_ids))
    recalls = np.empty(len(all_source1_ids))
    f05s = np.empty(len(all_source1_ids))

    n_singletons = 0
    n_singletons_correct = 0

    for i, s1 in enumerate(all_source1_ids):
        pred_set = predictions.get(s1, set())
        true_set = truth.get(s1, set())
        precision, recall, f05 = entity_prf(pred_set, true_set)
        precisions[i] = precision
        recalls[i] = recall
        f05s[i] = f05
        per_entity.append(
            EntityScore(
                source1_id=s1,
                precision=precision,
                recall=recall,
                f05=f05,
                n_true=len(true_set),
                n_pred=len(pred_set),
                n_tp=len(pred_set & true_set),
            )
        )
        if not true_set:
            n_singletons += 1
            if not pred_set:
                n_singletons_correct += 1

    return {
        "macro_f05": float(f05s.mean()) if len(f05s) else float("nan"),
        "macro_precision": float(precisions.mean()) if len(precisions) else float("nan"),
        "macro_recall": float(recalls.mean()) if len(recalls) else float("nan"),
        "singleton_accuracy": (n_singletons_correct / n_singletons) if n_singletons else float("nan"),
        "n_singletons": n_singletons,
        "n_entities": len(all_source1_ids),
        "per_entity": per_entity,
    }


def pair_level_precision_recall(
    predictions: Mapping[str, Set[str]], truth: Mapping[str, Set[str]]
) -> Dict[str, float]:
    """Micro (pair-level) precision/recall/F1 -- a diagnostic, NOT the leaderboard metric.

    Included because the challenge explicitly warns against optimizing only
    ordinary pair-level accuracy/F1; reporting it alongside the entity-level
    F0.5 makes that distinction visible in experiment logs.
    """
    tp = fp = fn = 0
    all_ids = set(predictions.keys()) | set(truth.keys())
    for s1 in all_ids:
        pred_set = predictions.get(s1, set())
        true_set = truth.get(s1, set())
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {"pair_precision": precision, "pair_recall": recall, "pair_f1": f1, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    # Minimal self-test matching the worked example in the challenge README:
    # pred = {S2-00047, S2-00193, S3-00812}; true = {S2-00047, S3-00812}
    # expected precision=2/3, recall=1.0, F0.5 = 0.714 (rounded)
    pred = {"S1-1": {"S2-00047", "S2-00193", "S3-00812"}}
    true = {"S1-1": {"S2-00047", "S3-00812"}}
    result = evaluate_entity_level_f05(pred, true)
    p, r, f = result["macro_precision"], result["macro_recall"], result["macro_f05"]
    print(f"precision={p:.3f} recall={r:.3f} f05={f:.3f} (expected ~0.667, 1.000, 0.714)")
    assert abs(p - 2 / 3) < 1e-9
    assert abs(r - 1.0) < 1e-9
    assert abs(f - 0.7142857142857143) < 1e-9

    # Singleton correctly predicted empty -> 1.0
    assert entity_prf(set(), set()) == (1.0, 1.0, 1.0)
    # Singleton with a false positive -> 0.0
    assert entity_prf({"S2-1"}, set()) == (0.0, 0.0, 0.0)
    # Missed entity entirely (pred empty, true non-empty) -> 0.0
    assert entity_prf(set(), {"S2-1"}) == (0.0, 0.0, 0.0)
    # Perfect match
    assert entity_prf({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == (1.0, 1.0, 1.0)
    print("All metrics self-tests passed.")
