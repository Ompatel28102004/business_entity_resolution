"""
Training / model-selection / threshold-tuning orchestration.

This is the script that turns raw TRAIN data into a frozen, ready-to-serve
pipeline artifact:

  1. Select a (possibly sampled) set of Source-1 TRAIN entities. Sampling is
     always ENTITY-aware (whole Source-1 rows, never individual pair rows)
     and, for sample sizes smaller than the full dataset, restricts the
     Source-2/Source-3 pool that gets normalized/vectorized to exactly what
     is needed: every true match for the sampled entities (guaranteeing
     positives are never lost) plus a random background sample (for
     blocking-derived hard-negative mining) -- see ``build_training_pool``.
     This is what makes ``--train-sample-size 10000`` finish in a couple of
     minutes instead of paying the full ~5M-row Source-2/Source-3
     normalization cost regardless of sample size.
  2. Split Source-1 TRAIN entities into train/validation (by entity, fixed
     seed -- no pair-level leakage).
  3. Generate candidates for both splits exactly the way inference.py will
     at test time (same blocking + TF-IDF code path).
  4. Label candidate pairs from the ground truth (hard negatives are simply
     every non-matching candidate that survived blocking -- see labeling.py).
  5. Train and compare Model A/B/C/D (rule-based, logistic regression,
     random forest, gradient boosting) on the TRAIN split.
  6. Score the VALIDATION split with each model, search the entity-level
     F0.5-optimal decision threshold for each, and record everything to
     experiments/results.csv (appended, one row per model per run).
  7. Persist the winning model + IDF tables + threshold to models/.

CLI:
    python -m src.train --sample-size 10000 --val-fraction 0.25 --seed 42

Prefer ``run.py`` (the project's single entry point) for normal use --
it wires this module up to config.yaml, CLI overrides, logging, resume
state, and system-info reporting. This module's own CLI is kept for direct/
standalone use and backward compatibility.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config, features, model as model_module, normalization, retrieval, thresholding
from .data_loader import (
    load_ground_truth,
    load_ground_truth_exploded,
    load_normalized_source,
    load_source_ids,
    load_source_subset,
)
from .inference import VectorizedOtherSide, candidates_for_chunk_and_source
from .labeling import candidate_recall, label_candidates_fast, truth_dict_from_wide
from .metrics import evaluate_entity_level_f05, pair_level_precision_recall
from .utils import chunked, log, log_mem, mem_mb, new_run_id, timer, write_json


def split_source1_ids(all_ids: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    """Random split of Source-1 ids into train/validation, fixed seed, no pair leakage."""
    rng = np.random.default_rng(seed)
    ids = np.array(all_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    return ids[n_val:].tolist(), ids[:n_val].tolist()


def sample_source1_ids(all_ids: List[str], sample_size: int, seed: int) -> List[str]:
    """Fixed-seed random sample of Source-1 ids for tractable dev/validation runs.

    ``sample_size <= 0`` means "use every id" (full data), matching the
    ``--train-sample-size 0`` = full-data convention used throughout the CLI.
    """
    if sample_size <= 0 or sample_size >= len(all_ids):
        return list(all_ids)
    rng = np.random.default_rng(seed)
    return list(rng.choice(np.asarray(all_ids), size=sample_size, replace=False))


def _split_by_prefix(ids) -> Tuple[set, set]:
    """Split a mixed id collection into (Source-2 ids, Source-3 ids)."""
    s2 = {i for i in ids if i.startswith("S2-")}
    s3 = {i for i in ids if i.startswith("S3-")}
    return s2, s3


def build_background_ids(all_ids: pd.Series, exclude_ids: set, n: int, seed: int) -> set:
    """Fixed-seed random sample of up to ``n`` ids from ``all_ids``, excluding ``exclude_ids``.

    Used to build the "background" pool of Source-2/Source-3 records that
    accompanies the guaranteed true matches in a sampled training pool, so
    blocking has a realistic population to mine hard negatives from.
    """
    if n <= 0:
        return set()
    pool = all_ids[~all_ids.isin(exclude_ids)] if exclude_ids else all_ids
    n = min(n, len(pool))
    if n <= 0:
        return set()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(pool.to_numpy(), size=n, replace=False)
    return set(sampled)


def build_training_pool(
    s1_ids_sample: List[str],
    split: str,
    truth_full: Dict[str, set],
    background_multiplier: int = 20,
    seed: int = config.RANDOM_SEED,
    source2_cap: Optional[int] = None,
    source3_cap: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build a normalized Source-1/2/3 pool restricted to what a sampled training run needs.

    Rather than normalizing the full ~5M/~5.3M-row Source-2/Source-3 tables
    regardless of how many Source-1 entities were sampled (the original,
    much slower behaviour -- still used automatically for the full-data
    ``sample_size=0`` case, see ``run_training``), this:

      1. Looks up every ground-truth true match for the sampled Source-1
         entities. These rows are ALWAYS included, so positives are never
         lost to sampling.
      2. Adds a random "background" sample of additional Source-2/Source-3
         records (default size: ``background_multiplier`` per sampled
         Source-1 entity, per source; override with ``source2_cap`` /
         ``source3_cap``) so the normal blocking/TF-IDF pipeline has a
         realistic population to mine hard negatives from.
      3. Loads and normalizes ONLY those rows (via PyArrow predicate
         pushdown, see ``data_loader.load_source_subset``) instead of the
         entire table.

    Candidate RECALL measured on this pool is identical to what full-table
    blocking would find (whether a blocking rule links two specific records
    depends only on their own text, not on which other rows happen to sit
    in the same table) -- see methodology.md for the full argument. Rarity/
    IDF-based features and the "rare token" blocking rule's document
    frequency are estimated from the pool rather than the true full corpus,
    which is a deliberate, documented approximation for fast iteration.

    Returns (s1_norm, s2_norm, s3_norm, pool_metadata).
    """
    s1_set = set(s1_ids_sample)
    true_ids: set = set()
    for s1 in s1_ids_sample:
        true_ids |= truth_full.get(s1, set())
    s2_true, s3_true = _split_by_prefix(true_ids)

    with timer("load Source-2/Source-3 id universe (id-only, cheap)"):
        all_s2_ids = load_source_ids(split, "source2")
        all_s3_ids = load_source_ids(split, "source3")

    bg2_size = source2_cap if source2_cap else max(len(s2_true), len(s1_ids_sample) * background_multiplier)
    bg3_size = source3_cap if source3_cap else max(len(s3_true), len(s1_ids_sample) * background_multiplier)
    bg2 = build_background_ids(all_s2_ids, s2_true, bg2_size, seed)
    bg3 = build_background_ids(all_s3_ids, s3_true, bg3_size, seed + 1)

    keep_s2 = s2_true | bg2
    keep_s3 = s3_true | bg3

    with timer(f"load+normalize Source-1 pool ({len(s1_set):,} rows)"):
        s1_raw = load_source_subset(split, "source1", s1_set)
        s1_norm = normalization.add_all_normalizations(s1_raw)

    with timer(f"load+normalize Source-2 pool ({len(keep_s2):,} rows = {len(s2_true):,} true + {len(bg2):,} background)"):
        s2_raw = load_source_subset(split, "source2", keep_s2)
        s2_norm = normalization.add_all_normalizations(s2_raw)

    with timer(f"load+normalize Source-3 pool ({len(keep_s3):,} rows = {len(s3_true):,} true + {len(bg3):,} background)"):
        s3_raw = load_source_subset(split, "source3", keep_s3)
        s3_norm = normalization.add_all_normalizations(s3_raw)

    pool_meta = {
        "mode": "background_pool",
        "n_source1_entities": len(s1_set),
        "n_source2_true_matches": len(s2_true),
        "n_source2_background": len(bg2),
        "n_source2_pool": len(keep_s2),
        "n_source2_full_table": int(len(all_s2_ids)),
        "n_source3_true_matches": len(s3_true),
        "n_source3_background": len(bg3),
        "n_source3_pool": len(keep_s3),
        "n_source3_full_table": int(len(all_s3_ids)),
    }
    return s1_norm, s2_norm, s3_norm, pool_meta


def build_feature_table(
    s1_ids: List[str],
    s1_norm_all: pd.DataFrame,
    s2_side: VectorizedOtherSide,
    s3_side: VectorizedOtherSide,
    name_vec,
    addr_vec,
    name_idf: dict,
    addr_idf: dict,
    truth_exploded: pd.DataFrame,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Generate candidates + features + labels for a list of Source-1 ids.

    Returns (feature_df, labels, recall_stats). ``recall_stats`` reports the
    candidate recall ceiling for this id set -- the upper bound the
    downstream classifier/threshold cannot exceed.
    """
    s1_subset = s1_norm_all[s1_norm_all["entity_id"].isin(set(s1_ids))].reset_index(drop=True)
    feat_frames, cand_frames = [], []

    for chunk in chunked(s1_subset, chunk_size):
        cand_s2 = candidates_for_chunk_and_source(chunk, s2_side, name_vec, addr_vec, config.MAX_BLOCK_SIZE, config.TFIDF_TOP_K)
        cand_s3 = candidates_for_chunk_and_source(chunk, s3_side, name_vec, addr_vec, config.MAX_BLOCK_SIZE, config.TFIDF_TOP_K)

        rule_cols = [c for c in cand_s2.columns if c.startswith("found_by_")] + ["n_blocking_rules"]

        merged_s2 = features.merge_pair_fields(cand_s2[["entity_id_s1", "entity_id_other"]], chunk, s2_side.norm_df)
        merged_s2 = merged_s2.assign(tfidf_score_name=cand_s2["tfidf_score_name"].to_numpy(), tfidf_score_address=cand_s2["tfidf_score_address"].to_numpy())
        feat_s2 = features.compute_pair_features(merged_s2, name_idf, addr_idf, rule_flags=cand_s2[rule_cols])

        merged_s3 = features.merge_pair_fields(cand_s3[["entity_id_s1", "entity_id_other"]], chunk, s3_side.norm_df)
        merged_s3 = merged_s3.assign(tfidf_score_name=cand_s3["tfidf_score_name"].to_numpy(), tfidf_score_address=cand_s3["tfidf_score_address"].to_numpy())
        feat_s3 = features.compute_pair_features(merged_s3, name_idf, addr_idf, rule_flags=cand_s3[rule_cols])

        feat_frames.append(pd.concat([feat_s2, feat_s3], ignore_index=True))
        cand_frames.append(pd.concat([cand_s2[["entity_id_s1", "entity_id_other"]], cand_s3[["entity_id_s1", "entity_id_other"]]], ignore_index=True))

    feat_all = pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()
    cand_all = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame(columns=["entity_id_s1", "entity_id_other"])

    truth = truth_dict_from_wide(load_ground_truth())
    truth = {k: v for k, v in truth.items() if k in set(s1_ids)}
    recall_stats = candidate_recall(cand_all, truth, s1_ids=s1_ids)

    labels = label_candidates_fast(feat_all, truth_exploded) if not feat_all.empty else pd.Series(dtype=int)
    return feat_all, labels, recall_stats


def run_training(
    sample_size: int = config.EXPERIMENT_SAMPLE_SIZE,
    val_fraction: float = config.VALIDATION_FRACTION,
    seed: int = config.RANDOM_SEED,
    chunk_size: int = 2000,
    models_to_try: Tuple[str, ...] = ("rule_based", "logistic_regression", "random_forest", "hist_gradient_boosting"),
    background_multiplier: int = 20,
    source2_cap: Optional[int] = None,
    source3_cap: Optional[int] = None,
    use_background_pool: bool = True,
    n_jobs_normalize: int = 1,
    run_id: Optional[str] = None,
    save_sample_metadata: bool = True,
) -> Dict[str, object]:
    """Full training + model comparison + threshold search. Returns a results summary dict.

    ``sample_size <= 0`` (or >= the full Source-1 count) always uses the
    full, un-pooled data path (every Source-2/Source-3 row is normalized) --
    this is the path a full-scale ``--train-sample-size 0`` run takes.
    ``sample_size > 0`` uses the memory/time-efficient background-pool path
    (``build_training_pool``) unless ``use_background_pool=False``.
    """
    run_id = run_id or new_run_id()
    t_start = time.perf_counter()

    truth_full = truth_dict_from_wide(load_ground_truth())

    # IMPORTANT: when sample_size<=0 (full data) or background pooling is
    # disabled, we deliberately never touch `load_source_ids` separately --
    # we load the full normalized Source-1 table first (exactly like the
    # original implementation) and sample from ITS ids. This keeps the
    # full-data path's data access pattern identical to the original
    # implementation (a single source of ids), which matters both for
    # correctness and for tests/notebooks that substitute their own
    # `load_normalized_source`.
    if sample_size <= 0 or not use_background_pool:
        with timer("load normalized TRAIN tables (full, un-pooled)"):
            s1_norm = load_normalized_source("train", "source1", n_jobs=n_jobs_normalize)
            s2_norm = load_normalized_source("train", "source2", n_jobs=n_jobs_normalize)
            s3_norm = load_normalized_source("train", "source3", n_jobs=n_jobs_normalize)
        total_n = len(s1_norm)
        is_full = sample_size <= 0 or sample_size >= total_n
        sampled_ids = sample_source1_ids(s1_norm["entity_id"].tolist(), sample_size, seed)
        pool_meta = {
            "mode": "full",
            "n_source1_entities": len(sampled_ids),
            "n_source2_pool": len(s2_norm),
            "n_source3_pool": len(s3_norm),
        }
    else:
        with timer("resolve Source-1 id universe (id-only, cheap)"):
            all_ids_full = load_source_ids("train", "source1")
        total_n = len(all_ids_full)
        is_full = sample_size >= total_n
        sampled_ids = sample_source1_ids(all_ids_full.tolist(), sample_size, seed)
        if is_full:
            with timer("load normalized TRAIN tables (full, un-pooled)"):
                s1_norm = load_normalized_source("train", "source1", n_jobs=n_jobs_normalize)
                s2_norm = load_normalized_source("train", "source2", n_jobs=n_jobs_normalize)
                s3_norm = load_normalized_source("train", "source3", n_jobs=n_jobs_normalize)
            pool_meta = {
                "mode": "full",
                "n_source1_entities": len(sampled_ids),
                "n_source2_pool": len(s2_norm),
                "n_source3_pool": len(s3_norm),
            }
        else:
            s1_norm, s2_norm, s3_norm, pool_meta = build_training_pool(
                sampled_ids, "train", truth_full, background_multiplier, seed, source2_cap, source3_cap,
            )
    log(f"train sample: {len(sampled_ids):,} / {total_n:,} Source-1 entities (full={is_full})")
    log_mem("after preparing normalized train pool")
    log(f"training pool metadata: {pool_meta}")

    train_ids, val_ids = split_source1_ids(sampled_ids, val_fraction, seed)
    log(f"sampled {len(sampled_ids)} S1 ids -> {len(train_ids)} train / {len(val_ids)} validation")

    with timer("fit TF-IDF vectorizers (train split)"):
        name_vec = retrieval.fit_field_vectorizer([s1_norm["name_alnum"], s2_norm["name_alnum"], s3_norm["name_alnum"]], ngram_range=config.TFIDF_NAME_NGRAM_RANGE)
        addr_vec = retrieval.fit_field_vectorizer([s1_norm["address_alnum"], s2_norm["address_alnum"], s3_norm["address_alnum"]], ngram_range=config.TFIDF_ADDRESS_NGRAM_RANGE)

    with timer("vectorize Source-2 / Source-3 (train)"):
        s2_side = VectorizedOtherSide(s2_norm, name_vec, addr_vec)
        s3_side = VectorizedOtherSide(s3_norm, name_vec, addr_vec)
    log_mem("after vectorizing")

    name_idf = features.build_idf_table(s1_norm["name_tokens"].tolist() + s2_norm["name_tokens"].tolist() + s3_norm["name_tokens"].tolist())
    addr_idf = features.build_idf_table(s1_norm["address_tokens"].tolist() + s2_norm["address_tokens"].tolist() + s3_norm["address_tokens"].tolist())

    truth_exploded = load_ground_truth_exploded()

    with timer("build TRAIN feature table"):
        train_feat, train_labels, train_recall = build_feature_table(
            train_ids, s1_norm, s2_side, s3_side, name_vec, addr_vec, name_idf, addr_idf, truth_exploded, chunk_size,
        )
    log(f"train candidate recall: {train_recall}")
    log(f"train feature table: {train_feat.shape}, positives={int(train_labels.sum())}")

    with timer("build VALIDATION feature table"):
        val_feat, val_labels, val_recall = build_feature_table(
            val_ids, s1_norm, s2_side, s3_side, name_vec, addr_vec, name_idf, addr_idf, truth_exploded, chunk_size,
        )
    log(f"validation candidate recall: {val_recall}")

    feat_cols = features.feature_columns(train_feat)
    val_truth = {k: v for k, v in truth_full.items() if k in set(val_ids)}

    if save_sample_metadata:
        sample_meta = {
            "run_id": run_id,
            "train_sample_size_requested": sample_size,
            "is_full_data": is_full,
            "n_source1_entities_selected": len(sampled_ids),
            "n_train_entities": len(train_ids),
            "n_val_entities": len(val_ids),
            "n_positive_pairs_train": int(train_labels.sum()) if len(train_labels) else 0,
            "n_negative_pairs_train": int((train_labels == 0).sum()) if len(train_labels) else 0,
            "n_positive_pairs_val": int(val_labels.sum()) if len(val_labels) else 0,
            "n_negative_pairs_val": int((val_labels == 0).sum()) if len(val_labels) else 0,
            "pool_metadata": pool_meta,
        }
        write_json(config.EXPERIMENTS_DIR / f"sample_metadata_{run_id}.json", sample_meta)
        log(f"wrote sample metadata -> experiments/sample_metadata_{run_id}.json")

    results_rows = []
    best = {"macro_f05": -1.0}
    for name in models_to_try:
        with timer(f"train + evaluate model: {name}"):
            model_t0 = time.perf_counter()
            if name == "rule_based":
                wrapper = model_module.RuleBasedModel()
            else:
                wrapper = model_module.train_model(name, train_feat, train_labels, feat_cols)
            model_train_time = time.perf_counter() - model_t0

            val_scores = val_feat[["entity_id_s1", "entity_id_other"]].copy()
            val_scores["score"] = wrapper.predict_proba_pair(val_feat)

            search = thresholding.search_best_threshold(val_scores, val_truth, val_ids)
            refined = thresholding.refine_threshold_search(val_scores, val_truth, val_ids, search.best_threshold)
            best_t = refined.best_threshold
            preds = thresholding.predictions_at_threshold(val_scores, best_t)
            entity_result = evaluate_entity_level_f05(preds, val_truth, val_ids)
            pair_result = pair_level_precision_recall(preds, val_truth)

            row = {
                "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_name": name,
                "model": name,  # kept for backward compatibility with earlier results.csv consumers
                "train_sample_size": sample_size,
                "is_full_data": is_full,
                "num_source1_entities": len(sampled_ids),
                "num_training_pairs": len(train_feat),
                "threshold": best_t,
                "entity_f05": entity_result["macro_f05"],
                "macro_f05": entity_result["macro_f05"],
                "macro_precision": entity_result["macro_precision"],
                "macro_recall": entity_result["macro_recall"],
                "singleton_accuracy": entity_result["singleton_accuracy"],
                "pair_precision": pair_result["pair_precision"],
                "pair_recall": pair_result["pair_recall"],
                "candidate_recall": val_recall["candidate_recall"],
                "train_candidate_recall": train_recall["candidate_recall"],
                "val_candidate_recall": val_recall["candidate_recall"],
                "n_train_pairs": len(train_feat),
                "n_val_pairs": len(val_feat),
                "training_time_sec": round(model_train_time, 2),
                "inference_time_sec": None,
                "peak_memory_mb": round(mem_mb(), 1),
            }
            results_rows.append(row)
            log(f"  {name}: macro_f05={row['macro_f05']:.4f} threshold={best_t:.2f} precision={row['macro_precision']:.4f} recall={row['macro_recall']:.4f} train_time={model_train_time:.2f}s")

            if row["macro_f05"] > best["macro_f05"]:
                best = dict(row)
                best["wrapper"] = wrapper

    total_time = time.perf_counter() - t_start
    for row in results_rows:
        row["total_run_time_sec"] = round(total_time, 2)

    results_df = pd.DataFrame(results_rows)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = config.EXPERIMENTS_DIR / "results.csv"
    if results_path.exists():
        # Append: every run (10k, 100k, 500k, full, ...) accumulates in the same
        # experiment-tracking table instead of overwriting prior measurements.
        existing = pd.read_csv(results_path)
        results_df = pd.concat([existing, results_df], ignore_index=True, sort=False)
    results_df.to_csv(results_path, index=False)
    log(f"wrote {results_path} ({len(results_df)} total rows)")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if best["model"] != "rule_based":
        best["wrapper"].save(config.MODELS_DIR / "final_model.joblib")
    joblib.dump({"name_idf": name_idf, "addr_idf": addr_idf}, config.MODELS_DIR / "idf_tables.joblib")
    joblib.dump(name_vec, config.MODELS_DIR / "name_vectorizer.joblib")
    joblib.dump(addr_vec, config.MODELS_DIR / "addr_vectorizer.joblib")
    (config.MODELS_DIR / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": best["threshold"], "model": best["model"], "run_id": run_id,
                "train_sample_size": sample_size, "macro_f05": best["macro_f05"],
            },
            indent=2,
        )
    )
    write_json(
        config.MODELS_DIR / "training_manifest.json",
        {"run_id": run_id, "sample_size": sample_size, "is_full_data": is_full, "best_model": best["model"], "total_time_sec": round(total_time, 2)},
    )
    log(f"run {run_id} complete in {total_time:.1f}s -- best model: {best['model']} (macro_f05={best['macro_f05']:.4f})")

    return {"results": results_df, "best": {k: v for k, v in best.items() if k != "wrapper"}, "run_id": run_id, "pool_metadata": pool_meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select the Business Entity Resolution model.")
    parser.add_argument("--sample-size", type=int, default=config.EXPERIMENT_SAMPLE_SIZE, help="0 = use all TRAIN Source-1 entities (full scale).")
    parser.add_argument("--val-fraction", type=float, default=config.VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--background-multiplier", type=int, default=20)
    parser.add_argument("--no-background-pool", action="store_true", help="Disable pooled sampling; always normalize full Source-2/3 tables.")
    parser.add_argument("--n-jobs-normalize", type=int, default=1)
    args = parser.parse_args()

    summary = run_training(
        sample_size=args.sample_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        chunk_size=args.chunk_size,
        background_multiplier=args.background_multiplier,
        use_background_pool=not args.no_background_pool,
        n_jobs_normalize=args.n_jobs_normalize,
    )
    print(summary["results"].to_string(index=False))
    print("BEST:", summary["best"])


if __name__ == "__main__":
    main()
