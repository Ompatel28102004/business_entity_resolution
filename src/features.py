"""
Pair feature engineering.

Given a set of candidate (Source-1, Source-2/3) pairs, this module builds a
rich numeric feature matrix combining:

  * fuzzy string-similarity scores (RapidFuzz) on several name/address
    representations,
  * token-set overlap statistics (Jaccard, common-token counts/ratios),
  * numeric/digit agreement (door numbers, PIN/ZIP-like tokens),
  * cross-field signals (country agreement, combined name/address scores),
  * blocking provenance (which rule(s) retrieved the pair -- "how many
    independent pieces of evidence support this candidate"),
  * IDF-based rarity features estimated from the TRAINING reference data
    only (never from an external database, per the fair-play rules).

All string-similarity functions are RapidFuzz (C++-backed, MIT licensed).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

DEFAULT_IDF = 12.0  # assigned to any token never seen in the training reference corpus (== "very rare")
RARE_IDF_THRESHOLD = 6.0

NAME_NORM_COLS = [
    "name_raw", "name_norm", "name_alnum", "name_core", "name_compact",
    "name_tokens", "name_sorted", "name_numbers", "name_first_token",
]
ADDR_NORM_COLS = [
    "address_raw", "address_norm", "address_alnum", "address_compact",
    "address_tokens", "address_sorted", "address_numbers", "address_postal",
    "address_first_number",
]
ID_COLS = ["entity_id", "country", "country_norm"]
PAIR_FEATURE_COLS = [
    "entity_id", "country_norm", "name_norm", "name_alnum", "name_core",
    "name_compact", "name_tokens", "name_sorted", "name_numbers",
    "address_alnum", "address_tokens", "address_numbers", "address_postal",
]


def build_idf_table(token_lists: Iterable[List[str]]) -> Dict[str, float]:
    """Estimate an IDF (inverse document frequency) score per token.

    ``token_lists`` should be the tokenized field (e.g. ``name_tokens``)
    from the TRAINING reference tables only (Source-1 + Source-2 + Source-3
    train). idf(t) = log((N + 1) / (df(t) + 1)) + 1, the standard smoothed
    IDF used by scikit-learn's TfidfVectorizer, so scores are comparable in
    spirit even though this table is used for hand-built rarity features
    rather than for a vectorizer.
    """
    df_counter: Counter = Counter()
    n_docs = 0
    for toks in token_lists:
        n_docs += 1
        df_counter.update(set(toks))
    return {
        tok: float(np.log((n_docs + 1) / (df + 1)) + 1.0)
        for tok, df in df_counter.items()
    }


def merge_pair_fields(
    pairs: pd.DataFrame, s1_norm: pd.DataFrame, other_norm: pd.DataFrame
) -> pd.DataFrame:
    """Attach every normalized S1 field and every normalized "other" field to a candidate-pairs frame.

    ``pairs`` must have columns entity_id_s1, entity_id_other (duplicates
    across blocking rules should already be dropped by the caller).
    """
    s1_cols = PAIR_FEATURE_COLS
    other_cols = PAIR_FEATURE_COLS

    left = pairs.merge(
        s1_norm[s1_cols].rename(columns={c: f"{c}_s1" for c in s1_cols}),
        left_on="entity_id_s1", right_on="entity_id_s1", how="left",
    )
    merged = left.merge(
        other_norm[other_cols].rename(columns={c: f"{c}_other" for c in other_cols}),
        left_on="entity_id_other", right_on="entity_id_other", how="left",
    )
    return merged


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def _common_count(a: list, b: list) -> int:
    return len(set(a) & set(b))


def _items_or_empty(items):
    if items is None:
        return []
    return items.tolist() if isinstance(items, np.ndarray) else items


def _rare_overlap(a: list, b: list, idf: Dict[str, float]) -> float:
    common = set(a) & set(b)
    return float(sum(idf.get(t, DEFAULT_IDF) for t in common if idf.get(t, DEFAULT_IDF) >= RARE_IDF_THRESHOLD))


def _avg_idf(tokens: list, idf: Dict[str, float]) -> float:
    if len(tokens) == 0:
        return 0.0
    return float(np.mean([idf.get(t, DEFAULT_IDF) for t in tokens]))


def _prefix_agree(a: str, b: str, n: int) -> int:
    return int(bool(a) and bool(b) and a[:n] == b[:n])


def _suffix_agree(a: str, b: str, n: int) -> int:
    return int(bool(a) and bool(b) and a[-n:] == b[-n:])


def compute_pair_features(
    merged: pd.DataFrame,
    name_idf: Dict[str, float],
    addr_idf: Dict[str, float],
    rule_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the full pairwise feature matrix for a merged candidate-pairs frame.

    ``rule_flags``, if given, must be a DataFrame aligned to ``merged``
    (same index / same entity_id_s1+entity_id_other order) with one boolean
    "found_by_<rule>" column per blocking rule; it is concatenated onto the
    output as-is plus a derived "n_blocking_rules" column.
    """
    n = len(merged)
    feats: Dict[str, np.ndarray] = {}

    name_a = merged["name_alnum_s1"].fillna("").tolist()
    name_b = merged["name_alnum_other"].fillna("").tolist()
    name_core_a = merged["name_core_s1"].fillna("").tolist()
    name_core_b = merged["name_core_other"].fillna("").tolist()
    name_sorted_a = merged["name_sorted_s1"].fillna("").tolist()
    name_sorted_b = merged["name_sorted_other"].fillna("").tolist()

    addr_a = merged["address_alnum_s1"].fillna("").tolist()
    addr_b = merged["address_alnum_other"].fillna("").tolist()

    # ---------------- name features ----------------
    feats["name_ratio"] = np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)])
    feats["name_wratio"] = np.array([fuzz.WRatio(a, b) / 100.0 for a, b in zip(name_a, name_b)])
    feats["name_token_sort_ratio"] = np.array([fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)])
    feats["name_token_set_ratio"] = np.array([fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)])
    feats["name_token_ratio"] = np.array([fuzz.token_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)])
    feats["name_partial_ratio"] = np.array([fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)])
    feats["name_core_ratio"] = np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(name_core_a, name_core_b)])
    feats["name_jaro_winkler"] = np.array([JaroWinkler.normalized_similarity(a, b) for a, b in zip(name_a, name_b)])
    feats["name_levenshtein_sim"] = np.array([Levenshtein.normalized_similarity(a, b) for a, b in zip(name_a, name_b)])
    feats["name_exact_norm_match"] = (merged["name_norm_s1"].fillna("") == merged["name_norm_other"].fillna("")).astype(int).to_numpy()
    feats["name_exact_compact_match"] = (merged["name_compact_s1"].fillna("") == merged["name_compact_other"].fillna("")).astype(int).to_numpy()
    feats["name_sorted_exact_match"] = np.array([1 if a == b and a != "" else 0 for a, b in zip(name_sorted_a, name_sorted_b)])

    name_tokens_s1 = merged["name_tokens_s1"]
    name_tokens_other = merged["name_tokens_other"]
    feats["name_token_jaccard"] = np.array([_jaccard(_items_or_empty(a), _items_or_empty(b)) for a, b in zip(name_tokens_s1, name_tokens_other)])
    common_name_tok = np.array([_common_count(_items_or_empty(a), _items_or_empty(b)) for a, b in zip(name_tokens_s1, name_tokens_other)])
    feats["name_common_token_count"] = common_name_tok
    min_len_tok = np.array([max(1, min(len(_items_or_empty(a)), len(_items_or_empty(b)))) for a, b in zip(name_tokens_s1, name_tokens_other)])
    feats["name_common_token_ratio"] = common_name_tok / min_len_tok
    feats["name_rare_token_overlap"] = np.array(
        [_rare_overlap(_items_or_empty(a), _items_or_empty(b), name_idf) for a, b in zip(name_tokens_s1, name_tokens_other)]
    )
    feats["name_avg_idf_s1"] = np.array([_avg_idf(_items_or_empty(a), name_idf) for a in name_tokens_s1])
    feats["name_avg_idf_other"] = np.array([_avg_idf(_items_or_empty(b), name_idf) for b in name_tokens_other])

    len_a = np.array([len(a) for a in name_a])
    len_b = np.array([len(b) for b in name_b])
    feats["name_len_diff"] = np.abs(len_a - len_b)
    feats["name_len_ratio"] = np.minimum(len_a, len_b) / np.maximum(1, np.maximum(len_a, len_b))
    feats["name_prefix_agree"] = np.array([_prefix_agree(a, b, 3) for a, b in zip(name_a, name_b)])
    feats["name_suffix_agree"] = np.array([_suffix_agree(a, b, 3) for a, b in zip(name_a, name_b)])

    name_numbers_s1 = merged["name_numbers_s1"]
    name_numbers_other = merged["name_numbers_other"]
    feats["name_digit_jaccard"] = np.array(
        [_jaccard(_items_or_empty(a), _items_or_empty(b)) for a, b in zip(name_numbers_s1, name_numbers_other)]
    )
    feats["name_digit_exact"] = np.array(
        [1 if sorted(_items_or_empty(a)) == sorted(_items_or_empty(b)) and (len(_items_or_empty(a)) or len(_items_or_empty(b))) else 0 for a, b in zip(name_numbers_s1, name_numbers_other)]
    )

    if "tfidf_score_name" in merged.columns:
        feats["name_tfidf_cosine"] = merged["tfidf_score_name"].fillna(0.0).to_numpy()
    else:
        feats["name_tfidf_cosine"] = np.zeros(n)

    # ---------------- address features ----------------
    feats["addr_ratio"] = np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(addr_a, addr_b)])
    feats["addr_wratio"] = np.array([fuzz.WRatio(a, b) / 100.0 for a, b in zip(addr_a, addr_b)])
    feats["addr_token_sort_ratio"] = np.array([fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(addr_a, addr_b)])
    feats["addr_token_set_ratio"] = np.array([fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(addr_a, addr_b)])
    feats["addr_token_ratio"] = np.array([fuzz.token_ratio(a, b) / 100.0 for a, b in zip(addr_a, addr_b)])
    feats["addr_partial_ratio"] = np.array([fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(addr_a, addr_b)])

    addr_tokens_s1 = merged["address_tokens_s1"]
    addr_tokens_other = merged["address_tokens_other"]
    feats["addr_token_jaccard"] = np.array([_jaccard(_items_or_empty(a), _items_or_empty(b)) for a, b in zip(addr_tokens_s1, addr_tokens_other)])
    common_addr_tok = np.array([_common_count(_items_or_empty(a), _items_or_empty(b)) for a, b in zip(addr_tokens_s1, addr_tokens_other)])
    feats["addr_common_token_count"] = common_addr_tok
    min_len_addr_tok = np.array([max(1, min(len(_items_or_empty(a)), len(_items_or_empty(b)))) for a, b in zip(addr_tokens_s1, addr_tokens_other)])
    feats["addr_common_token_ratio"] = common_addr_tok / min_len_addr_tok
    feats["addr_rare_token_overlap"] = np.array(
        [_rare_overlap(_items_or_empty(a), _items_or_empty(b), addr_idf) for a, b in zip(addr_tokens_s1, addr_tokens_other)]
    )

    addr_numbers_s1 = merged["address_numbers_s1"]
    addr_numbers_other = merged["address_numbers_other"]
    feats["addr_number_jaccard"] = np.array(
        [_jaccard(_items_or_empty(a), _items_or_empty(b)) for a, b in zip(addr_numbers_s1, addr_numbers_other)]
    )
    feats["addr_number_exact_seq"] = np.array(
        [1 if _items_or_empty(a) == _items_or_empty(b) and (len(_items_or_empty(a)) or len(_items_or_empty(b))) else 0 for a, b in zip(addr_numbers_s1, addr_numbers_other)]
    )
    postal_s1 = merged["address_postal_s1"].fillna("")
    postal_other = merged["address_postal_other"].fillna("")
    feats["addr_postal_agree"] = ((postal_s1 == postal_other) & (postal_s1 != "")).astype(int).to_numpy()
    feats["addr_postal_either_missing"] = ((postal_s1 == "") | (postal_other == "")).astype(int).to_numpy()

    alen_a = np.array([len(a) for a in addr_a])
    alen_b = np.array([len(b) for b in addr_b])
    feats["addr_len_diff"] = np.abs(alen_a - alen_b)
    feats["addr_len_ratio"] = np.minimum(alen_a, alen_b) / np.maximum(1, np.maximum(alen_a, alen_b))
    feats["addr_both_present"] = ((alen_a > 0) & (alen_b > 0)).astype(int)

    if "tfidf_score_address" in merged.columns:
        feats["addr_tfidf_cosine"] = merged["tfidf_score_address"].fillna(0.0).to_numpy()
    else:
        feats["addr_tfidf_cosine"] = np.zeros(n)

    # ---------------- cross-field features ----------------
    country_s1 = merged["country_norm_s1"].fillna("")
    country_other = merged["country_norm_other"].fillna("")
    feats["country_exact_match"] = (country_s1 == country_other).astype(int).to_numpy()
    feats["country_missing_s1"] = (country_s1 == "").astype(int).to_numpy()
    feats["country_missing_other"] = (country_other == "").astype(int).to_numpy()

    feats["combined_avg_sim"] = 0.5 * feats["name_ratio"] + 0.5 * feats["addr_ratio"]
    feats["combined_weighted_sim"] = 0.6 * feats["name_ratio"] + 0.4 * feats["addr_ratio"]
    feats["combined_max_sim"] = np.maximum(feats["name_ratio"], feats["addr_ratio"])
    feats["combined_min_sim"] = np.minimum(feats["name_ratio"], feats["addr_ratio"])

    feats["is_source3"] = merged["entity_id_other"].str.startswith("S3-").astype(int).to_numpy()

    out = pd.DataFrame(feats)
    out.insert(0, "entity_id_s1", merged["entity_id_s1"].to_numpy())
    out.insert(1, "entity_id_other", merged["entity_id_other"].to_numpy())

    if rule_flags is not None:
        rule_flags = rule_flags.reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), rule_flags], axis=1)
        rule_cols = [c for c in rule_flags.columns if c.startswith("found_by_")]
        if rule_cols:
            out["n_blocking_rules"] = out[rule_cols].sum(axis=1)

    return out


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of numeric feature columns (excluding id columns) for modelling."""
    exclude = {"entity_id_s1", "entity_id_other"}
    return [c for c in df.columns if c not in exclude]
