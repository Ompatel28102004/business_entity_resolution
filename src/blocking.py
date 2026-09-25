"""
Candidate generation / blocking.

Generates the UNION of several complementary, independently-computed
blocking rules between a Source-1 table and one Source-2/Source-3 table.
Every rule is implemented as a hash join (pandas ``merge`` on a shared key
column) which is O(n + m) per rule -- this is what makes it possible to run
candidate generation on multi-million-row tables without ever forming an
explicit Cartesian product.

Rules implemented (see methodology.md for the recall/precision trade-off
measured for each one):

  A. exact normalized name            (key_name_norm)
  B. exact normalized address         (key_addr_norm)
  C. name token overlap (first token) (key_name_first_token [+ country])
  D. first-two-token block            (key_name_first2)
  E. rare-token block (any shared alnum name token, IDF-weighted at feature
     time; implemented here as a token->id inverted index join)
  F. address-number signature         (key_addr_number [+ country])
  G. name prefix / compact block      (key_name_compact prefix)
  H/I. character n-gram TF-IDF retrieval for name/address -- see retrieval.py
     (unioned in by the caller, not this module, since it needs a fitted
     vectorizer index).

Every rule tags which pairs it produced, so downstream feature engineering
can use "found_by_<rule>" and "number_of_blocking_rules" as features, and
so each rule's incremental contribution to recall can be measured directly
(Phase 10 requirement).
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import config
from .utils import timer


def _capped_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, max_block_size: int) -> pd.DataFrame:
    """Inner-join ``left``/``right`` on ``on``, dropping over-large blocks.

    Extremely common keys (e.g. an empty string, or a very short generic
    first token) can otherwise create O(block^2) candidate explosions while
    contributing almost nothing to recall (a huge indiscriminate block is
    not informative). We drop any key value whose left-count * right-count
    would exceed ``max_block_size**2`` -- keeping the join itself O(n+m) to
    *compute* (we only need group sizes, not the full join, to decide what
    to drop).
    """
    if left.empty or right.empty:
        return left.iloc[0:0].merge(right.iloc[0:0], on=on)

    left_counts = left[on].value_counts()
    right_counts = right[on].value_counts()
    shared = left_counts.index.intersection(right_counts.index)
    if len(shared) == 0:
        return left.iloc[0:0].merge(right.iloc[0:0], on=on)

    # keys with a manageable fan-out on both sides (and non-empty key)
    ok_keys = shared[
        (left_counts.loc[shared].values <= max_block_size)
        & (right_counts.loc[shared].values <= max_block_size)
    ]
    ok_keys = ok_keys[ok_keys != ""]
    if len(ok_keys) == 0:
        return left.iloc[0:0].merge(right.iloc[0:0], on=on)

    l = left[left[on].isin(ok_keys)]
    r = right[right[on].isin(ok_keys)]
    return l.merge(r, on=on, suffixes=("_s1", "_other"))


def block_exact_name(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int) -> pd.DataFrame:
    """Rule A: exact match on the normalized (punctuation-free) business name."""
    l = s1[["entity_id", "name_norm"]].rename(columns={"name_norm": "key"})
    r = other[["entity_id", "name_norm"]].rename(columns={"name_norm": "key"})
    out = _capped_merge(l, r, "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].assign(rule="exact_name")


def block_exact_address(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int) -> pd.DataFrame:
    """Rule B: exact match on the normalized business address."""
    l = s1[["entity_id", "address_norm"]].rename(columns={"address_norm": "key"})
    r = other[["entity_id", "address_norm"]].rename(columns={"address_norm": "key"})
    out = _capped_merge(l, r, "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].assign(rule="exact_address")


def block_name_first_token_country(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int) -> pd.DataFrame:
    """Rule C: first significant name token + country (country is an open-set feature here, not a filter)."""
    l = s1[["entity_id", "name_first_token", "country_norm"]].copy()
    l["key"] = l["name_first_token"] + "|" + l["country_norm"]
    r = other[["entity_id", "name_first_token", "country_norm"]].copy()
    r["key"] = r["name_first_token"] + "|" + r["country_norm"]
    out = _capped_merge(l[["entity_id", "key"]], r[["entity_id", "key"]], "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].assign(rule="name_first_token_country")


def block_name_first_two_tokens(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int) -> pd.DataFrame:
    """Rule D: first two name tokens (helps when a common single first token is ambiguous)."""
    def first_two(tokens_series: pd.Series) -> pd.Series:
        return tokens_series.map(lambda t: " ".join(t[:2]) if len(t) >= 2 else "")

    l = s1[["entity_id", "name_tokens"]].copy()
    l["key"] = first_two(l["name_tokens"])
    r = other[["entity_id", "name_tokens"]].copy()
    r["key"] = first_two(r["name_tokens"])
    out = _capped_merge(l[["entity_id", "key"]], r[["entity_id", "key"]], "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].assign(rule="name_first_two_tokens")


def block_address_number_country(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int) -> pd.DataFrame:
    """Rule F: first address number + country. Strong signal: door/plot numbers rarely collide by chance."""
    l = s1[["entity_id", "address_first_number", "country_norm"]].copy()
    l["key"] = l["address_first_number"] + "|" + l["country_norm"]
    r = other[["entity_id", "address_first_number", "country_norm"]].copy()
    r["key"] = r["address_first_number"] + "|" + r["country_norm"]
    out = _capped_merge(l[["entity_id", "key"]], r[["entity_id", "key"]], "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].assign(rule="address_number_country")


def block_name_compact_prefix(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int, prefix_len: int = 6) -> pd.DataFrame:
    """Rule G: shared prefix of the fully-compacted (punctuation/space-free) name."""
    l = s1[["entity_id", "name_compact"]].copy()
    l["key"] = l["name_compact"].str.slice(0, prefix_len)
    r = other[["entity_id", "name_compact"]].copy()
    r["key"] = r["name_compact"].str.slice(0, prefix_len)
    out = _capped_merge(l[["entity_id", "key"]], r[["entity_id", "key"]], "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].assign(rule="name_compact_prefix")


def block_rare_name_token(
    s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int, other_token_doc_freq: pd.Series | None = None,
    rarity_threshold: int = 50,
) -> pd.DataFrame:
    """Rule E: any shared *rare* alnum name token (inverted-index join).

    Common tokens ("the", "inc", "market"...) are excluded via a document
    frequency threshold computed on the ``other`` side, so this rule targets
    exactly the informative, discriminative tokens (e.g. a distinctive
    surname or brand word) that make two noisy name variants recognisable
    as the same business, without blowing up on generic words.
    """
    l = s1[["entity_id", "name_tokens"]].explode("name_tokens").rename(columns={"name_tokens": "key"})
    l = l[l["key"].notna() & (l["key"].str.len() >= 3)]
    r = other[["entity_id", "name_tokens"]].explode("name_tokens").rename(columns={"name_tokens": "key"})
    r = r[r["key"].notna() & (r["key"].str.len() >= 3)]

    if other_token_doc_freq is None:
        other_token_doc_freq = r["key"].value_counts()
    rare_keys = other_token_doc_freq[other_token_doc_freq <= rarity_threshold].index
    r = r[r["key"].isin(rare_keys)]
    l = l[l["key"].isin(rare_keys)]

    out = _capped_merge(l, r, "key", max_block_size)
    return out[["entity_id_s1", "entity_id_other"]].drop_duplicates().assign(rule="rare_name_token")


ALL_HASH_RULES = [
    block_exact_name,
    block_exact_address,
    block_name_first_token_country,
    block_name_first_two_tokens,
    block_address_number_country,
    block_name_compact_prefix,
]


def generate_hash_candidates(
    s1: pd.DataFrame,
    other: pd.DataFrame,
    max_block_size: int = config.MAX_BLOCK_SIZE,
    include_rare_token_rule: bool = True,
) -> pd.DataFrame:
    """Union every hash-based blocking rule between ``s1`` and ``other``.

    Returns a long DataFrame with columns (entity_id_s1, entity_id_other,
    rule). The same pair can appear multiple times (once per rule that
    retrieved it) -- callers that just need the candidate *set* should
    ``drop_duplicates`` on the id pair; callers that want "how many rules
    agreed" should keep the duplicates and aggregate.
    """
    frames: List[pd.DataFrame] = []
    for rule_fn in ALL_HASH_RULES:
        with timer(f"  blocking rule: {rule_fn.__name__}"):
            frames.append(rule_fn(s1, other, max_block_size))
    if include_rare_token_rule:
        with timer("  blocking rule: block_rare_name_token"):
            frames.append(block_rare_name_token(s1, other, max_block_size))
    result = pd.concat(frames, axis=0, ignore_index=True)
    return result


def summarize_candidates(cand: pd.DataFrame) -> Dict[str, object]:
    """Summary stats for a candidate-pairs long-frame (see Phase 10 metrics)."""
    dedup = cand.drop_duplicates(subset=["entity_id_s1", "entity_id_other"])
    per_s1 = dedup.groupby("entity_id_s1").size()
    rule_counts = cand.groupby("rule")["entity_id_s1"].count().to_dict()
    return {
        "n_unique_pairs": int(len(dedup)),
        "n_s1_with_candidates": int(per_s1.shape[0]),
        "avg_candidates_per_s1": float(per_s1.mean()) if len(per_s1) else 0.0,
        "median_candidates_per_s1": float(per_s1.median()) if len(per_s1) else 0.0,
        "max_candidates_per_s1": int(per_s1.max()) if len(per_s1) else 0,
        "pairs_per_rule": rule_counts,
    }
