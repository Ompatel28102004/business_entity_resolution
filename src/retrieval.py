"""
Character n-gram TF-IDF retrieval (rules H/I from the blocking spec).

Complements the exact/hash blocking rules in ``blocking.py`` by retrieving
the top-K most textually similar Source-2/3 records for each Source-1
record, using character n-gram TF-IDF + cosine similarity. This is what
catches misspellings, transliteration noise, and other variations that
don't share an exact token/prefix with the true match.

Memory/CPU note: computing a full S1 x Other cosine-similarity matrix is
infeasible at this dataset's scale (millions x millions). We therefore
process Source-1 rows in bounded-size chunks: for each chunk we compute a
sparse ``chunk @ other.T`` product (cheap because TF-IDF rows are sparse)
and extract the top-K nonzero entries per row without ever densifying
anything. Peak memory is bounded by chunk_size * nnz-per-row, independent
of the total corpus size -- this is the piece of the pipeline most worth
running on a bigger SageMaker instance (more RAM => bigger chunks => fewer
Python-level iterations).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .utils import log, timer


def fit_vectorizer(
    corpus: Iterable[str],
    ngram_range: Tuple[int, int] = (3, 5),
    analyzer: str = "char_wb",
    max_features: int | None = 200_000,
) -> TfidfVectorizer:
    """Fit a character n-gram TF-IDF vectorizer on a text corpus.

    ``analyzer="char_wb"`` (character n-grams within word boundaries) is
    used by default: it is robust to typos/abbreviations while still
    respecting token boundaries, which empirically retrieves cleaner
    candidates than plain ``"char"`` for short business-name/address
    strings (see notebooks/03 ablation of char vs char_wb / n-gram ranges).
    """
    vec = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range, max_features=max_features, lowercase=False,
    )
    vec.fit(corpus)
    return vec


def top_k_candidates(
    s1_matrix: sp.csr_matrix,
    other_matrix: sp.csr_matrix,
    s1_ids: np.ndarray,
    other_ids: np.ndarray,
    k: int = config.TFIDF_TOP_K,
    chunk_size: int = 2000,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Retrieve the top-K cosine-similarity neighbours in ``other`` for every row of ``s1``.

    Both matrices must be L2-row-normalized TF-IDF matrices in CSR format
    (``TfidfVectorizer`` output already is L2-normalized by default), so
    cosine similarity reduces to a plain sparse dot product.

    Returns a long DataFrame: entity_id_s1, entity_id_other, tfidf_score.
    """
    other_t = other_matrix.T.tocsr()
    rows_s1, rows_other, scores = [], [], []

    n = s1_matrix.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = s1_matrix[start:end] @ other_t  # sparse (chunk, n_other)
        sim_chunk = sim_chunk.tocsr()
        for local_i in range(sim_chunk.shape[0]):
            row_start, row_end = sim_chunk.indptr[local_i], sim_chunk.indptr[local_i + 1]
            if row_end == row_start:
                continue
            cols = sim_chunk.indices[row_start:row_end]
            vals = sim_chunk.data[row_start:row_end]
            if vals.shape[0] > k:
                top_pos = np.argpartition(vals, -k)[-k:]
            else:
                top_pos = np.arange(vals.shape[0])
            top_cols = cols[top_pos]
            top_vals = vals[top_pos]
            keep = top_vals >= min_score
            top_cols = top_cols[keep]
            top_vals = top_vals[keep]
            if top_cols.shape[0] == 0:
                continue
            global_i = start + local_i
            rows_s1.append(np.full(top_cols.shape[0], global_i))
            rows_other.append(top_cols)
            scores.append(top_vals)

    if not rows_s1:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_other", "tfidf_score"])

    idx_s1 = np.concatenate(rows_s1)
    idx_other = np.concatenate(rows_other)
    all_scores = np.concatenate(scores)

    return pd.DataFrame(
        {
            "entity_id_s1": s1_ids[idx_s1],
            "entity_id_other": other_ids[idx_other],
            "tfidf_score": all_scores,
        }
    )


def fit_field_vectorizer(
    text_series_list: list[pd.Series],
    ngram_range: Tuple[int, int] = (3, 5),
    fit_sample_size: int | None = 300_000,
) -> TfidfVectorizer:
    """Fit one shared TF-IDF vectorizer across several text Series (e.g. S1+S2+S3 names).

    Sampling (see ``generate_tfidf_candidates`` docstring) keeps this bounded
    regardless of total corpus size. Fit once per field per split and reuse
    across every Source-1 chunk / Source-2 / Source-3 pass for consistent,
    cheap ``transform`` calls.
    """
    corpus = pd.concat(text_series_list, axis=0)
    if fit_sample_size is not None and len(corpus) > fit_sample_size:
        corpus = corpus.sample(n=fit_sample_size, random_state=config.RANDOM_SEED)
    return fit_vectorizer(corpus, ngram_range=ngram_range)


def generate_tfidf_candidates(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str,
    ngram_range: Tuple[int, int] = (3, 5),
    k: int = config.TFIDF_TOP_K,
    chunk_size: int = 2000,
    fit_sample_size: int | None = 300_000,
    vectorizer: TfidfVectorizer | None = None,
) -> pd.DataFrame:
    """End-to-end TF-IDF candidate generation for one text field (name or address).

    Fits the vectorizer on a bounded-size random sample of the UNION of
    s1_df[text_col] and other_df[text_col] (``fit_sample_size``, default
    300k documents) rather than the full multi-million-row corpus: vocabulary
    selection converges well before seeing every document, and this keeps
    ``fit`` time/memory bounded regardless of total corpus size -- the
    difference between "fits on a laptop" and "fits on any machine". The
    fitted vectorizer is then used to ``transform`` every row (no sampling
    at transform time). Pass a pre-fitted ``vectorizer`` to reuse across
    calls (e.g. the same fitted name vectorizer for both the Source-2 and
    Source-3 candidate passes).

    Fitting only ever sees text from the current split (train text at
    validation time, test text at final-inference time) -- no leakage
    either direction since this is an unsupervised vectorizer, not a
    label-aware model.
    """
    if vectorizer is None:
        corpus = pd.concat([s1_df[text_col], other_df[text_col]], axis=0)
        if fit_sample_size is not None and len(corpus) > fit_sample_size:
            corpus = corpus.sample(n=fit_sample_size, random_state=config.RANDOM_SEED)
        vec = fit_vectorizer(corpus, ngram_range=ngram_range)
    else:
        vec = vectorizer
    s1_matrix = vec.transform(s1_df[text_col])
    other_matrix = vec.transform(other_df[text_col])
    cands = top_k_candidates(
        s1_matrix,
        other_matrix,
        s1_df["entity_id"].to_numpy(),
        other_df["entity_id"].to_numpy(),
        k=k,
        chunk_size=chunk_size,
    )
    cands["rule"] = f"tfidf_{text_col}"
    return cands
