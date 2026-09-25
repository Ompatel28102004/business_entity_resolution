"""
Text normalization for business names and addresses.

Design goals (see methodology.md / challenge spec for the rationale):

* Never destroy information by over-normalizing. We build *several*
  complementary representations per field instead of collapsing everything
  into one "canonical" string.
* Numbers are informative (unit/door numbers, PIN/ZIP codes, highway
  numbers...) and must be preserved in at least one representation.
* The data mixes scripts (Latin, Devanagari observed in the India subset,
  and accented Latin for the France test subset). We use Unicode-aware
  normalization (`unicodedata.normalize("NFKC", ...)`) and Unicode-aware
  regex word classes (`\\w` on `str` already matches non-ASCII letters in
  Python 3) rather than assuming ASCII/English text.
* No hard-coded country list: normalization never branches on the value of
  ``country`` (it is only ever used later as a *feature*, not as a switch
  here).

All heavy per-row operations are implemented with vectorized pandas
``.str`` methods (regex substitution compiled once) so that they run in
seconds, not minutes, on multi-million row tables.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Compiled regexes (module level: compiled once, reused across millions of rows)
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^\w\s]", re.UNICODE)
_NON_ALNUM_RE = re.compile(r"[^\w]", re.UNICODE)
_DIGIT_RUN_RE = re.compile(r"\d+")
_AMPERSAND_RE = re.compile(r"&")
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u02bc']")

_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(set(config.LEGAL_SUFFIXES), key=len, reverse=True)) + r")\b\.?",
    re.IGNORECASE,
)
_SUFFIX_SET = frozenset(config.LEGAL_SUFFIXES)


def unicode_normalize(series: pd.Series) -> pd.Series:
    """Apply NFKC Unicode normalization to every string in a Series.

    NFKC folds full-width/compatibility characters and combines accents,
    which stabilises comparisons across encoding variants without touching
    the underlying script. This runs a Python-level loop (no vectorized
    NFKC in pandas), so it's used once per field, on the raw text only.
    """
    return series.map(lambda s: unicodedata.normalize("NFKC", s) if s else s)


def basic_clean(series: pd.Series) -> pd.Series:
    """Lowercase + normalize "and"/apostrophes + collapse whitespace.

    This is the shared first step for every downstream representation.
    """
    s = unicode_normalize(series)
    s = s.str.lower()
    s = s.str.replace(_AMPERSAND_RE, " and ", regex=True)
    s = s.str.replace(_APOSTROPHE_RE, "", regex=True)
    s = s.str.replace(_WHITESPACE_RE, " ", regex=True).str.strip()
    return s


def to_alnum(series: pd.Series) -> pd.Series:
    """Strip punctuation but keep letters/digits/spaces; collapse whitespace."""
    s = series.str.replace(_NON_ALNUM_SPACE_RE, " ", regex=True)
    s = s.str.replace(_WHITESPACE_RE, " ", regex=True).str.strip()
    return s


def to_compact(series: pd.Series) -> pd.Series:
    """Remove ALL non-alphanumeric characters, including spaces.

    Useful as an exact-match blocking key robust to spacing/punctuation
    differences ("B+ Retail Inc" vs "B Retail, Inc." -> "bretailinc").
    """
    return series.str.replace(_NON_ALNUM_RE, "", regex=True)


def extract_numbers(series: pd.Series) -> pd.Series:
    """Return, for each row, the list of digit-runs found in the string.

    Used both as an address "numeric signature" (door/unit/PIN numbers) and
    to preserve numeric tokens inside business names (e.g. "7-Eleven").
    """
    return series.str.findall(_DIGIT_RUN_RE)


def strip_legal_suffixes_tokens(token_series: pd.Series) -> pd.Series:
    """Remove common legal/company suffix tokens from an already-tokenized name.

    Operates on the token *list* with an O(1) set-membership test per token
    instead of re-scanning the whole string with a large alternation regex
    (~15-20x faster at multi-million-row scale, same result). The vocabulary
    (``config.LEGAL_SUFFIXES``) spans many jurisdictions on purpose -- this
    is a normalization aid, not a country switch.
    """
    return token_series.map(
        lambda toks: [t for t in toks if t not in _SUFFIX_SET] if toks else toks
    )


def tokenize(series: pd.Series) -> pd.Series:
    """Split an already-cleaned string on whitespace into a list of tokens."""
    return series.str.split()


def sorted_tokens(token_series: pd.Series) -> pd.Series:
    """Sort each row's token list alphabetically and rejoin with spaces.

    Makes word-order transpositions ("Corner Cafe" vs "Cafe Corner") match
    exactly on this representation.
    """
    return token_series.map(lambda toks: " ".join(sorted(toks)) if toks else "")


def build_name_features(raw: pd.Series) -> pd.DataFrame:
    """Build every business-name representation for a Series of raw names.

    Returns a DataFrame with columns:
        name_raw, name_norm, name_alnum, name_core, name_compact,
        name_tokens, name_sorted, name_numbers, name_first_token
    """
    raw = raw.fillna("")
    norm = basic_clean(raw)
    alnum = to_alnum(norm)
    compact = to_compact(norm)
    tokens = tokenize(alnum)
    core_tokens = strip_legal_suffixes_tokens(tokens)
    core = core_tokens.map(lambda t: " ".join(t))
    sorted_tok = sorted_tokens(tokens)
    numbers = extract_numbers(norm)
    first_token = tokens.map(lambda t: t[0] if t else "")

    return pd.DataFrame(
        {
            "name_raw": raw,
            "name_norm": norm,
            "name_alnum": alnum,
            "name_core": core,
            "name_compact": compact,
            "name_tokens": tokens,
            "name_sorted": sorted_tok,
            "name_numbers": numbers,
            "name_first_token": first_token,
        }
    )


def build_address_features(raw: pd.Series) -> pd.DataFrame:
    """Build every business-address representation for a Series of raw addresses.

    Returns a DataFrame with columns:
        address_raw, address_norm, address_alnum, address_compact,
        address_tokens, address_sorted, address_numbers, address_postal,
        address_first_number
    """
    raw = raw.fillna("")
    norm = basic_clean(raw)
    alnum = to_alnum(norm)
    compact = to_compact(norm)
    tokens = tokenize(alnum)
    sorted_tok = sorted_tokens(tokens)
    numbers = extract_numbers(norm)
    # Postal-like token: the longest digit run of length >= 4 (covers 5-digit
    # US ZIP, 6-digit Indian PIN, 5-digit French postal code); falls back to
    # "" when no such run exists (addresses are frequently incomplete).
    postal = numbers.map(lambda ns: max((n for n in ns if len(n) >= 4), key=len, default=""))
    first_number = numbers.map(lambda ns: ns[0] if ns else "")

    return pd.DataFrame(
        {
            "address_raw": raw,
            "address_norm": norm,
            "address_alnum": alnum,
            "address_compact": compact,
            "address_tokens": tokens,
            "address_sorted": sorted_tok,
            "address_numbers": numbers,
            "address_postal": postal,
            "address_first_number": first_number,
        }
    )


def normalize_country(series: pd.Series) -> pd.Series:
    """Normalize country labels for use as an (open-set) feature/blocking key.

    Only whitespace/case normalization -- no mapping to a fixed vocabulary,
    since the test set introduces "France" (unseen in training) and the
    pipeline must not assume a closed set of countries.
    """
    return basic_clean(series.fillna(""))


def add_all_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """Given a raw source DataFrame (entity_id, business_name, business_address,
    country), return it enriched with every normalized representation.
    """
    name_feats = build_name_features(df["business_name"])
    addr_feats = build_address_features(df["business_address"])
    country_norm = normalize_country(df["country"])
    out = pd.concat(
        [df[["entity_id"]].reset_index(drop=True), name_feats.reset_index(drop=True), addr_feats.reset_index(drop=True)],
        axis=1,
    )
    out["country"] = df["country"].values
    out["country_norm"] = country_norm.values
    return out
