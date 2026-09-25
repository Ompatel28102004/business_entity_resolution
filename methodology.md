# Methodology — Business Entity Resolution

**Status note (read first):** Sections 1-5 and the data statistics below
report numbers actually measured on the real challenge data (via
`cache/audit_scratch.py`, cached at `experiments/data_audit_report.json`
and reproduced in `notebooks/01_data_audit.ipynb`). Sections describing the
model comparison, ablation study, and final threshold (7, 9-13) describe the
**method** precisely — the exact code that runs it is complete and
correctness-tested on synthetic data (`tests/smoke_test*.py`, including the
official validator) — but the **numbers** must come from running
`src/train.py` / `src/inference.py` on the AWS SageMaker instance described
in `aws/README_AWS.md`, per the project decision not to run full-scale
(multi-million-row) computation on the ~12GB-RAM development laptop. Those
sections are marked `[PENDING AWS RUN]` and should be updated with the
contents of `experiments/results.csv` after that run, before final
submission.

## 1. Problem understanding

Three independent sources describe overlapping sets of real-world
businesses with no shared identifier. Source-1 is a deduplicated reference
set; Source-2 and Source-3 are noisy, possibly-duplicated records of the
same (and other) businesses. For every Source-1 entity we must return the
set of Source-2/Source-3 ids that refer to the same business — zero, one,
or many. Evaluation is the macro average, across every Source-1 entity, of
a precision-weighted F-beta (beta=0.5), so **false merges are punished twice
as hard, relative to precision, as missed matches**, and correctly
predicting "no match" for a true singleton is worth full credit.

## 2. Data statistics (measured)

| table | rows | unique entity_id | unique name | unique address | % empty name | % empty address |
|---|---:|---:|---:|---:|---:|---:|
| train_source1 | 2,206,821 | 2,206,821 | 1,539,229 | 2,130,606 | 0.0% | 0.0% |
| train_source2 | 5,034,616 | 5,034,616 | 4,402,009 | 4,337,262 | 0.0% | 3.36% |
| train_source3 | 5,285,603 | 5,285,603 | 4,651,609 | 4,632,765 | 0.0% | 3.33% |
| test_source1 | 1,732,544 | 1,732,544 | 1,238,867 | 1,677,483 | 0.0% | 0.0% |
| test_source2 | 4,887,273 | 4,887,273 | 4,311,041 | 4,224,784 | 0.0% | 2.65% |
| test_source3 | 5,082,316 | 5,082,316 | 4,521,929 | 4,456,436 | 0.0% | 2.68% |

Country distribution: train is `{US: 1,323,633 / 60.0%, India: 883,188 /
40.0%}` (Source-1); test is `{India: 809,986 / 46.8%, US: 663,106 / 38.3%,
France: 259,452 / 15.0%}` (Source-1) — **France has zero training
examples**, which is why nothing in `normalization.py` / `blocking.py`
branches on a fixed country list.

Ground truth (train, 2,206,821 Source-1 entities, 7,638,365 true pairs):

* **5.58% are true singletons** (123,247 entities, zero matches).
* 5.40% have exactly one match (119,157).
* 89.0% have multiple matches (1,964,417), averaging ~3.7 matches each
  among non-singletons (match counts up to 11, with a long right tail down
  to a handful of entities).
* Of the 7,638,365 true pairs, 3,693,619 point into Source-2 and 3,944,746
  into Source-3 — roughly balanced.
* **Every Source-2/Source-3 id is claimed by at most one Source-1 parent**
  (checked exhaustively: 0 violations, max fan-in = 1). This motivated the
  optional "one parent per match" consistency post-processing rule in
  `thresholding.resolve_one_parent_per_match` (kept only if it measurably
  improves validation F0.5 — see Section 11).
* Name lengths average ~24-25 characters across all tables; address lengths
  average ~46-57. ~70% of Source-1 business names are unique — the other
  30% recur across genuinely different businesses (common names), which is
  why name-only blocking keys are combined with address/country signals
  rather than used alone.

Full breakdown, including per-source name/address length percentiles and
the complete match-count histogram, is in
`experiments/data_audit_report.json` / `notebooks/01_data_audit.ipynb`.

## 3. Data normalization

Implemented in `src/normalization.py`. Per the challenge's explicit
guidance, we build *several* representations instead of collapsing each
field into one canonical string, so no downstream stage is starved of
information another stage needs:

* `*_norm`: NFKC Unicode-normalized, lowercased, `&`->`and`, apostrophes
  stripped, whitespace collapsed. Base for every other representation.
* `*_alnum`: punctuation removed, digits and Unicode letters kept.
* `name_core`: `name_alnum` with legal/company suffix *tokens* removed
  (Inc, Ltd, Pvt, LLC, GmbH, SARL, ... — a broad, multi-jurisdiction list,
  removed by O(1) set membership on the token list rather than a slow
  string-level regex).
* `*_compact`: every non-alphanumeric character removed, including spaces
  — an exact-match key robust to punctuation/spacing differences.
* `*_tokens` / `*_sorted`: whitespace-split tokens, and the same tokens
  re-joined after sorting (catches word-order transpositions).
* `*_numbers` (`name_numbers`, `address_numbers`): every digit run
  extracted and preserved as its own list — numbers are informative
  (door/unit numbers, highway numbers, "7-Eleven") and are never discarded.
* `address_postal`: the longest digit run of length >= 4 (covers 5-digit
  US ZIP, 6-digit Indian PIN, 5-digit French postal codes) when present;
  many addresses in this dataset simply omit a postal code (consistent with
  the audit's "missing address components" noise pattern), so this feature
  is treated as optional evidence, never a required key.

No external database, geocoding, or business lookup is used anywhere (see
Section 17 / fair-play compliance).

## 4. Candidate generation / blocking

Implemented in `src/blocking.py` (hash-join rules) and `src/retrieval.py`
(TF-IDF retrieval), unioned in `src/inference.py::candidates_for_chunk_and_source`.

**Hash-join rules** (each an O(n+m) pandas merge, with over-large blocks
capped via `config.MAX_BLOCK_SIZE` to avoid a common/empty key exploding
into a near-Cartesian block):

| rule | key |
|---|---|
| A. exact_name | `name_norm` |
| B. exact_address | `address_norm` |
| C. name_first_token_country | `name_first_token` + `country_norm` |
| D. name_first_two_tokens | first two tokens of `name_alnum` |
| E. rare_name_token | any shared name token with document frequency <= 50 (inverted index join) |
| F. address_number_country | `address_first_number` + `country_norm` |
| G. name_compact_prefix | first 6 characters of `name_compact` |

**TF-IDF retrieval** (rules H/I): character-within-word-boundary
(`analyzer="char_wb"`) n-grams, n-gram range `(3, 5)`, top-`K`
(`config.TFIDF_TOP_K = 20`) cosine-similarity neighbours per Source-1 entity,
computed separately for `name_alnum` and `address_alnum`, separately against
Source-2 and Source-3. The vectorizer is fit on a bounded random sample
(300k documents) of the current split's text for scalability, then used to
`transform` every row — vocabulary selection converges well before seeing
every document, and this keeps `fit` bounded regardless of total corpus
size. Retrieval itself processes Source-1 in chunks and takes the top-K of
a sparse `chunk @ other.T` product without ever densifying anything, so
peak memory does not depend on the total corpus size either.

Every candidate pair is tagged with **which rule(s)** retrieved it
(`found_by_<rule>` boolean columns + `n_blocking_rules` count), which
becomes both a diagnostic (Section 4 of the challenge spec: measure each
rule's recall contribution) and a model feature (more independent pieces of
evidence agreeing on a pair is itself informative).

`[PENDING AWS RUN]` Per-rule and union candidate recall / average-candidates-
per-entity / runtime, measured on a fixed-seed sample of TRAIN Source-1
entities: run `notebooks/02_candidate_generation.ipynb` (or
`python -m src.train`, which reports `candidate_recall` in
`experiments/results.csv`) and record the resulting table here.

## 5. Feature engineering

Implemented in `src/features.py` (~60 columns per candidate pair):

* **Name**: RapidFuzz `ratio`, `WRatio`, `token_sort_ratio`,
  `token_set_ratio`, `token_ratio`, `partial_ratio` (on `name_alnum`), a
  second `ratio` on `name_core` (suffix-stripped), Jaro-Winkler and
  normalized-Levenshtein similarity, exact-match indicators on `name_norm`/
  `name_compact`/`name_sorted`, token Jaccard / common-token count & ratio,
  IDF-weighted rare-token overlap, average token IDF per side, length
  diff/ratio, prefix/suffix agreement, digit-set Jaccard and exact-sequence
  match, and the TF-IDF cosine score carried over from retrieval.
* **Address**: the same fuzzy-ratio family, token Jaccard/common-token
  stats, rare-token overlap, address-number Jaccard and exact-sequence
  match, postal-token agreement (+ a "both missing" flag, since ~3% of
  Source-2/3 addresses are empty), length diff/ratio, a both-present flag,
  and the address TF-IDF cosine score.
* **Cross-field**: country exact-match + missingness indicators (never a
  hard filter), combined average/weighted-average/max/min of name and
  address similarity, and a Source-2-vs-Source-3 indicator.
* **Blocking provenance**: one `found_by_<rule>` boolean per rule plus
  `n_blocking_rules`.
* **Rarity**: token IDF tables (`features.build_idf_table`) estimated from
  the TRAIN reference tables (Source-1+2+3) only — never from an external
  database — with a fixed "very rare" default IDF assigned to any token
  never seen in training (relevant for the unseen-country France subset).

## 6. Model architecture

Implemented in `src/model.py`; four families compared under the same
feature set and threshold-search procedure:

* **Model A — rule-based baseline**: fixed linear combination
  (`0.5*name_token_sort_ratio + 0.3*addr_token_sort_ratio + 0.2*exact-match
  bonus`), no training, kept as an honest zero-parameter reference point.
* **Model B — Logistic Regression** (scaled features, `class_weight="balanced"`).
* **Model C — Random Forest / Extra Trees** (`n_estimators=300`,
  `max_depth=16`, `class_weight="balanced_subsample"`).
* **Model D — HistGradientBoostingClassifier** (scikit-learn's
  histogram-based gradient boosting; `max_depth=8`, `max_iter=300`,
  `learning_rate=0.08`) — the strongest CPU-native option tried, BSD-3
  licensed (ships inside scikit-learn), comfortably satisfying the
  challenge's MIT/Apache-2.0-compatible, <=8B-parameter constraint (this is
  a few-hundred-tree ensemble, not a billion-parameter neural network).

All four share the `predict_proba_pair(X) -> np.ndarray` interface so
`thresholding.py`/`inference.py` are model-agnostic.

## 7. Training strategy

`src/train.py::run_training`: Source-1 TRAIN entities are split by
**entity id** (never by pair, to avoid leaking a Source-1 entity's other
true pairs across the split) into train/validation
(`config.VALIDATION_FRACTION = 0.25`, fixed seed `42`). Candidates are
generated for each split with the *exact same code path* `inference.py`
uses at test time. Models are fit on the TRAIN split's labelled candidate
pairs and evaluated on the VALIDATION split with the official entity-level
F0.5 metric, at a threshold searched per Section 9.

For full-scale runs, `--sample-size 0` uses every TRAIN Source-1 entity;
`config.EXPERIMENT_SAMPLE_SIZE` (40,000) is the default for a first,
tractable pass (timing/memory sanity check) before committing to the full
run on AWS.

## 8. Hard-negative generation

No separate random-negative sampling step exists, by design: every
candidate pair that is *not* a true match but survived at least one
blocking rule (exact key collision or top-K TF-IDF retrieval) is, by
construction, a hard negative — textually or structurally similar to the
query yet wrong. Random pairs sampled from the full cross product would
overwhelmingly be trivially-different businesses that add little training
signal; the blocking-derived negatives concentrate the classifier's
attention on the actually-confusable cases (common business names, shared
address numbers, near-duplicate spellings of *different* underlying
businesses).

## 9. Validation methodology

`metrics.evaluate_entity_level_f05` implements the exact formula from the
problem statement (`F0.5 = 1.25*P*R / (0.25*P + R)`, macro-averaged over
Source-1 entities, singletons scored as 1.0 when correctly predicted empty
and 0.0 on any false merge). It is unit-tested against the worked example
in the challenge README (`python -m src.metrics`; verified locally:
precision=0.667, recall=1.000, F0.5=0.714, matching the spec exactly) and
against several hand-constructed edge cases (both-empty, pred-empty/true-
non-empty, pred-non-empty/true-empty, perfect match).

A second, independently-seeded sample (`config.SECOND_SPLIT_SAMPLE_SIZE`)
is used to confirm the chosen model/threshold is not an artifact of one
lucky split, per the challenge's robustness guidance — `[PENDING AWS RUN]`.

## 10. Entity-level F0.5 optimization

`src/thresholding.py` never uses a fixed 0.5 cut. `search_best_threshold`
grid-searches a coarse range (0.30-0.95, step 0.05), `refine_threshold_search`
then does a fine search (step 0.01) around the coarse optimum, both scored
by the exact entity-level F0.5 metric (plus pair precision/recall and
average predicted matches per entity, for diagnosis). `search_source_specific_thresholds`
optionally tunes separate Source-1->Source-2 / Source-1->Source-3 thresholds,
falling back to the global threshold whenever a subgroup has fewer than
`min_support=500` validation candidate pairs, to avoid overfitting a tiny
subgroup. `apply_decision_policy` additionally supports an optional
top1-vs-runner-up score-gap ambiguity guard. Every one of these
refinements is applied only if it measurably improves validation F0.5 —
`[PENDING AWS RUN]` for the actual comparison numbers.

## 11. Singleton handling

Singletons are handled implicitly by the same thresholding mechanism (an
entity with no candidate scoring above threshold is predicted empty) rather
than a separate binary "has a match at all" classifier, plus two optional
strengthening mechanisms, both gated on measured validation improvement:

* The score-gap ambiguity guard (`min_score_gap` in
  `apply_decision_policy`): when the top candidate isn't clearly ahead of
  the runner-up, the entity's single best-scoring candidate is treated as
  insufficiently confident and dropped.
* The one-parent-per-match consistency rule
  (`thresholding.resolve_one_parent_per_match`), motivated by the measured
  fact that no true Source-2/3 id is ever shared between two Source-1
  parents in the training data (Section 2): when the same candidate id is
  accepted by more than one Source-1 query, only the highest-scoring
  assignment is kept.

## 12. Error analysis

`notebooks/05_validation_and_thresholds.ipynb` sorts validation entities by
per-entity F0.5, prints the raw name/address/country for the worst cases
next to their true and predicted matches, and asks for each to be
categorized (typo, abbreviation/DBA, address component reordering, missing
address component, common-name collision, transliteration, numeric
conflict, country-specific pattern) with a proposed targeted fix, re-run
against notebook 03/04 to confirm the fix actually helps before keeping it.
`[PENDING AWS RUN]` for the concrete categorized examples and outcomes.

## 13. Ablation experiments

`experiments/results.csv` (written by `src/train.py`) is the ablation
table: one row per model (rule-based / logistic regression / random forest
/ gradient boosting), each with its own threshold-searched
`macro_f05`/`macro_precision`/`macro_recall`/`singleton_accuracy`/
`pair_precision`/`pair_recall`/candidate-recall/pair-count columns. The
notebooks additionally isolate the incremental effect of each blocking rule
(notebook 02) and each feature group (notebook 03, mean-value-by-label
comparison) before the full model comparison. `[PENDING AWS RUN]` for
populated results.

## 14. Final model selection

Per the challenge's explicit instruction, the final model is whichever of
Model A/B/C/D achieves the highest **validation entity-level F0.5** in
`experiments/results.csv` — not assumed in advance. `src/train.py`
automatically persists that winning model (plus its IDF tables, fitted
vectorizers, and searched threshold) to `models/`, and `src/inference.py`
loads exactly those frozen artifacts for the test run, so there is a single
source of truth for "what got submitted" and it is always the empirically
best-scoring configuration measured, not a hard-coded default. `[PENDING
AWS RUN]` for the final winning model/threshold and its validation numbers.

## 15. Computational considerations

* Every heavy table read goes through a one-time streaming TSV->parquet
  conversion (`pyarrow.csv.open_csv` + `ParquetWriter` in bounded-size
  batches — see `data_loader.convert_tsv_to_parquet`), so subsequent loads
  are columnar and fast rather than re-parsing multi-hundred-MB TSVs.
* Candidate generation is O(n+m) hash joins plus bounded-chunk sparse
  TF-IDF products — never an explicit Cartesian product of Source-1 x
  Source-2/3.
* `src/inference.py` processes Source-1 in chunks (`config.S1_CHUNK_SIZE`)
  while Source-2/3 are vectorized once and read-only thereafter, and writes
  output incrementally (`output.append_output_chunk`) so peak memory is
  bounded by chunk size, independent of total dataset size (1.7M+ test
  Source-1 entities).
* `utils.parallel_map_df` provides process-level parallelism for the
  string-heavy, GIL-bound normalization step, to actually use a many-vCPU
  SageMaker instance's cores (`--n-jobs-normalize`).
* See `aws/README_AWS.md` for the recommended instance size
  (`ml.m5.4xlarge`, 16 vCPU / 64GB RAM) and rationale.

## 16. Final inference pipeline

`src/inference.py::run_pipeline` (CLI: `python -m src.inference --split
test`): loads the frozen model/IDF/vectorizer/threshold artifacts from
`models/`, processes every test Source-1 entity in chunks, generates
candidates against the full test Source-2/3 pool with the identical
blocking+retrieval code used during training/validation, scores them,
applies the frozen decision policy, and appends validated rows to
`output/matching_results.tsv` and `output/candidate_pairs.tsv` — the latter
being, by construction, the exact candidate set fed to the model (never an
earlier, superseded blocking pass), and a strict superset of every accepted
match (`output.check_matches_subset_of_candidates` / the official
validator's cross-check). `src/validation.py` then runs the official
`student_resource/utils/validate_submission.py` against the produced files.

## 17. Limitations

* **Cross-script matching** (e.g. a Devanagari-script name vs. its Latin
  transliteration) is not specifically modelled — no transliteration model
  or external lookup is used (fair-play rules prohibit external data), so
  recall on such pairs relies only on shared numeric tokens / address
  overlap, and is expected to be weaker than same-script fuzzy matches.
* **France (test-only country)** has no training examples; country is only
  ever used as an open-set feature/blocking key, but no France-specific
  normalization patterns could be learned or validated ahead of time — this
  is an inherent generalization risk the pipeline is designed to minimize
  (never hard-coded to US/India) but cannot eliminate.
* **TF-IDF vectorizer fitting uses a bounded random sample** (300k docs) of
  each split's text rather than the full multi-million-document corpus, for
  fit-time scalability; this is a standard practical approximation but
  means extremely rare n-grams that only ever appear beyond the sample
  might be under-weighted.
* Numbers in Sections 7, 9-14 are marked `[PENDING AWS RUN]` and must be
  filled in from `experiments/results.csv` after running `src/train.py` /
  `src/inference.py` at full scale, per `aws/README_AWS.md`, before this
  document is considered final for submission.
