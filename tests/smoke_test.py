"""Cheap, laptop-safe smoke test of the full pipeline on a tiny synthetic dataset.

This does NOT touch the real 22M-row challenge data; it only exists to catch
integration bugs (wrong column names, shape mismatches, etc.) before the
full-scale run happens on AWS. Run: python cache/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import blocking, features, inference, metrics, model as model_module, output, retrieval, thresholding
from src.normalization import add_all_normalizations

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Build a tiny synthetic dataset with KNOWN matches + noise patterns.
# ---------------------------------------------------------------------------
base_businesses = [
    ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US"),
    ("Prime Money", "17560 Ellis Road, Tahlequah, OK", "US"),
    ("B+ Retail Inc", "1712 Montebello Avenue, Phoenix, AZ", "US"),
    ("Prabhav Business Center", "797, Lake Town Block A, Kolkata, West Bengal", "India"),
    ("Custom Wealth Services LLC", "5559 Orville Avenue, Columbus, OH", "US"),
    ("Smart Healthcare Private Limited", "303, 3rd Floor Sakar 5, Ashram Road, Ahmedabad, Gujarat", "India"),
    ("Le Petit Cafe", "12 Rue de Paris, Lyon", "France"),
    ("Global Traders Co", "44 Marine Drive, Mumbai, Maharashtra", "India"),
    ("Sunrise Bakery", "220 Main Street, Austin, TX", "US"),
    ("Kelly Advisory Inc", "301 1st Street, Chokio, MN", "US"),
]

def noisy_variant(name, addr, country, seed):
    r = np.random.default_rng(seed)
    n = name
    if r.random() < 0.5:
        n = n.replace("Inc", "Incorporated").replace("LLC", "Limited Liability Co").replace("Private Limited", "Pvt Ltd")
    if r.random() < 0.3:
        n = n.upper()
    if r.random() < 0.3 and len(n) > 5:
        i = r.integers(0, len(n) - 1)
        n = n[:i] + n[i + 1] + n[i] + n[i + 2:]  # swap two chars (typo)
    a = addr
    if r.random() < 0.3:
        a = a.replace("Street", "St").replace("Drive", "Dr").replace("Avenue", "Ave").replace("Road", "Rd")
    if r.random() < 0.3:
        parts = [p.strip() for p in a.split(",")]
        r.shuffle(parts)
        a = ", ".join(parts)
    return n, a, country


s1_rows, s2_rows, s3_rows, gt = [], [], [], {}
for i, (name, addr, country) in enumerate(base_businesses):
    s1_id = f"S1-{i:04d}"
    s1_rows.append({"entity_id": s1_id, "business_name": name, "business_address": addr, "country": country})
    matches = set()
    # 1-2 matches in S2, 1 in S3 for most; a couple of singletons
    if i % 5 != 4:
        for j in range(rng.integers(1, 3)):
            n, a, c = noisy_variant(name, addr, country, seed=1000 * i + j)
            s2_id = f"S2-{i:04d}{j}"
            s2_rows.append({"entity_id": s2_id, "business_name": n, "business_address": a, "country": c})
            matches.add(s2_id)
        n, a, c = noisy_variant(name, addr, country, seed=2000 + i)
        s3_id = f"S3-{i:04d}"
        s3_rows.append({"entity_id": s3_id, "business_name": n, "business_address": a, "country": c})
        matches.add(s3_id)
    gt[s1_id] = matches

# Add unrelated "noise" rows to S2/S3 (hard negatives: different businesses, same country)
for k in range(60):
    name, addr, country = base_businesses[rng.integers(0, len(base_businesses))]
    n, a, c = noisy_variant(name + " Other Branch", addr, country, seed=9000 + k)
    (s2_rows if k % 2 == 0 else s3_rows).append(
        {"entity_id": f"{'S2' if k % 2 == 0 else 'S3'}-N{k:04d}", "business_name": n, "business_address": a, "country": c}
    )

s1_df = pd.DataFrame(s1_rows)
s2_df = pd.DataFrame(s2_rows)
s3_df = pd.DataFrame(s3_rows)
print(f"synthetic sizes: s1={len(s1_df)} s2={len(s2_df)} s3={len(s3_df)}")

s1_norm = add_all_normalizations(s1_df)
s2_norm = add_all_normalizations(s2_df)
s3_norm = add_all_normalizations(s3_df)

# ---------------------------------------------------------------------------
# 2. Candidate generation (hash blocking + TF-IDF) using the real inference helpers.
# ---------------------------------------------------------------------------
name_vec = retrieval.fit_field_vectorizer([s1_norm["name_alnum"], s2_norm["name_alnum"], s3_norm["name_alnum"]], fit_sample_size=None)
addr_vec = retrieval.fit_field_vectorizer([s1_norm["address_alnum"], s2_norm["address_alnum"], s3_norm["address_alnum"]], fit_sample_size=None)

chunk_corpus = pd.Series(["alpha query", "alpha distant", "query alpha", "unrelated", "alpha query exact"])
chunk_vectorizer = retrieval.fit_vectorizer(chunk_corpus, ngram_range=(2, 3))
chunk_s1_matrix = chunk_vectorizer.transform(pd.Series(["alpha query"]))
chunk_other_matrix = chunk_vectorizer.transform(chunk_corpus)
chunk_expected = retrieval.top_k_candidates(
    chunk_s1_matrix, chunk_other_matrix, np.array(["S1-check"]),
    np.array([f"S2-check-{i}" for i in range(len(chunk_corpus))]), k=2,
)
chunk_actual = retrieval.top_k_candidates_chunked(
    chunk_s1_matrix, chunk_corpus, chunk_vectorizer, np.array(["S1-check"]),
    np.array([f"S2-check-{i}" for i in range(len(chunk_corpus))]),
    k=2, other_chunk_size=2, s1_chunk_size=1,
)
assert set(chunk_expected["entity_id_other"]) == set(chunk_actual["entity_id_other"])
assert np.allclose(
    chunk_expected.sort_values("entity_id_other")["tfidf_score"].to_numpy(),
    chunk_actual.sort_values("entity_id_other")["tfidf_score"].to_numpy(),
)

s2_side = inference.VectorizedOtherSide(s2_norm, source_name="Source-2")
s3_side = inference.VectorizedOtherSide(s3_norm, source_name="Source-3")

cand_s2 = inference.candidates_for_chunk_and_source(
    s1_norm, s2_side, name_vec, addr_vec, max_block_size=400, tfidf_k=10, tfidf_other_chunk_size=11,
)
cand_s3 = inference.candidates_for_chunk_and_source(
    s1_norm, s3_side, name_vec, addr_vec, max_block_size=400, tfidf_k=10, tfidf_other_chunk_size=11,
)
print(f"candidates: s2={len(cand_s2)} s3={len(cand_s3)}")

# recall check
cand_pairs_all = pd.concat([cand_s2[["entity_id_s1", "entity_id_other"]], cand_s3[["entity_id_s1", "entity_id_other"]]])
cand_set = set(zip(cand_pairs_all["entity_id_s1"], cand_pairs_all["entity_id_other"]))
total_true = sum(len(v) for v in gt.values())
found_true = sum(1 for s1, matches in gt.items() for m in matches if (s1, m) in cand_set)
print(f"candidate recall: {found_true}/{total_true} = {found_true/total_true:.3f}")
assert found_true / total_true > 0.8, "candidate recall too low on synthetic smoke test"

# ---------------------------------------------------------------------------
# 3. Features + labels + a quick model.
# ---------------------------------------------------------------------------
name_idf = features.build_idf_table(pd.concat([s1_norm["name_tokens"], s2_norm["name_tokens"], s3_norm["name_tokens"]]))
addr_idf = features.build_idf_table(pd.concat([s1_norm["address_tokens"], s2_norm["address_tokens"], s3_norm["address_tokens"]]))

rule_cols = [c for c in cand_s2.columns if c.startswith("found_by_")] + ["n_blocking_rules"]
merged_s2 = features.merge_pair_fields(cand_s2[["entity_id_s1", "entity_id_other"]], s1_norm, s2_norm)
merged_s2 = merged_s2.assign(tfidf_score_name=cand_s2["tfidf_score_name"].to_numpy(), tfidf_score_address=cand_s2["tfidf_score_address"].to_numpy())
feat_s2 = features.compute_pair_features(merged_s2, name_idf, addr_idf, rule_flags=cand_s2[rule_cols])
array_merged_s2 = merged_s2.copy()
for column in (
    "name_tokens_s1", "name_tokens_other", "name_numbers_s1", "name_numbers_other",
    "address_tokens_s1", "address_tokens_other", "address_numbers_s1", "address_numbers_other",
):
    array_merged_s2[column] = array_merged_s2[column].map(np.asarray)
array_features_s2 = features.compute_pair_features(
    array_merged_s2, name_idf, addr_idf, rule_flags=cand_s2[rule_cols],
)
pd.testing.assert_frame_equal(feat_s2, array_features_s2)

merged_s3 = features.merge_pair_fields(cand_s3[["entity_id_s1", "entity_id_other"]], s1_norm, s3_norm)
merged_s3 = merged_s3.assign(tfidf_score_name=cand_s3["tfidf_score_name"].to_numpy(), tfidf_score_address=cand_s3["tfidf_score_address"].to_numpy())
feat_s3 = features.compute_pair_features(merged_s3, name_idf, addr_idf, rule_flags=cand_s3[rule_cols])

feat_all = pd.concat([feat_s2, feat_s3], ignore_index=True)
print("feature matrix shape:", feat_all.shape)

labels = feat_all.apply(lambda r: int(r.entity_id_other in gt.get(r.entity_id_s1, set())), axis=1)
print("label balance:", labels.value_counts().to_dict())

feat_cols = features.feature_columns(feat_all)
wrapper = model_module.train_model("logistic_regression", feat_all, labels, feat_cols)
scores = wrapper.predict_proba_pair(feat_all)
scored = feat_all[["entity_id_s1", "entity_id_other"]].copy()
scored["score"] = scores

# ---------------------------------------------------------------------------
# 4. Threshold search + entity-level F0.5 + output writing.
# ---------------------------------------------------------------------------
all_s1_ids = s1_df["entity_id"].tolist()
search = thresholding.search_best_threshold(scored, gt, all_s1_ids)
print("best threshold:", search.best_threshold)
print(search.table.to_string(index=False))

preds = thresholding.predictions_at_threshold(scored, search.best_threshold)
result = metrics.evaluate_entity_level_f05(preds, gt, all_s1_ids)
print(f"macro_f05={result['macro_f05']:.3f} precision={result['macro_precision']:.3f} recall={result['macro_recall']:.3f} singleton_acc={result['singleton_accuracy']:.3f}")

cand_map = cand_pairs_all.drop_duplicates().groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()
out_dir = Path(__file__).resolve().parent / "_smoke_output"
match_df = output.write_matching_results(preds, all_s1_ids, out_dir / "matching_results.tsv")
cand_df = output.write_candidate_pairs(cand_map, all_s1_ids, out_dir / "candidate_pairs.tsv")
offenders = output.check_matches_subset_of_candidates(preds, cand_map)
print("matches-not-in-candidates offenders:", offenders)
assert not offenders, "final matches must be a subset of candidates"

print("SMOKE TEST PASSED")
