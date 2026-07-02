
import os
import sys
import tempfile
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.parser import parse_all_candidates, normalize
from src.features import (
    compute_all_features,
    score_quantitative_evidence,
    detect_honeypot,
)
from src.scorer import compute_final_scores, select_top_n
from src.reasoning import generate_reasoning
from src.config import (
    MULTIPLICATIVE_FLOORS,
    CAREER_EXPONENT,
    CAREER_DESCRIPTION_KEYWORDS,
)

CANDIDATES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "candidates.jsonl"
)


def _load_small_batch(n=20):
    return parse_all_candidates(CANDIDATES_PATH)[:n]


def _run_pipeline(candidates, tfidf_lookup=None):
    features_list = [
        compute_all_features(c, tfidf_lookup or {}) for c in candidates
    ]
    scored_df = compute_final_scores(features_list)
    top_df = select_top_n(scored_df, n=min(100, len(scored_df)))
    return features_list, scored_df, top_df



def test_full_pipeline_runs():
    candidates = _load_small_batch(20)
    features_list, scored_df, top_df = _run_pipeline(candidates)

    assert len(features_list) == 20, "Should compute features for all candidates"
    assert len(scored_df) == 20, "Should score all candidates"
    assert len(top_df) <= 20, "Top-N should not exceed input size"
    assert "final_score" in scored_df.columns
    assert "rank" in top_df.columns



def test_all_feature_groups_present():
    candidates = _load_small_batch(5)
    features_list, scored_df, _ = _run_pipeline(candidates)

    expected_groups = [
        "group_a_title",
        "group_b_desc",
        "group_c_skills",
        "group_d_gate",
        "group_e_experience",
        "group_f_location",
        "group_g_behavioral",
        "group_h_honeypot",
        "group_i_education",
        "group_j_semantic",
        "group_k_quantitative",
    ]

    for group in expected_groups:
        assert group in scored_df.columns, f"Missing feature group: {group}"

    for feat in features_list:
        for group in expected_groups:
            assert group in feat, f"Missing {group} in feature dict for {feat.get('candidate_id')}"



def test_scores_valid_and_ordered():
    candidates = _load_small_batch(20)
    _, _, top_df = _run_pipeline(candidates)

    scores = top_df["final_score"].tolist()

    for s in scores:
        assert 0.0 <= s <= 1.0, f"Score {s} out of range [0, 1]"

    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1], (
            f"Score at rank {i+1} ({scores[i]}) > score at rank {i} ({scores[i-1]})"
        )



def test_ranks_sequential():
    candidates = _load_small_batch(20)
    _, _, top_df = _run_pipeline(candidates)

    ranks = top_df["rank"].tolist()
    expected = list(range(1, len(ranks) + 1))
    assert ranks == expected, f"Ranks not sequential: got {ranks[:5]}..."



def test_multiplicative_config():
    assert CAREER_EXPONENT > 0, "Career exponent must be positive"
    assert CAREER_EXPONENT >= 1.0, "Career exponent should amplify (>= 1.0)"

    for key, floor in MULTIPLICATIVE_FLOORS.items():
        assert 0.0 < floor < 1.0, f"Floor for {key} ({floor}) should be in (0, 1)"



def test_group_k_quantitative():
    rich = {
        "all_career_descriptions": (
        ),
    }
    vague = {
        "all_career_descriptions": "worked on various projects in the team.",
    }
    empty = {"all_career_descriptions": ""}

    assert score_quantitative_evidence(rich) >= 0.85, "Rich evidence should score high"
    assert score_quantitative_evidence(vague) == 0.10, "Vague descriptions should score low"
    assert score_quantitative_evidence(empty) == 0.10, "Empty descriptions should score 0.10"



def test_honeypot_two_flags_required():
    one_flag = {
        "skills": [{"proficiency": "expert", "duration_months": 0, "name": "python"}],
        "years_of_experience": 5,
        "total_career_months": 48,
        "career_history": [{"duration_months": 48}],
    }
    assert detect_honeypot(one_flag) == 0.0, "1 flag should NOT trigger honeypot"

    two_flags = {
        "skills": [{"proficiency": "expert", "duration_months": 0, "name": "python"}],
        "years_of_experience": 50,
        "total_career_months": 36,
        "career_history": [{"duration_months": 36}],
    }
    assert detect_honeypot(two_flags) == 1.0, "2 flags should trigger honeypot"



def test_normalization_keyword_alignment():
    assert "fine-tuning" in normalize("we fine-tuned the model")
    assert "fine-tuning" in normalize("finetuning transformers")
    assert "fine-tuning" in normalize("fine tuned BERT")

    assert normalize("using an llm") == "using an llm"
    assert "llm" in normalize("training llms on custom data")

    assert "vector database" in normalize("stored in vector db")

    all_keywords = []
    for group in CAREER_DESCRIPTION_KEYWORDS.values():
        all_keywords.extend(group["keywords"])
    assert "fine-tuning" in all_keywords, "fine-tuning should be in config keywords"
    assert "vector database" in all_keywords, "vector database should be in config keywords"
    assert "llm" in all_keywords, "llm should be in config keywords"



def test_reasoning_no_tautology():
    candidates = _load_small_batch(20)
    features_list, _, top_df = _run_pipeline(candidates)
    feat_lookup = {f["candidate_id"]: f for f in features_list}
    cand_lookup = {c["candidate_id"]: c for c in candidates}

    tautological_phrase = "ranked below top 10 due to combined scoring across all dimensions"
    reasoning_texts = []

    for _, row in top_df.iterrows():
        cid = row["candidate_id"]
        r = generate_reasoning(
            cand_lookup.get(cid, {}), feat_lookup.get(cid, {}), int(row["rank"])
        )
        assert tautological_phrase not in r, (
            f"Tautological fallback found at rank {int(row['rank'])}: {r[:100]}"
        )
        reasoning_texts.append(r)

    unique_reasonings = set(reasoning_texts)
    assert len(unique_reasonings) > 1, "All reasoning strings are identical"



def test_csv_output_format():
    candidates = _load_small_batch(10)
    features_list, _, top_df = _run_pipeline(candidates)
    feat_lookup = {f["candidate_id"]: f for f in features_list}
    cand_lookup = {c["candidate_id"]: c for c in candidates}

    reasoning_list = []
    for _, row in top_df.iterrows():
        cid = row["candidate_id"]
        r = generate_reasoning(
            cand_lookup.get(cid, {}), feat_lookup.get(cid, {}), int(row["rank"])
        )
        reasoning_list.append(r)
    top_df = top_df.copy()
    top_df["reasoning"] = reasoning_list

    submission = top_df[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission = submission.rename(columns={"final_score": "score"})

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        submission.to_csv(f, index=False)
        tmp_path = f.name

    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert headers == ["candidate_id", "rank", "score", "reasoning"], (
                f"Wrong CSV headers: {headers}"
            )
            rows = list(reader)
            assert len(rows) == len(top_df), "Row count mismatch"
            for row in rows:
                assert row["candidate_id"].startswith("CAND_"), f"Bad candidate_id: {row['candidate_id']}"
                assert int(row["rank"]) >= 1, f"Bad rank: {row['rank']}"
                assert float(row["score"]) >= 0, f"Bad score: {row['score']}"
                assert len(row["reasoning"]) > 10, f"Reasoning too short: {row['reasoning']}"
    finally:
        os.unlink(tmp_path)



if __name__ == "__main__":
    tests = [
        test_full_pipeline_runs,
        test_all_feature_groups_present,
        test_scores_valid_and_ordered,
        test_ranks_sequential,
        test_multiplicative_config,
        test_group_k_quantitative,
        test_honeypot_two_flags_required,
        test_normalization_keyword_alignment,
        test_reasoning_no_tautology,
        test_csv_output_format,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__} — {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*50}")
    sys.exit(1 if failed > 0 else 0)
