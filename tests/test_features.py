
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    score_career_titles,
    score_career_descriptions,
    score_skills,
    gate_anti_negative,
    score_experience_band,
    score_location,
    score_behavioral,
    detect_honeypot,
    score_education_assessment,
    compute_all_features,
)


def _make_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000001",
        "headline": "",
        "summary": "",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 7,
        "current_title": "ML Engineer",
        "current_company": "Startup Inc",
        "current_company_size": "51-200",
        "current_industry": "Technology",
        "career_history": [
            {
                "company": "Startup Inc",
                "title": "ML Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 84,
                "is_current": True,
                "industry": "Technology",
                "company_size": "51-200",
                "description": "Built recommendation systems using embeddings and FAISS. "
                               "Deployed ranking models to production with real-time inference.",
            }
        ],
        "all_career_descriptions": "built recommendation systems using embeddings and faiss. "
                                    "deployed ranking models to production with real-time inference.",
        "all_career_titles": "ml engineer",
        "total_career_months": 84,
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
            {"name": "Machine Learning", "proficiency": "advanced", "endorsements": 8, "duration_months": 24},
        ],
        "skill_names_lower": {"python", "machine learning"},
        "education": [
            {"institution": "IIT Delhi", "degree": "B.Tech", "field_of_study": "Computer Science", "tier": "tier_1"}
        ],
        "best_education_tier": 4,  
        "has_relevant_degree": True,
        "profile_completeness_score": 85,
        "signup_date": "2024-01-01",
        "last_active_date": "2026-06-10",
        "open_to_work_flag": True,
        "profile_views_received_30d": 10,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.7,
        "avg_response_time_hours": 12,
        "skill_assessment_scores": {"Python": 85, "Machine Learning": 78},
        "connection_count": 500,
        "endorsements_received": 30,
        "notice_period_days": 30,
        "expected_salary_range": {"min": 20, "max": 35},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 45,
        "search_appearance_30d": 15,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.75,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return base


class TestGroupA:

    def test_ml_engineer_high_score(self):
        cand = _make_candidate()
        score = score_career_titles(cand)
        assert score > 0.5, f"ML Engineer should score high, got {score}"

    def test_hr_manager_zero(self):
        cand = _make_candidate(
            current_title="HR Manager",
            career_history=[{
                "company": "Corp", "title": "HR Manager",
                "duration_months": 60, "is_current": True,
                "description": "Managed human resources",
                "industry": "HR", "company_size": "", "start_date": "", "end_date": None,
            }],
            all_career_titles="hr manager",
        )
        score = score_career_titles(cand)
        assert score < 0.2, f"HR Manager should score low, got {score}"

    def test_html_developer_not_matched_as_ml(self):
        cand = _make_candidate(
            current_title="HTML Developer",
            career_history=[{
                "company": "Web Co", "title": "HTML Developer",
                "duration_months": 24, "is_current": True,
                "description": "Frontend HTML/CSS work",
                "industry": "Tech", "company_size": "", "start_date": "", "end_date": None,
            }],
            all_career_titles="html developer",
        )
        score = score_career_titles(cand)
        assert score < 0.2, f"HTML Developer should NOT match ML keywords, got {score}"

    def test_no_career(self):
        cand = _make_candidate(career_history=[])
        assert score_career_titles(cand) == 0.0


class TestGroupB:

    def test_rich_description_high(self):
        cand = _make_candidate()
        score = score_career_descriptions(cand)
        assert score > 0.2, f"Rich description should score well, got {score}"

    def test_empty_description_zero(self):
        cand = _make_candidate(
            all_career_descriptions="",
            summary="",
            headline="",
        )
        score = score_career_descriptions(cand)
        assert score == 0.0

    def test_roadmap_not_false_positive_on_map(self):
        cand = _make_candidate(
            all_career_descriptions="created a product roadmap for the team. managed stakeholders.",
            summary="", headline="",
        )
        score = score_career_descriptions(cand)
        assert score < 0.3, f"'roadmap' should not trigger MAP keyword, got {score}"


class TestGroupD:

    def test_ml_engineer_passes(self):
        cand = _make_candidate()
        assert gate_anti_negative(cand) == 1.0

    def test_hr_manager_gated(self):
        cand = _make_candidate(
            current_title="HR Manager",
            all_career_descriptions="managed human resources and recruiting",
            career_history=[{
                "company": "Corp", "title": "HR Manager",
                "duration_months": 60, "is_current": True,
                "description": "HR duties",
                "industry": "HR", "company_size": "", "start_date": "", "end_date": None,
            }],
        )
        gate = gate_anti_negative(cand)
        assert gate == 0.0, f"HR Manager with no ML evidence should be gated, got {gate}"

    def test_full_tcs_career_heavy_penalty(self):
        cand = _make_candidate(
            current_title="Software Developer",
            career_history=[{
                "company": "Tata Consultancy Services",
                "title": "Software Developer",
                "duration_months": 60, "is_current": True,
                "description": "Developed enterprise Java applications",
                "industry": "IT Services", "company_size": "10001+",
                "start_date": "", "end_date": None,
            }],
            all_career_descriptions="developed enterprise java applications",
        )
        gate = gate_anti_negative(cand)
        assert gate <= 0.10, f"Full TCS career should be heavily penalized, got {gate}"

    def test_cv_engineer_no_nlp_penalized(self):
        cand = _make_candidate(
            current_title="Computer Vision Engineer",
            career_history=[{
                "company": "Vision Corp", "title": "Computer Vision Engineer",
                "duration_months": 48, "is_current": True,
                "description": "Developed object detection and image segmentation pipelines",
                "industry": "Tech", "company_size": "", "start_date": "", "end_date": None,
            }],
            all_career_descriptions="developed object detection and image segmentation pipelines",
        )
        gate = gate_anti_negative(cand)
        assert gate <= 0.20, f"CV Engineer without NLP/IR should be penalized, got {gate}"

    def test_cv_engineer_with_nlp_mild_penalty(self):
        cand = _make_candidate(
            current_title="Computer Vision Engineer",
            career_history=[{
                "company": "AI Corp", "title": "Computer Vision Engineer",
                "duration_months": 48, "is_current": True,
                "description": "Developed vision models and NLP text retrieval systems for search",
                "industry": "Tech", "company_size": "", "start_date": "", "end_date": None,
            }],
            all_career_descriptions="developed vision models and nlp text retrieval systems for search",
        )
        gate = gate_anti_negative(cand)
        assert 0.5 < gate <= 0.80, f"CV Engineer WITH NLP should get mild penalty, got {gate}"

    def test_junior_title_penalty(self):
        cand = _make_candidate(current_title="Junior ML Engineer")
        gate = gate_anti_negative(cand)
        assert gate == 0.60, f"Junior title should get 0.60 penalty, got {gate}"


class TestGroupE:

    def test_sweet_spot(self):
        assert score_experience_band(_make_candidate(years_of_experience=7)) == 1.0

    def test_too_junior(self):
        assert score_experience_band(_make_candidate(years_of_experience=2)) == 0.30

    def test_too_senior_manager(self):
        cand = _make_candidate(years_of_experience=20, current_title="Engineering Director")
        assert score_experience_band(cand) == 0.35


class TestGroupF:

    def test_india_preferred_city(self):
        cand = _make_candidate(country="India", location="Pune, Maharashtra")
        assert score_location(cand) == 1.0

    def test_indiana_not_matched(self):
        cand = _make_candidate(country="Indiana", location="Indianapolis")
        score = score_location(cand)
        assert score < 0.5, f"'Indiana' should NOT match as India, got {score}"


class TestGroupH:

    def test_expert_zero_months(self):
        cand = _make_candidate(
            skills=[{"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 0}],
            skill_names_lower={"python"},
        )
        assert detect_honeypot(cand) == 1.0

    def test_legitimate_candidate(self):
        cand = _make_candidate()
        assert detect_honeypot(cand) == 0.0


class TestGroupI:

    def test_tier1_cs_degree_bonus(self):
        cand = _make_candidate(
            best_education_tier=4,  
            has_relevant_degree=True,
            skill_assessment_scores={"Python": 90, "Machine Learning": 85},
        )
        score = score_education_assessment(cand)
        assert score > 0.7, f"Tier 1 CS degree + strong assessments should score high, got {score}"

    def test_no_education_neutral(self):
        cand = _make_candidate(
            best_education_tier=0,
            has_relevant_degree=False,
            skill_assessment_scores={},
        )
        score = score_education_assessment(cand)
        assert 0.4 <= score <= 0.6, f"No education should be neutral ~0.5, got {score}"


class TestComputeAll:

    def test_all_groups_present(self):
        cand = _make_candidate()
        features = compute_all_features(cand)
        expected_keys = [
            "candidate_id", "group_a_title", "group_b_desc", "group_c_skills",
            "group_d_gate", "group_e_experience", "group_f_location",
            "group_g_behavioral", "group_h_honeypot", "group_i_education",
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_all_scores_in_range(self):
        cand = _make_candidate()
        features = compute_all_features(cand)
        for key, val in features.items():
            if key == "candidate_id":
                continue
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range [0, 1]"


def run_tests():
    test_classes = [
        TestGroupA, TestGroupB, TestGroupD, TestGroupE,
        TestGroupF, TestGroupH, TestGroupI, TestComputeAll,
    ]
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for method_name in sorted(dir(instance)):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except AssertionError as e:
                    failed += 1
                    errors.append(f"  FAIL: {cls.__name__}.{method_name}: {e}")
                except Exception as e:
                    failed += 1
                    errors.append(f"  ERROR: {cls.__name__}.{method_name}: {e}")

    print(f"\nFeature tests: {passed} passed, {failed} failed")
    for err in errors:
        print(err)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
