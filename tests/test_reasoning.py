
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reasoning import generate_reasoning
from difflib import SequenceMatcher


def _make_candidate(cid="CAND_0000001", title="ML Engineer", company="AI Corp",
                    years=7, **overrides):
    base = {
        "candidate_id": cid,
        "current_title": title,
        "current_company": company,
        "years_of_experience": years,
        "country": "India",
        "location": "Pune",
        "career_history": [
            {
                "company": company, "title": title,
                "duration_months": int(years * 12), "is_current": True,
                "description": "Built recommendation systems using FAISS and embeddings. "
                               "Evaluated using NDCG and MRR metrics.",
                "industry": "Technology", "company_size": "51-200",
                "start_date": "2020-01-01", "end_date": None,
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
            {"name": "Machine Learning", "proficiency": "advanced", "endorsements": 8, "duration_months": 30},
        ],
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.7,
        "notice_period_days": 30,
        "github_activity_score": 45,
        "willing_to_relocate": True,
    }
    base.update(overrides)
    return base


def _make_features(a=0.8, b=0.6, c=0.5, d=1.0, e=1.0, f=1.0, g=0.8, h=0.0, i=0.7):
    return {
        "group_a_title": a,
        "group_b_desc": b,
        "group_c_skills": c,
        "group_d_gate": d,
        "group_e_experience": e,
        "group_f_location": f,
        "group_g_behavioral": g,
        "group_h_honeypot": h,
        "group_i_education": i,
    }


class TestReasoningTiers:

    def test_rank1_is_enthusiastic(self):
        cand = _make_candidate()
        features = _make_features()
        reasoning = generate_reasoning(cand, features, 1)
        positive_words = ["strong", "top-tier", "excellent", "highly", "outstanding",
                          "compelling", "standout"]
        has_positive = any(w in reasoning.lower() for w in positive_words)
        assert has_positive, f"Rank 1 should have enthusiastic tone: {reasoning[:100]}"

    def test_rank95_acknowledges_weakness(self):
        cand = _make_candidate(
            cid="CAND_0000095", title="Operations Manager",
            company="Wipro", years=12,
            career_history=[{
                "company": "Wipro", "title": "Operations Manager",
                "duration_months": 144, "is_current": True,
                "description": "Managed business operations and team coordination",
                "industry": "IT Services", "company_size": "10001+",
                "start_date": "2014-01-01", "end_date": None,
            }],
            skills=[],
            open_to_work_flag=False,
        )
        features = _make_features(a=0, b=0.05, c=0, d=0.4, e=0.7, f=0.6, g=0.4)
        reasoning = generate_reasoning(cand, features, 95)
        weakness_words = ["weak", "limited", "borderline", "marginal",
                          "thin", "concern", "lacking", "no"]
        has_weakness = any(w in reasoning.lower() for w in weakness_words)
        assert has_weakness, f"Rank 95 should acknowledge weaknesses: {reasoning[:200]}"

    def test_rank50_is_honest(self):
        cand = _make_candidate(cid="CAND_0000050", title="Data Analyst", company="TCS", years=5)
        features = _make_features(a=0.1, b=0.15, c=0.2, d=0.6, e=1.0)
        reasoning = generate_reasoning(cand, features, 50)
        honest_words = ["partial", "some", "mixed", "adjacent", "moderate",
                        "qualified", "gaps", "limited"]
        has_honest = any(w in reasoning.lower() for w in honest_words)
        assert has_honest, f"Rank 50 should be honest about gaps: {reasoning[:200]}"


class TestReasoningQuality:

    def test_reasoning_length(self):
        cand = _make_candidate()
        features = _make_features()
        reasoning = generate_reasoning(cand, features, 1)
        assert len(reasoning) <= 500, f"Reasoning too long: {len(reasoning)} chars"
        assert len(reasoning) > 50, f"Reasoning too short: {len(reasoning)} chars"

    def test_no_mid_word_truncation(self):
        cand = _make_candidate()
        features = _make_features()
        reasoning = generate_reasoning(cand, features, 1)
        if not reasoning.endswith("..."):
            last_char = reasoning[-1]
            assert last_char in ".!?", f"Reasoning doesn't end cleanly: ...{reasoning[-20:]}"

    def test_reasoning_diversity(self):
        cands_and_ranks = [
            (_make_candidate(cid="CAND_0000001", title="ML Engineer", company="Google", years=7), 1),
            (_make_candidate(cid="CAND_0000015", title="Data Scientist", company="Flipkart", years=5,
                career_history=[{
                    "company": "Flipkart", "title": "Data Scientist",
                    "duration_months": 60, "is_current": True,
                    "description": "Built recommendation engine using collaborative filtering and embeddings.",
                    "industry": "E-commerce", "company_size": "1001-5000",
                    "start_date": "2021-01-01", "end_date": None,
                }],
                skills=[
                    {"name": "Recommendation Systems", "proficiency": "advanced", "endorsements": 12, "duration_months": 48},
                    {"name": "Python", "proficiency": "advanced", "endorsements": 20, "duration_months": 60},
                ]), 15),
            (_make_candidate(cid="CAND_0000050", title="Backend Engineer", company="TCS", years=10,
                career_history=[{
                    "company": "TCS", "title": "Backend Engineer",
                    "duration_months": 120, "is_current": True,
                    "description": "Developed Java microservices for enterprise applications.",
                    "industry": "IT Services", "company_size": "10001+",
                    "start_date": "2016-01-01", "end_date": None,
                }],
                skills=[
                    {"name": "Java", "proficiency": "expert", "endorsements": 30, "duration_months": 120},
                ]), 50),
            (_make_candidate(cid="CAND_0000080", title="Operations Manager", company="Wipro", years=12,
                career_history=[{
                    "company": "Wipro", "title": "Operations Manager",
                    "duration_months": 144, "is_current": True,
                    "description": "Managed IT operations and team coordination.",
                    "industry": "IT Services", "company_size": "10001+",
                    "start_date": "2014-01-01", "end_date": None,
                }],
                skills=[]), 80),
        ]
        reasonings = []
        for cand, rank in cands_and_ranks:
            features = _make_features(
                a=max(0, 1.0 - rank * 0.01),
                b=max(0, 0.8 - rank * 0.01),
                c=max(0, 0.6 - rank * 0.005),
            )
            reasonings.append(generate_reasoning(cand, features, rank))
        for i in range(len(reasonings)):
            for j in range(i + 1, len(reasonings)):
                sim = SequenceMatcher(None, reasonings[i], reasonings[j]).ratio()
                assert sim < 0.85, (
                    f"Reasoning too similar ({sim:.2f}) between rank {i+1} and {j+1}\n"
                    f"  R{i+1}: {reasonings[i][:80]}\n"
                    f"  R{j+1}: {reasonings[j][:80]}"
                )

    def test_concerns_even_for_rank1(self):
        cand = _make_candidate(notice_period_days=120)
        features = _make_features()
        reasoning = generate_reasoning(cand, features, 1)
        assert "notice" in reasoning.lower() or "120" in reasoning, \
            f"Rank 1 with 120-day notice should mention it: {reasoning}"


def run_tests():
    test_classes = [TestReasoningTiers, TestReasoningQuality]
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

    print(f"\nReasoning tests: {passed} passed, {failed} failed")
    for err in errors:
        print(err)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
