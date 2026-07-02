
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scorer import compute_final_scores, select_top_100


def _make_features(cid, a=0.5, b=0.5, c=0.5, d=1.0, e=0.85, f=0.85, g=0.7, h=0.0, i=0.5):
    return {
        "candidate_id": cid,
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


class TestComputeFinalScores:

    def test_honeypot_gets_zero(self):
        features = [_make_features("CAND_0000001", h=1.0)]
        df = compute_final_scores(features)
        assert df.iloc[0]["final_score"] == 0.0

    def test_hard_gated_near_zero(self):
        features = [_make_features("CAND_0000001", d=0.0)]
        df = compute_final_scores(features)
        assert df.iloc[0]["final_score"] == 0.0

    def test_perfect_candidate_high_score(self):
        features = [_make_features("CAND_0000001", a=1.0, b=1.0, c=1.0,
                                   d=1.0, e=1.0, f=1.0, g=1.0, h=0.0, i=1.0)]
        df = compute_final_scores(features)
        score = df.iloc[0]["final_score"]
        assert score > 0.7, f"Perfect candidate should score > 0.7, got {score}"

    def test_soft_gate_reduces_score(self):
        perfect = [_make_features("CAND_0000001", a=1.0, b=1.0, c=1.0,
                                  d=1.0, e=1.0, f=1.0, g=1.0, h=0.0, i=1.0)]
        gated = [_make_features("CAND_0000002", a=1.0, b=1.0, c=1.0,
                                d=0.40, e=1.0, f=1.0, g=1.0, h=0.0, i=1.0)]
        df_perfect = compute_final_scores(perfect)
        df_gated = compute_final_scores(gated)
        assert df_gated.iloc[0]["final_score"] > 0.0
        assert df_gated.iloc[0]["final_score"] < df_perfect.iloc[0]["final_score"]

    def test_sorting_order(self):
        features = [
            _make_features("CAND_0000003", b=0.1),
            _make_features("CAND_0000001", b=0.9),
            _make_features("CAND_0000002", b=0.5),
        ]
        df = compute_final_scores(features)
        scores = df["final_score"].tolist()
        assert scores[0] >= scores[1] >= scores[2], "Scores should be sorted descending"

    def test_tiebreak_by_candidate_id(self):
        features = [
            _make_features("CAND_0000002"),
            _make_features("CAND_0000001"),
        ]
        df = compute_final_scores(features)
        ids = df["candidate_id"].tolist()
        assert ids[0] == "CAND_0000001"


class TestSelectTop100:

    def test_ranks_sequential(self):
        features = [_make_features(f"CAND_{i:07d}", b=1.0 - i*0.005) for i in range(150)]
        df = compute_final_scores(features)
        top = select_top_100(df)
        assert len(top) == 100
        assert list(top["rank"]) == list(range(1, 101))

    def test_scores_non_increasing(self):
        features = [_make_features(f"CAND_{i:07d}", b=1.0 - i*0.005) for i in range(150)]
        df = compute_final_scores(features)
        top = select_top_100(df)
        scores = top["final_score"].tolist()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Scores not non-increasing at rank {i+1}: {scores[i]} < {scores[i+1]}"

    def test_scores_rounded(self):
        features = [_make_features(f"CAND_{i:07d}") for i in range(150)]
        df = compute_final_scores(features)
        top = select_top_100(df)
        for score in top["final_score"]:
            decimal_str = f"{score:.10f}"
            parts = decimal_str.split(".")
            assert len(parts[1].rstrip("0")) <= 6, f"Score {score} has > 6 decimal places"


class TestScoreSpread:

    def test_meaningful_spread(self):
        import random
        random.seed(42)
        features = []
        for i in range(200):
            features.append(_make_features(
                f"CAND_{i:07d}",
                a=random.random(),
                b=random.random(),
                c=random.random(),
                d=random.choice([1.0, 1.0, 1.0, 0.4, 0.1]),
                e=random.choice([0.3, 0.6, 0.85, 1.0]),
                f=random.choice([0.15, 0.40, 0.60, 0.85, 1.0]),
                g=0.5 + random.random() * 0.5,
                h=random.choice([0.0] * 19 + [1.0]),  
                i=0.3 + random.random() * 0.7,
            ))
        df = compute_final_scores(features)
        top = select_top_100(df)
        scores = top["final_score"].tolist()
        non_zero = [s for s in scores if s > 0]
        if len(non_zero) > 5:
            import statistics
            std = statistics.stdev(non_zero)
            assert std > 0.03, f"Score spread too narrow: std={std:.4f}"


def run_tests():
    test_classes = [TestComputeFinalScores, TestSelectTop100, TestScoreSpread]
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

    print(f"\nScorer tests: {passed} passed, {failed} failed")
    for err in errors:
        print(err)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
