
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching import match_keyword, count_keyword_hits, any_keyword_match


class TestMatchKeyword:

    def test_ai_matches_ai_engineer(self):
        assert match_keyword("ai", "AI Engineer at Google")

    def test_ai_does_not_match_thailand(self):
        assert not match_keyword("ai", "Traveled to Thailand")

    def test_ai_does_not_match_said(self):
        assert not match_keyword("ai", "He said something")

    def test_ml_matches_ml_pipeline(self):
        assert match_keyword("ml", "Built an ML Pipeline")

    def test_ml_does_not_match_html(self):
        assert not match_keyword("ml", "HTML and CSS developer")

    def test_bert_matches_fine_tuned_bert(self):
        assert match_keyword("bert", "Fine-tuned BERT for NER")

    def test_bert_does_not_match_albert(self):
        assert not match_keyword("bert", "Albert Einstein was a physicist")

    def test_map_does_not_match_roadmap(self):
        assert not match_keyword("map", "Created a product roadmap")

    def test_recall_matches_precision_recall(self):
        assert match_keyword("recall", "precision/recall curves")

    def test_ltr_matches_ltr_standalone(self):
        assert match_keyword("ltr", "implemented LTR models")

    def test_t5_matches_t5_model(self):
        assert match_keyword("t5", "fine-tuned T5 for summarization")

    def test_python_matches(self):
        assert match_keyword("python", "Expert in Python development")

    def test_empty_keyword(self):
        assert not match_keyword("", "some text")

    def test_empty_text(self):
        assert not match_keyword("ai", "")

    def test_both_empty(self):
        assert not match_keyword("", "")

    def test_case_insensitive(self):
        assert match_keyword("python", "PYTHON developer")
        assert match_keyword("PYTHON", "python developer")

    def test_special_chars_a_b_test(self):
        assert match_keyword("a/b test", "Ran A/B test experiments")

    def test_ci_cd_matches(self):
        assert match_keyword("ci/cd", "Set up CI/CD pipelines")

    def test_hyphenated_keyword(self):
        assert match_keyword("fine-tuning", "Experience with fine-tuning LLMs")

    def test_nlp_does_not_match_unrelated(self):
        assert match_keyword("nlp", "NLP engineer")

    def test_ir_boundary(self):
        assert not match_keyword("ir", "He was the first to arrive")


class TestCountKeywordHits:

    def test_multiple_hits(self):
        text = "Built FAISS index for vector search with embeddings"
        keywords = ["faiss", "vector search", "embeddings", "pinecone"]
        assert count_keyword_hits(keywords, text) == 3

    def test_no_hits(self):
        text = "Managed a team of designers"
        keywords = ["python", "ml", "ai", "embeddings"]
        assert count_keyword_hits(keywords, text) == 0

    def test_empty_text(self):
        assert count_keyword_hits(["python", "ml"], "") == 0

    def test_empty_keywords(self):
        assert count_keyword_hits([], "some text about ML") == 0


class TestAnyKeywordMatch:

    def test_any_match_found(self):
        assert any_keyword_match(["python", "java", "go"], "Python developer")

    def test_no_match_found(self):
        assert not any_keyword_match(["python", "java"], "Ruby developer")

    def test_empty_text(self):
        assert not any_keyword_match(["python"], "")


def run_tests():
    test_classes = [TestMatchKeyword, TestCountKeywordHits, TestAnyKeywordMatch]
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
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

    print(f"\nMatching tests: {passed} passed, {failed} failed")
    for err in errors:
        print(err)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
