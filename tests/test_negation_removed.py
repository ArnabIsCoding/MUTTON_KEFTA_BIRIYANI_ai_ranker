
import os
import sys
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import matching
from src.matching import match_keyword, count_keyword_hits, any_keyword_match


def test_no_negation_terms_constant():
    assert not hasattr(matching, "NEGATION_TERMS"), (
    )


def test_no_is_negated_function():
    assert not hasattr(matching, "_is_negated"), (
    )


def test_no_negation_in_source():
    source = inspect.getsource(matching)
    assert "negat" not in source.lower(), (
    )


def test_match_keyword_ignores_negation_context():
    assert match_keyword("ml", "no experience with ml pipeline") is True
    assert match_keyword("python", "doesn't know python well") is True
    assert match_keyword("ranking", "never worked on ranking systems") is True
    assert match_keyword("retrieval", "not focused on retrieval") is True
    assert match_keyword("embeddings", "without embeddings knowledge") is True


def test_count_keyword_hits_no_negation():
    text = "no experience with ml but knows python and has not done ranking"
    keywords = ["ml", "python", "ranking"]
    assert count_keyword_hits(keywords, text) == 3, (
    )


def test_any_keyword_match_no_negation():
    text = "never worked with elasticsearch or faiss"
    keywords = ["elasticsearch", "faiss"]
    assert any_keyword_match(keywords, text) is True


def test_word_boundary_still_works():
    assert match_keyword("rank", "he works on ranking systems") is False
    assert match_keyword("ranking", "he works on ranking systems") is True

    assert match_keyword("embed", "working with embeddings") is False
    assert match_keyword("embeddings", "working with embeddings") is True

    assert match_keyword("vector database", "uses vector database for search") is True
    assert match_keyword("vector database", "vectordatabase") is False  


if __name__ == "__main__":
    tests = [
        test_no_negation_terms_constant,
        test_no_is_negated_function,
        test_no_negation_in_source,
        test_match_keyword_ignores_negation_context,
        test_count_keyword_hits_no_negation,
        test_any_keyword_match_no_negation,
        test_word_boundary_still_works,
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
