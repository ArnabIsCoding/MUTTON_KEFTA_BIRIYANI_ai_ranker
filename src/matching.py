
import re
from functools import lru_cache


@lru_cache(maxsize=1024)
def _compile_pattern(keyword: str) -> re.Pattern:
    cleaned = keyword.strip().lower()
    escaped = re.escape(cleaned)
    return re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)


@lru_cache(maxsize=256)
def _compile_combined_pattern(keywords_tuple: tuple) -> re.Pattern:
    escaped = [re.escape(kw.strip().lower()) for kw in keywords_tuple]
    combined = r'\b(?:' + '|'.join(escaped) + r')\b'
    return re.compile(combined, re.IGNORECASE)


def match_keyword(keyword: str, text: str) -> bool:
    if not keyword or not text:
        return False
    cleaned = keyword.strip().lower()
    text_lower = text if text.islower() else text.lower()
    if cleaned not in text_lower:
        return False
    pattern = _compile_pattern(keyword)
    return bool(pattern.search(text_lower))


def count_keyword_hits(keywords: list[str], text: str) -> int:
    if not text or not keywords:
        return 0
    text_lower = text if text.islower() else text.lower()
    candidates = [kw for kw in keywords if kw.strip().lower() in text_lower]
    if not candidates:
        return 0
    pattern = _compile_combined_pattern(tuple(candidates))
    matched_keywords = set(m.group(0).lower() for m in pattern.finditer(text_lower))
    return len(matched_keywords)


def any_keyword_match(keywords: list[str], text: str) -> bool:
    if not text or not keywords:
        return False
    text_lower = text if text.islower() else text.lower()
    candidates = [kw for kw in keywords if kw.strip().lower() in text_lower]
    if not candidates:
        return False
    pattern = _compile_combined_pattern(tuple(candidates))
    return bool(pattern.search(text_lower))

