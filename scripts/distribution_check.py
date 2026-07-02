
import csv
import sys
import re
from collections import Counter
from difflib import SequenceMatcher


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/distribution_check.py <submission.csv> [candidates.jsonl]")
        sys.exit(1)

    csv_path = sys.argv[1]
    candidates_path = sys.argv[2] if len(sys.argv) > 2 else None

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    scores = [float(r["score"]) for r in rows]
    reasonings = [r["reasoning"] for r in rows]

    print("=" * 60)
    print(" DISTRIBUTION CHECK")
    print("=" * 60)

    print(f"\n📊 Score Distribution:")
    print(f"  Rank  1: {scores[0]:.6f}")
    print(f"  Rank 10: {scores[9]:.6f}")
    print(f"  Rank 25: {scores[24]:.6f}")
    print(f"  Rank 50: {scores[49]:.6f}")
    print(f"  Rank 75: {scores[74]:.6f}")
    print(f"  Rank 100: {scores[99]:.6f}")
    print(f"  Range (50-100): {scores[49] - scores[99]:.6f}")
    print(f"  Range (1-100):  {scores[0] - scores[99]:.6f}")

    import statistics
    print(f"  Mean:   {statistics.mean(scores):.6f}")
    print(f"  StdDev: {statistics.stdev(scores):.6f}")

    print(f"\n💬 Reasoning Quality:")
    avg_len = sum(len(r) for r in reasonings) / len(reasonings)
    min_len = min(len(r) for r in reasonings)
    max_len = max(len(r) for r in reasonings)
    print(f"  Avg length: {avg_len:.0f} chars")
    print(f"  Min length: {min_len} chars")
    print(f"  Max length: {max_len} chars")

    high_sim_count = 0
    pairs_checked = 0
    for i in range(min(20, len(reasonings))):
        for j in range(i + 1, min(20, len(reasonings))):
            sim = SequenceMatcher(None, reasonings[i], reasonings[j]).ratio()
            pairs_checked += 1
            if sim > 0.7:
                high_sim_count += 1
                print(f"  ⚠️  High similarity ({sim:.2f}) between rank {i+1} and rank {j+1}")
    if high_sim_count == 0:
        print(f"  ✅ No high-similarity pairs found in top 20 ({pairs_checked} pairs checked)")
    else:
        print(f"  ❌ {high_sim_count}/{pairs_checked} pairs have >70% similarity — TEMPLATED RISK")

    print(f"\n🔍 Concern Acknowledgment:")
    concern_words = ["concern", "however", "limited", "weaker", "thin",
                     "borderline", "despite", "lacking", "no evidence", "below"]
    for rank_idx in [74, 84, 94, 99]:  
        reasoning_lower = reasonings[rank_idx].lower()
        has_concern = any(w in reasoning_lower for w in concern_words)
        status = "✅" if has_concern else "❌"
        print(f"  {status} Rank {rank_idx + 1}: {'acknowledges concerns' if has_concern else 'NO concern language found'}")

    cv_keywords = ["computer vision", "cv engineer", "image processing"]
    cv_in_reasoning = 0
    for r in reasonings:
        if any(kw in r.lower() for kw in cv_keywords):
            cv_in_reasoning += 1
    if cv_in_reasoning > 0:
        print(f"\n⚠️  {cv_in_reasoning} candidates mention CV/vision in reasoning")

    print(f"\n{'=' * 60}")
    print(" CHECK COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
