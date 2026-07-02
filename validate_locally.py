
import argparse
import csv
import re
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Validate a Redrob AI Ranker submission CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="team_submission.csv",
        help="Path to the submission CSV file (default: team_submission.csv)",
    )
    args = parser.parse_args()

    results = []  

    try:
        with open(args.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read file: {e}")
        sys.exit(1)

    if not rows:
        print("ERROR: File is empty.")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    check = "Row count (100 data rows + 1 header)"
    if len(data_rows) == 100:
        results.append((check, True, f"{len(data_rows)} data rows"))
    else:
        results.append((check, False, f"Expected 100 data rows, got {len(data_rows)}"))

    expected_header = ["candidate_id", "rank", "score", "reasoning"]
    check = "Header columns"
    if header == expected_header:
        results.append((check, True, ", ".join(header)))
    else:
        results.append(
            (check, False, f"Expected {expected_header}, got {header}")
        )

    candidate_ids = []
    ranks = []
    scores = []
    reasonings = []
    parse_errors = []

    for i, row in enumerate(data_rows, start=2):  
        if len(row) != 4:
            parse_errors.append(f"Row {i}: expected 4 columns, got {len(row)}")
            continue
        candidate_ids.append(row[0].strip())
        ranks.append(row[1].strip())
        scores.append(row[2].strip())
        reasonings.append(row[3].strip())

    if parse_errors:
        results.append(("Column count per row", False, "; ".join(parse_errors[:5])))
    else:
        results.append(("Column count per row", True, "All rows have 4 columns"))

    cand_pattern = re.compile(r"^CAND_\d{7}$")
    bad_ids = [
        f"Row {i+2}: '{cid}'"
        for i, cid in enumerate(candidate_ids)
        if not cand_pattern.match(cid)
    ]
    check = "candidate_id format (CAND_XXXXXXX)"
    if not bad_ids:
        results.append((check, True, "All candidate_ids match pattern"))
    else:
        results.append((check, False, f"{len(bad_ids)} invalid: " + "; ".join(bad_ids[:5])))

    check = "candidate_id uniqueness"
    seen = set()
    dupes = []
    for cid in candidate_ids:
        if cid in seen:
            dupes.append(cid)
        seen.add(cid)
    if not dupes:
        results.append((check, True, f"{len(seen)} unique candidate_ids"))
    else:
        results.append(
            (check, False, f"{len(dupes)} duplicate(s): " + ", ".join(dupes[:5]))
        )

    int_ranks = []
    bad_ranks = []
    for i, r in enumerate(ranks):
        try:
            int_ranks.append(int(r))
        except ValueError:
            bad_ranks.append(f"Row {i+2}: '{r}'")

    check = "Ranks are integers 1-100, each once"
    if bad_ranks:
        results.append(
            (check, False, f"Non-integer ranks: " + "; ".join(bad_ranks[:5]))
        )
    elif sorted(int_ranks) == list(range(1, 101)):
        results.append((check, True, "Ranks 1-100 all present exactly once"))
    else:
        missing = set(range(1, 101)) - set(int_ranks)
        extra = set(int_ranks) - set(range(1, 101))
        detail = ""
        if missing:
            detail += f"Missing: {sorted(missing)[:10]}  "
        if extra:
            detail += f"Extra/out-of-range: {sorted(extra)[:10]}"
        results.append((check, False, detail.strip()))

    float_scores = []
    bad_scores = []
    for i, s in enumerate(scores):
        try:
            float_scores.append(float(s))
        except ValueError:
            bad_scores.append(f"Row {i+2}: '{s}'")

    check = "Scores are valid floats"
    if bad_scores:
        results.append(
            (check, False, f"Invalid scores: " + "; ".join(bad_scores[:5]))
        )
    else:
        results.append((check, True, "All scores are valid floats"))

    check = "Scores non-increasing by rank"
    if not bad_ranks and not bad_scores and len(int_ranks) == len(float_scores):
        ranked = sorted(zip(int_ranks, float_scores, candidate_ids), key=lambda x: x[0])
        violations = []
        for j in range(1, len(ranked)):
            if ranked[j][1] > ranked[j - 1][1]:
                violations.append(
                    f"Rank {ranked[j-1][0]} score={ranked[j-1][1]} < "
                    f"Rank {ranked[j][0]} score={ranked[j][1]}"
                )
        if not violations:
            results.append((check, True, "Scores are non-increasing by rank"))
        else:
            results.append(
                (check, False, f"{len(violations)} violation(s): " + "; ".join(violations[:5]))
            )
    else:
        results.append((check, False, "Skipped due to earlier parse errors"))

    check = "Tie-break (equal scores → ascending candidate_id)"
    if not bad_ranks and not bad_scores and len(int_ranks) == len(float_scores):
        ranked = sorted(zip(int_ranks, float_scores, candidate_ids), key=lambda x: x[0])
        tie_violations = []
        for j in range(1, len(ranked)):
            if ranked[j][1] == ranked[j - 1][1]:
                if ranked[j][2] < ranked[j - 1][2]:
                    tie_violations.append(
                        f"Rank {ranked[j-1][0]} ({ranked[j-1][2]}) and "
                        f"Rank {ranked[j][0]} ({ranked[j][2]}) have same score "
                        f"{ranked[j][1]} but candidate_ids not ascending"
                    )
        if not tie_violations:
            results.append((check, True, "Tie-break ordering correct"))
        else:
            results.append(
                (check, False, f"{len(tie_violations)} violation(s): " + "; ".join(tie_violations[:5]))
            )
    else:
        results.append((check, False, "Skipped due to earlier parse errors"))

    check = "Reasoning non-empty"
    empty_reasons = [
        f"Row {i+2}" for i, r in enumerate(reasonings) if not r
    ]
    if not empty_reasons:
        results.append((check, True, "All rows have non-empty reasoning"))
    else:
        results.append(
            (check, False, f"{len(empty_reasons)} empty: " + ", ".join(empty_reasons[:10]))
        )

    print("=" * 64)
    print("  REDROB AI RANKER — Submission Validation Report")
    print(f"  File: {args.csv_path}")
    print("=" * 64)

    all_passed = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        icon = "[OK]" if passed else "[!!]"
        print(f"\n  {icon} [{status}] {name}")
        print(f"          {detail}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 64)
    if all_passed:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        failed = sum(1 for _, p, _ in results if not p)
        print(f"  RESULT: {failed} CHECK(S) FAILED")
    print("=" * 64)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
