
import argparse
import csv
import json
import math
import os
import sys


GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")


def load_ground_truth(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: Ground truth file not found: {path}")
        print("Run 'python scripts/hand_label.py' first to create labels.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_submission(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"ERROR: Submission file not found: {path}")
        sys.exit(1)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "candidate_id": row["candidate_id"].strip(),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
            })
    rows.sort(key=lambda x: x["rank"])
    return rows


def dcg_at_k(ranked_ids: list[str], ground_truth: dict, k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(ranked_ids))):
        cid = ranked_ids[i]
        rel = ground_truth.get(cid, 0)
        dcg += rel / math.log2(i + 2)  
    return dcg


def idcg_at_k(ground_truth: dict, k: int) -> float:
    grades = sorted(ground_truth.values(), reverse=True)
    dcg = 0.0
    for i in range(min(k, len(grades))):
        dcg += grades[i] / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_ids: list[str], ground_truth: dict, k: int) -> float:
    ideal = idcg_at_k(ground_truth, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, ground_truth, k) / ideal


def mean_average_precision(ranked_ids: list[str], ground_truth: dict, threshold: int = 2) -> float:
    relevant_ids = {cid for cid, grade in ground_truth.items() if grade >= threshold}
    if not relevant_ids:
        return 0.0

    relevant_found = 0
    precision_sum = 0.0

    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant_ids:
            relevant_found += 1
            precision_sum += relevant_found / i

    if relevant_found == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], ground_truth: dict, k: int, threshold: int = 2) -> float:
    relevant = 0
    for i in range(min(k, len(ranked_ids))):
        cid = ranked_ids[i]
        if ground_truth.get(cid, 0) >= threshold:
            relevant += 1
    return relevant / min(k, len(ranked_ids)) if ranked_ids else 0.0


def evaluate_submission(submission: list[dict], ground_truth: dict, name: str = "Submission") -> dict:
    ranked_ids = [row["candidate_id"] for row in submission]

    n10 = ndcg_at_k(ranked_ids, ground_truth, 10)
    n50 = ndcg_at_k(ranked_ids, ground_truth, 50)
    map_score = mean_average_precision(ranked_ids, ground_truth)
    p10 = precision_at_k(ranked_ids, ground_truth, 10)

    composite = 0.50 * n10 + 0.30 * n50 + 0.15 * map_score + 0.05 * p10

    return {
        "name": name,
        "ndcg_10": n10,
        "ndcg_50": n50,
        "map": map_score,
        "p_10": p10,
        "composite": composite,
    }


def print_report(metrics: dict, ground_truth: dict, submission: list[dict]):
    name = metrics["name"]

    print(f"\n  {'='*64}")
    print(f"  Evaluation Report: {name}")
    print(f"  {'='*64}")

    ranked_ids = [row["candidate_id"] for row in submission]
    rank_lookup = {row["candidate_id"]: row["rank"] for row in submission}

    print(f"\n  Labeled Candidates in Submission:")
    print(f"  {'ID':<16} {'Grade':>6} {'Rank':>6} {'Status':<20}")
    print(f"  {'-'*50}")

    found_in_top10 = 0
    found_in_top100 = 0
    for cid, grade in sorted(ground_truth.items(), key=lambda x: -x[1]):
        rank = rank_lookup.get(cid, None)
        if rank is not None:
            found_in_top100 += 1
            if rank <= 10:
                found_in_top10 += 1
            status = f"Rank {rank}"
            if rank <= 10:
                status += " * TOP 10"
            elif rank <= 50:
                status += " + TOP 50"
        else:
            status = "NOT IN TOP 100"
        print(f"  {cid:<16} {grade:>6} {(rank or '-'):>6} {status:<20}")

    print(f"\n  {'-'*50}")
    print(f"  Labeled candidates found in top 100: {found_in_top100}/{len(ground_truth)}")
    print(f"  Labeled candidates found in top 10:  {found_in_top10}/{len(ground_truth)}")

    print(f"\n  Metrics:")
    print(f"  {'-'*50}")
    print(f"  NDCG@10   (50% weight):  {metrics['ndcg_10']:.4f}")
    print(f"  NDCG@50   (30% weight):  {metrics['ndcg_50']:.4f}")
    print(f"  MAP       (15% weight):  {metrics['map']:.4f}")
    print(f"  P@10      ( 5% weight):  {metrics['p_10']:.4f}")
    print(f"  {'-'*50}")
    print(f"  COMPOSITE:               {metrics['composite']:.4f}")
    print(f"  {'='*64}")


def print_comparison(before: dict, after: dict):
    print(f"\n  {'='*64}")
    print(f"  Comparison: {before['name']} -> {after['name']}")
    print(f"  {'='*64}")
    print(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Delta':>10} {'':>5}")
    print(f"  {'-'*55}")

    metrics = [
        ("NDCG@10 (50%)", "ndcg_10"),
        ("NDCG@50 (30%)", "ndcg_50"),
        ("MAP     (15%)", "map"),
        ("P@10    ( 5%)", "p_10"),
        ("COMPOSITE", "composite"),
    ]

    for label, key in metrics:
        b = before[key]
        a = after[key]
        delta = a - b
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {label:<20} {b:>10.4f} {a:>10.4f} {delta:>+10.4f} {arrow:>5}")

    print(f"  {'='*64}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ranking submission against hand-labeled ground truth."
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="Path to submission CSV (single evaluation mode)",
    )
    parser.add_argument(
        "--before",
        type=str,
        default=None,
        help="Path to 'before' submission CSV (comparison mode)",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Path to 'after' submission CSV (comparison mode)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=GROUND_TRUTH_PATH,
        help="Path to ground truth JSON",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} ground truth labels.")

    grade_dist = {}
    for g in ground_truth.values():
        grade_dist[g] = grade_dist.get(g, 0) + 1
    print(f"Grade distribution: {dict(sorted(grade_dist.items(), reverse=True))}")

    if args.before and args.after:
        sub_before = load_submission(args.before)
        sub_after = load_submission(args.after)
        met_before = evaluate_submission(sub_before, ground_truth, os.path.basename(args.before))
        met_after = evaluate_submission(sub_after, ground_truth, os.path.basename(args.after))
        print_report(met_before, ground_truth, sub_before)
        print_report(met_after, ground_truth, sub_after)
        print_comparison(met_before, met_after)
    elif args.submission:
        sub = load_submission(args.submission)
        met = evaluate_submission(sub, ground_truth, os.path.basename(args.submission))
        print_report(met, ground_truth, sub)
    else:
        default_path = "team_submission.csv"
        if not os.path.exists(default_path):
            print(f"No submission specified and {default_path} not found.")
            print("Usage: python scripts/evaluate.py --submission path/to/submission.csv")
            sys.exit(1)
        sub = load_submission(default_path)
        met = evaluate_submission(sub, ground_truth, default_path)
        print_report(met, ground_truth, sub)


if __name__ == "__main__":
    main()
