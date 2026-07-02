
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_all_candidates
from src.features import compute_all_features
from src.scorer import compute_final_scores, select_top_100


def run_ablation(candidates, disable_group=None):
    features_list = []
    for cand in candidates:
        feats = compute_all_features(cand)
        if disable_group:
            feats[disable_group] = 0.5  
        features_list.append(feats)

    scored_df = compute_final_scores(features_list)
    top_df = select_top_100(scored_df)
    return set(top_df["candidate_id"].tolist())


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--candidates", default="sample_candidates.json")
    args = parser.parse_args()

    print("Loading candidates...")
    candidates = parse_all_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.\n")

    baseline = run_ablation(candidates)
    print(f"Baseline top 100: {len(baseline)} candidates\n")

    groups = [
        ("group_a_title", "A — Career Title"),
        ("group_b_desc", "B — Career Description"),
        ("group_c_skills", "C — Skills"),
        ("group_d_gate", "D — Anti-Negative Gate"),
        ("group_e_experience", "E — Experience Band"),
        ("group_f_location", "F — Location"),
        ("group_g_behavioral", "G — Behavioral"),
        ("group_i_education", "I — Education"),
    ]

    print(f"{'Group':<30} {'Overlap':>8} {'Dropped':>8} {'Added':>8}")
    print("-" * 60)

    for group_key, group_name in groups:
        ablated = run_ablation(candidates, disable_group=group_key)
        overlap = len(baseline & ablated)
        dropped = len(baseline - ablated)
        added = len(ablated - baseline)
        print(f"{group_name:<30} {overlap:>8} {dropped:>8} {added:>8}")

    print(f"\nHigher 'Dropped' = more influential group (changes who makes top 100)")


if __name__ == "__main__":
    main()
