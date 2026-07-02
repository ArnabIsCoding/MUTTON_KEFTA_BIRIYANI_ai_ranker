
import argparse
import json
import os
import time
import pandas as pd

from src.parser import parse_all_candidates
from src.features import compute_all_features
from src.scorer import compute_final_scores, select_top_n
from src.reasoning import generate_reasoning


def load_tfidf_scores(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}

    if path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranker — Hackathon Submission"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="sample_candidates.json",
        help="Path to candidates file (.json or .jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="team_submission.csv",
        help="Path for the output CSV submission file",
    )
    parser.add_argument(
        "--tfidf-scores",
        type=str,
        default="data/tfidf_scores.json",
        help="Path to pre-computed TF-IDF similarity scores (.json or .pkl)",
    )
    args = parser.parse_args()

    start_time = time.time()

    print(f"[1/6] Parsing candidates from {args.candidates}...")
    candidates = parse_all_candidates(args.candidates)
    print(f"      Loaded {len(candidates)} candidates.")
    t1 = time.time()
    print(f"      Parse time: {t1 - start_time:.2f}s")

    tfidf_path = getattr(args, 'tfidf_scores', 'data/tfidf_scores.json')
    print(f"[2/6] Loading TF-IDF scores from {tfidf_path}...")
    tfidf_lookup = load_tfidf_scores(tfidf_path)
    if tfidf_lookup:
        print(f"      Loaded {len(tfidf_lookup)} TF-IDF scores.")
    else:
        print("      No TF-IDF scores found — Group J will score 0.0 for all.")
    t2 = time.time()

    from datetime import datetime
    valid_dates = []
    for c in candidates:
        d_str = c.get("last_active_date")
        if d_str:
            try:
                valid_dates.append(datetime.strptime(d_str, "%Y-%m-%d").date())
            except (ValueError, TypeError):
                pass
    
    from src.config import REFERENCE_DATE
    dynamic_ref_date = max(valid_dates) if valid_dates else REFERENCE_DATE
    print(f"      Calculated dynamic reference date: {dynamic_ref_date}")

    print("[3/6] Computing feature scores (11 groups A-K)...")
    features_list = []
    for i, cand in enumerate(candidates):
        features_list.append(compute_all_features(cand, tfidf_lookup, dynamic_ref_date))
        if (i + 1) % 10000 == 0:
            print(f"      Processed {i + 1}/{len(candidates)} candidates...")
    t3 = time.time()
    print(f"      Feature computation time: {t3 - t2:.2f}s")

    print("[4/6] Computing scores (career-dominant multiplicative formula)...")
    scored_df = compute_final_scores(features_list)

    top_df = select_top_n(scored_df, n=100)
    t4 = time.time()
    print(f"      Scoring time: {t4 - t3:.2f}s")

    print("[5/6] Generating fact-based reasoning for top candidates...")
    cand_lookup = {c["candidate_id"]: c for c in candidates}
    feat_lookup = {f["candidate_id"]: f for f in features_list}

    reasoning_list = []
    for _, row in top_df.iterrows():
        cid = row["candidate_id"]
        rank = int(row["rank"])
        cand_data = cand_lookup.get(cid, {})
        feat_data = feat_lookup.get(cid, {})
        reasoning = generate_reasoning(cand_data, feat_data, rank)
        reasoning_list.append(reasoning)

    top_df = top_df.copy()
    top_df["reasoning"] = reasoning_list
    t5 = time.time()
    print(f"      Reasoning generation time: {t5 - t4:.2f}s")

    print(f"[6/6] Writing submission to {args.out}...")
    submission = top_df[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission = submission.rename(columns={"final_score": "score"})
    submission["reasoning"] = submission["reasoning"].str.replace("—", "-", regex=False)
    submission.to_csv(args.out, index=False, encoding="utf-8-sig")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" Submission generated: {args.out}")
    print(f" Total candidates processed: {len(candidates)}")
    print(f" Top 100 selected, ranks 1-{len(top_df)}")
    print(f" TF-IDF scores loaded: {'Yes' if tfidf_lookup else 'No'}")
    print(f" Total time: {total_time:.2f}s")
    print(f"{'='*60}")

    honeypot_count = int(scored_df["group_h_honeypot"].sum())
    gated_count = int((scored_df["group_d_gate"] < 0.5).sum())
    print(f"\n Honeypots detected: {honeypot_count}")
    print(f" Anti-negative gated (score < 0.5): {gated_count}")
    print(f" Honeypots in top 100: {int(top_df['group_h_honeypot'].sum())}")

    print(f"\n Top 5 candidates:")
    for _, row in top_df.head(5).iterrows():
        print(f"   Rank {int(row['rank'])}: {row['candidate_id']} | "
              f"score={row['final_score']:.4f} | "
              f"A={row['group_a_title']:.2f} B={row['group_b_desc']:.2f} "
              f"C={row['group_c_skills']:.2f} D={row['group_d_gate']:.2f} "
              f"G={row['group_g_behavioral']:.2f} H={row['group_h_honeypot']:.0f} "
              f"I={row['group_i_education']:.2f} J={row['group_j_semantic']:.2f} "
              f"K={row['group_k_quantitative']:.2f}")

    print(f"\n Bottom 5 of top 100:")
    for _, row in top_df.tail(5).iterrows():
        print(f"   Rank {int(row['rank'])}: {row['candidate_id']} | "
              f"score={row['final_score']:.4f}")


if __name__ == "__main__":
    main()

