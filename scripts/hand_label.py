
import argparse
import json
import os
import random
import sys


ML_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "deep learning", "ai engineer",
    "data scientist", "nlp", "natural language", "search engineer",
    "ranking", "recommendation", "retrieval", "applied scientist",
    "ml", "artificial intelligence", "data science",
]

CONSULTING_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "deloitte", "hcl", "genpact",
    "mindtree", "tech mahindra", "ltimindtree", "mphasis",
]

NON_TECH_KEYWORDS = [
    "hr", "human resources", "marketing", "sales", "accountant",
    "content writer", "graphic designer", "operations", "civil engineer",
    "mechanical engineer", "customer support", "receptionist",
    "recruiter", "clerk", "administrative",
]

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_all_candidates

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")

def load_candidates(path: str) -> list[dict]:
    return parse_all_candidates(path)


def matches_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def stratified_sample(candidates: list[dict], existing_labels: dict) -> list[dict]:
    labeled_ids = set(existing_labels.keys())
    pool = [c for c in candidates if c.get("candidate_id", "") not in labeled_ids]

    random.seed(42)  

    ml_pool, consulting_pool, nontech_pool, overseas_pool, other_pool = [], [], [], [], []

    for c in pool:
        title = c.get("current_title", "")
        company = c.get("current_company", "")
        country = c.get("country", "India")

        if matches_any(title, ML_TITLE_KEYWORDS):
            ml_pool.append(c)
        elif matches_any(company, CONSULTING_COMPANIES):
            consulting_pool.append(c)
        elif matches_any(title, NON_TECH_KEYWORDS):
            nontech_pool.append(c)
        elif country.lower() != "india":
            overseas_pool.append(c)
        else:
            other_pool.append(c)

    def safe_sample(pool, n):
        return random.sample(pool, min(n, len(pool)))

    sampled = []
    sampled.extend(safe_sample(ml_pool, 10))
    sampled.extend(safe_sample(consulting_pool, 10))
    sampled.extend(safe_sample(nontech_pool, 10))
    sampled.extend(safe_sample(overseas_pool, 10))
    sampled.extend(safe_sample(other_pool, 10))

    random.shuffle(sampled)
    return sampled


def display_candidate(c: dict, index: int, total: int):
    print(f"\n{'='*70}")
    print(f"  Candidate {index}/{total}")
    print(f"{'='*70}")
    print(f"  ID:         {c.get('candidate_id', 'N/A')}")
    print(f"  Title:      {c.get('current_title', 'N/A')}")
    print(f"  Company:    {c.get('current_company', 'N/A')}")
    print(f"  Experience: {c.get('years_of_experience', 'N/A')} years")
    print(f"  Country:    {c.get('country', 'N/A')}")
    print(f"  Location:   {c.get('location', 'N/A')}")

    skills = c.get("skills", [])[:5]
    if skills:
        skill_strs = []
        for s in skills:
            if isinstance(s, dict):
                name = s.get("name", "")
                prof = s.get("proficiency", "")
                skill_strs.append(f"{name} ({prof})" if prof else name)
            elif isinstance(s, str):
                skill_strs.append(s)
        print(f"  Top Skills: {', '.join(skill_strs)}")

    career = c.get("career_history", [])
    if career:
        titles = [f"{r.get('title', '?')} @ {r.get('company', '?')}" for r in career[:5]]
        print(f"  Career:     {' -> '.join(titles)}")

    descs = []
    for role in career:
        desc = role.get("description", "")
        if desc:
            descs.append(desc)
    combined = " | ".join(descs)[:300]
    if combined:
        print(f"  Desc:       {combined}...")

    print(f"{'_'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified candidate sampler and relevance labeler."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="candidates.jsonl",
        help="Path to candidates file (.json or .jsonl)",
    )
    args = parser.parse_args()

    existing_labels = {}
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            existing_labels = json.load(f)
        print(f"Loaded {len(existing_labels)} existing labels from {GROUND_TRUTH_PATH}")

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    sampled = stratified_sample(candidates, existing_labels)
    print(f"\nSampled {len(sampled)} candidates for labeling.")
    print(f"Already labeled: {len(existing_labels)}")

    if not sampled:
        print("No new candidates to label. All sampled candidates already labeled.")
        return

    print("""
Labeling Guide (for Senior AI Engineer - Candidate Discovery & Ranking):
  3 = Highly relevant (ML/NLP/IR experience, retrieval/ranking systems, 5-9yr)
  2 = Somewhat relevant (adjacent ML experience, some relevant skills)
  1 = Marginally relevant (some tech background, few matching skills)
  0 = Irrelevant (wrong domain, no ML, non-technical)
  s = Skip this candidate
  q = Quit and save
""")

    labels = dict(existing_labels)  
    labeled_count = 0
    quit_flag = False

    for i, c in enumerate(sampled, 1):
        cid = c.get("candidate_id", "")
        if cid in labels:
            continue

        display_candidate(c, i, len(sampled))

        while True:
            choice = input("  Grade (0/1/2/3/s/q): ").strip().lower()
            if choice in ("0", "1", "2", "3"):
                labels[cid] = int(choice)
                labeled_count += 1
                print(f"  -> Labeled {cid} as {choice}")
                break
            elif choice == "s":
                print(f"  -> Skipped {cid}")
                break
            elif choice == "q":
                print(f"\nQuitting. Labeled {labeled_count} new candidates.")
                quit_flag = True
                break
            else:
                print("  Invalid input. Enter 0, 1, 2, 3, s, or q.")

        if quit_flag:
            break

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSaved {len(labels)} total labels to {GROUND_TRUTH_PATH}")
    print(f"  (New: {labeled_count}, Existing: {len(existing_labels)})")


if __name__ == "__main__":
    main()
