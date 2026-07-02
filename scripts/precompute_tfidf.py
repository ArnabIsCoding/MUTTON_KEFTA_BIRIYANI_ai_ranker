
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


JD_TEXT = """
Senior AI Engineer — Candidate Discovery & Ranking

We're building an AI-powered talent discovery engine. This role focuses on
retrieval, ranking, and recommendation of candidates from large pools.

Must-Have:
- 5-9 years in ML/NLP/IR
- Built retrieval or ranking systems (search, recommendation, embeddings)
- Vector databases (FAISS, Pinecone, Qdrant, Milvus, Weaviate)
- Evaluation metrics: NDCG, MRR, MAP, A/B testing
- Python (production-grade), deployed ML models at scale
- LLM fine-tuning (LoRA, QLoRA, PEFT) and RAG pipelines

Nice-to-Have:
- Learning-to-rank (LTR) frameworks
- HR-tech or recruiting domain experience
- Open-source contributions
- Distributed systems knowledge

Key responsibilities:
- Design and implement candidate retrieval and ranking pipelines
- Build embedding-based search using vector stores
- Fine-tune LLMs for talent matching and JD parsing
- Evaluate ranking quality with NDCG, MRR, MAP
- Deploy models with low-latency inference in production
- A/B test ranking improvements
    candidates = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                candidates = data
            elif isinstance(data, dict) and "candidates" in data:
                candidates = data["candidates"]
            else:
                candidates = [data]
    return candidates


def build_candidate_text(candidate: dict) -> str:
    parts = []
    profile = candidate.get("profile", {})

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    if title:
        parts.append(title)
    if company:
        parts.append(company)

    summary = profile.get("summary", "") or profile.get("headline", "")
    if summary:
        parts.append(summary)

    career = candidate.get("career_history", [])
    for role in career:
        desc = role.get("description", "")
        if desc:
            parts.append(desc)
        role_title = role.get("title", "")
        if role_title:
            parts.append(role_title)

    skills = candidate.get("skills", [])
    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get("name", "")
            if name:
                parts.append(name)
        elif isinstance(skill, str):
            parts.append(skill)

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute TF-IDF similarity scores for candidates."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="candidates.jsonl",
        help="Path to candidates file (.json or .jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/tfidf_scores.json",
        help="Output path for the similarity scores JSON",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Max features for TfidfVectorizer (default: 5000)",
    )
    args = parser.parse_args()

    start_time = time.time()

    print(f"[1/5] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates)} candidates.")

    print("[2/5] Building text corpus...")
    candidate_ids = []
    corpus = []
    for cand in candidates:
        cid = cand.get("candidate_id", "")
        text = build_candidate_text(cand)
        candidate_ids.append(cid)
        corpus.append(text)

    corpus.append(JD_TEXT)
    print(f"      Corpus size: {len(corpus)} documents (candidates + JD)")

    print(f"[3/5] Fitting TfidfVectorizer (max_features={args.max_features}, bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),       
        sublinear_tf=True,        
        min_df=5,                 
        max_df=0.95,              
        stop_words="english",     
        dtype=np.float32,         
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    t3 = time.time()
    print(f"      TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"      Fit time: {t3 - start_time:.2f}s")

    print("[4/5] Computing cosine similarity to JD...")
    jd_vector = tfidf_matrix[-1]          
    candidate_matrix = tfidf_matrix[:-1]  

    similarities = cosine_similarity(candidate_matrix, jd_vector).flatten()

    tfidf_scores = {}
    for cid, sim in zip(candidate_ids, similarities):
        tfidf_scores[cid] = round(float(sim), 6)

    t4 = time.time()
    print(f"      Similarity computation time: {t4 - t3:.2f}s")

    sim_values = list(tfidf_scores.values())
    print(f"      Score range: [{min(sim_values):.4f}, {max(sim_values):.4f}]")
    print(f"      Mean: {np.mean(sim_values):.4f}, Median: {np.median(sim_values):.4f}")
    print(f"      Std: {np.std(sim_values):.4f}")

    sorted_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n      Top 10 by TF-IDF similarity:")
    for cid, score in sorted_scores[:10]:
        print(f"        {cid}: {score:.4f}")

    print(f"\n[5/5] Saving to {args.out}...")
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tfidf_scores, f, indent=None)  

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" TF-IDF pre-computation complete")
    print(f" Candidates processed: {len(candidates)}")
    print(f" Output: {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB)")
    print(f" Total time: {total_time:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
