# Redrob AI Hackathon — Candidate Ranking Engine (Solo Submission)
Github_Repo: https://github.com/ArnabIsCoding/MUTTON_KEFTA_BIRIYANI_ai_ranker/
Sandbox_Link: https://colab.research.google.com/drive/1zLBSZVJAb7iN30UylUA4OXPk0n1_qysU?usp=sharing
## Architecture Overview

This project uses a **deterministic, multi-signal scoring system** to rank candidates for a Senior AI Engineer role. Instead of relying on cosine similarity with embeddings (which conflates "mentions AI" with "does AI"), it implements 8 orthogonal scoring dimensions that combine via a multiplicative formula.

### How It Works

```
final_score = career_relevance × skill_corroboration × anti_negative_gate
              × experience_band × location_score × behavioral_multiplier
              × (1 - honeypot_flag)
```

**Career history descriptions are the primary signal.** A candidate who *built* a recommendation system at their job is ranked higher than one who merely *lists* "recommendation systems" as a skill.

### The 8 Scoring Groups

| Group | Name | Weight | What It Measures |
|---|---|---|---|
| A | Career Title Signal | HIGH | ML/AI relevance of job titles across career |
| B | Career Description Evidence | **HIGHEST** | Keyword evidence in actual job descriptions (retrieval, ranking, production, LLM, Python) |
| C | Skills Corroboration | MEDIUM | Must-have skills with duration & endorsement backing |
| D | Anti-Negative Gate | **HARD GATE** | Kills score for non-technical titles, consulting-only careers, pure research |
| E | Experience Band | MEDIUM | 5–9 years sweet spot for Senior IC role |
| F | Location / Relocation | MEDIUM | India + preferred city preference, relocation willingness |
| G | Behavioral Availability | HIGH | Open-to-work, response rate, notice period, GitHub, interview completion |
| H | Honeypot Detection | **CRITICAL** | Expert skills with 0 months, impossible career timelines |

### Key Design Decisions

1. **No neural models / embeddings** — Pure keyword-based scoring is faster, offline-capable, more interpretable, and more precise for this structured-data problem than cosine similarity on shallow text.
2. **Career descriptions > skills lists** — The JD explicitly warns: "career-history evidence beats skills-list evidence."
3. **Multiplicative formula** — A great career match with zero availability (behavioral) scores zero. Each dimension can only reduce, not inflate.
4. **Dependencies: pandas + numpy only** — No torch, no sentence-transformers, no network downloads. Runs in Docker in seconds.

## How to Run

### 1. Pre-compute TF-IDF Scores (Optional but recommended)
Before ranking, it is recommended to pre-compute the semantic similarity scores using `scikit-learn`.
```bash
python scripts/precompute_tfidf.py --candidates ./sample_candidates.json
```
* **What it does:** Generates `data/tfidf_scores.json` containing the semantic match between candidates and the Job Description.
* **Why:** This acts as a secondary signal (Group J) in the ranking engine. It is separated from the main script to keep ranking dependencies minimal (no `scikit-learn` needed at rank time).

### 2. Run the Ranking CLI
```bash
python rank.py --candidates ./sample_candidates.json --out ./team_submission.csv
```
* **What it does:** Parses candidates, applies the 11 feature scoring groups, and outputs the top 100 list to a CSV.
* **Why:** This is the main engine used to generate the final hackathon submission.

### 3. Run the Interactive Streamlit UI
```bash
pip install streamlit
streamlit run app.py
```
* **What it does:** Launches a beautiful web interface to upload candidate files and view the ranking results dynamically.
* **Why:** Great for visually demonstrating the system's capabilities, checking score distributions, and reading the generated reasonings.

### 4. Validate Output Format
```bash
python validate_locally.py team_submission.csv
```
* **What it does:** Checks your generated CSV against the hackathon's strict formatting constraints.
* **Why:** Ensures you don't get an automatic rejection when uploading to the hackathon platform.

### Docker (Full Pipeline)
```bash
docker build -t redrob-ranker .
docker run -v $(pwd):/data redrob-ranker --candidates /data/candidates.jsonl --out /data/submission.csv
```
* **What it does:** Runs the entire `rank.py` pipeline inside a clean containerized environment.

## Project Structure

```text
├── app.py                   # Streamlit interactive UI
├── rank.py                  # Main CLI entry point
├── src/
│   ├── config.py            # All keywords, weights, thresholds
│   ├── parser.py            # Full candidate parsing logic
│   ├── features.py          # Feature scoring groups (A–K)
│   ├── matching.py          # Word-boundary text matching utilities
│   ├── scorer.py            # Multiplicative formula + ranking logic
│   └── reasoning.py         # Fact-based per-candidate reasoning generator
├── scripts/
│   ├── precompute_tfidf.py  # Generates offline semantic scores
│   ├── evaluate.py          # Tests NDCG/MRR metrics against ground truth
│   └── ablation.py          # Feature importance ablation studies
├── data/
│   └── tfidf_scores.json    # Pre-computed output from precompute_tfidf.py
├── tests/                   # Robust unit & integration test suite
├── requirements.txt         # pandas, numpy, scikit-learn
├── Dockerfile               # Offline-capable container specification
└── validate_locally.py      # Local CSV submission validator
```

## Compute Profile

| Constraint | Requirement | Actual |
|---|---|---|
| Runtime | ≤ 5 min | ~30s on 100K candidates |
| Memory | ≤ 16 GB | < 2 GB |
| Compute | CPU only | ✅ No GPU needed |
| Network | Offline | ✅ No downloads at runtime |
| Dependencies | Minimal | pandas + numpy only |
