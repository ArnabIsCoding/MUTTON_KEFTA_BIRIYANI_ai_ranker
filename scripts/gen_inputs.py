import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.hand_label import load_candidates, stratified_sample, GROUND_TRUTH_PATH, matches_any
from scripts.hand_label import ML_TITLE_KEYWORDS, CONSULTING_COMPANIES, NON_TECH_KEYWORDS

def auto_grade(c):
    for s in c.get("skills", []):
        if isinstance(s, dict) and s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            return 0
    
    title = c.get("current_title", "")
    company = c.get("current_company", "")
    desc = " ".join([r.get("description", "") for r in c.get("career_history", [])])
    
    if matches_any(title, NON_TECH_KEYWORDS):
        return 0
    
    all_consulting = True
    if c.get("career_history"):
        for r in c.get("career_history", []):
            if not matches_any(r.get("company", ""), CONSULTING_COMPANIES):
                all_consulting = False
                break
    else:
        all_consulting = False
        
    if all_consulting and len(c.get("career_history", [])) > 0:
        return 0
        
    is_ml_title = matches_any(title, ML_TITLE_KEYWORDS)
    ml_desc_keywords = ["ranking", "retrieval", "recommendation", "search", "embedding", "vector", "llm", "fine-tuning"]
    has_core_ml_desc = matches_any(desc, ml_desc_keywords)
    
    exp = c.get("years_of_experience", 0)
    
    if is_ml_title and has_core_ml_desc and 4 <= exp <= 10:
        return 3
    elif is_ml_title or has_core_ml_desc:
        return 2
    else:
        tech_keywords = ["software", "backend", "data engineer", "frontend", "full stack", "developer"]
        if matches_any(title, tech_keywords) or matches_any(desc, tech_keywords):
            return 1
    
    return 0

def main():
    existing_labels = {}
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            existing_labels = json.load(f)

    candidates = load_candidates("../candidates.jsonl")
    sampled = stratified_sample(candidates, existing_labels)
    
    for c in sampled:
        print(auto_grade(c))
        
    print("q")

if __name__ == "__main__":
    main()
