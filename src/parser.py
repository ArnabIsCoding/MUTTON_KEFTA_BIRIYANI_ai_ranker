
import json
import re

def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'fine[\s-]?tuned\b', 'fine-tuning', text)
    text = re.sub(r'fine[\s-]?tunes?\b', 'fine-tuning', text)
    text = re.sub(r'\bfinetuning\b', 'fine-tuning', text)
    text = re.sub(r'\bfinetune\b', 'fine-tuning', text)
    text = re.sub(r'vector\s*db\b', 'vector database', text)
    text = re.sub(r'\bllms\b', 'llm', text)
    text = re.sub(r'recommendation\s*engine', 'recommendation system', text)
    return text


def load_candidates(file_path: str) -> list[dict]:
    candidates = []

    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

    return candidates


def parse_candidate(raw: dict) -> dict:
    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})

    parsed = {
        "candidate_id":        raw.get("candidate_id", ""),
        "headline":            profile.get("headline", ""),
        "summary":             profile.get("summary", ""),
        "location":            profile.get("location", ""),
        "country":             profile.get("country", ""),
        "years_of_experience": profile.get("years_of_experience", 0),
        "current_title":       profile.get("current_title", ""),
        "current_company":     profile.get("current_company", ""),
        "current_company_size": profile.get("current_company_size", ""),
        "current_industry":    profile.get("current_industry", ""),
    }

    career_history = raw.get("career_history", [])
    parsed["career_history"] = []
    for role in career_history:
        parsed["career_history"].append({
            "company":          role.get("company", ""),
            "title":            role.get("title", ""),
            "start_date":       role.get("start_date", ""),
            "end_date":         role.get("end_date"),
            "duration_months":  role.get("duration_months", 0),
            "is_current":       role.get("is_current", False),
            "industry":         role.get("industry", ""),
            "company_size":     role.get("company_size", ""),
            "description":      role.get("description", ""),
        })

    parsed["all_career_descriptions"] = normalize(" ".join(
        role.get("description", "") for role in career_history
    ))

    parsed["all_career_titles"] = " | ".join(
        role.get("title", "") for role in career_history
    ).lower()

    parsed["total_career_months"] = sum(
        role.get("duration_months", 0) for role in career_history
    )

    skills_data = raw.get("skills", [])
    parsed["skills"] = []
    for skill in skills_data:
        parsed["skills"].append({
            "name":             skill.get("name", ""),
            "proficiency":      skill.get("proficiency", "beginner"),
            "endorsements":     skill.get("endorsements", 0),
            "duration_months":  skill.get("duration_months", 0),
        })

    parsed["skill_names_lower"] = set(
        s.get("name", "").lower() for s in skills_data
    )

    education = raw.get("education", [])
    parsed["education"] = []
    for edu in education:
        parsed["education"].append({
            "institution":    edu.get("institution", ""),
            "degree":         edu.get("degree", ""),
            "field_of_study": edu.get("field_of_study", ""),
            "tier":           edu.get("tier", "unknown"),
        })

    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    parsed["best_education_tier"] = max(
        (tier_rank.get(e.get("tier", "unknown"), 0) for e in education),
        default=0,
    )

    cs_fields = {"computer science", "cs", "computer engineering",
                 "artificial intelligence", "machine learning",
                 "data science", "information technology", "it",
                 "electronics", "ece", "electrical engineering",
                 "mathematics", "statistics", "applied mathematics"}
    parsed["has_relevant_degree"] = any(
        any(kw in edu.get("field_of_study", "").lower() for kw in cs_fields)
        for edu in education
    )

    parsed["profile_completeness_score"] = signals.get("profile_completeness_score", 0)
    parsed["signup_date"]                = signals.get("signup_date", "")
    parsed["last_active_date"]           = signals.get("last_active_date", "")
    parsed["open_to_work_flag"]          = signals.get("open_to_work_flag", False)
    parsed["profile_views_received_30d"] = signals.get("profile_views_received_30d", 0)
    parsed["applications_submitted_30d"] = signals.get("applications_submitted_30d", 0)
    parsed["recruiter_response_rate"]    = signals.get("recruiter_response_rate", 0.0)
    parsed["avg_response_time_hours"]    = signals.get("avg_response_time_hours", 0)
    parsed["skill_assessment_scores"]    = signals.get("skill_assessment_scores", {})
    parsed["connection_count"]           = signals.get("connection_count", 0)
    parsed["endorsements_received"]      = signals.get("endorsements_received", 0)
    parsed["notice_period_days"]         = signals.get("notice_period_days", 90)
    parsed["expected_salary_range"]      = signals.get("expected_salary_range_inr_lpa", {})
    parsed["preferred_work_mode"]        = signals.get("preferred_work_mode", "")
    parsed["willing_to_relocate"]        = signals.get("willing_to_relocate", False)
    parsed["github_activity_score"]      = signals.get("github_activity_score", -1)
    parsed["search_appearance_30d"]      = signals.get("search_appearance_30d", 0)
    parsed["saved_by_recruiters_30d"]    = signals.get("saved_by_recruiters_30d", 0)
    parsed["interview_completion_rate"]  = signals.get("interview_completion_rate", 0.0)
    parsed["offer_acceptance_rate"]      = signals.get("offer_acceptance_rate", -1)
    parsed["verified_email"]             = signals.get("verified_email", False)
    parsed["verified_phone"]             = signals.get("verified_phone", False)
    parsed["linkedin_connected"]         = signals.get("linkedin_connected", False)

    return parsed


def parse_all_candidates(file_path: str) -> list[dict]:
    raw_candidates = load_candidates(file_path)
    return [parse_candidate(c) for c in raw_candidates]
