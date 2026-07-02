
import re
from datetime import date, datetime
from src.matching import match_keyword, any_keyword_match, count_keyword_hits
from src.config import (
    AI_ML_TITLE_KEYWORDS,
    NON_TECHNICAL_TITLES,
    CV_SPEECH_ROBOTICS_TITLES,
    JUNIOR_TITLES,
    CAREER_DESCRIPTION_KEYWORDS,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    CONSULTING_FIRMS,
    RESEARCH_ONLY_TITLES,
    PREFERRED_CITIES,
    LOCATION_SCORES,
    BEHAVIORAL_WEIGHTS,
    OPEN_TO_WORK_SCORE,
    LAST_ACTIVE_THRESHOLDS,
    NOTICE_PERIOD_THRESHOLDS,
    INTERVIEW_COMPLETION_THRESHOLDS,
    GITHUB_ACTIVITY_THRESHOLDS,
    HONEYPOT_TIMELINE_RATIO,
    REFERENCE_DATE,
    EDUCATION_TIER_BONUS,
    RELEVANT_DEGREE_MULTIPLIER,
    RELEVANT_ASSESSMENT_KEYWORDS,
    SALARY_CONCERN_THRESHOLD_LPA,
    SALARY_RED_FLAG_THRESHOLD_LPA,
)


def score_career_titles(candidate: dict) -> float:
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    total_months = 0
    ml_months = 0
    has_current_ml_title = False

    current_title_lower = candidate.get("current_title", "").lower()

    for role in career:
        title_lower = role.get("title", "").lower()
        duration = role.get("duration_months", 0)
        total_months += duration

        is_ml_title = any_keyword_match(AI_ML_TITLE_KEYWORDS, title_lower)
        if is_ml_title:
            ml_months += duration

        if role.get("is_current", False) and is_ml_title:
            has_current_ml_title = True

    if any_keyword_match(AI_ML_TITLE_KEYWORDS, current_title_lower):
        has_current_ml_title = True

    if total_months == 0:
        fraction_ml = 0.0
    else:
        fraction_ml = ml_months / total_months

    score = fraction_ml * 0.7
    if has_current_ml_title:
        score += 0.3
    elif ml_months > 0:
        score += 0.10

    return min(score, 1.0)


def score_career_descriptions(candidate: dict) -> float:
    text = candidate.get("all_career_descriptions", "")
    text += " " + candidate.get("summary", "").lower()
    text += " " + candidate.get("headline", "").lower()

    total_weighted_score = 0.0
    max_possible = 0.0
    categories_hit = 0

    for category_name, category_data in CAREER_DESCRIPTION_KEYWORDS.items():
        weight = category_data["weight"]
        keywords = category_data["keywords"]
        max_possible += weight

        hits = count_keyword_hits(keywords, text)
        if hits > 0:
            categories_hit += 1
            if hits == 1:
                category_score = 0.30
            elif hits == 2:
                category_score = 0.50
            elif hits == 3:
                category_score = 0.65
            elif hits == 4:
                category_score = 0.80
            else:
                category_score = 1.0
            total_weighted_score += weight * category_score

    if max_possible == 0:
        return 0.0

    raw_score = total_weighted_score / max_possible

    breadth_bonus = min(categories_hit * 0.05, 0.15)

    return min(raw_score + breadth_bonus, 1.0)


def score_skills(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    skill_names_lower = candidate.get("skill_names_lower", set())

    skill_lookup = {}
    for s in skills:
        name_lower = s.get("name", "").lower()
        skill_lookup[name_lower] = s

    must_have_score = 0.0
    must_have_count = 0
    for target_skill in MUST_HAVE_SKILLS:
        matched = False
        for skill_name in skill_names_lower:
            if match_keyword(target_skill, skill_name) or match_keyword(skill_name, target_skill):
                matched = True
                detail = skill_lookup.get(skill_name, {})
                duration = detail.get("duration_months", 0)
                endorsements = detail.get("endorsements", 0)

                skill_pts = 0.5
                skill_pts += min(duration / 36, 1.0) * 0.3
                skill_pts += min(endorsements / 20, 1.0) * 0.2

                must_have_score += skill_pts
                must_have_count += 1
                break

    nice_to_have_count = 0
    for target_skill in NICE_TO_HAVE_SKILLS:
        for skill_name in skill_names_lower:
            if match_keyword(target_skill, skill_name) or match_keyword(skill_name, target_skill):
                nice_to_have_count += 1
                break

    must_have_normalized = min(must_have_score / 5.0, 0.80)

    nice_bonus = min(nice_to_have_count * 0.04, 0.20)

    return min(must_have_normalized + nice_bonus, 1.0)


def gate_anti_negative(candidate: dict) -> float:
    current_title_lower = candidate.get("current_title", "").lower()
    career = candidate.get("career_history", [])
    all_descriptions = candidate.get("all_career_descriptions", "")

    current_is_non_tech = any_keyword_match(NON_TECHNICAL_TITLES, current_title_lower)

    if current_is_non_tech:
        ml_evidence_keywords = []
        for cat in CAREER_DESCRIPTION_KEYWORDS.values():
            ml_evidence_keywords.extend(cat["keywords"][:8])
        has_ml_evidence = any_keyword_match(ml_evidence_keywords, all_descriptions)

        if not has_ml_evidence:
            return 0.0   

        non_tech_months = 0
        total_months = 0
        for role in career:
            title_lower = role.get("title", "").lower()
            duration = role.get("duration_months", 0)
            total_months += duration
            if any_keyword_match(NON_TECHNICAL_TITLES, title_lower):
                non_tech_months += duration

        if total_months > 0:
            non_tech_ratio = non_tech_months / total_months
            if non_tech_ratio > 0.80:
                return 0.10   
            elif non_tech_ratio > 0.50:
                return 0.25   
        return 0.15   

    is_cv_robotics = any_keyword_match(CV_SPEECH_ROBOTICS_TITLES, current_title_lower)

    if is_cv_robotics:
        nlp_ir_keywords = [
            "nlp", "natural language", "retrieval", "ranking", "search",
            "recommendation", "embedding", "text", "language model",
            "information retrieval", "semantic search",
        ]
        has_nlp_ir = any_keyword_match(nlp_ir_keywords, all_descriptions)
        if not has_nlp_ir:
            return 0.20   
        else:
            return 0.70   

    if career:
        consulting_months = 0
        total_months = 0
        current_is_consulting = False

        for role in career:
            company_lower = role.get("company", "").lower()
            industry_lower = role.get("industry", "").lower()
            duration = role.get("duration_months", 0)
            total_months += duration

            is_consulting = any_keyword_match(CONSULTING_FIRMS, company_lower)
            if is_consulting:
                consulting_months += duration
                if role.get("is_current", False):
                    current_is_consulting = True

        if total_months > 0:
            consulting_ratio = consulting_months / total_months
            if consulting_ratio >= 1.0:
                ml_rescue_kw = ["embedding", "retrieval", "ranking", "deployed",
                                "ml", "model", "recommendation", "search"]
                has_ml = any_keyword_match(ml_rescue_kw, all_descriptions)
                return 0.25 if has_ml else 0.10
            elif consulting_ratio >= 0.75:
                return 0.50 if not current_is_consulting else 0.35
            elif consulting_ratio >= 0.50:
                return 0.75 if not current_is_consulting else 0.60

    if career:
        all_research = all(
            any_keyword_match(RESEARCH_ONLY_TITLES, role.get("title", "").lower())
            for role in career
        )
        if all_research:
            production_keywords = ["deployed", "production", "scaled", "users",
                                   "api", "microservice", "inference"]
            has_production = any_keyword_match(production_keywords, all_descriptions)
            if not has_production:
                return 0.15   

    if any_keyword_match(JUNIOR_TITLES, current_title_lower):
        return 0.60   

    return 1.0   


def score_experience_band(candidate: dict) -> float:
    years = candidate.get("years_of_experience", 0)
    current_title = candidate.get("current_title", "").lower()

    if 5 <= years <= 9:
        return 1.00
    elif 4 <= years < 5 or 9 < years <= 12:
        return 0.85
    elif 3 <= years < 4:
        return 0.60
    elif 12 < years <= 15:
        return 0.70
    elif years < 3:
        return 0.30
    else:  
        management_keywords = ["manager", "director", "vp", "head of", "chief"]
        if any_keyword_match(management_keywords, current_title):
            return 0.35   
        return 0.50   


def score_location(candidate: dict) -> float:
    country = candidate.get("country", "").strip().lower()
    location = candidate.get("location", "").lower()
    willing = candidate.get("willing_to_relocate", False)

    is_india = country == "india"
    is_preferred_city = any_keyword_match(PREFERRED_CITIES, location)

    if is_india and is_preferred_city:
        return LOCATION_SCORES["india_preferred_city"]
    elif is_india and willing:
        return LOCATION_SCORES["india_other_relocate"]
    elif is_india:
        return LOCATION_SCORES["india_other_no_relocate"]
    elif willing:
        return LOCATION_SCORES["outside_india_relocate"]
    else:
        return LOCATION_SCORES["outside_india_no_relocate"]


def score_behavioral(candidate: dict, reference_date=None) -> float:
    if reference_date is None:
        reference_date = REFERENCE_DATE

    sub_scores = {}

    otw = candidate.get("open_to_work_flag", False)
    sub_scores["open_to_work"] = OPEN_TO_WORK_SCORE.get(otw, 0.40)

    last_active_str = candidate.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_ago = (reference_date - last_active).days
            recency_score = 0.20
            for threshold_days, threshold_score in LAST_ACTIVE_THRESHOLDS:
                if days_ago <= threshold_days:
                    recency_score = threshold_score
                    break
            sub_scores["last_active_recency"] = recency_score
        except (ValueError, TypeError):
            sub_scores["last_active_recency"] = 0.50
    else:
        sub_scores["last_active_recency"] = 0.50

    rr = candidate.get("recruiter_response_rate", 0.0)
    sub_scores["recruiter_response_rate"] = max(0.30, 0.30 + 0.70 * rr)

    notice = candidate.get("notice_period_days", 90)
    notice_score = 0.30
    for threshold_days, threshold_score in NOTICE_PERIOD_THRESHOLDS:
        if notice <= threshold_days:
            notice_score = threshold_score
            break
    sub_scores["notice_period"] = notice_score

    icr = candidate.get("interview_completion_rate", 0.0)
    ic_score = 0.50
    for threshold_val, threshold_score in INTERVIEW_COMPLETION_THRESHOLDS:
        if icr < threshold_val:
            ic_score = threshold_score
            break
    sub_scores["interview_completion"] = ic_score

    gh = candidate.get("github_activity_score", -1)
    if gh == -1:
        gh_score = 0.65
    elif gh <= 20:
        gh_score = 0.80
    elif gh <= 40:
        gh_score = 0.90
    else:
        gh_score = 1.00
    sub_scores["github_activity"] = gh_score

    extra_score = 0.50   

    saved = candidate.get("saved_by_recruiters_30d", 0)
    if saved >= 10:
        extra_score += 0.08
    elif saved >= 5:
        extra_score += 0.05
    elif saved >= 1:
        extra_score += 0.02

    completeness = candidate.get("profile_completeness_score", 0)
    if completeness >= 80:
        extra_score += 0.05
    elif completeness >= 60:
        extra_score += 0.02

    apps = candidate.get("applications_submitted_30d", 0)
    if apps > 0:
        extra_score += 0.03

    oar = candidate.get("offer_acceptance_rate", -1)
    if oar >= 0.5:
        extra_score += 0.04
    elif oar >= 0 and oar < 0.3:
        extra_score -= 0.03   

    verified_count = sum([
        candidate.get("verified_email", False),
        candidate.get("verified_phone", False),
        candidate.get("linkedin_connected", False),
    ])
    extra_score += verified_count * 0.02

    work_mode = candidate.get("preferred_work_mode", "").lower()
    if work_mode in ["hybrid", "onsite", "flexible"]:
        extra_score += 0.03
    elif work_mode == "remote":
        extra_score -= 0.02

    salary = candidate.get("expected_salary_range", {})
    if isinstance(salary, dict):
        max_salary = salary.get("max", 0)
        min_salary = salary.get("min", 0)
        effective_max = max(max_salary, min_salary) if max_salary and min_salary else max_salary
        if effective_max > SALARY_RED_FLAG_THRESHOLD_LPA:
            extra_score -= 0.05
        elif effective_max > SALARY_CONCERN_THRESHOLD_LPA:
            extra_score -= 0.02

    sub_scores["extra_signals"] = max(0.1, min(extra_score, 1.0))

    extra2 = 0.0

    views = candidate.get("profile_views_received_30d", 0)
    if views >= 20:
        extra2 += 0.05
    elif views >= 10:
        extra2 += 0.03
    elif views >= 3:
        extra2 += 0.01

    appearances = candidate.get("search_appearance_30d", 0)
    if appearances >= 100:
        extra2 += 0.04
    elif appearances >= 30:
        extra2 += 0.02

    connections = candidate.get("connection_count", 0)
    if connections >= 500:
        extra2 += 0.03
    elif connections >= 200:
        extra2 += 0.01

    endorsements = candidate.get("endorsements_received", 0)
    if endorsements >= 30:
        extra2 += 0.03
    elif endorsements >= 10:
        extra2 += 0.01

    signup_str = candidate.get("signup_date", "")
    if signup_str:
        try:
            signup = datetime.strptime(signup_str, "%Y-%m-%d").date()
            days_since_signup = (reference_date - signup).days
            if days_since_signup >= 365:       
                extra2 += 0.03
            elif days_since_signup >= 180:     
                extra2 += 0.01
        except (ValueError, TypeError):
            pass

    response_time = candidate.get("avg_response_time_hours", 0)
    if response_time > 0:
        if response_time <= 24:
            extra2 += 0.04     
        elif response_time <= 72:
            extra2 += 0.02     
        elif response_time >= 168:
            extra2 -= 0.02     

    sub_scores["extra_signals"] = max(0.1, min(
        sub_scores["extra_signals"] + extra2, 1.0
    ))

    total = 0.0
    for signal_name, weight in BEHAVIORAL_WEIGHTS.items():
        total += weight * sub_scores.get(signal_name, 0.50)

    return min(total, 1.0)


def detect_honeypot(candidate: dict) -> float:
    flags = 0

    for skill in candidate.get("skills", []):
        if (skill.get("proficiency", "").lower() == "expert" and
                skill.get("duration_months", 0) == 0):
            flags += 1
            break

    years_exp = candidate.get("years_of_experience", 0)
    total_career_months = candidate.get("total_career_months", 0)
    career_years_from_history = total_career_months / 12.0 if total_career_months > 0 else 0

    if career_years_from_history >= 3 and years_exp > 4.0 * career_years_from_history:
        flags += 1

    career = candidate.get("career_history", [])
    if len(career) >= 2:
        total_duration = sum(r.get("duration_months", 0) for r in career)
        if total_duration > 0 and years_exp > 0:
            ratio = total_duration / (years_exp * 12)
            if ratio > 3.5:
                flags += 1

    return 1.0 if flags >= 2 else 0.0


def score_education_assessment(candidate: dict) -> float:
    score = 0.50   

    tier_numeric = candidate.get("best_education_tier", 0)
    tier_bonus = EDUCATION_TIER_BONUS.get(tier_numeric, 0.0)

    has_relevant = candidate.get("has_relevant_degree", False)
    if has_relevant:
        tier_bonus = min(tier_bonus * RELEVANT_DEGREE_MULTIPLIER, 0.30)

    score += tier_bonus

    assessments = candidate.get("skill_assessment_scores", {})
    if assessments and isinstance(assessments, dict):
        relevant_scores = []
        for skill_name, skill_score in assessments.items():
            if any(kw in skill_name.lower() for kw in RELEVANT_ASSESSMENT_KEYWORDS):
                relevant_scores.append(skill_score)

        if relevant_scores:
            avg_score = sum(relevant_scores) / len(relevant_scores)
            score += min(avg_score / 100.0 * 0.20, 0.20)
        elif assessments:
            all_avg = sum(assessments.values()) / len(assessments)
            score += min(all_avg / 100.0 * 0.10, 0.10)

    return min(score, 1.0)


def score_semantic_tfidf(candidate: dict, tfidf_lookup: dict) -> float:
    if not tfidf_lookup:
        return 0.0
    cid = candidate.get("candidate_id", "")
    raw_sim = tfidf_lookup.get(cid, 0.0)
    return min(raw_sim / 0.30, 1.0)


def score_quantitative_evidence(candidate: dict) -> float:
    text = candidate.get("all_career_descriptions", "")
    if not text:
        return 0.10  

    patterns = [
        r'\d+\s*%',                                  
        r'\d+[kmb]\b',                                
        r'\d+\s*(?:ms|millisecond|second)',            
        r'p(?:50|90|95|99)\b',                         
        r'\d+x\s',                                     
        r'(?:million|thousand|billion)\s',              
        r'\d+\s*(?:user|request|quer|transaction)',     
        r'(?:reduced|improved|increased|decreased|optimized|boosted)\s+\w+\s+(?:by|from|to)\s+\d',
        r'\d+\s*(?:node|server|instance|replica|gpu)',  
        r'(?:dag|pipeline|workflow|job)s?\s+\w*\s*\d',  
    ]

    hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    if hits >= 5:
        return 1.0
    elif hits >= 4:
        return 0.85
    elif hits >= 3:
        return 0.70
    elif hits >= 2:
        return 0.55
    elif hits >= 1:
        return 0.40
    return 0.10   


def compute_all_features(candidate: dict, tfidf_lookup: dict = None, reference_date=None) -> dict:
    return {
        "candidate_id":       candidate["candidate_id"],
        "group_a_title":      score_career_titles(candidate),
        "group_b_desc":       score_career_descriptions(candidate),
        "group_c_skills":     score_skills(candidate),
        "group_d_gate":       gate_anti_negative(candidate),
        "group_e_experience": score_experience_band(candidate),
        "group_f_location":   score_location(candidate),
        "group_g_behavioral": score_behavioral(candidate, reference_date),
        "group_h_honeypot":   detect_honeypot(candidate),
        "group_i_education":  score_education_assessment(candidate),
        "group_j_semantic":   score_semantic_tfidf(candidate, tfidf_lookup or {}),
        "group_k_quantitative": score_quantitative_evidence(candidate),
    }
