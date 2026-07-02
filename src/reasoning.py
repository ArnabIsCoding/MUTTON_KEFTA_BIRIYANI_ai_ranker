
from src.matching import match_keyword


_EVIDENCE_TERMS = [
    "FAISS", "Elasticsearch", "Pinecone", "Qdrant", "Milvus", "Weaviate",
    "Chroma", "BM25", "HNSW",
    "embeddings", "vector search", "vector database", "semantic search",
    "hybrid search", "retrieval", "ranking", "re-ranking", "reranking",
    "recommendation", "recommender", "search engine",
    "NDCG", "MRR", "MAP", "A/B test", "precision@", "recall@",
    "learning to rank", "LTR",
    "deployed", "deployment", "production", "inference", "model serving",
    "scaled", "latency", "p99", "real-time",
    "LLM", "RAG", "fine-tuning", "finetuning", "LoRA", "QLoRA", "PEFT",
    "transformer", "BERT", "GPT", "T5", "LLaMA",
    "Hugging Face", "HuggingFace", "LangChain", "LlamaIndex",
    "sentence transformer", "embedding model",
    "Python", "FastAPI", "PyTorch", "TensorFlow", "scikit-learn",
    "Spark", "Kubernetes", "Docker", "MLOps",
]

_TIER1_OPENERS = [
    "Strong candidate —",
    "Top-tier match —",
    "Excellent fit —",
    "Highly relevant profile —",
    "Outstanding alignment —",
    "Compelling background —",
    "Standout candidate —",
]

_TIER2_OPENERS = [
    "Solid candidate with",
    "Good match —",
    "Relevant background —",
    "Credible profile with",
    "Capable candidate —",
    "Decent alignment —",
]

_TIER3_OPENERS = [
    "Partial match —",
    "Some relevance —",
    "Mixed profile —",
    "Has adjacent experience —",
    "Moderate fit —",
    "Qualified in some areas —",
]

_TIER4_OPENERS = [
    "Included for availability signals and some skill overlap — career evidence is thin.",
    "Borderline inclusion —",
    "Marginal fit —",
    "Weak alignment —",
    "Limited relevance —",
    "Included primarily for skill-list overlap —",
]

KNOWN_PRODUCT_COMPANIES = {
    'zomato': 'food-tech with search & recommendation',
    'swiggy': 'food-tech with ranking & logistics',
    'flipkart': 'e-commerce with search & ranking',
    'amazon': 'e-commerce with search, ranking & recommendation',
    'google': 'search & ML infrastructure',
    'microsoft': 'enterprise AI & search',
    'netflix': 'recommendation & personalization',
    'meta': 'ranking, recommendation & NLP',
    'apple': 'on-device ML & Siri NLP',
    'razorpay': 'fintech with ML fraud detection',
    'paytm': 'fintech with recommendation',
    'cred': 'fintech with personalization',
    'linkedin': 'professional search & ranking',
    'uber': 'marketplace matching & pricing ML',
    'ola': 'ride-hailing with matching & pricing ML',
    'dream11': 'fantasy sports with recommendation',
    'meesho': 'social commerce with search & ranking',
    'unacademy': 'ed-tech with recommendation',
    'phonepe': 'fintech with ML',
    'freshworks': 'SaaS with AI features',
    'salesforce': 'CRM with AI & search',
    'adobe': 'creative tools with ML',
    'sarvam ai': 'AI-native NLP company',
    'observe.ai': 'conversational AI',
    'yellow.ai': 'conversational AI',
    'haptik': 'conversational AI & NLP',
    'nykaa': 'e-commerce with recommendation',
}


def _pick_opener(openers: list[str], candidate: dict) -> str:
    cid = candidate.get("candidate_id", "")
    years = candidate.get("years_of_experience", 0)
    n_skills = len(candidate.get("skills", []))
    h = sum(ord(c) for c in str(cid)) + int(years * 7) + n_skills * 3
    return openers[h % len(openers)]


def _variant_hash(candidate: dict, n_variants: int) -> int:
    cid = str(candidate.get("candidate_id", ""))
    years = candidate.get("years_of_experience", 0)
    return (sum(ord(c) for c in cid) + int(years * 7)) % n_variants


def generate_reasoning(candidate: dict, features: dict, rank: int) -> str:
    if rank <= 10:
        text = _tier1_reasoning(candidate, features, rank)
    elif rank <= 40:
        text = _tier2_reasoning(candidate, features, rank)
    elif rank <= 75:
        text = _tier3_reasoning(candidate, features, rank)
    else:
        text = _tier4_reasoning(candidate, features, rank)

    concerns = _build_concerns(candidate, features, rank)
    if concerns:
        label = "Note" if rank <= 10 else "Concerns"
        concern_str = f" {label}: " + "; ".join(concerns) + "."
        text += concern_str

    return _truncate_at_sentence(text, 450)




def _tier1_reasoning(candidate: dict, features: dict, rank: int) -> str:
    opener = _pick_opener(_TIER1_OPENERS, candidate)
    title = candidate.get("current_title", "Unknown")
    company = candidate.get("current_company", "")
    years = candidate.get("years_of_experience", 0)

    best = _find_best_career_role(candidate)
    evidence = _extract_specific_evidence(
        best.get("description", "") if best else ""
    )
    skills = _get_relevant_skills(candidate)
    skill_str = ", ".join(skills[:5]) if skills else "relevant ML skills"

    beh = []
    if candidate.get("open_to_work_flag", False):
        beh.append("open to work")
    gh = candidate.get("github_activity_score", -1)
    if gh > 30:
        beh.append(f"GitHub score {gh:.0f}")
    rr = candidate.get("recruiter_response_rate", 0)
    if rr >= 0.6:
        beh.append(f"{rr:.0%} response rate")
    beh_str = "Behavioral: " + ", ".join(beh) + "." if beh else ""

    variant = _variant_hash(candidate, 3)

    if variant == 2:
        company_key = company.strip().lower() if company else ""
        company_context = KNOWN_PRODUCT_COMPANIES.get(company_key)
        if company_context and best and evidence:
            parts = [
                opener,
                f"{company} ({company_context}) — {title}, {years:.0f} yrs total experience,"
                f" brings direct domain experience.",
                f"Career at {best['company']} references"
                f" {', '.join(evidence[:4])}.",
                f"Skills: {skill_str}.",
            ]
            return " ".join(parts)
        variant = 0

    if variant == 1:
        parts = [opener]
        if best and evidence:
            parts.append(
                f"Career descriptions reference {', '.join(evidence[:4])}"
                f" — strongest evidence comes from {best['title']} role"
                f" at {best['company']}."
            )
            parts.append(
                f"Currently {title} at {company} with {years:.0f} yrs total experience"
                f" and {skill_str}."
            )
        elif best:
            parts.append(
                f"Career as {best['title']} at {best['company']}"
                f" shows relevant engineering depth."
            )
            parts.append(
                f"Currently {title} at {company} with {years:.0f} yrs total experience"
                f" and {skill_str}."
            )
        else:
            parts.append(
                f"{title} at {company}, {years:.0f} yrs total experience."
                f" Strong title alignment. Skills: {skill_str}."
            )
        if beh_str:
            parts.append(beh_str)
        return " ".join(parts)

    parts = [opener]
    if best and evidence:
        parts.append(
            f"{title} at {company}, {years:.0f} yrs total experience."
            f" Career includes {best['title']} at {best['company']}"
            f" with descriptions referencing {', '.join(evidence[:4])}."
        )
    elif best:
        parts.append(
            f"{title} at {company}, {years:.0f} yrs total experience."
            f" Career as {best['title']} at {best['company']}"
            f" shows relevant engineering depth."
        )
    else:
        parts.append(
            f"{title} at {company}, {years:.0f} yrs total experience."
            f" Strong title alignment for this role."
        )
    if skills:
        parts.append(f"Key skills: {skill_str}.")
    if beh_str:
        parts.append(beh_str)
    return " ".join(parts)


def _tier2_reasoning(candidate: dict, features: dict, rank: int) -> str:
    opener = _pick_opener(_TIER2_OPENERS, candidate)
    title = candidate.get("current_title", "Unknown")
    company = candidate.get("current_company", "")
    years = candidate.get("years_of_experience", 0)

    best = _find_best_career_role(candidate)
    evidence = _extract_specific_evidence(
        best.get("description", "") if best else ""
    )
    desc_score = features.get("group_b_desc", 0)
    skills = _get_relevant_skills(candidate)
    skill_str = ", ".join(skills[:4]) if skills else "some relevant skills"
    evidence_str = ", ".join(evidence[:3]) if evidence else "general ML work"

    limits = []
    if desc_score < 0.4:
        limits.append("weaker evidence of retrieval/ranking work in career descriptions")
    skill_score = features.get("group_c_skills", 0)
    if skill_score < 0.4:
        limits.append("limited skill corroboration for core JD requirements")
    exp_score = features.get("group_e_experience", 0)
    if exp_score < 0.7:
        limits.append(f"experience band ({years:.0f} yrs) outside ideal range")
    loc_score = features.get("group_f_location", 0)
    if loc_score < 0.6:
        loc = candidate.get("location", candidate.get("country", ""))
        limits.append(f"location ({loc}) not in preferred cities")
    if not limits:
        weakest_dim = None
        weakest_gap = 0
        dim_labels = {
            "group_b_desc": "career description evidence depth",
            "group_c_skills": "breadth of corroborated must-have skills",
            "group_g_behavioral": "behavioral availability signals",
            "group_f_location": "location proximity to preferred cities",
            "group_e_experience": "experience band alignment",
        }
        for dim_key, dim_label in dim_labels.items():
            val = features.get(dim_key, 0)
            gap = 1.0 - val
            if gap > weakest_gap:
                weakest_gap = gap
                weakest_dim = dim_label
        if weakest_dim:
            limits.append(f"comparatively lower {weakest_dim} vs top-ranked candidates")
        else:
            limits.append("aggregate score across all dimensions placed below top 10")
    limits_str = "; ".join(limits[:2])

    variant = _variant_hash(candidate, 3)

    if variant == 1:
        parts = [
            opener,
            f"Profiles strongly on {skill_str}, backed by {evidence_str}"
            f" in career at {company}, {years:.0f} yrs total experience.",
            f"Placed at rank {rank} due to {limits_str}.",
        ]
        return " ".join(parts)

    if variant == 2:
        parts = [
            opener,
            f"Despite relevant {evidence_str} evidence from {company} tenure,"
            f" placed here because {limits_str}.",
            f"Key strengths: {skill_str}.",
        ]
        return " ".join(parts)

    parts = [opener]
    if best and evidence:
        parts.append(
            f"{title} at {company}, {years:.0f} yrs total experience."
            f" Career references {evidence_str}."
        )
    else:
        parts.append(f"{title} at {company}, {years:.0f} yrs total experience.")
    if skills:
        parts.append(f"Skills include {skill_str}.")
    parts.append("Ranked here because: " + limits_str + ".")
    return " ".join(parts)


def _tier3_reasoning(candidate: dict, features: dict, rank: int) -> str:
    opener = _pick_opener(_TIER3_OPENERS, candidate)
    title = candidate.get("current_title", "Unknown")
    company = candidate.get("current_company", "")
    years = candidate.get("years_of_experience", 0)

    best = _find_best_career_role(candidate)
    evidence = _extract_specific_evidence(
        best.get("description", "") if best else ""
    )
    desc_score = features.get("group_b_desc", 0)
    skills = _get_relevant_skills(candidate)
    skill_str = ", ".join(skills[:3]) if skills else "no directly relevant ML/retrieval skills"

    gaps = []
    if desc_score < 0.2:
        gaps.append("no production deployment evidence")
    skill_score = features.get("group_c_skills", 0)
    if skill_score < 0.3:
        gaps.append("weak skill overlap with JD")
    gate = features.get("group_d_gate", 1)
    if gate < 0.5:
        gaps.append("current role pattern suggests non-core-ML background")

    missing = []
    if desc_score < 0.3:
        missing.append("hands-on retrieval/ranking evidence")
    if skill_score < 0.3:
        missing.append("core skill coverage (vector search, embeddings, LLM)")
    if gate < 0.5:
        missing.append("product-company ML background")
    if not missing:
        missing.append("depth of domain-specific experience")

    if evidence:
        positive_note = f"mentions of {', '.join(evidence[:2])}"
    elif desc_score >= 0.15:
        positive_note = "some ML experience"
    else:
        positive_note = "a relevant skill list"

    variant = _variant_hash(candidate, 2)

    if variant == 1:
        parts = [
            opener,
            f"{title} at {company}, {years:.0f} yrs total experience.",
            f"Has {positive_note}, but missing {'; '.join(missing[:2])}"
            f" that the JD requires.",
            f"Listed skills: {skill_str}.",
        ]
        return " ".join(parts)

    parts = [opener]
    parts.append(f"{title} at {company}, {years:.0f} yrs total experience.")
    if evidence:
        parts.append(
            f"Descriptions mention {', '.join(evidence[:2])}"
            f" but limited retrieval/ranking depth."
        )
    elif desc_score >= 0.15:
        parts.append(
            "Some relevant keywords in descriptions."
        )
    else:
        parts.append(
            "Missing evidence in job descriptions."
        )
    if skills:
        parts.append(f"Lists {skill_str}.")
    else:
        parts.append("No directly relevant ML/retrieval skills listed.")
    if gaps:
        parts.append("Gaps: " + "; ".join(gaps) + ".")
    return " ".join(parts)


def _tier4_reasoning(candidate: dict, features: dict, rank: int) -> str:
    opener = _pick_opener(_TIER4_OPENERS, candidate)
    title = candidate.get("current_title", "Unknown")
    company = candidate.get("current_company", "")
    years = candidate.get("years_of_experience", 0)

    desc_score = features.get("group_b_desc", 0)
    skill_score = features.get("group_c_skills", 0)

    weaknesses = []
    if desc_score < 0.15:
        weaknesses.append(
        )
    elif desc_score < 0.3:
        weaknesses.append(
        )
    if skill_score < 0.2:
        weaknesses.append("very few skills match core JD requirements")
    gate = features.get("group_d_gate", 1)
    if gate < 0.5:
        weaknesses.append("role/company pattern suggests consulting or non-ML background")
    exp_score = features.get("group_e_experience", 0)
    if exp_score < 0.5:
        weaknesses.append(f"experience ({years:.0f} yrs) is far from the 5-9yr sweet spot")

    skills = _get_relevant_skills(candidate)
    main_weakness = weaknesses[0] if weaknesses else "no standout signals across scoring dimensions"
    any_positive = (
        f"listing {', '.join(skills[:2])}"
        if skills
        else "minimal skill-list overlap"
    )

    variant = _variant_hash(candidate, 2)

    if variant == 1:
        parts = [
            opener,
            f"{title} at {company}, {years:.0f} yrs total experience.",
            f"Bottom-line: {main_weakness}.",
            f"Included at rank {rank} for {any_positive}.",
        ]
        return " ".join(parts)

    parts = [opener]
    parts.append(f"{title} at {company}, {years:.0f} yrs total experience.")
    if weaknesses:
        parts.append("Weaknesses: " + "; ".join(weaknesses[:3]) + ".")
    else:
        parts.append(
        )
    if skills:
        parts.append(f"Does list {', '.join(skills[:2])}.")
    return " ".join(parts)




def _find_best_career_role(candidate: dict) -> dict | None:
    career = candidate.get("career_history", [])
    if not career:
        return None

    relevance_keywords = [
        "embedding", "retrieval", "ranking", "recommendation",
        "search", "nlp", "machine learning", "deployed", "production",
        "llm", "rag", "fine-tuning", "transformer", "vector",
        "model", "inference", "pipeline", "FAISS", "Elasticsearch",
    ]

    best_role = None
    best_score = -1

    for role in career:
        desc = role.get("description", "")
        title = role.get("title", "")
        score = sum(1 for kw in relevance_keywords if match_keyword(kw, desc))
        score += sum(
            0.5 for kw in relevance_keywords if match_keyword(kw, title)
        )
        if role.get("is_current", False):
            score += 2
        score += min(role.get("duration_months", 0) / 24, 2)

        if score > best_score:
            best_score = score
            best_role = role

    return best_role if best_score > 0 else (career[0] if career else None)


def _extract_specific_evidence(description: str) -> list[str]:
    if not description:
        return []

    found = []
    for term in _EVIDENCE_TERMS:
        if match_keyword(term, description):
            found.append(term)
        if len(found) >= 6:
            break
    return found


def _get_relevant_skills(candidate: dict) -> list[str]:
    relevant_keywords = [
        "python", "embedding", "vector", "ranking", "retrieval",
        "recommendation", "nlp", "machine learning", "deep learning",
        "llm", "fine-tuning", "transformer", "bert", "gpt",
        "rag", "search", "tensorflow", "pytorch", "keras",
        "faiss", "elasticsearch", "pinecone", "qdrant",
        "scikit", "sklearn", "spark", "data science",
        "fastapi", "flask", "docker", "kubernetes",
        "lora", "peft", "huggingface",
    ]

    skills = candidate.get("skills", [])
    result = []
    for skill in skills:
        name = skill.get("name", "")
        name_lower = name.lower()
        if any(kw in name_lower for kw in relevant_keywords):
            duration = skill.get("duration_months", 0)
            if duration >= 24:
                result.append(f"{name} ({duration // 12}yr)")
            elif duration >= 12:
                result.append(f"{name} ({duration}mo)")
            else:
                result.append(name)

    return result


def _build_concerns(candidate: dict, features: dict, rank: int) -> list[str]:
    concerns = []

    notice = candidate.get("notice_period_days", 90)
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")
    elif notice > 60:
        concerns.append(f"notice period is {notice} days")

    rr = candidate.get("recruiter_response_rate", 0)
    if rr < 0.2:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    elif rr < 0.4 and rank > 20:
        concerns.append(f"moderate recruiter response ({rr:.0%})")

    loc_score = features.get("group_f_location", 0)
    if loc_score < 0.5:
        country = candidate.get("country", "")
        location = candidate.get("location", "")
        willing = candidate.get("willing_to_relocate", False)
        if not willing and country.lower() != "india":
            concerns.append(
                f"located in {location or country}, not willing to relocate"
            )
        elif not willing:
            concerns.append(
                f"in {location}, not willing to relocate to preferred cities"
            )
        elif rank > 30:
            concerns.append(f"location ({location or country}) not ideal")

    gate = features.get("group_d_gate", 1)
    honeypot = features.get("group_h_honeypot", 0)
    if honeypot > 0:
        concerns.append("flagged as possible honeypot (impossible skill/timeline data)")
    if gate <= 0.10:
        concerns.append("career history is entirely at consulting/IT services firms")
    elif gate < 0.5 and rank > 15:
        concerns.append("current role pattern is non-technical for this position")

    if rank > 40:
        desc_score = features.get("group_b_desc", 0)
        if desc_score < 0.15:
            concerns.append("no career evidence of ML/retrieval work")

    if rank <= 10:
        return concerns[:2]
    elif rank <= 40:
        return concerns[:3]
    else:
        return concerns[:4]


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    last_excl = truncated.rfind("! ")
    last_q = truncated.rfind("? ")
    if truncated.endswith(".") or truncated.endswith("!") or truncated.endswith("?"):
        return truncated

    best = max(last_period, last_excl, last_q)
    if best > max_chars * 0.5:  
        return text[: best + 1]

    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return text[:last_space] + "..."

    return truncated + "..."
