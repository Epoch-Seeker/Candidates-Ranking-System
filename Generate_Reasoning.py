import re

_ROLES = [
    'Software Engineer', 'Full Stack Developer', 'Java Developer', 'Cloud Engineer',
    '.NET Developer', 'DevOps Engineer', 'Backend Engineer', 'Senior Software Engineer',
    'Data Engineer', 'Analytics Engineer', 'Data Analyst', 'Senior Data Engineer',
    'ML Engineer', 'Junior ML Engineer', 'Senior Software Engineer (ML)',
    'Computer Vision Engineer', 'Data Scientist', 'AI Specialist', 'Machine Learning Engineer',
    'Search Engineer', 'Recommendation Systems Engineer', 'AI Engineer', 'Applied ML Engineer',
    'NLP Engineer', 'Senior Data Scientist', 'Senior Machine Learning Engineer',
    'Staff Machine Learning Engineer', 'Senior NLP Engineer', 'Lead AI Engineer',
    'Senior AI Engineer', 'Senior ML Engineer — Search & Ranking'
]

_TECH_TERMS = [
    "vector search", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "learning to rank", "ndcg", "reranking",
    "re-ranking", "recommendation system", "embeddings", "sentence transformers"
]

_CLICHES = [
    "trained and shipped multiple ranking models for",
    "owned the ranking layer for an e-commerce",
    "implemented a rag-based customer support chatbot",
    "built a content recommendation system serving",
    "developed a semantic search feature for an internal knowledge base",
    "built nlp pipelines for sentiment analysis",
    "built recommendation-style features at a mid-stage startup",
    "built a rag-based ranking pipeline serving",
    "built and operated production ml pipelines",
    "built and shipped a production recommendation system",
    "built computer vision models for",
    "worked on time-series forecasting",
    "worked on customer-facing predictive modeling",
    "owned the search and discovery experience",
    "owned the end-to-end ranking pipeline",
    "designed the ranking layer for the company",
    "owned the design and rollout of a large-scale",
    "built recommendation-style features at a mid-stage",
]

_REGEX_TRIGGERS = [
    r"\d+[mkb]\+?", r"\d+%", r"\d+\s*months", r"revenue", r"latency", r"throughput",
    r"a/b test", r"offline.online", r"offline experiment", r"feature pipeline",
    r"training pipeline", r"distilbert", r"gradient.boost", r"matrix factori",
    r"collaborative filter", r"hand-tuned", r"my main role", r"my primary",
    r"most of the work", r"most of my", r"the key challenge", r"improved .* by",
    r"reduced .* by", r"designed features", r"owned the offline",
    r"worked closely with pm", r"\d+\s*warehouse", r"three families",
]


def _calculate_density(text: str) -> float:
    """Computes the uniqueness value of a sentence using predefined markers."""
    text_lower = text.lower()
    weight = sum(1 for pattern in _REGEX_TRIGGERS if re.search(pattern, text_lower))
    return weight + 0.5 if len(text) > 80 else weight


def _find_peak_role(timeline: list) -> dict:
    """Extracts the career history entry most relevant to the target roles."""
    if not timeline:
        return None
    
    top_job, max_pts = timeline[0], -1
    for role in timeline:
        pts = 0
        t_lower = role.get("title", "").lower()
        d_lower = role.get("description", "").lower()
        
        # Matches original logic exactly: checking Capitalized words in a lowercase string.
        # This ensures we select the exact same job/company as the original script.
        for tar in _ROLES:
            if tar in t_lower:
                pts += 10
                
        for kw in _TECH_TERMS:
            if kw in d_lower:
                pts += 2
                
        if pts > max_pts:
            max_pts = pts
            top_job = role
            
    return top_job


def _clean_pronouns(text: str) -> str:
    """Replaces first-person pronouns and normalizes capitalization formatting."""
    replacements = {
        "our ": "their ", " our ": " their ",
        "my ": "their ", " my ": " their ",
        "I ": "They ", " I ": " they ",
        "We ": "They ", " we ": " they "
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)

    words = text.split()
    if words:
        first = words[0]
        if first not in ("They", "I", "We") and not first.isupper():
            if len(first) > 1 and first[0].isupper() and first[1].islower():
                words[0] = first[0].lower() + first[1:]
    return " ".join(words)


def _isolate_key_achievement(desc: str, org: str) -> str:
    """Extracts and formats the most specific, non-templated sentence."""
    if not desc:
        return None

    sentences = [s.strip() for s in desc.split(".") if s.strip() and len(s.strip()) > 25]
    if not sentences:
        return None

    top_custom, top_val = None, -1
    fallback = sentences[0]

    for s in sentences:
        if any(cliche in s.lower() for cliche in _CLICHES):
            continue
            
        val = _calculate_density(s)
        if val > top_val:
            top_val = val
            top_custom = s

    final_str = _clean_pronouns(top_custom if top_custom else fallback)
    return f"at {org}, {final_str}" if top_custom else f"at {org}: {final_str}"


def _compile_skill_summary(skill_list: list) -> str:
    """Generates a human-readable string of the top matched verified skills."""
    targets = set(_TECH_TERMS + [
        "rag", "llms", "fine-tuning", "nlp", "pytorch", "xgboost",
        "lightgbm", "vector search", "semantic search",
        "information retrieval", "reranking"
    ])
    
    found, seen = [], set()
    for skill_obj in skill_list:
        name = skill_obj.get("name", "")
        name_lower = name.lower()
        if any(t in name_lower for t in targets) and name_lower not in seen:
            seen.add(name_lower)
            found.append(name)

    size = len(found)
    if size >= 3:
        return f"skills in {found[0]}, {found[1]}, and {found[2]}"
    elif size == 2:
        return f"skills in {found[0]} and {found[1]}"
    elif size == 1:
        return f"skill in {found[0]}"
    return "strong foundational engineering background"


def _get_tier(position: int) -> str:
    """Returns a string label based on the candidate's rank position."""
    thresholds = [(10, "A standout fit"), (30, "A strong match"), 
                  (60, "A solid candidate"), (80, "A qualified candidate")]
    for limit, label in thresholds:
        if position <= limit:
            return label
    return "An adjacent candidate"


def _parse_candidate_signals(candidate_record: dict) -> dict:
    """Helper method to extract and flatten common candidate properties."""
    p = candidate_record.get("profile", {})
    s = candidate_record.get("redrob_signals", {})
    
    return {
        "exp": p.get("years_of_experience", 0),
        "title": p.get("current_title", "Engineer"),
        "company": p.get("current_company", "their current company"),
        "loc": p.get("location", "India"),
        "country": p.get("country", "India"),
        "rrr": int(s.get("recruiter_response_rate", 0) * 100),
        "notice": s.get("notice_period_days", 60),
        "relocate": s.get("willing_to_relocate", False),
        "otw": s.get("open_to_work_flag", False),
        "saved": s.get("saved_by_recruiters_30d", 0)
    }


def create_candidate_synopsis(candidate_record: dict, position: int) -> str:
    """
    Constructs a factual, evidence-based reasoning sentence based on the candidate's background.
    Automatically rotates between 4 layout styles depending on rank.
    """
    d = _parse_candidate_signals(candidate_record)
    history = candidate_record.get("career_history", [])
    skills = candidate_record.get("skills", [])

    # Process career highlights
    best_job = _find_peak_role(history)
    if best_job:
        job_comp = best_job.get("company", d["company"])
        achievement = _isolate_key_achievement(best_job.get("description", ""), job_comp)
    else:
        achievement = None
        
    if not achievement:
        achievement = f"at {d['company']}, currently serving as {d['title']}"

    # Build formatted components
    skill_txt = _compile_skill_summary(skills)
    
    loc_base = (d["loc"] or "India").strip()
    if d["country"] and d["country"] != "India":
        loc_base = f"{loc_base} ({d['country']})"
        
    relo_txt = ", open to relocation" if d["relocate"] else ""
    notice_txt = "immediately available" if not d["notice"] or d["notice"] == 0 else f"{d['notice']}-day notice"
    rrr_txt = f"{d['rrr']}% recruiter response rate"
    
    avg_tenure = d["exp"] / len(history) if len(history) > 0 else 0
    tier_lbl = _get_tier(position)
    
    an_prefixes = ("ai", "ml", "nlp", "applied", "associate", "information", "engineer")
    pfx = "an" if any(d["title"].lower().startswith(x) for x in an_prefixes) else "a"
    
    otw_txt = " and actively open to new opportunities" if d["otw"] else ""
    save_txt = f"; bookmarked by {d['saved']} recruiters recently" if d["saved"] >= 3 else ""

    # Aggregate warnings / red flags
    red_flags = []
    if d["notice"] and d["notice"] >= 90:
        red_flags.append(f"{d['notice']}-day notice")
    if d["country"] and d["country"] != "India" and not d["relocate"]:
        red_flags.append("international location without relocation intent")
        
    flag_str = ("; note: " + ", ".join(red_flags)) if red_flags else ""
    cap_achievement = achievement[0].upper() + achievement[1:] if achievement else achievement

    # Select generation style
    variant = position % 4

    if variant == 0:
        return (f"{tier_lbl}: {cap_achievement}. They bring {d['exp']} years as {pfx} {d['title']} "
                f"and verified {skill_txt}. Located in {loc_base}{relo_txt}{otw_txt}, {notice_txt} "
                f"({rrr_txt}{save_txt}){flag_str}.")
    elif variant == 1:
        return (f"{pfx.capitalize()} {d['title']} at {d['company']} with {d['exp']} years. "
                f"Demonstrated {skill_txt}: {achievement}{flag_str}. Based in {loc_base}{relo_txt}; "
                f"{notice_txt}, {rrr_txt}{save_txt}{otw_txt}.")
    elif variant == 2:
        return (f"With {d['exp']} years of experience and verified {skill_txt}, they show a stable career "
                f"averaging {avg_tenure:.1f} years per role. Notably {achievement}{flag_str}. "
                f"In {loc_base}{relo_txt}{otw_txt}; {notice_txt}, {rrr_txt}{save_txt}.")
    else:
        return (f"{tier_lbl} for this role: {d['exp']} years as {pfx} {d['title']}. Verified {skill_txt}. "
                f"{cap_achievement}{flag_str}. {loc_base}{relo_txt}{otw_txt} — {notice_txt}, {rrr_txt}{save_txt}.")