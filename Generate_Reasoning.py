import re

TARGET_TITLES = [
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

CORE_KEYWORDS = [
    "vector search", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "learning to rank", "ndcg", "reranking",
    "re-ranking", "recommendation system", "embeddings", "sentence transformers"
]

TEMPLATE_SENTENCE_FRAGMENTS = [
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

SPECIFICITY_MARKERS = [
    r"\d+[mkb]\+?",           # 50M+, 10K+, 1B+
    r"\d+%",                   # 12%, 80%
    r"\d+\s*months",           # 9 months
    r"revenue",
    r"latency",
    r"throughput",
    r"a/b test",
    r"offline.online",
    r"offline experiment",
    r"feature pipeline",
    r"training pipeline",
    r"distilbert",
    r"gradient.boost",
    r"matrix factori",
    r"collaborative filter",
    r"hand-tuned",
    r"my main role",
    r"my primary",
    r"most of the work",
    r"most of my",
    r"the key challenge",
    r"improved .* by",
    r"reduced .* by",
    r"designed features",
    r"owned the offline",
    r"worked closely with pm",
    r"\d+\s*warehouse",
    r"three families",
]

def specificity_score(sentence):
    """Score how specific / unique a sentence is based on markers."""
    sl = sentence.lower()
    score = 0
    for pattern in SPECIFICITY_MARKERS:
        if re.search(pattern, sl):
            score += 1
    # Longer non-template sentences generally carry more information
    if len(sentence) > 80:
        score += 0.5
    return score

def get_most_relevant_job(history):
    """Return the career history entry most relevant to the JD."""
    if not history:
        return None
    best_job = history[0]
    best_score = -1
    for job in history:
        score = 0
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        for tar in TARGET_TITLES:
            if tar in title:
                score += 10
        for kw in CORE_KEYWORDS:
            if kw in desc:
                score += 2
        if score > best_score:
            best_score = score
            best_job = job
    return best_job

def is_template_sentence(sentence_lower):
    """Return True if the sentence matches a known template fragment."""
    for frag in TEMPLATE_SENTENCE_FRAGMENTS:
        if frag in sentence_lower:
            return True
    return False

def find_best_sentence(description, company):
    """
    From a job description, find the most specific, non-template sentence.
    Falls back to a constructed fact if nothing beats the templates.
    Returns a cleaned string starting with 'they' or a noun phrase.
    """
    if not description:
        return None

    sentences = [s.strip() for s in description.split(".") if s.strip() and len(s.strip()) > 25]
    if not sentences:
        return None

    # Score all sentences; prefer non-template ones
    best_non_template = None
    best_non_template_score = -1
    best_template = sentences[0]  # fallback

    for sent in sentences:
        sl = sent.lower()
        if is_template_sentence(sl):
            continue
        score = specificity_score(sent)
        if score > best_non_template_score:
            best_non_template_score = score
            best_non_template = sent

    chosen = best_non_template if best_non_template else best_template

    # Clean third-person pronouns
    chosen = chosen.replace("our ", "their ").replace(" our ", " their ")
    chosen = chosen.replace("my ", "their ").replace(" my ", " their ")
    chosen = chosen.replace("I ", "They ").replace(" I ", " they ")
    chosen = chosen.replace("We ", "They ").replace(" we ", " they ")

    # Normalise capitalisation of first word
    words = chosen.split()
    if words:
        first = words[0]
        if first not in ("They", "I", "We") and not first.isupper():
            if len(first) > 1 and first[0].isupper() and first[1].islower():
                words[0] = first[0].lower() + first[1:]
        chosen = " ".join(words)

    # Prefix company context if we fell back to a template sentence
    if best_non_template is None:
        return f"at {company}: {chosen}"
    return f"at {company}, {chosen}"

def get_relevant_skills_str(skills):
    """Return a formatted string of the top relevant verified skills."""
    relevant_found = []
    seen = set()
    all_targets = (CORE_KEYWORDS +
                   ["rag", "llms", "fine-tuning", "nlp", "pytorch", "xgboost",
                    "lightgbm", "vector search", "semantic search",
                    "information retrieval", "reranking"])
    for s in skills:
        s_name = s.get("name", "")
        s_lower = s_name.lower()
        if any(t in s_lower for t in all_targets) and s_lower not in seen:
            seen.add(s_lower)
            relevant_found.append(s_name)

    if len(relevant_found) >= 3:
        return f"skills in {relevant_found[0]}, {relevant_found[1]}, and {relevant_found[2]}"
    elif len(relevant_found) == 2:
        return f"skills in {relevant_found[0]} and {relevant_found[1]}"
    elif len(relevant_found) == 1:
        return f"skill in {relevant_found[0]}"
    else:
        return "strong foundational engineering background"
    
def get_rank_label(rank):
    """Return a rank-appropriate quality label."""
    if rank <= 10:
        return "A standout fit"
    elif rank <= 30:
        return "A strong match"
    elif rank <= 60:
        return "A solid candidate"
    elif rank <= 80:
        return "A qualified candidate"
    else:
        return "An adjacent candidate"
    
def capitalize_first(s):
    if not s:
        return s
    return s[0].upper() + s[1:]


def generate_reasoning(cand, rank):
    """
    Generates a high-quality, factual, evidence-first 1-2 sentence explanation.
    Four structural styles rotate by rank. No template phrases like 'Earning a top
    rank'. Accomplishment is extracted from the most specific (non-template)
    sentence in the candidate's most relevant job description.
    """
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    history = cand.get("career_history", [])
    signals = cand.get("redrob_signals", {})

    exp = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "their current company")
    loc = profile.get("location", "India")
    country = profile.get("country", "India")
    rrr = int(signals.get("recruiter_response_rate", 0) * 100)
    notice = signals.get("notice_period_days", 60)
    willing_relocate = signals.get("willing_to_relocate", False)
    open_to_work = signals.get("open_to_work_flag", False)
    saved = signals.get("saved_by_recruiters_30d", 0)

    # Accomplishment from most relevant job
    relevant_job = get_most_relevant_job(history)
    if relevant_job:
        job_comp = relevant_job.get("company", company)
        job_desc = relevant_job.get("description", "")
        accomplishment = find_best_sentence(job_desc, job_comp)
    else:
        accomplishment = None
    if not accomplishment:
        accomplishment = f"at {company}, currently serving as {title}"

    # Skills string
    skills_str = get_relevant_skills_str(skills)

    # Location string
    loc_clean = (loc or "India").strip()
    if country and country != "India":
        loc_clean = f"{loc_clean} ({country})"
    relocate_txt = ", open to relocation" if willing_relocate else ""

    # Notice and response rate
    if not notice or notice == 0:
        notice_txt = "immediately available"
    else:
        notice_txt = f"{notice}-day notice"
    rrr_str = f"{rrr}% recruiter response rate"

    # Tenure
    num_jobs = len(history)
    avg_tenure = exp / num_jobs if num_jobs > 0 else 0

    # A/An prefix for title
    an_starters = ["ai", "ml", "nlp", "applied", "associate", "information", "engineer"]
    prefix = "an" if any(title.lower().startswith(x) for x in an_starters) else "a"

    # Rank label (replaces "Earning a top rank")
    rank_label = get_rank_label(rank)

    # Open to work supplement
    otw_txt = " and actively open to new opportunities" if open_to_work else ""
    saved_txt = f"; bookmarked by {saved} recruiters recently" if saved >= 3 else ""

    # Concern flags for lower ranks
    concerns = []
    if notice and notice >= 90:
        concerns.append(f"{notice}-day notice")
    if country and country != "India" and not willing_relocate:
        concerns.append("international location without relocation intent")
    concern_str = ("; note: " + ", ".join(concerns)) if concerns else ""

    style_idx = rank % 4

    if style_idx == 0:
        # Style A: Accomplishment-first
        reasoning = (
            f"{rank_label}: {capitalize_first(accomplishment)}. "
            f"They bring {exp} years as {prefix} {title} and verified {skills_str}. "
            f"Located in {loc_clean}{relocate_txt}{otw_txt}, {notice_txt} ({rrr_str}{saved_txt}){concern_str}."
        )
    elif style_idx == 1:
        # Style B: Role + company first
        reasoning = (
            f"{capitalize_first(prefix)} {title} at {company} with {exp} years. "
            f"Demonstrated {skills_str}: {accomplishment}{concern_str}. "
            f"Based in {loc_clean}{relocate_txt}; {notice_txt}, {rrr_str}{saved_txt}{otw_txt}."
        )
    elif style_idx == 2:
        # Style C: Skills + trajectory first
        reasoning = (
            f"With {exp} years of experience and verified {skills_str}, "
            f"they show a stable career averaging {avg_tenure:.1f} years per role. "
            f"Notably {accomplishment}{concern_str}. "
            f"In {loc_clean}{relocate_txt}{otw_txt}; {notice_txt}, {rrr_str}{saved_txt}."
        )
    else:
        # Style D: JD-fit framing
        reasoning = (
            f"{rank_label} for this role: {exp} years as {prefix} {title}. "
            f"Verified {skills_str}. {capitalize_first(accomplishment)}{concern_str}. "
            f"{loc_clean}{relocate_txt}{otw_txt} — {notice_txt}, {rrr_str}{saved_txt}."
        )

    return reasoning
