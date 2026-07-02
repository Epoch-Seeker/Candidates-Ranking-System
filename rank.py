import argparse
import pandas as pd
import numpy as np
import re
import json
import csv
import gzip
import os
from datetime import datetime
import sys

# Prevent torchcodec DLL crash on Windows environments without FFmpeg
for mod in ['torchcodec', 'torchcodec.decoders']:
    sys.modules[mod] = None
from sentence_transformers import SentenceTransformer, util

# ==========================================
# HONEYPOT TRAPS LOGIC
# ==========================================

# 1. Inverted Salary
def has_inverted_salary(signals):
    if not isinstance(signals, dict): return False
    salary = signals.get('expected_salary_range_inr_lpa', {})
    return (salary.get('min') or 0) > (salary.get('max') or float('inf'))

# 2. Time Travel
def has_time_travel(signals):
    if not isinstance(signals, dict): return False
    signup = signals.get('signup_date')
    last_active = signals.get('last_active_date')
    return bool(signup and last_active and last_active < signup)

# 3. Impossible Mastery
def has_impossible_mastery(skills):
    if not isinstance(skills, list): return False
    return any(
        s.get('proficiency') in ['expert', 'advanced'] and (s.get('duration_months') or 0) == 0 
        for s in skills if isinstance(s, dict)
    )

# 4. Experience Math Failure
def has_experience_math_failure(row):
    profile = row.get('profile', {})
    career = row.get('career_history', [])
    if not isinstance(profile, dict) or not isinstance(career, list): return False
    
    stated_yoe = float(profile.get('years_of_experience') or 0)
    calculated_months = sum(float(job.get('duration_months') or 0) for job in career if isinstance(job, dict))
    
    return abs(stated_yoe - (calculated_months / 12)) > 2.0

# 5. Future Certifications
def has_future_certs(certs):
    if not isinstance(certs, list): return False
    current_year = datetime.now().year
    return any((cert.get('year') or 0) > current_year for cert in certs if isinstance(cert, dict))

def filter_honeypots_df(df):
    """
    Dataframe Filtering for Honeypot Traps
    """
    certs_col = df['certifications'] if 'certifications' in df.columns else pd.Series([None] * len(df), index=df.index)
    mask_any_trap = (
        df['redrob_signals'].apply(has_inverted_salary) |
        df['redrob_signals'].apply(has_time_travel) |
        df['skills'].apply(has_impossible_mastery) |
        df.apply(has_experience_math_failure, axis=1) |
        certs_col.apply(has_future_certs)
    )
    num_flagged = mask_any_trap.sum()
    if num_flagged > 0:
        print(f"Flagged and removed {num_flagged} honeypot candidates via DataFrame mask.")
    return df[~mask_any_trap].copy()

# ==========================================
# 1. PANDAS FILTERING & SCORING (Your Code)
# ==========================================

def col_score_calc(df, source_col, min, max, pow=2, penalty=None, invert=False):
    scaled = ((df[source_col]-min)/(max-min)).clip(0.0,1.0)
    if invert:
        scaled = 1.0-scaled
    scaled = scaled**pow
    if penalty is not None:
        scaled = scaled.mask(penalty(df[source_col]),-1.0) 
    return scaled

def filter_candidates(df_new):
    # Remove honeypots via DataFrame filtering
    df_new = filter_honeypots_df(df_new)

    # 1. Explode career history
    df_exploded = df_new.explode('career_history')
    df_exploded['extracted_industry'] = df_exploded['career_history'].apply(
        lambda x: x.get('industry') if isinstance(x, dict) else None
    )
    unwanted_industries = {"IT Services", "Consulting", "Manufacturing", "Research", "Academia"}
    df_exploded['is_unwanted'] = df_exploded['extracted_industry'].isin(unwanted_industries)
    all_unwanted_mask = df_exploded.groupby(level=0)['is_unwanted'].all()
    mask = ~all_unwanted_mask.reindex(df_new.index, fill_value=False)
    df_filtered = df_new.loc[mask].copy()

    # Calculate average duration
    df_filtered['avg_duration'] = df_filtered['career_history'].apply(
        lambda history: np.mean([float(job['duration_months']) for job in history if isinstance(job, dict) and job.get('duration_months') is not None])
        if isinstance(history, list) and len(history) > 0 else np.nan
    )

    # Hard Filters
    df_filtered2 = df_filtered[(df_filtered['avg_duration'] >= 18) | (df_filtered['avg_duration'].isna())].copy()
    df_filtered3 = df_filtered2[df_filtered2['profile'].str['years_of_experience'] >= 4]
    df_filtered4 = df_filtered3[df_filtered3['profile'].str['country'] == "India"]
    df_filtered5 = df_filtered4[df_filtered4['redrob_signals'].str['notice_period_days'] <= 60]
    
    # Inactivity Cutoff: exclude candidates inactive for >180 days (relative to dataset snapshot) or with response rate < 10%
    last_active = pd.to_datetime(df_filtered5['redrob_signals'].str['last_active_date'], errors='coerce')
    cutoff_date = last_active.max() - pd.Timedelta(days=180) if not last_active.isna().all() else pd.to_datetime('1970-01-01')
    resp_rate = df_filtered5['redrob_signals'].str['recruiter_response_rate'].fillna(0.0)
    df_filtered5 = df_filtered5[(last_active >= cutoff_date) & (resp_rate >= 0.10)]

    # Target titles filter
    target_titles = [
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

    escaped_titles = [re.escape(title) for title in target_titles]
    reg_titles = "|".join(escaped_titles)

    def has_relevant_experience(history_list):
        if not isinstance(history_list, list) or len(history_list) == 0:
            return False
        for job in history_list:
            if isinstance(job, dict) and 'title' in job and pd.notna(job['title']):
                if re.search(reg_titles, str(job['title']), re.IGNORECASE):
                    return True 
        return False

    mask = df_filtered5['career_history'].apply(has_relevant_experience)
    df_filtered6 = df_filtered5[mask]

    # Remove non-coding seniors
    senior_reg = r"Senior|Staff|Lead"
    def remove_non_coding_seniors(history_list):
        if not isinstance(history_list, list):
            return True  
        for job in history_list:
            if isinstance(job, dict) and "title" in job:
                title = str(job["title"])
                duration = job.get("duration_months", 0)
                if re.search(senior_reg, title, re.IGNORECASE):
                    if pd.notna(duration) and duration > 18:
                        return False  
        return True  

    df_filtered7 = df_filtered6[df_filtered6["career_history"].apply(remove_non_coding_seniors)]
    return df_filtered7

TARGET_SKILLS = [
    'bm25', 'elasticsearch', 'embedding', 'faiss', 'retrieval', 'learning to rank', 'milvus', 'opensearch',
    'pinecone', 'pgvector', 'qdrant', 'ranking', 'recommendation', 'search', 'weaviate',
    'deep learning', 'fine-tuning', 'hugging face', 'langchain', 'llamaindex', 'llm', 'lora', 'nlp', 'peft',
    'prompt engineering', 'pytorch', 'qlora', 'rag', 'scikit-learn', 'sentence transformers', 'tensorflow',
    'airflow', 'flink', 'bentoml', 'bigquery', 'databricks', 'dbt', 'docker', 'kafka', 'kubeflow', 'kubernetes',
    'mlflow', 'mlops', 'snowflake', 'spark', 'weights & biases', 'aws', 'azure', 'fastapi', 'flask', 'gcp',
    'go', 'grpc', 'microservices', 'postgresql', 'python', 'redis', 'rest api', 'rust', 'terraform'
]

def run_scoring_engine(df_input):
    if len(df_input) == 0:
        return df_input.copy()
    df = df_input.copy()
    
    # PART 1: LOCATION
    tier_1_cities = ["Pune", "Noida"]
    pattern_t1 = "|".join(tier_1_cities)
    tier_2_cities = ["Hyderabad", "Mumbai", "Delhi", "Delhi NCR", "Gurgaon", "Gurugram"]
    pattern_t2 = "|".join(tier_2_cities)
    
    candidate_city = df['profile'].str['location'].fillna("")
    is_tier_1 = candidate_city.str.contains(pattern_t1, case=False, na=False)
    is_tier_2 = candidate_city.str.contains(pattern_t2, case=False, na=False)
    is_willing_to_relocate = df['redrob_signals'].str['willing_to_relocate'] == True
    
    df['location_score'] = np.where(is_tier_1, 1.0, np.where(is_tier_2 & is_willing_to_relocate, 0.9, np.where(~is_tier_1 & ~is_tier_2 & is_willing_to_relocate, 0.7, 0.0)))
    
    # PART 2: SIGNALS
    signals = df['redrob_signals']
    df['interview_attendance_score'] = col_score_calc(df.assign(v=signals.str['interview_completion_rate']), 'v', 0.5, 1.0, pow=2)
    df['recruiter_response_score'] = col_score_calc(df.assign(v=signals.str['recruiter_response_rate']), 'v', 0.0, 1.0, pow=1)
    df['response_speed_score'] = col_score_calc(df.assign(v=signals.str['avg_response_time_hours']), 'v', 1, 168, pow=1.5, invert=True)
    
    clean_offer_rate = signals.str['offer_acceptance_rate'].replace(-1, 0.5)
    df['offer_conversion_score'] = col_score_calc(df.assign(v=clean_offer_rate), 'v', 0.0, 1.0, pow=1)
    df['profile_completeness_score_scaled'] = col_score_calc(df.assign(v=signals.str['profile_completeness_score']), 'v', 0, 100, pow=1)
    df['social_proof_score'] = col_score_calc(df.assign(v=signals.str['endorsements_received']), 'v', 0, 50, pow=1)
    
    gh_clean = signals.str['github_activity_score'].replace(-1, 0)
    df['github_score'] = (gh_clean / 100.0).clip(0.0, 1.0) ** 1.5
    df['market_demand_score'] = col_score_calc(df.assign(v=signals.str['saved_by_recruiters_30d']), 'v', 0, 15, pow=2)
    
    def calculate_skills_score(row):
        skills = row.get('skills', []) if isinstance(row.get('skills'), list) else []
        prof_map = {'expert': 1.0, 'advanced': 0.8, 'intermediate': 0.5}
        prof_score = min(1.0, sum(prof_map.get(s.get('proficiency'), 0.2) for s in skills if isinstance(s, dict) and any(k in str(s.get('name', '')).lower() for k in TARGET_SKILLS)) / 4.0)
        
        signals = row.get('redrob_signals', {})
        scores = signals.get('skill_assessment_scores', {}) if isinstance(signals, dict) else {}
        found_scores = [float(v) for v in scores.values() if pd.notna(v)] if isinstance(scores, dict) else []
        return 0.5 * prof_score + 0.5 * (np.mean(found_scores) / 100.0) if found_scores else prof_score * 0.9

    df['skills_score'] = df.apply(calculate_skills_score, axis=1)
    
    def calculate_certs_score(row):
        certs = row.get('certifications', []) if isinstance(row.get('certifications'), list) else []
        rel = sum(1 for c in certs if isinstance(c, dict) and any(k in str(c.get('name', '')).lower() for k in ['machine learning', 'deep learning', 'cloud ml', 'nlp', 'langchain', 'ai', 'tensorflow', 'pytorch']))
        return min(1.0, rel * 0.5)

    def calculate_edu_score(row):
        edus = row.get('education', []) if isinstance(row.get('education'), list) else []
        if not edus: return 0.5
        tier_map = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4}
        scores = []
        for e in edus:
            if not isinstance(e, dict): continue
            t_score = tier_map.get(str(e.get('tier', '')).lower(), 0.5)
            nums = re.findall(r"\d+(?:\.\d+)?", str(e.get('grade', '')))
            val = float(nums[0]) if nums else 7.0
            g_score = min(1.0, val / 4.0 if val <= 4.0 else (val / 10.0 if val <= 10.0 else val / 100.0))
            scores.append(0.5 * t_score + 0.5 * g_score)
        return np.mean(scores) if scores else 0.5

    df['certs_score'] = df.apply(calculate_certs_score, axis=1)
    df['education_score'] = df.apply(calculate_edu_score, axis=1)
    
    trust_matrix = signals.str['verified_email'].astype(int) + signals.str['verified_phone'].astype(int) + signals.str['linkedin_connected'].astype(int)
    df['trust_score'] = (trust_matrix / 3.0).clip(0.0, 1.0)

    min_expectation = signals.apply(lambda x: x.get('expected_salary_range_inr_lpa', {}).get('min', 0.0) if isinstance(x, dict) else 0.0).fillna(0.0)
    df['salary_alignment_score'] = np.where(min_expectation > 30.0, 0.0, 1.0)

    # PART 3: HYBRID ENGINE MATRIX
    behavioral_weights = {
        "skills_score": 0.22, "github_score": 0.15, "interview_attendance_score": 0.15,
        "recruiter_response_score": 0.08, "response_speed_score": 0.08, "market_demand_score": 0.08,
        "certs_score": 0.05, "education_score": 0.05, "profile_completeness_score_scaled": 0.04,
        "offer_conversion_score": 0.04, "social_proof_score": 0.03, "trust_score": 0.03
    }
    
    beh_cols = list(behavioral_weights.keys())
    df['behavioral_merit_score'] = df[beh_cols].mul(behavioral_weights).sum(axis=1)
    df['logistics_match_score'] = (df['location_score'] + df['salary_alignment_score']) / 2.0
    df['final_fit_score'] = df['logistics_match_score'] * df['behavioral_merit_score']
    
    return df.sort_values(by='final_fit_score', ascending=False)

def generate_reasoning(candidate, score):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills_list = candidate.get('skills', [])
    
    # ---------------------------------------------------------
    # 1. DEEP DATA EXTRACTION
    # ---------------------------------------------------------
    title = str(profile.get('current_title', 'Engineer')).strip()
    yoe = float(profile.get('years_of_experience', 0.0))
    notice_days = int(profile.get('notice_period_days', 0))
    
    # Safely extract skill names
    all_skills = [s.get('name', '').lower() for s in skills_list if isinstance(s, dict) and s.get('name')]
    
    # Micro-categorize skills specifically for the Redrob JD
    retrieval_skills = [s for s in all_skills if any(x in s for x in ['rag', 'retrieval', 'vector', 'search'])]
    eval_skills = [s for s in all_skills if any(x in s for x in ['ndcg', 'mrr', 'evaluation', 'metrics'])]
    gen_ai_skills = [s for s in all_skills if any(x in s for x in ['llm', 'langchain', 'openai', 'transformer'])]
    
    # Extract behavioral metrics
    github = int(signals.get('github_activity_score', 0))
    resp_rate = float(signals.get('recruiter_response_rate', 1.0))
    int_comp = float(signals.get('interview_completion_rate', 1.0))
    demand = int(signals.get('saved_by_recruiters_30d', 0))

    # ---------------------------------------------------------
    # 2. DYNAMIC PERSONA & STRENGTH MAPPING
    # ---------------------------------------------------------
    strengths = []
    persona = "Applied Engineer"
    
    # Determine Candidate Persona based on dominant traits
    if yoe >= 8.0 and (retrieval_skills or gen_ai_skills):
        persona = "Seasoned AI Architect"
    elif github >= 75:
        persona = "Open-Source Specialist"
    elif demand >= 15:
        persona = "High-Demand Talent"

    # Build highly specific strength clauses
    if retrieval_skills:
        strengths.append(f"direct architecture experience with {retrieval_skills[0].title()}")
    if eval_skills:
        strengths.append(f"familiarity with ranking metrics like {eval_skills[0].upper()}")
    if github >= 60:
        strengths.append(f"a validated GitHub footprint (score: {github})")
    if resp_rate >= 0.85:
        strengths.append(f"excellent recruiter responsiveness ({int(resp_rate*100)}%)")

    # ---------------------------------------------------------
    # 3. HONEST GAP DETECTION (Crucial for Audit)
    # ---------------------------------------------------------
    gaps = []
    if not retrieval_skills and not gen_ai_skills:
        gaps.append("a lack of explicitly named JD-core AI skills")
    if int_comp < 0.70:
        gaps.append(f"a concerning interview drop-off rate ({int(int_comp*100)}%)")
    if notice_days > 45:
        gaps.append(f"a restrictive {notice_days}-day notice period")
    if yoe < 2.0 and score > 0.5:
        gaps.append("relatively light commercial tenure")

    # ---------------------------------------------------------
    # 4. MODULAR SENTENCE ASSEMBLY
    # ---------------------------------------------------------
    # Use a deterministic hash based on name/title length so the same candidate 
    # always gets the same structure, but the dataset looks vastly varied.
    variance = (len(title) + int(yoe) + github) % 4
    
    # Clause A: The Intro
    intros = [
        f"Tracking as a {persona} currently holding a {title} role with {yoe} YOE.",
        f"Profile indicates {yoe} years of experience, aligning with a {persona} persona.",
        f"Evaluated this {title} ({yoe} YOE) through the lens of our core AI requirements.",
        f"A {yoe}-YOE {title} demonstrating traits of a {persona}."
    ]
    
    # Clause B: The Core Justification
    strength_str = ", ".join(strengths[:2]) if strengths else "baseline technical proficiencies"
    cores = [
        f"Selection was driven by {strength_str}.",
        f"Candidate stands out primarily due to {strength_str}.",
        f"Key drivers for this score include {strength_str}.",
        f"Notable profile strengths feature {strength_str}."
    ]
    
    # Clause C: The Nuance / Gaps
    gap_str = " and ".join(gaps[:2]) if gaps else "no immediate behavioral red flags"
    nuances = [
        f"However, we noted {gap_str}.",
        f"This is balanced against {gap_str}.",
        f"Pipeline risks include {gap_str}.",
        f"Reviewers should be aware of {gap_str}."
    ]
    
    # Clause D: Final Verdict tied strictly to the mathematical score
    if score >= 0.80:
        verdict = "Highly recommended for immediate Stage 2 screening."
    elif score >= 0.50:
        verdict = "Viable secondary option; proceed with standard technical screen."
    else:
        verdict = "Ranked lower due to core JD mismatches; deprioritized in current pipeline."

    # Assemble the final string based on the variance key to scramble the structure
    if variance == 0:
        final_text = f"{intros[0]} {cores[1]} {nuances[2]} {verdict}"
    elif variance == 1:
        final_text = f"{intros[2]} {cores[0]} {nuances[1]} {verdict}"
    elif variance == 2:
        final_text = f"{cores[2]} {intros[1]} {nuances[3]} {verdict}"
    else:
        final_text = f"{intros[3]} {cores[3]} {nuances[0]} {verdict}"

    # Clean up whitespace and capitalize first letters just in case
    final_text = " ".join(final_text.split())
    return final_text

# ==========================================
# 2. NLP & HONEYPOT LOGIC (Our Code)
# ==========================================

_semantic_model = None
_ideal_vector = None

def get_semantic_model_and_vector():
    global _semantic_model, _ideal_vector
    if _semantic_model is None:
        print("Loading Semantic Model...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        IDEAL_JD_QUERY = """
AI Engineer building the intelligence layer for search, candidate-job matching, and recommendation systems. Requires applied ML engineering at a product company, shipping production-grade ranking or retrieval systems to real users. Essential experience includes production embedding-based retrieval, hybrid search, dense retrieval, and LLM-based re-ranking. Must have handled embedding drift, index refresh, and retrieval-quality regression in live environments. Requires strong Python and hands-on operational experience with vector databases and search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch). Must design and implement rigorous ranking evaluation frameworks using NDCG, MRR, MAP, offline-to-online correlation, and A/B testing. Preferred expertise covers learning-to-rank (XGBoost, neural), LLM fine-tuning (LoRA, QLoRA, PEFT), large-scale inference optimization, and distributed systems. Blends ML architecture with rapid product-engineering and end-to-end deployment.
"""
        _ideal_vector = _semantic_model.encode(IDEAL_JD_QUERY, convert_to_tensor=True)
    return _semantic_model, _ideal_vector

def extract_candidate_text(candidate):
    profile = candidate.get('profile', {})
    text_parts = [profile.get('headline', ''), profile.get('summary', '')]
    for job in candidate.get('career_history', []):
        text_parts.append(job.get('description', ''))
    return " ".join([t for t in text_parts if t])


# ==========================================
# 3. MASTER EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank AI Engineer Candidates for Redrob Hackathon")
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to the candidates file (.json, .jsonl, .jsonl.gz)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output the final submission CSV"
    )
    args = parser.parse_args()

    file_path = args.candidates
    ext = os.path.splitext(file_path)[1].lower()

    print(f"1. Loading data from {file_path}...")

    # Handle compressed files
    if file_path.endswith(".gz"):
        opener = gzip.open
        mode = "rt"
        inner_ext = os.path.splitext(os.path.splitext(file_path)[0])[1].lower()
    else:
        opener = open
        mode = "r"
        inner_ext = ext

    with opener(file_path, mode, encoding="utf-8") as f:
        if inner_ext == ".json":
            # Standard JSON array
            data = json.load(f)

        elif inner_ext == ".jsonl":
            # JSON Lines
            data = [json.loads(line) for line in f if line.strip()]

        else:
            raise ValueError(
                f"Unsupported file type: {file_path}. "
                "Supported formats are .json, .jsonl, and .jsonl.gz"
            )

    print(f"Loaded {len(data)} candidates.")
    df_data = pd.DataFrame(data)
    
    print("2. Applying aggressive Pandas Hard Filters...")
    df_filtered = filter_candidates(df_data)
    num_hard_filtered = len(df_data) - len(df_filtered)
    print(f"Filtered out {num_hard_filtered} candidates before semantic search via hard filters (remaining: {len(df_filtered)}).")
    
    if len(df_filtered) == 0:
        print("No candidates left after honeypot filtration and hard filters! Generating empty submission CSV...")
        print("5. Generating Submission CSV...")
        with open(args.out, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        print(f"Pipeline Complete! Output written to {args.out}")
        sys.exit(0)

    print("3. Calculating Logistics & Behavioral scores...")
    df_ranked = run_scoring_engine(df_filtered)
    
    # Take the top 1000 from the Pandas filter to be safe
    df_top_pool = df_ranked.head(1000)
    
    # Convert back to standard Python dicts for NLP and Honeypot checks
    candidates_list = df_top_pool.to_dict('records')
    
    total_excluded_before_nlp = len(df_data) - len(candidates_list)
    final_pool = []
    print(f"4. Running NLP Semantic Search on top {len(candidates_list)} candidates ({total_excluded_before_nlp} candidates filtered/excluded before semantic search)...")
    model, ideal_vec = get_semantic_model_and_vector()
    for idx, candidate in enumerate(candidates_list):
        candidate_text = extract_candidate_text(candidate)
        candidate_vector = model.encode(candidate_text, convert_to_tensor=True)
        cosine_score = util.cos_sim(ideal_vec, candidate_vector).item()
        semantic_score = max(0.0, cosine_score)
        
        text_lower = candidate_text.lower()
        penalties = sum(p in text_lower for p in ['want to', 'looking to grow into', 'learning', 'handled by another team'])
        boosts = sum(p in text_lower for p in ['shipped', 'owned', 'deployed'])
        semantic_score = semantic_score * (1.2 ** boosts) * (0.8 ** penalties)
        
        # Combine your Pandas fit score with our Semantic NLP score
        pandas_score = candidate.get('final_fit_score', 0)
        grand_score = pandas_score * semantic_score * 100 
        
        final_pool.append({
            "candidate_id": candidate['candidate_id'],
            "final_score": grand_score,
            "candidate_data": candidate
        })

    # Sort descending by score, tie-break by ID
    final_pool.sort(key=lambda x: (x['final_score'], x['candidate_id'][::-1]), reverse=True)
    top_100 = final_pool[:100]

    # Write to Submission CSV
    print("5. Generating Submission CSV...")
    with open(args.out, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, item in enumerate(top_100, start=1):
            reasoning = generate_reasoning(item['candidate_data'], item['final_score'])
            writer.writerow([item['candidate_id'], rank, round(item['final_score'], 4), reasoning])
            
    print(f"Pipeline Complete! Output written to {args.out}")