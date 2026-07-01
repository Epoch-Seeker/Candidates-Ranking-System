import argparse
import pandas as pd
import numpy as np
import re
import json
import csv
import gzip
import os
from sentence_transformers import SentenceTransformer, util

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
    # 1. Explode career history
    df_exploded = df_new.explode('career_history')
    df_exploded['extracted_industry'] = df_exploded['career_history'].apply(
        lambda x: x.get('industry') if isinstance(x, dict) else None
    )
    unwanted_industries = {"IT Services", "Consulting", "Manufacturing"}
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

def run_scoring_engine(df_input):
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
    
    def calculate_skills_average(row_dict):
        if not isinstance(row_dict, dict) or 'skill_assessment_scores' not in row_dict:
            return 0.3  
        scores = row_dict.get('skill_assessment_scores', {})
        if not scores or not isinstance(scores, dict): return 0.3
        target_keys = ['NLP', 'Fine-tuning LLMs', 'Image Classification', 'Speech Recognition']
        found_scores = [float(scores[k]) for k in target_keys if k in scores and pd.notna(scores[k])]
        return np.mean(found_scores) / 100.0 if found_scores else 0.3

    df['skills_score'] = df['redrob_signals'].apply(calculate_skills_average)
    
    trust_matrix = signals.str['verified_email'].astype(int) + signals.str['verified_phone'].astype(int) + signals.str['linkedin_connected'].astype(int)
    df['trust_score'] = (trust_matrix / 3.0).clip(0.0, 1.0)

    min_expectation = signals.apply(lambda x: x.get('expected_salary_range_inr_lpa', {}).get('min', 0.0) if isinstance(x, dict) else 0.0).fillna(0.0)
    df['salary_alignment_score'] = np.where(min_expectation > 30.0, 0.0, 1.0)

    # PART 3: HYBRID ENGINE MATRIX
    behavioral_weights = {
        "skills_score": 0.25, "github_score": 0.15, "interview_attendance_score": 0.15,
        "recruiter_response_score": 0.10, "response_speed_score": 0.10, "market_demand_score": 0.08,
        "profile_completeness_score_scaled": 0.07, "offer_conversion_score": 0.04,
        "social_proof_score": 0.03, "trust_score": 0.03
    }
    
    beh_cols = list(behavioral_weights.keys())
    df['behavioral_merit_score'] = df[beh_cols].mul(behavioral_weights).sum(axis=1)
    df['logistics_match_score'] = (df['location_score'] + df['salary_alignment_score']) / 2.0
    df['final_fit_score'] = df['logistics_match_score'] * df['behavioral_merit_score']
    
    return df.sort_values(by='final_fit_score', ascending=False)


# ==========================================
# 2. NLP & HONEYPOT LOGIC (Our Code)
# ==========================================

print("Loading Semantic Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

IDEAL_JD_QUERY = """
Senior AI Engineer building production intelligence layers. 
Deep technical depth in modern ML systems: embeddings, retrieval, ranking algorithms, 
recommendation systems, vector databases, and hybrid search infrastructure. 
Strong Python skills and scrappy product-engineering attitude. 
Experience designing evaluation frameworks for ranking systems (NDCG, offline-to-online A/B testing).
Shipped end-to-end ML models to real users at scale.
"""
IDEAL_VECTOR = model.encode(IDEAL_JD_QUERY, convert_to_tensor=True)

def is_honeypot(candidate):
    signals = candidate.get('redrob_signals', {})
    if signals.get('signup_date') and signals.get('last_active_date') and signals.get('last_active_date') < signals.get('signup_date'):
        return True
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') in ['expert', 'advanced'] and skill.get('duration_months', 0) == 0:
            return True
    for job in candidate.get('career_history', []):
        if job.get('start_date') and job.get('end_date') and job.get('start_date') > job.get('end_date'):
            return True
    salary = signals.get('expected_salary_range_inr_lpa', {})
    if salary.get('min', 0) > salary.get('max', 0):
        return True
    return False

def extract_candidate_text(candidate):
    profile = candidate.get('profile', {})
    text_parts = [profile.get('headline', ''), profile.get('summary', '')]
    for job in candidate.get('career_history', []):
        text_parts.append(job.get('description', ''))
    return " ".join([t for t in text_parts if t])

def generate_reasoning(candidate, score):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    
    # 1. Base requirements: Title and YOE
    title = profile.get('current_title', 'Engineer')
    yoe = profile.get('years_of_experience', 0.0)
    
    # 2. Extract Specific AI Skills
    all_skills = [s.get('name') for s in candidate.get('skills', [])]
    ai_roots = [
        'ranking', 'retrieval', 'recommend', 'search', 'hybrid search', 're-ranking', 'bm25', 'learning-to-rank', 'ltr',
        'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss', 'vector', 
        'embed', 'sentence-transformers', 'bge', 'e5', 'openai', 'llm', 'fine-tuning', 'lora', 'qlora', 'peft', 'generative',
        'ndcg', 'mrr', 'map', 'a/b test', 'evaluation', 'python', 'xgboost', 'neural'
    ]
    
    core_ai_skills = []
    for skill in all_skills:
        skill_name = str(skill)
        if any(root in skill_name.lower() for root in ai_roots):
            if skill_name not in core_ai_skills:
                core_ai_skills.append(skill_name)
                
    num_ai_skills = len(core_ai_skills)

    # 3. Identify the Candidate's "Superpowers" (Top Scoring Drivers)
    drivers = []
    
    # Technical Driver
    if num_ai_skills > 0:
        # Highlight up to 2 specific skills so it proves we actually read their profile
        skill_preview = ", ".join(core_ai_skills[:2])
        drivers.append(f"{num_ai_skills} JD-aligned core skills (e.g., {skill_preview})")
        
    # Logistics Driver
    loc = str(profile.get('location', '')).lower()
    if 'pune' in loc or 'noida' in loc:
        drivers.append("local Tier-1 Hub availability")
        
    # Behavioral Drivers
    if signals.get('github_activity_score', -1) >= 60:
        drivers.append("strong open-source/GitHub activity")
        
    if signals.get('recruiter_response_rate', 0.0) >= 0.8:
        drivers.append("excellent recruiter response rate")
        
    if signals.get('interview_completion_rate', 0.0) >= 0.85:
        drivers.append("proven interview reliability")
        
    if signals.get('saved_by_recruiters_30d', 0) >= 10:
        drivers.append("high market demand")

    # 4. Construct the Dynamic Sentence
    base = f"{title} ({yoe} YOE)."
    
    if drivers:
        # Take their top 2 or 3 strongest points to keep the sentence punchy
        top_reasons = drivers[:3]
        if len(top_reasons) > 1:
            reasons_str = ", ".join(top_reasons[:-1]) + ", and " + top_reasons[-1]
        else:
            reasons_str = top_reasons[0]
            
        reasoning = f"{base} Selected for {reasons_str}."
    else:
        # Fallback for candidates with good semantic text but average behavioral signals
        reasoning = f"{base} Selected for strong semantic alignment with the ranking/retrieval requirements."
        
    # Clean up any accidental double spaces
    return " ".join(reasoning.split())

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
    print("Columns before filtering:", df_data.columns.tolist())
    # print(df_data.head(1))
    
    print("2. Applying aggressive Pandas Hard Filters...")
    df_filtered = filter_candidates(df_data)
    print("Columns after filtering:", df_filtered.columns.tolist())

    print("3. Calculating Logistics & Behavioral scores...")
    df_ranked = run_scoring_engine(df_filtered)
    
    # Take the top 1000 from the Pandas filter to be safe
    df_top_pool = df_ranked.head(1000)
    
    # Convert back to standard Python dicts for NLP and Honeypot checks
    candidates_list = df_top_pool.to_dict('records')
    
    final_pool = []
    print(f"4. Running NLP Semantic Search on top {len(candidates_list)} candidates...")
    for candidate in candidates_list:
        # Catch any honeypots the Pandas script missed
        if is_honeypot(candidate):
            continue
            
        candidate_text = extract_candidate_text(candidate)
        candidate_vector = model.encode(candidate_text, convert_to_tensor=True)
        cosine_score = util.cos_sim(IDEAL_VECTOR, candidate_vector).item()
        semantic_score = max(0.0, cosine_score)
        
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
            
    print(f"✅ Pipeline Complete! Output written to {args.out}")