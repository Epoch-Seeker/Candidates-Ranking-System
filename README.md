# RedRob Intelligent Candidate Discovery & Ranking Engine

This repository implements an end-to-end AI recruitment and ranking pipeline designed specifically for the **Senior AI Engineer (Founding Team)** role at Redrob AI. 

Instead of simple keyword matching—which falls into common recruitment traps—this engine evaluates candidates across **data integrity (honeypot detection)**, **strict disqualifiers (hard filters)**, **multi-dimensional behavioral merit (scoring matrix)**, and **semantic execution depth (NLP search with phrasing modifiers)**.

> **🌐 Live Sandbox Demo:** Try out the ranking engine interactively on our hosted Streamlit application: [https://candidates-ranking-system.streamlit.app/](https://candidates-ranking-system.streamlit.app/)

---

## 1. Honeypot & Trap Detection Methods

Synthetic datasets and real-world applicant pools often contain adversarial or corrupted profiles designed to trick automated keyword scrapers. Before candidate ranking begins, our pipeline runs vectorized Pandas masks to detect and eliminate five distinct honeypot traps:

| Trap Name | Detection Logic | Why It's Eliminated |
| :--- | :--- | :--- |
| **Inverted Salary** | `min_salary > max_salary` | Catches corrupted data payloads or adversarial inputs where salary range bounds are flipped. |
| **Time Travel** | `last_active_date < signup_date` | Catches logical impossibilities where a candidate's last activity chronologically precedes their account creation. |
| **Impossible Mastery** | `proficiency == 'expert'/'advanced'` with `duration_months == 0` | Eliminates keyword-stuffers claiming expert/advanced mastery in complex technical tools without a single month of actual usage. |
| **Experience Math Failure** | `abs(stated_yoe - sum(job_durations)/12) > 2.0` | Identifies deceptive profiles where total claimed Years of Experience (YOE) diverges significantly (>2 years) from their cumulative career history. |
| **Future Certifications** | `cert.year > current_year` | Filters out fabricated credentials claiming completion dates in the future. |

> **Execution Result:** Flagged and removed **24,966** honeypot candidates via vectorized DataFrame masks.

---

## 2. Aggressive Hard Filters (Pandas)

As stated in the job description: *"We'd rather see 10 great matches than 1000 maybes."* Before scoring, we apply strict disqualification rules implemented in `filter_candidates()` derived directly from the JD's non-negotiable requirements:

| Hard Filter | Filter Condition in Code | JD Reasoning |
| :--- | :--- | :--- |
| **Unwanted Industries Disqualifier** | Excludes candidates whose entire career history is solely in `{"IT Services", "Consulting", "Manufacturing", "Research", "Academia"}`. | *"People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career... If you've spent your career in pure research environments without any production deployment — we will not move forward."* |
| **Job Hopper / Tenure Cutoff** | Excludes candidates whose average job tenure across roles is `< 18 months` (`avg_duration >= 18` or `isna()`). | *"Title-chasers... switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years."* |
| **Experience Cutoff** | Excludes candidates with total experience `< 4 years` (`years_of_experience >= 4`). | *"Experience Required: 5–9 years... Some people hit 'senior engineer' judgment at 4 years... We'll seriously consider candidates outside the band if other signals are strong."* |
| **Country Cutoff** | Excludes candidates outside India (`country == "India"`). | *"Outside India: case-by-case, but we don't sponsor work visas."* |
| **Notice Period Cutoff** | Excludes candidates with a notice period `> 60 days` (`notice_period_days <= 60`). | *"We'd love sub-30-day notice... 30+ day notice candidates are still in scope but the bar gets higher."* |
| **Target Engineering Titles** | Excludes candidates who have never held a relevant Software, ML, AI, Backend, or Data Engineering title in their career history. | *"A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit, no matter how perfect their skill list looks."* |
| **Non-Coding Seniors Disqualifier** | Excludes candidates whose career history has titles matching `'Senior,Staff,Lead'` with a duration `> 18 months`. | *"If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into 'architecture' or 'tech lead' roles — we will probably not move forward. This role writes code."* |
| **Inactivity & Ghoster Cutoff** | Excludes candidates inactive for `> 180 days` (relative to dataset snapshot) or with a recruiter response rate `< 10%`. | *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."* |

> **Execution Result:** Filtered out **97,533** candidates before semantic search via hard filters (leaving **2,467** high-signal candidates for behavioral scoring and the top 1,000 for deep NLP semantic search).

---

## 3. Hybrid Scoring Engine & Parameter Matrix

Candidates surviving the hard filters are evaluated through a two-part scoring engine: **Logistics Match Score** and **Behavioral Merit Score** (`Final Fit Score = Logistics * Behavioral`).

### A. Behavioral Merit Matrix (Weighted at 100%)

| Parameter | Weight | Calculation & Scale | JD Reasoning & Justification |
| :--- | :---: | :--- | :--- |
| **Skills Proficiency** (`skills_score`) | **22%** | Direct evaluation of `skills` array against 50+ AI/ML/Retreival keywords (`expert`=1.0, `advanced`=0.8, `intermediate`=0.5). Capped at 4+ senior skills. | *"We need deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, vector DBs."* Rewards actual proficiency rather than keyword stuffing. |
| **GitHub Activity** (`github_score`) | **15%** | `(github_activity_score / 100)^1.5` | *"We need to see how you think, not just trust that you can think."* Strong open-source code quality and shipper mentality is critical. |
| **Interview Reliability** (`interview_attendance_score`) | **15%** | `(interview_completion_rate)^2` (Squared penalty for missed interviews). | A Series A founding team moves fast; unreliable candidates waste critical executive scheduling bandwidth. |
| **Recruiter Responsiveness** (`recruiter_response_score`) | **8%** | Linear scaling of `recruiter_response_rate` (0.0 to 1.0). | Identifies active, high-intent candidates who respect communication channels and move through pipelines smoothly. |
| **Response Speed** (`response_speed_score`) | **8%** | Inverted log-scale of `avg_response_time_hours` (1 hr = 1.0, 168+ hrs = 0.0). | *"We disagree openly and decide quickly."* Fast communication correlates strongly with high startup velocity and engagement. |
| **Market Demand** (`market_demand_score`) | **8%** | `(saved_by_recruiters_30d / 15)^2` | External market validation; highly bookmarked candidates represent validated top-tier talent in active demand. |
| **Certifications** (`certs_score`) | **5%** | `+0.5` per relevant AI/ML/Cloud cert (AWS ML, GCP ML, Deep Learning, NLP, LangChain). Capped at 1.0. | Subordinate to practical skills. Rewards specialized theoretical mastery while ignoring noise certifications (Six Sigma, Scrum). |
| **Education Tier & Grade** (`education_score`) | **5%** | `50% Tier + 50% Grade`. Tier 1 = 1.0 down to Tier 4 = 0.4. Automatically parses GPA/percentage to [0, 1]. | Kept at low weightage to avoid elitism, while still rewarding academic excellence and institutional rigor. |
| **Profile Completeness** (`profile_completeness_score_scaled`) | **4%** | Linear scale of `profile_completeness_score` (0 to 100). | *"We work async-first and write a lot."* Thorough documentation on thorough profiles reflects strong written communication. |
| **Offer Conversion** (`offer_conversion_score`) | **4%** | `offer_acceptance_rate` (defaults to 0.5 if no history). | Screens out habitual offer shoppers who accept offers purely to leverage counter-offers elsewhere. |
| **Social Proof** (`social_proof_score`) | **3%** | `min(1.0, endorsements_received / 50)` | Peer validation of technical capabilities and collaborative teamwork. |
| **Trust & Verification** (`trust_score`) | **3%** | `(verified_email + verified_phone + linkedin_connected) / 3.0` | Basic identity verification to ensure platform integrity and authenticity. |

### B. Logistics Match Score
* **Location Alignment**: Candidates in **Tier-1 hubs (Pune/Noida)** receive `1.0`. Candidates in **Tier-2 hubs (Hyderabad/Mumbai/Delhi/NCR/Bangalore)** willing to relocate receive `0.9`. Others willing to relocate receive `0.7`.
* **Salary Alignment**: Candidates with expected minimum salary `<= 60 LPA` receive `1.0`; extreme outliers above budget receive `0.0`.

---

## 4. Semantic NLP Search & Phrasing Modifiers

For the top 1,000 candidates emerging from the Pandas scoring engine, we run deep semantic search using `SentenceTransformer('all-MiniLM-L6-v2')`. 

We generate embedding vectors for the candidate's combined headline, summary, and career history descriptions, comparing them via Cosine Similarity against an ideal **Senior AI Engineer JD Query** prioritizing embeddings, retrieval, hybrid search, ranking algorithms, and NDCG/A/B evaluation frameworks.

### Phrasing Multipliers (Execution vs. Aspirational)
To differentiate true shippers from passive learners or title-chasers, the semantic score is dynamically adjusted based on resume phrasing:
* **Historical Execution Boost (+20% per phrase)**: Each occurrence of action-oriented shipping language (`"shipped"`, `"owned"`, `"deployed"`) multiplies the semantic score by **`1.2x`** (up to a 72.8% boost for candidates exhibiting all three).
* **Aspirational & Dependency Penalty (-20% per phrase)**: Each occurrence of passive or dependent phrasing (`"want to"`, `"looking to grow into"`, `"learning"`, `"handled by another team"`) penalizes the score by multiplying by **`0.8x`**.

$$\text{Final Grand Score} = \text{Pandas Fit Score} \times \text{Adjusted Semantic Score} \times 100$$

---

## 5. Dynamic Reasoning Generator

For the final **Top 100 submitted candidates**, the pipeline automatically generates a human-readable, evidence-based explanation (`reasoning`) justifying why they were selected. It highlights their top 2-3 specific "superpowers":
1. **JD-Aligned Core Skills**: Names specific matching skills (e.g., *“9 JD-aligned core skills (e.g., Go, Kubeflow)”*).
2. **Top-Tier Certifications**: Explicitly cites specialized credentials (e.g., *“top-tier AI/ML certification (AWS Certified Machine Learning Specialty)”*).
3. **Behavioral & Logistics Highlights**: Notes local Tier-1 hub availability, high GitHub activity (>=60), proven interview reliability (>=85%), or strong recruiter responsiveness.

---

## 6. Top 5 Ranked Candidates & JD Alignment Case Studies

The following table summarizes the Top 5 candidates emerging from our ranking engine, demonstrating how our hybrid pipeline successfully surfaces true AI/ML "shippers" over keyword stuffers:

| Rank  | Candidate ID   | Score        | Current Title & YOE                       | Generated Reasoning from Submission                                                                                                                                                                                        |
| ----- | -------------- | ------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | `CAND_0036437` | **549.5849** | Search Engineer (4.8 YOE)                 | Search Engineer at Rephrase.ai (4.8 YOE). Selected for proven e-commerce learning-to-rank experience, high recruiter response rate (87%), and demonstrated skills in OpenSearch, Elasticsearch, and Sentence Transformers. |
| **2** | `CAND_0017960` | **451.4879** | Recommendation Systems Engineer (7.7 YOE) | Recommendation Systems Engineer at Sarvam AI (7.7 YOE). Selected for strong semantic search background (Qdrant, FAISS), offline-to-online correlation expertise, and high market demand (bookmarked by 43 recruiters).     |
| **3** | `CAND_0006418` | **351.6436** | Machine Learning Engineer (5.7 YOE)       | Machine Learning Engineer at Verloop.io (5.7 YOE). Selected for strong evaluation rigor, verified skills in Weaviate and Embeddings, and excellent recruiter response rate (92%).                                          |
| **4** | `CAND_0099806` | **256.4864** | Machine Learning Engineer (4.6 YOE)       | Machine Learning Engineer at Mad Street Den (4.6 YOE). Selected for direct experience migrating keyword systems to embedding-based retrieval, strong RAG/FAISS skills, and offline-online correlation ownership.           |
| **5** | `CAND_0041669` | **168.0092** | Recommendation Systems Engineer (8.0 YOE) | Recommendation Systems Engineer at CRED (8.0 YOE). Selected for massive scale experience (10M+ users), comprehensive vector DB skills (Milvus, FAISS, Pinecone), and high market demand (bookmarked by 37 recruiters).     |

### Why These Candidates Perfectly Match the Job Description:

#### 1. `CAND_0036437` — The E-Commerce Search & RAG Shipper (Rank 1)

- **What they built**: Owned the ranking layer for an e-commerce search product at Rephrase.ai, successfully evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months, improving revenue-per-search by **12%**. Also shipped a production RAG-based feature and built an end-to-end churn prediction model utilizing MLflow and Kubeflow.
- **JD Alignment**: Perfectly maps to the requirement for someone who _"understood retrieval and ranking before it became fashionable."_ Their ability to design relevance labeling pipelines (mixing click-through data with explicit human judgments) and handle infrastructure/data quality proves they can _"ship a working ranker in a week."_

#### 2. `CAND_0017960` — The Semantic Search Builder & Discovery Feed Expert (Rank 2)

- **What they built**: Built semantic search infrastructure from scratch utilizing `sentence-transformers` and FAISS, while managing critical operational components like A/B testing, drift monitoring, and retraining schedules. Additionally, implemented a RAG-based support chatbot utilizing OpenAI embeddings and Pinecone that cut average ticket resolution time by **31%**.
- **JD Alignment**: Directly satisfies the need for _"production experience with embeddings-based retrieval systems"_ and vector databases. Their hands-on experience training discovery feed ranking models (XGBoost, LightGBM) and owning the offline-to-online correlation analysis fulfills the core ranking and evaluation requirements of the JD.

#### 3. `CAND_0006418` — The Evaluation Rigor & ML Ops Specialist (Rank 3)

- **What they built**: Shipped a new RAG-based feature and immediately took ownership of its evaluation framework. Trained multiple discovery feed ranking models using XGBoost/LightGBM and managed robust production ML pipelines using MLflow, Kubeflow, and internal feature stores, complete with data drift detection and alerting.
- **JD Alignment**: Speaks directly to the JD's focus on benchmarking: _"I care a lot about evaluation rigor — too many teams ship models without offline benchmarks they trust."_ Their experience defining optimization targets (click-through vs. dwell time vs. downstream conversion) with PMs shows high product-engineering maturity.

#### 4. `CAND_0099806` — The Embedding-Based Retrieval Migrator (Rank 4)

- **What they built**: Successfully led a product team through the exact migration journey from legacy keyword-based search to modern embedding-based retrieval. Designed comprehensive feature families (content metadata, user behavior signals, item engagement history) for discovery feeds and owned the offline-online correlation analysis for A/B tests.
- **JD Alignment**: The candidate's recent project acts as a proven blueprint for the JD's _"Weeks 4-8"_ objective to ship a v2 ranking system involving embeddings. As they note, they have _"spent enough time debugging production ranking issues to know which signals matter and which are noise."_

#### 5. `CAND_0041669` — The 10M+ Scale Recommendation & RAG Architect (Rank 5)

- **What they built**: Designed a massive-scale content recommendation system serving **10M+ users**, utilizing sentence-transformer embeddings for cold starts and gradient-boosted models for warm users, which improved 7-day retention by **6%**. Also evolved an e-commerce search product to learning-to-rank (+12% revenue) and shipped a Pinecone-backed RAG chatbot that reduced ticket resolution time by **31%**.
- **JD Alignment**: Brings elite, proven scale to the table with 8 years of robust experience spanning NLP, recommendation systems, and applied AI. Their ability to manage massive feature pipelines (~200 features) and build A/B testing infrastructure from the ground up makes them an ideal senior IC hire capable of owning a highly impactful piece of the ML stack.

---

## 7. Setup & Quick Reproduction Instructions

To reproduce the submission CSV in a standard CPU environment (or inside the Stage 3 sandboxed Docker container) within the mandated 5-minute wall-clock and 16 GB memory limits:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Ranking Pipeline
Execute the following single command from the repository root:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### 3. Validate Submission Format
To verify that the generated CSV strictly conforms to the Redrob Hackathon v4 specification:
```bash
python validate_submission.py submission.csv
```

### 4. Interactive Live Sandbox
You can also test and verify our ranking pipeline interactively via our hosted Streamlit cloud application:
[https://candidates-ranking-system.streamlit.app/](https://candidates-ranking-system.streamlit.app/)


