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

For the final **Top 100 submitted candidates**, the pipeline automatically generates a factual, evidence-based synopsis (`reasoning`) justifying their selection. The generator synthesizes career data, verified skills, and recruiter signals into a concise summary using the following logic:

* **Peak Role & Achievement Extraction:** Scans the candidate's career history to identify the most relevant position, weighted by target titles and ML/Search keywords. It extracts the highest-density achievement sentence using regex triggers (metrics, technical architectures) while filtering out generic clichés and normalizing pronouns to the third person.
* **Targeted Skill Compilation:** Identifies and highlights up to three verified, domain-relevant skills (e.g., Vector Search, RAG, PyTorch, LLMs) to summarize technical alignment.
* **Logistics & Behavioral Signals:** Integrates operational metrics including total years of experience, average tenure, recruiter response rate (RRR), notice period, relocation willingness, open-to-work status, and 30-day recruiter bookmark counts.
* **Automated Risk Flagging:** Appends explicit warnings for potential hiring friction, such as notice periods of 90+ days or international base locations lacking relocation intent.
* **Dynamic Formatting & Tiering:** Assigns a qualitative label (e.g., "A standout fit", "A strong match") based on the candidate's rank threshold (Top 10, Top 30, etc.). To maintain readability at scale, the generator rotates between four distinct structural templates based on the candidate's index position.
  
---

## 6. Top 5 Ranked Candidates & JD Alignment Case Studies

The following table summarizes the Top 5 candidates emerging from our ranking engine, demonstrating how our hybrid pipeline successfully surfaces true AI/ML "shippers" over keyword stuffers:

| Rank | Candidate ID | Score | Current Title & YOE | Generated Reasoning from Submission |
| --- | --- | --- | --- | --- |
| **1** | `CAND_0041669` | **68.0092** | Recommendation Systems Engineer (8.0 YOE) | A Recommendation Systems Engineer at CRED with 8.0 years. Demonstrated skills in FAISS, Milvus, and Semantic Search: at Mad Street Den, deployment cut average ticket resolution time by 31% for the supported categories. Based in Noida, Uttar Pradesh; 60-day notice, 77% recruiter response rate; bookmarked by 37 recruiters recently and actively open to new opportunities. |
| **2** | `CAND_0099806` | **56.4864** | Machine Learning Engineer (4.6 YOE) | With 4.6 years of experience and verified skills in RAG, FAISS, and Embeddings, they show a stable career averaging 2.3 years per role. Notably at Mad Street Den, owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. In Bhubaneswar, Odisha, open to relocation and actively open to new opportunities; 30-day notice, 76% recruiter response rate; bookmarked by 9 recruiters recently. |
| **3** | `CAND_0006418` | **51.6436** | Machine Learning Engineer (5.7 YOE) | A standout fit for this role: 5.7 years as a Machine Learning Engineer. Verified skills in Semantic Search, Embeddings, and Weaviate. At Verloop.io, owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Gurgaon, Haryana, open to relocation and actively open to new opportunities — 60-day notice, 92% recruiter response rate; bookmarked by 9 recruiters recently. |
| **4** | `CAND_0017960` | **51.4879** | Recommendation Systems Engineer (7.7 YOE) | A standout fit: At Sarvam AI, deployment cut average ticket resolution time by 31% for the supported categories. They bring 7.7 years as a Recommendation Systems Engineer and verified skills in Qdrant and Information Retrieval. Located in Bangalore, Karnataka, open to relocation and actively open to new opportunities, 60-day notice (72% recruiter response rate; bookmarked by 43 recruiters recently). |
| **5** | `CAND_0036437` | **49.5849** | Search Engineer (4.8 YOE) | A Search Engineer at Rephrase.ai with 4.8 years. Demonstrated skills in OpenSearch, Elasticsearch, and Sentence Transformers: at Rephrase.ai, final model improved revenue-per-search by 12%. Based in Kolkata, West Bengal, open to relocation; 30-day notice, 87% recruiter response rate; bookmarked by 15 recruiters recently. |

### Why These Candidates Perfectly Match the Job Description:

#### 1. `CAND_0041669` — The E-Commerce Ranking & RAG Veteran (Rank 1)

* **What they built**: Evolved an e-commerce search product from hand-tuned scoring to a learning-to-rank model over 9 months, improving revenue-per-search by 12%. Shipped a RAG customer support chatbot (Pinecone, OpenAI embeddings, custom BLEU/ROUGE/human eval framework) that cut resolution time by 31%.
* **JD Alignment**: Directly satisfies the 5–9 YOE requirement and the preference for candidates with learning-to-rank experience. They are located in the preferred Noida hub. Furthermore, their high market demand is validated by 37 recent recruiter bookmarks.



#### 2. `CAND_0099806` — The Migration & Analytics Specialist (Rank 2)

* **What they built**: Led a migration from legacy keyword search to embedding-based retrieval. Trained discovery feed ranking models (XGBoost/LightGBM) and owned critical offline-to-online correlation analysis for determining which offline metrics predict A/B test outcomes.
* **JD Alignment**: Perfectly executes the mandate for the first 90 days: auditing BM25/keyword systems and migrating to embeddings/hybrid retrieval. They explicitly fulfill the core requirement of understanding offline-to-online correlation and A/B test interpretation. Their sub-30-day notice period aligns with immediate hiring preferences.



#### 3. `CAND_0006418` — The Evaluation Rigor & MLOps Architect (Rank 3)

* **What they built**: Built and shipped a RAG-based feature and an accompanying offline benchmark framework. Designed and operated ML pipelines utilizing MLflow and Kubeflow, deploying model monitoring stacks focused on data drift detection.
* **JD Alignment**: Meets the strict mandate for evaluating ranking systems rigorously; the JD explicitly warns that candidates without this specific experience will struggle. Their operational experience in data drift detection aligns directly with the requirement for handling embedding drift in production. Located in Gurgaon, facilitating the flexible Tier-1 NCR cadence.



#### 4. `CAND_0017960` — The Semantic Search Operations Expert (Rank 4)

* **What they built**: Constructed semantic search infrastructure from scratch utilizing sentence-transformers and FAISS. Managed the operational lifecycle spanning A/B testing, drift monitoring, and retraining schedules. Deployed a RAG chatbot improving ticketing resolution by 31%.
* **JD Alignment**: Checks every operational requirement in the JD's absolute necessities: production deployment of embeddings, vector database infrastructure (FAISS), and active handling of embedding drift and retraining. Demonstrates high availability and engagement via 43 recent bookmarks and a strong recruiter response rate.



#### 5. `CAND_0036437` — The Full-Stack Ranking Shipper (Rank 5)

* **What they built**: Advanced a ranking layer from hand-tuned functions to learning-to-rank, yielding a 12% boost in revenue-per-search. Built production ML pipelines encompassing experiment tracking and data drift alerting for churn prediction.
* **JD Alignment**: Reflects the required "scrappy product-engineering attitude" by prioritizing shipped outcomes (revenue improvements) alongside operational infrastructure (Kubeflow, MLflow). Their 30-day notice period and 87% response rate represent a highly actionable signal for rapid pipeline movement.

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


