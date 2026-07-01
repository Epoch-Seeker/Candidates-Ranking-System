# 🏆 Redrob AI Engineer Ranker

An intelligent, two-stage candidate discovery and ranking system built for the Redrob Intelligent Candidate Discovery Challenge. 

This system moves beyond basic keyword matching by combining strict logistical filtering with deep semantic search. It accurately identifies top-tier AI Engineers while successfully avoiding dataset traps like keyword-stuffers, non-coding architects, and synthetic honeypots.

---

## 🧠 System Architecture

Our ranking engine (`rank.py`) processes the 100,000-candidate dataset in a highly optimized pipeline designed to execute well under the 5-minute compute limit on standard CPU hardware.

### Stage 1: The Hard Filter (Pandas Engine)
Before any heavy NLP models are initialized, we aggressively prune the dataset using vector-based Pandas operations:
* **Trap Evasion:** Automatically drops known honeypots (e.g., negative duration jobs, overlapping timelines, inverted salary expectations).
* **Logistics Matching:** Filters for required locations (Pune/Noida/Remote options), notice periods (≤60 days), and maximum budget caps.
* **Title Relevance:** Drops irrelevant industries (IT Services, Manufacturing) and title-chasers who haven't written code recently.

### Stage 2: Semantic NLP Search
We process the surviving high-quality candidates through a local embedding model (`all-MiniLM-L6-v2`).
* We convert the core requirements of the Redrob Job Description (Ranking, Retrieval, NDCG/MRR evaluation, Vector DBs, LLM fine-tuning) into an ideal target vector.
* We compare this against the candidate's actual parsed career history to calculate a **Semantic Fit Score**. This ensures we find candidates who *actually* built recommendation systems, even if they didn't explicitly use the buzzword "RAG" in their skills list.

### Stage 3: Hybrid Scoring & Dynamic Reasoning
* The Semantic NLP score is multiplied by a **Behavioral Merit Score** (a weighted matrix of recruiter response rates, interview attendance, GitHub activity, and market demand).
* Finally, an **Explainable AI (XAI)** reasoning string is dynamically generated for the top 100 candidates, highlighting exactly *why* they were selected (e.g., specific skill matches, tier-1 hub location, open-source contributions).

---

## 📂 Repository Structure

* `rank.py`: The master execution script containing the Pandas filters, NLP model, and CSV generator.
* `app.py`: The Streamlit sandbox UI for quick visual evaluation.
* `requirements.txt`: Python dependencies required to run the pipeline.
* `submission_metadata.yaml`: Team details and configuration.
* `sample_candidates.json`: A lightweight sample dataset used for the Streamlit sandbox.

---

## 🚀 How to Run the Pipeline (CLI)

This script is fully compliant with the Stage 3 automated reproduction requirements. It accepts dynamic file paths via CLI arguments and runs entirely offline.

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the end-to-end ranking algorithm
python rank.py --candidates ./path/to/candidates.jsonl --out ./submission.csv