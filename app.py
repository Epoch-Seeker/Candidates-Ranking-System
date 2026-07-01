import streamlit as st
import subprocess
import pandas as pd
import json
import os
import sys
import tempfile
from datetime import datetime

st.set_page_config(page_title="Redrob Hackathon Sandbox", layout="wide")

st.title("Redrob AI Engineer Ranker - Sandbox")
st.markdown("Evaluate candidates using our 5-stage hybrid ranking pipeline (honeypot removal, hard filters, behavioral scoring, and NLP semantic search). Fulfills **Section 10.5 requirement**.")

# 1. Input selection
mode = st.radio(
    "Select Candidate Dataset:",
    ["Pre-loaded Sample (1,000 High-Signal Candidates)", "Upload Custom Dataset (.json / .jsonl / .gz)"],
    horizontal=True
)

target_file = "sample_candidates.json" if mode.startswith("Pre-loaded") else None
if not mode.startswith("Pre-loaded"):
    uploaded_file = st.file_uploader("Upload Candidate Pool File", type=["json", "jsonl", "gz"])
    if uploaded_file:
        ext = ".jsonl.gz" if uploaded_file.name.endswith(".jsonl.gz") else os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            target_file = tmp.name
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")

OUTPUT_FILE = "submission.csv"

# 2. Run Algorithm with Live Stats & Session State
if target_file and st.button("Run Ranking Algorithm", type="primary"):
    with st.status("Executing 5-Stage Ranking Engine...", expanded=True) as status_box:
        start_time = datetime.now()
        process = subprocess.Popen(
            [sys.executable, "-u", "rank.py", "--candidates", target_file, "--out", OUTPUT_FILE],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        
        live_log = []
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
                live_log.append(clean_line)
                if "Loaded" in clean_line:
                    st.write(f"📂 **{clean_line}**")
                elif "honeypot" in clean_line:
                    st.write(f"🛡️ **{clean_line}**")
                elif "Filtered out" in clean_line:
                    st.write(f"🎯 **{clean_line}**")
                elif "No candidates left" in clean_line:
                    st.warning(f"⚠️ **{clean_line}**")
                elif "Running NLP" in clean_line:
                    st.write(f"🧠 **{clean_line}**")
                elif "Generating Submission" in clean_line:
                    st.write("📝 **Generating submission.csv...**")
        process.wait()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if process.returncode == 0:
            status_box.update(label=f"Ranking Complete in {elapsed:.2f}s!", state="complete", expanded=False)
            st.session_state["results"] = {
                "df": pd.read_csv(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else pd.DataFrame(),
                "log": "\n".join(live_log),
                "elapsed": elapsed,
                "file": target_file
            }
            # Cache raw candidate data for the Inspector
            try:
                import gzip
                opener = gzip.open if target_file.endswith(".gz") else open
                with opener(target_file, "rt", encoding="utf-8") as f:
                    data = json.load(f) if target_file.endswith(".json") else [json.loads(l) for l in f if l.strip()]
                st.session_state["cand_map"] = {c["candidate_id"]: c for c in data if "candidate_id" in c}
            except:
                st.session_state["cand_map"] = {}
        else:
            status_box.update(label="Execution Failed!", state="error")
            st.error("\n".join(live_log))

# 3. Render Results from Session State
if "results" in st.session_state:
    res = st.session_state["results"]
    df = res["df"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Execution Time", f"{res['elapsed']:.2f}s")
    col2.metric("Ranked Candidates", str(len(df)))
    col3.metric("Top Fit Score", f"{df['score'].max():.2f}" if len(df) > 0 else "0.00")
    
    if len(df) == 0:
        st.warning("⚠️ **No candidates left after honeypot filtration and hard disqualification filters!** An empty submission.csv was generated.")
    else:
        st.subheader("Ranked Leaderboard")
        st.dataframe(df, use_container_width=True)
        
        with open(OUTPUT_FILE, "rb") as f:
            st.download_button("Download submission.csv", data=f, file_name="submission.csv", mime="text/csv")
        
        # 4. Candidate Inspector (Persistent & Detailed)
        with st.expander("Candidate Inspector (View Raw JSON & Profile Breakdown)", expanded=True):
            sel_id = st.selectbox("Select Candidate ID to Inspect:", df["candidate_id"].tolist())
            if sel_id:
                row = df[df["candidate_id"] == sel_id].iloc[0]
                st.markdown(f"### Rank #{row['rank']} | Candidate `{sel_id}` | Fit Score: `{row['score']}`")
                st.info(f"**Dynamic Reasoning:** {row['reasoning']}")
                
                cand_obj = st.session_state.get("cand_map", {}).get(sel_id)
                if cand_obj:
                    prof = cand_obj.get("profile", {})
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Title:** {prof.get('current_title', 'N/A')}")
                    c2.write(f"**Experience:** {prof.get('years_of_experience', 0)} YOE")
                    c3.write(f"**Location:** {prof.get('location', 'N/A')}")
                    
                    skills = [s.get("name") for s in cand_obj.get("skills", []) if s.get("name")]
                    if skills: st.write(f"**Skills ({len(skills)}):** {', '.join(skills[:15])}" + ("..." if len(skills)>15 else ""))
                    
                    st.markdown("#### Full Raw Profile JSON")
                    st.json(cand_obj)
                else:
                    st.write("Raw JSON profile not available for this candidate.")
                    
        with st.expander("Show Terminal Output (CLI Logs)"):
            st.code(res["log"])
