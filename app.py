import streamlit as st
import subprocess
import pandas as pd
import os
import sys
import json

st.set_page_config(page_title="Redrob Hackathon Sandbox", layout="wide")

st.title("🏆 Redrob AI Engineer Ranker - Sandbox")
st.markdown("""
This sandbox fulfills the **Section 10.5 requirement**. 
It runs our `rank.py` algorithm end-to-end on a pre-loaded sample dataset and outputs the ranked CSV.
""")

# We use the sample file for the sandbox to ensure it runs well under the 5-minute limit
SAMPLE_FILE = "top_500_candidates.json"
OUTPUT_FILE = "submission.csv"

st.write(f"**Target Dataset:** `{SAMPLE_FILE}`")

if st.button("🚀 Run Ranking Algorithm", type="primary"):
    with st.spinner("Initializing models and processing candidates (this takes a few seconds)..."):
        
        # This executes your exact CLI command in the background
        result = subprocess.run(
            [sys.executable, "rank.py", "--candidates", SAMPLE_FILE, "--out", OUTPUT_FILE],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            st.success("Ranking complete! Algorithm executed successfully.")
            
            # Show the terminal output so judges can see your print statements
            with st.expander("Show Terminal Output"):
                st.code(result.stdout)
            
            # Display the generated CSV as a beautiful table
            if os.path.exists(OUTPUT_FILE):
                st.subheader("Top Ranked Candidates")
                df = pd.read_csv(OUTPUT_FILE)
                st.dataframe(df, use_container_width=True)
                
                # Provide a download button
                with open(OUTPUT_FILE, "rb") as f:
                    st.download_button(
                        label="⬇️ Download submission.csv",
                        data=f,
                        file_name="submission.csv",
                        mime="text/csv"
                    )
        else:
            st.error("An error occurred during execution.")
            st.code(result.stderr)