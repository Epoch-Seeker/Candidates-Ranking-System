# Redrob AI Engineer Candidate Ranking

This project contains a candidate ranking script `rank.py` for selecting top candidates for a Senior AI Engineer position, evaluating candidates based on logistic requirements, behavioral merits, and semantic alignment with the job description.

## Requirements

Before running the ranking script, make sure you have installed all the necessary dependencies. You can install them using `pip`:

```bash
pip install pandas numpy sentence-transformers
```

## Running the Ranking Script

To run the ranking script and produce the output submission CSV, use the following single command:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Where:
- `--candidates`: The input candidates data file (JSONL format, can also be `.gz`).
- `--out`: The output file path for the submission CSV.
