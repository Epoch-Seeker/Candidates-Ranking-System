import kagglehub
import shutil
import os

path = kagglehub.dataset_download("sunnyks01/rebrob-hackathon-dataset")

src = os.path.join(path, "candidates.jsonl")
dst = "./candidates.jsonl"

shutil.copy(src, dst)

print("Copied to:", os.path.abspath(dst))