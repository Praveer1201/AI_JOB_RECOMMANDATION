import pandas as pd
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ================= CREATE MODEL FOLDER =================
os.makedirs("model", exist_ok=True)

# ================= LOAD DATASET =================
df = pd.read_csv("data/jobs.csv", encoding="latin-1")

# ================= BASIC CLEANING =================
df = df.dropna()

# Ensure required column exists
if "skills" not in df.columns:
    raise ValueError("Dataset must contain a column named 'skills'")

# ================= TEXT CLEANING (IMPORTANT AI UPGRADE) =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove symbols safely
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["skills"] = df["skills"].apply(clean_text)

# ================= TF-IDF VECTORIZATION =================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)   # captures single + paired skills (AI upgrade)
)

job_vectors = vectorizer.fit_transform(df["skills"])

# ================= SAVE MODELS =================
with open("model/tfidf.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/job_vectors.pkl", "wb") as f:
    pickle.dump(job_vectors, f)

with open("model/jobs_data.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ Model trained and saved successfully!")
print("📁 Files created:")
print(" - model/tfidf.pkl")
print(" - model/job_vectors.pkl")
print(" - model/jobs_data.pkl")

