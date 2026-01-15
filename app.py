import streamlit as st
import pickle
import urllib.parse
import pdfplumber
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Job Matcher",
    page_icon="🤖",
    layout="wide"
)

# ================= PREMIUM DARK UI =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

h1, h2, h3, p, li {
    color: #e5e7eb;
}

.card {
    background: rgba(30,41,59,0.85);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    margin-bottom: 24px;
}

.primary-btn {
    background: linear-gradient(135deg,#2563eb,#1e40af);
    color: white;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: 600;
    text-decoration: none;
}

.badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ================= HELPERS =================
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except:
        pass
    return text.lower()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def create_job_links(skills):
    linkedin = f"https://www.linkedin.com/jobs/search/?keywords={urllib.parse.quote(skills)}"
    naukri = f"https://www.naukri.com/{re.sub(r'[,\s]+','-', skills)}-jobs"
    return linkedin, naukri

# ================= LOAD AI MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    with open("model/jobs_data.pkl", "rb") as f:
        jobs = pickle.load(f)
    jobs["semantic"] = jobs["skills"].astype(str)
    embeddings = model.encode(jobs["semantic"].tolist(), convert_to_numpy=True)
    return jobs, embeddings

jobs_data, job_embeddings = load_data()

# ================= HERO SECTION =================
st.markdown("""
<div style="text-align:center; padding:50px 0 10px;">
    <h1>🤖 AI Job Recommendation System</h1>
    <p style="font-size:18px; max-width:750px; margin:auto; color:#cbd5f5;">
        Semantic AI-powered system that understands your skills
        and intelligently matches you with relevant jobs.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= HERO IMAGE (FIXED) =================
if os.path.exists("assets/hero.png"):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("assets/hero.png", width=320)

# ================= INPUT + INFO =================
col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Your Profile")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    user_skills = st.text_input(
        "Or enter skills manually",
        placeholder="Python, Data Science, AI Ethics, SQL"
    )
    recommend = st.button("🔍 Find My Match")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 How AI Matches Jobs</h3>
        <ul>
            <li>Understands meaning, not keywords</li>
            <li>Matches using semantic similarity</li>
            <li>Handles related skills automatically</li>
            <li>Ranks jobs intelligently</li>
        </ul>
        <p style="color:#93c5fd;"><b>Powered by NLP & Semantic AI</b></p>
    </div>
    """, unsafe_allow_html=True)

# ================= RECOMMENDATIONS =================
if recommend:
    final_input = ""

    if uploaded_file:
        with st.spinner("Reading resume..."):
            final_input += extract_text_from_pdf(uploaded_file)

    if user_skills.strip():
        final_input += " " + clean_text(user_skills)

    if final_input.strip() == "":
        st.warning("⚠️ Please upload a resume or enter skills.")
    else:
        user_embedding = model.encode([final_input], convert_to_numpy=True)
        similarity = cosine_similarity(user_embedding, job_embeddings)[0]

        jobs_data["Similarity"] = similarity
        top_jobs = jobs_data.sort_values(by="Similarity", ascending=False).head(5)

        st.subheader("📌 Recommended Jobs")

        for _, row in top_jobs.iterrows():
            score = round(row["Similarity"] * 100, 2)
            skills = str(row["skills"])
            linkedin, naukri = create_job_links(skills)

            st.markdown(f"""
            <div class="card">
                <h3>🧾 Job Match</h3>
                <p><b>Required Skills:</b> {skills}</p>
                <p><b>AI Match Score:</b> {score}%</p>
                <a href="{linkedin}" target="_blank"
                   style="padding:8px 14px; background:#2563eb; color:white;
                          text-decoration:none; border-radius:8px;
                          font-weight:600; margin-right:10px;">
                   Apply on LinkedIn
                </a>
                <a href="{naukri}" target="_blank"
                   style="padding:8px 14px; background:#1d4ed8; color:white;
                          text-decoration:none; border-radius:8px;
                          font-weight:600;">
                   Apply on Naukri
                </a>
            </div>
            """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<hr style="border-color:#334155;">
<p style="text-align:center; color:#94a3b8;">
© 2026 AI Job Recommendation System • Semantic AI Powered
</p>
""", unsafe_allow_html=True)

