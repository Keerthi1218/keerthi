import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------
# 1. Simulated dataset
data = [
    {
        "job_title": "Data Scientist",
        "skills": "python, machine learning, statistics, data visualization",
        "education": "bachelor",
        "interests": "data, analytics, research",
        "experience": "mid",
        "description": "Analyzes data to extract insights and build predictive models."
    },
    {
        "job_title": "Software Engineer",
        "skills": "java, c++, algorithms, problem solving",
        "education": "bachelor",
        "interests": "coding, development, problem solving",
        "experience": "junior",
        "description": "Designs and develops software applications and systems."
    },
    {
        "job_title": "Graphic Designer",
        "skills": "photoshop, creativity, adobe illustrator, visual design",
        "education": "associate",
        "interests": "art, creativity, media",
        "experience": "entry",
        "description": "Creates visual concepts to communicate ideas."
    },
    {
        "job_title": "Project Manager",
        "skills": "leadership, communication, scheduling, budgeting",
        "education": "bachelor",
        "interests": "management, organization, planning",
        "experience": "senior",
        "description": "Oversees projects to ensure timely delivery within budget."
    },
    {
        "job_title": "Marketing Specialist",
        "skills": "seo, content creation, social media, communication",
        "education": "bachelor",
        "interests": "marketing, branding, communication",
        "experience": "mid",
        "description": "Develops strategies to promote products and brands."
    },
    {
        "job_title": "Cybersecurity Analyst",
        "skills": "network security, python, risk assessment, cryptography",
        "education": "bachelor",
        "interests": "security, technology, risk management",
        "experience": "mid",
        "description": "Protects an organization's computer systems and networks."
    },
    {
        "job_title": "Mechanical Engineer",
        "skills": "cad, thermodynamics, mechanics, problem solving",
        "education": "bachelor",
        "interests": "engineering, mechanics, design",
        "experience": "mid",
        "description": "Designs and tests mechanical devices and systems."
    },
    {
        "job_title": "Financial Analyst",
        "skills": "excel, finance, accounting, data analysis",
        "education": "bachelor",
        "interests": "finance, economics, data",
        "experience": "junior",
        "description": "Provides investment and financial recommendations."
    },
    {
        "job_title": "Teacher",
        "skills": "communication, patience, subject knowledge, mentoring",
        "education": "bachelor",
        "interests": "teaching, education, helping others",
        "experience": "mid",
        "description": "Educates and supports students in learning."
    },
    {
        "job_title": "UX Designer",
        "skills": "wireframing, user research, creativity, prototyping",
        "education": "bachelor",
        "interests": "design, user experience, psychology",
        "experience": "mid",
        "description": "Improves user satisfaction with products by enhancing usability."
    },
]

df = pd.DataFrame(data)

# ---------------------------------
# 2. Data preparation
def combine_text_features(row):
    return f"{row['skills']} {row['education']} {row['interests']} {row['experience']}"

df['combined_features'] = df.apply(combine_text_features, axis=1)

# Vectorizer for combined features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_features'])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['job_title'])

# ---------------------------------
# 3. Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Vectorizer for skills ONLY
skill_vectorizer = CountVectorizer()
skill_vectorizer.fit(df['skills'])

# ---------------------------------
# 4. Streamlit UI
st.title("Career Path Recommendation System")

st.markdown("""
Enter your profile information to get the top 3 recommended careers.
""")

skills_input = st.text_input("Enter your skills (comma separated):", "")
education_input = st.selectbox("Highest Education Level:", ['highschool', 'associate', 'bachelor', 'master', 'phd'])
interests_input = st.text_input("Enter your interests (comma separated):", "")
experience_input = st.selectbox("Your Experience Level:", ['entry', 'junior', 'mid', 'senior'])

if st.button("Recommend Careers"):

    # Combine text
    user_features = f"{skills_input} {education_input} {interests_input} {experience_input}"
    user_vector = vectorizer.transform([user_features])

    # Model predictions
    pred_probs = model.predict_proba(user_vector)[0]

    # ---------- Corrected Skill Similarity ----------
    user_skills_vec = skill_vectorizer.transform([skills_input.lower()])
    career_skills_vec = skill_vectorizer.transform(df['skills'].str.lower())

    skill_sim = cosine_similarity(user_skills_vec, career_skills_vec)[0]

    # Combined score
    final_scores = 0.7 * pred_probs + 0.3 * skill_sim

    top3 = final_scores.argsort()[::-1][:3]

    st.subheader("Top 3 Career Recommendations")
    for idx in top3:
        st.markdown(f"## ⭐ {df.iloc[idx]['job_title']}")
        st.write(df.iloc[idx]['description'])
        st.markdown(f"**Required Skills:** {df.iloc[idx]['skills']}")
        st.markdown(f"**Education:** {df.iloc[idx]['education'].capitalize()}")
        st.markdown(f"**Experience:** {df.iloc[idx]['experience'].capitalize()}")
        st.markdown("---")

# ---------------------------------
# Sidebar
st.sidebar.title("How it Works")
st.sidebar.info("""
- Text features (skills, education, interests, experience) are combined.
- A Random Forest model predicts possible career matches.
- Skill similarity (cosine similarity) improves accuracy.
- Final ranking = 70% ML model + 30% skill similarity.
""")
