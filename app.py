# ============================================================
# PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# Streamlit + KNN
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Learning Recommendation",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #eeeeee;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🎓 Personalized Learning Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based personalized learning guidance using K-Nearest Neighbors</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

FILE_NAME = "PLR 3 FINAL.xlsx"

try:
    df = pd.read_excel(FILE_NAME)
except Exception as e:
    st.error(f"❌ Unable to load dataset: {e}")
    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Student ID",
    "Student Name",
    "College",
    "Branch",
    "Year",
    "CGPA",
    "Learning History",
    "Skill Gap"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "❌ These columns are missing from the Excel file: "
        + ", ".join(missing_columns)
    )
    st.write("Available columns:", list(df.columns))
    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["CGPA"] = pd.to_numeric(df["CGPA"], errors="coerce")

df["Year"] = df["Year"].fillna(df["Year"].median())
df["CGPA"] = df["CGPA"].fillna(df["CGPA"].median())

text_columns = [
    "College",
    "Branch",
    "Learning History",
    "Skill Gap"
]

for col in text_columns:
    df[col] = df[col].fillna("Unknown").astype(str)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 Student Profile")

student_name = st.sidebar.text_input(
    "Student Name",
    value="New Student"
)

college = st.sidebar.selectbox(
    "College",
    sorted(df["College"].unique())
)

branch = st.sidebar.selectbox(
    "Branch",
    sorted(df["Branch"].unique())
)

year = st.sidebar.number_input(
    "Year",
    min_value=1,
    max_value=6,
    value=1
)

cgpa = st.sidebar.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.1
)

learning_history = st.sidebar.text_input(
    "Learning History",
    value="Python, statistics, ML basics"
)

skill_gap = st.sidebar.text_input(
    "Skill Gap",
    value="Machine Learning"
)


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

features = [
    "College",
    "Branch",
    "Year",
    "CGPA",
    "Learning History",
    "Skill Gap"
]

X = df[features]


# ============================================================
# PREPROCESSING
# ============================================================

categorical_features = [
    "College",
    "Branch",
    "Learning History",
    "Skill Gap"
]

numeric_features = [
    "Year",
    "CGPA"
]


# Handle different scikit-learn versions
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", encoder, categorical_features),
        ("numeric", StandardScaler(), numeric_features)
    ]
)


# ============================================================
# TRAIN KNN MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "knn",
            NearestNeighbors(
                n_neighbors=min(5, len(df)),
                metric="cosine"
            )
        )
    ]
)

model.fit(X)


# ============================================================
# NEW STUDENT DATA
# ============================================================

new_student = pd.DataFrame({
    "College": [college],
    "Branch": [branch],
    "Year": [year],
    "CGPA": [cgpa],
    "Learning History": [learning_history],
    "Skill Gap": [skill_gap]
})


# ============================================================
# GENERATE RECOMMENDATION
# ============================================================

if st.sidebar.button("🚀 Generate Recommendation"):

    st.session_state["generate"] = True


# ============================================================
# RESULTS
# ============================================================

if st.session_state.get("generate", False):

    st.markdown("## 🎯 Personalized Recommendation")

    # Transform new student
    new_student_transformed = model.named_steps[
        "preprocessor"
    ].transform(new_student)

    # Get nearest students
    distances, indices = model.named_steps[
        "knn"
    ].kneighbors(new_student_transformed)

    similar_students = df.iloc[indices[0]].copy()

    # --------------------------------------------------------
    # TOP INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👩‍🎓 Student",
            student_name
        )

    with col2:
        st.metric(
            "📊 CGPA",
            f"{cgpa:.2f}"
        )

    with col3:
        st.metric(
            "🎯 Skill Gap",
            skill_gap
        )

    st.divider()

    # --------------------------------------------------------
    # RECOMMENDATION LOGIC
    # --------------------------------------------------------

    history_text = learning_history.lower()
    skill_text = skill_gap.lower()

    recommendations = []

    if "machine" in skill_text or "ml" in skill_text:
        recommendations.extend([
            "Machine Learning Fundamentals",
            "Supervised Learning",
            "Unsupervised Learning",
            "Scikit-learn"
        ])

    elif "deep" in skill_text:
        recommendations.extend([
            "Deep Learning",
            "Neural Networks",
            "TensorFlow / PyTorch",
            "Computer Vision"
        ])

    elif "python" in skill_text:
        recommendations.extend([
            "Python Programming",
            "Pandas",
            "NumPy",
            "Data Analysis"
        ])

    elif "sql" in skill_text:
        recommendations.extend([
            "SQL Fundamentals",
            "Advanced SQL",
            "Database Management",
            "Data Analysis"
        ])

    elif "cyber" in skill_text:
        recommendations.extend([
            "Cybersecurity Fundamentals",
            "Network Security",
            "Ethical Hacking",
            "Information Security"
        ])

    elif "cloud" in skill_text:
        recommendations.extend([
            "Cloud Computing Fundamentals",
            "AWS / Azure",
            "Cloud Security",
            "DevOps Basics"
        ])

    else:
        recommendations.extend([
            "Programming Fundamentals",
            "Data Structures & Algorithms",
            "Python Programming",
            "Problem Solving"
        ])

    # Remove duplicates
    recommendations = list(dict.fromkeys(recommendations))

    # --------------------------------------------------------
    # LEARNING PATH
    # --------------------------------------------------------

    st.subheader("📚 Recommended Learning Path")

    for i, item in enumerate(recommendations, 1):

        st.markdown(
            f"""
            <div class="card">
                <b>{i}. {item}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SIMILAR STUDENTS
    # --------------------------------------------------------

    st.subheader("👥 Similar Students")

    display_columns = [
        "Student ID",
        "Student Name",
        "College",
        "Branch",
        "Year",
        "CGPA",
        "Learning History",
        "Skill Gap"
    ]

    display_columns = [
        col for col in display_columns
        if col in similar_students.columns
    ]

    st.dataframe(
        similar_students[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ML EXPLANATION
    # --------------------------------------------------------

    st.subheader("🤖 How ML Generated This Recommendation")

    st.write(
        "The system converts student information into numerical "
        "features using One-Hot Encoding and Standard Scaling."
    )

    st.write(
        "The K-Nearest Neighbors (KNN) algorithm then compares "
        "the new student's profile with existing students in the dataset."
    )

    st.write(
        "The system identifies the most similar students and uses "
        "their learning characteristics to personalize the learning direction."
    )

    # --------------------------------------------------------
    # SIMILARITY SCORE
    # --------------------------------------------------------

    similarity_score = max(
        0,
        min(100, (1 - distances[0][0]) * 100)
    )

    st.subheader("📈 Profile Similarity")

    st.progress(
        int(similarity_score)
    )

    st.write(
        f"Your profile is approximately **{similarity_score:.1f}% "
        "similar** to the closest student profile in the dataset."
    )


# ============================================================
# DATASET - COLLAPSED BY DEFAULT
# ============================================================

st.divider()

with st.expander("📊 View Dataset", expanded=False):

    st.write(
        f"Dataset contains **{len(df)} students** "
        f"and **{len(df.columns)} columns**."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <center>
    🎓 <b>Personalized Learning Recommendation System</b><br>
    Built with Python • Machine Learning • KNN • Streamlit
    </center>
    """,
    unsafe_allow_html=True
)
