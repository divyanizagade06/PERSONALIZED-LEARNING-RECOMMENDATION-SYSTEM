import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Personalized Learning Recommendation",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f3fa;
}

.title-box {
    background-color: #d7b4df;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
}

.title-box h1 {
    color: #30243b;
    margin: 0;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
    margin-bottom: 15px;
}

.path {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border-left: 6px solid #7b3f98;
    margin-bottom: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.10);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class="title-box">
<h1>🎓 PERSONALIZED LEARNING RECOMMENDATION SYSTEM</h1>
</div>
""", unsafe_allow_html=True)

st.write("")

st.write(
    "ML-based system that finds students with similar learning profiles "
    "and generates a personalized learning path."
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_excel("PLR 3.xlsx")

    return data


df = load_data()


# =========================================================
# CLEAN DATA
# =========================================================

df = df.drop_duplicates()

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["CGPA"] = pd.to_numeric(
    df["CGPA"],
    errors="coerce"
)

df["Assessment Score"] = pd.to_numeric(
    df["Assessment Score"],
    errors="coerce"
)

df["Time Available (hrs/day)"] = pd.to_numeric(
    df["Time Available (hrs/day)"],
    errors="coerce"
)

df["Attendance (%)"] = pd.to_numeric(
    df["Attendance (%)"],
    errors="coerce"
)

df = df.dropna()


# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Students",
        len(df)
    )

with col2:
    st.metric(
        "Average CGPA",
        round(df["CGPA"].mean(), 2)
    )

with col3:
    st.metric(
        "Assessment",
        round(df["Assessment Score"].mean(), 2)
    )

with col4:
    st.metric(
        "Attendance",
        f'{df["Attendance (%)"].mean():.1f}%'
    )

with col5:
    st.metric(
        "Study Hours",
        round(
            df["Time Available (hrs/day)"].mean(),
            2
        )
    )


# =========================================================
# ML FEATURES
# =========================================================

features = [

    "College",
    "Branch",
    "Year",
    "CGPA",
    "Learning History",
    "Skill Gap",
    "Career Goal",
    "Assessment Score",
    "Time Available (hrs/day)",
    "Attendance (%)",
    "Requirement Type",
    "Current Level"

]


categorical_features = [

    "College",
    "Branch",
    "Learning History",
    "Skill Gap",
    "Career Goal",
    "Requirement Type",
    "Current Level"

]


numeric_features = [

    "Year",
    "CGPA",
    "Assessment Score",
    "Time Available (hrs/day)",
    "Attendance (%)"

]


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numeric",

            StandardScaler(),

            numeric_features
        )

    ]

)


# =========================================================
# KNN MODEL
# =========================================================

knn_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "knn",

            NearestNeighbors(
                n_neighbors=5,
                metric="cosine"
            )

        )

    ]

)


# Train model

knn_model.fit(
    df[features]
)


# =========================================================
# STUDENT INPUT
# =========================================================

st.header("👨‍🎓 Student Information")


col1, col2, col3 = st.columns(3)


with col1:

    college = st.selectbox(
        "College",
        sorted(df["College"].unique())
    )

    branch = st.selectbox(
        "Branch",
        sorted(df["Branch"].unique())
    )

    year = st.selectbox(
        "Year",
        sorted(df["Year"].unique())
    )


with col2:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    assessment = st.number_input(
        "Assessment Score",
        min_value=0,
        max_value=100,
        value=60
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )


with col3:

    study_hours = st.number_input(
        "Available Study Hours/Day",
        min_value=0.5,
        max_value=12.0,
        value=2.0,
        step=0.5
    )

    learning_history = st.selectbox(
        "Learning History",
        sorted(df["Learning History"].unique())
    )

    skill_gap = st.selectbox(
        "Current Skill Gap",
        sorted(df["Skill Gap"].unique())
    )


career_goal = st.selectbox(
    "🎯 Career Goal",
    sorted(df["Career Goal"].unique())
)


requirement_type = st.selectbox(
    "Requirement Type",
    sorted(df["Requirement Type"].unique())
)


current_level = st.selectbox(
    "Current Level",
    sorted(df["Current Level"].unique())
)


# =========================================================
# GENERATE RECOMMENDATION
# =========================================================

if st.button(
    "🚀 Generate Personalized Learning Path",
    use_container_width=True
):

    # -----------------------------------------------------
    # CREATE NEW STUDENT
    # -----------------------------------------------------

    new_student = pd.DataFrame({

        "College": [college],

        "Branch": [branch],

        "Year": [year],

        "CGPA": [cgpa],

        "Learning History": [learning_history],

        "Skill Gap": [skill_gap],

        "Career Goal": [career_goal],

        "Assessment Score": [assessment],

        "Time Available (hrs/day)": [study_hours],

        "Attendance (%)": [attendance],

        "Requirement Type": [requirement_type],

        "Current Level": [current_level]

    })


    # -----------------------------------------------------
    # FIND SIMILAR STUDENTS
    # -----------------------------------------------------

    distances, indices = knn_model.named_steps[
        "knn"
    ].kneighbors(

        knn_model.named_steps[
            "preprocessor"
        ].transform(new_student)

    )


    similar_students = df.iloc[
        indices[0]
    ]


    # =====================================================
    # RECOMMENDATION SECTION
    # =====================================================

    st.success(
        "✅ Personalized learning path generated!"
    )


    st.header("🎯 Your Personalized Recommendation")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            f"""
            **Career Goal**

            {career_goal}
            """
        )


    with col2:

        st.warning(
            f"""
            **Skill Gap**

            {skill_gap}
            """
        )


    with col3:

        st.success(
            f"""
            **Current Level**

            {current_level}
            """
        )


    # =====================================================
    # FIND COMMON SKILLS
    # =====================================================

    skill_counts = (
        similar_students["Skill Gap"]
        .value_counts()
    )


    common_skill = skill_counts.index[0]


    # =====================================================
    # COURSE RECOMMENDATION
    # =====================================================

    career = career_goal.lower()
    skill = skill_gap.lower()


    if "data scientist" in career:

        course = "Python for Data Science + Statistics + Machine Learning"

        project = "Student Performance Prediction using Machine Learning"

    elif "data analyst" in career:

        course = "Excel + SQL + Python + Power BI"

        project = "Student / Business Analytics Dashboard"

    elif "ai/ml" in career or "ml engineer" in career:

        course = "Python + Machine Learning + Deep Learning"

        project = "Machine Learning Prediction System"

    elif "cyber" in career:

        course = "Networking + Linux + Cybersecurity"

        project = "Network Security Monitoring System"

    elif "cloud" in career:

        course = "Linux + AWS/Azure + Cloud Fundamentals"

        project = "Cloud Deployment Project"

    elif "devops" in career:

        course = "Linux + Git + Docker + CI/CD"

        project = "CI/CD Automation Project"

    elif "full stack" in career:

        course = "HTML + CSS + JavaScript + React + Backend"

        project = "Full Stack Web Application"

    elif "software" in career:

        course = "Programming + DSA + OOP + Git"

        project = "Software Application Development"

    elif "web" in career:

        course = "HTML + CSS + JavaScript + React"

        project = "Responsive Web Application"

    elif "business analyst" in career:

        course = "Excel + SQL + Business Analytics"

        project = "Business Intelligence Dashboard"

    elif "ui/ux" in career:

        course = "UI/UX Design + Figma + User Research"

        project = "Mobile/Web UI UX Case Study"

    else:

        course = "Programming Fundamentals + Data Analysis"

        project = "Data Analysis Project"


    # =====================================================
    # LEARNING PATH
    # =====================================================

    st.header("🗺️ Personalized Learning Path")


    learning_path = [

        (
            "STEP 1",
            "Foundation",
            "Strengthen your basic concepts according to your current level."
        ),

        (
            "STEP 2",
            "Skill Gap",
            f"Focus on improving: {skill_gap}"
        ),

        (
            "STEP 3",
            "Core Course",
            course
        ),

        (
            "STEP 4",
            "Hands-on Practice",
            "Solve coding exercises, quizzes and real-world problems."
        ),

        (
            "STEP 5",
            "Project",
            project
        ),

        (
            "STEP 6",
            "Advanced Learning",
            f"Learn advanced concepts related to {career_goal}."
        ),

        (
            "STEP 7",
            "Career Preparation",
            "Build portfolio, GitHub projects and prepare for interviews."
        )

    ]


    for number, title, description in learning_path:

        st.markdown(

            f"""
            <div class="path">

            <h3>{number} — {title}</h3>

            <p>{description}</p>

            </div>
            """,

            unsafe_allow_html=True
        )


    # =====================================================
    # STUDY PLAN
    # =====================================================

    st.header("⏰ Personalized Study Plan")


    daily_hours = study_hours


    if daily_hours <= 1:

        plan = {
            "Concept Learning": "30 minutes",
            "Practice": "20 minutes",
            "Revision": "10 minutes"
        }

    elif daily_hours <= 2:

        plan = {
            "Concept Learning": "45 minutes",
            "Practice": "45 minutes",
            "Project": "20 minutes",
            "Revision": "10 minutes"
        }

    else:

        plan = {
            "Concept Learning": "60 minutes",
            "Practice": "60 minutes",
            "Project": "60 minutes",
            "Revision": "30 minutes"
        }


    for activity, time in plan.items():

        st.write(
            f"### 📌 {activity} — {time}"
        )


    # =====================================================
    # SIMILAR STUDENTS
    # =====================================================

    st.header("👥 Similar Students Found by ML")


    display_columns = [

        "Student ID",
        "Student Name",
        "Branch",
        "CGPA",
        "Skill Gap",
        "Career Goal",
        "Current Level"

    ]


    st.dataframe(

        similar_students[
            display_columns
        ],

        use_container_width=True,

        hide_index=True

    )


    # =====================================================
    # ML EXPLANATION
    # =====================================================

    st.header("🤖 How ML Generated This Recommendation")


    st.write(
        """
        The system converts student information into numerical features
        using One-Hot Encoding and Standard Scaling.

        The K-Nearest Neighbors (KNN) algorithm then compares the new
        student's profile with existing students in the dataset.

        The system identifies the most similar students and uses their
        learning characteristics to personalize the learning direction.
        """
    )


# =========================================================
# DATASET
# =========================================================

with st.expander("📊 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Personalized Learning Recommendation System | "
    "Machine Learning + Streamlit"
)
