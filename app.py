# ============================================================
# PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# UI/UX - STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Learning Recommendation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.recommendation {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #f5f3ff
    );
    padding: 25px;
    border-radius: 18px;
    border-left: 6px solid #667eea;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.metric-title {
    color: #6b7280;
    font-size: 14px;
}

.metric-value {
    font-size: 25px;
    font-weight: 700;
    color: #111827;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel("PLR 3 FINAL.xlsx")

    df = df.drop_duplicates()
    df = df.fillna("Unknown")

    return df


df = load_data()


# ============================================================
# TRAIN MODEL
# ============================================================

features = [
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

target = "Recommended Course"


numerical_features = [
    "Year",
    "CGPA",
    "Assessment Score",
    "Time Available (hrs/day)",
    "Attendance (%)"
]


categorical_features = [
    "Branch",
    "Learning History",
    "Skill Gap",
    "Career Goal",
    "Requirement Type",
    "Current Level"
]


X = df[features]
y = df[target]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


pipeline.fit(X, y)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🎓 Personalized Learning Recommendation</h1>

<p>
AI-powered learning recommendations based on your academic
performance, skills, career goals and learning profile.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 PLR System")

    st.write(
        "Create a personalized learning path "
        "for every student."
    )

    st.divider()

    st.markdown("### 📌 Navigation")

    page = st.radio(
        "Select",
        [
            "Student Recommendation",
            "Dataset Overview"
        ]
    )

    st.divider()

    st.caption(
        "Personalized Learning Recommendation System"
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Students",
            len(df)
        )

    with col2:
        st.metric(
            "Features",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Courses",
            df["Recommended Course"].nunique()
        )

    with col4:
        st.metric(
            "Clusters",
            df["Learning_Profile_Cluster"].nunique()
        )

    st.divider()

    st.subheader("📋 Student Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )


# ============================================================
# STUDENT RECOMMENDATION PAGE
# ============================================================

else:

    st.header("👤 Student Profile")

    st.write(
        "Enter the student's information to generate "
        "a personalized learning recommendation."
    )


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🎓 Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        branch = st.selectbox(
            "Branch",
            sorted(
                df["Branch"].astype(str).unique()
            )
        )

    with col2:

        year = st.selectbox(
            "Year",
            sorted(
                pd.to_numeric(
                    df["Year"],
                    errors="coerce"
                )
                .dropna()
                .astype(int)
                .unique()
            )
        )

    with col3:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📈 Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        assessment = st.slider(
            "Assessment Score",
            0,
            100,
            75
        )

    with col2:

        attendance = st.slider(
            "Attendance (%)",
            0,
            100,
            85
        )

    with col3:

        time_available = st.slider(
            "Study Time (hrs/day)",
            1,
            12,
            2
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # LEARNING PROFILE
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧠 Learning Profile")

    col1, col2 = st.columns(2)

    with col1:

        learning_history = st.selectbox(
            "Learning History",
            sorted(
                df["Learning History"]
                .astype(str)
                .unique()
            )
        )

    with col2:

        current_level = st.selectbox(
            "Current Level",
            sorted(
                df["Current Level"]
                .astype(str)
                .unique()
            )
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CAREER & SKILLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🎯 Career & Skills")

    col1, col2 = st.columns(2)

    with col1:

        skill_gap = st.selectbox(
            "Skill Gap",
            sorted(
                df["Skill Gap"]
                .astype(str)
                .unique()
            )
        )

    with col2:

        career_goal = st.selectbox(
            "Career Goal",
            sorted(
                df["Career Goal"]
                .astype(str)
                .unique()
            )
        )

    requirement_type = st.selectbox(
        "Requirement Type",
        sorted(
            df["Requirement Type"]
            .astype(str)
            .unique()
        )
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    st.markdown("### 🚀 Generate Your Learning Path")

    generate = st.button(
        "✨ Get Personalized Recommendation"
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if generate:

        student = pd.DataFrame([{

            "Branch": branch,
            "Year": year,
            "CGPA": cgpa,
            "Learning History": learning_history,
            "Skill Gap": skill_gap,
            "Career Goal": career_goal,
            "Assessment Score": assessment,
            "Time Available (hrs/day)": time_available,
            "Attendance (%)": attendance,
            "Requirement Type": requirement_type,
            "Current Level": current_level

        }])


        # Prediction

        recommendation = pipeline.predict(
            student
        )[0]


        # ----------------------------------------------------
        # FIND RELATED DATA
        # ----------------------------------------------------

        related = df[
            df["Recommended Course"]
            == recommendation
        ]


        if len(related) > 0:

            skill = related[
                "Recommended Skill Area"
            ].mode()[0]

            project = related[
                "Recommended Project"
            ].mode()[0]

            avg_hours = round(
                pd.to_numeric(
                    related[
                        "Suggested Daily Study Hours"
                    ],
                    errors="coerce"
                ).mean(),
                1
            )

        else:

            skill = "Skill development"
            project = "Practical project"
            avg_hours = time_available


        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.success(
            "🎉 Personalized learning path generated!"
        )

        st.markdown(
            '<div class="recommendation">',
            unsafe_allow_html=True
        )

        st.markdown(
            "## 🎯 Your Recommended Course"
        )

        st.markdown(
            f"# {recommendation}"
        )

        st.write(
            "This recommendation is generated using "
            "your academic performance, learning history, "
            "skill gap and career goal."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # RECOMMENDATION CARDS
        # ====================================================

        st.markdown("### 📚 Your Personalized Learning Path")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                '<div class="metric-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="metric-title">'
                'Recommended Skill'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="metric-value">'
                f'{skill}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                '<div class="metric-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="metric-title">'
                'Recommended Project'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="metric-value">'
                f'{project}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                '<div class="metric-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="metric-title">'
                'Suggested Study'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="metric-value">'
                f'{avg_hours} hrs/day'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # STUDENT SUMMARY
        # ====================================================

        st.markdown("### 👤 Student Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.info(
                f"""
                **Branch:** {branch}

                **Year:** {year}

                **CGPA:** {cgpa}

                **Current Level:** {current_level}

                **Career Goal:** {career_goal}
                """
            )

        with summary_col2:

            st.info(
                f"""
                **Assessment Score:** {assessment}%

                **Attendance:** {attendance}%

                **Study Time:** {time_available} hrs/day

                **Skill Gap:** {skill_gap}

                **Requirement:** {requirement_type}
                """
            )


        # ====================================================
        # LEARNING PLAN
        # ====================================================

        st.markdown("### 🗺️ Suggested Learning Journey")

        st.markdown(
            f"""
            **01 — Build Skills**  
            Focus on **{skill}**

            ↓

            **02 — Learn Course**  
            Complete **{recommendation}**

            ↓

            **03 — Build Project**  
            Work on **{project}**

            ↓

            **04 — Career Preparation**  
            Align your learning with your goal of
            **{career_goal}**
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Personalized Learning Recommendation System | "
    "Machine Learning + Streamlit"
)
