import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIGURATION
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

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.header {
    background: linear-gradient(90deg, #cda7d9, #e4c5e8);
    padding: 18px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 15px;
}

.header h1 {
    color: #30243b;
    font-size: 30px;
    font-weight: 700;
}

.card {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.12);
    margin-bottom: 15px;
}

.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.12);
}

.metric-title {
    font-size: 15px;
    color: #555;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #252525;
}

.path-card {
    background-color: white;
    padding: 20px;
    border-left: 6px solid #7b3f98;
    border-radius: 8px;
    margin-bottom: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.10);
}

.path-number {
    color: #7b3f98;
    font-size: 18px;
    font-weight: bold;
}

.recommendation {
    background: linear-gradient(135deg, #e9d7ef, #f7eef9);
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel("PLR 3 FINAL(5).xlsx")

    return df


df = load_data()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header">
    <h1>🎓 PERSONALIZED LEARNING RECOMMENDATION SYSTEM</h1>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TRAIN ML MODELS
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

course_target = "Recommended Course"
project_target = "Recommended Project"


# Keep only required columns
model_df = df[features + [course_target, project_target]].copy()

model_df = model_df.dropna()


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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ---------------------------------------------------------
# COURSE MODEL
# ---------------------------------------------------------

X = model_df[features]
y_course = model_df[course_target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_course,
    test_size=0.20,
    random_state=42,
    stratify=y_course
)


course_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

course_model.fit(X_train, y_train)


# ---------------------------------------------------------
# PROJECT MODEL
# ---------------------------------------------------------

y_project = model_df[project_target]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X,
    y_project,
    test_size=0.20,
    random_state=42
)


project_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

project_model.fit(X_train_p, y_train_p)


# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Total Students</div>
        <div class="metric-value">{len(df)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Avg CGPA</div>
        <div class="metric-value">{df["CGPA"].mean():.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Avg Assessment</div>
        <div class="metric-value">{df["Assessment Score"].mean():.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Avg Attendance</div>
        <div class="metric-value">{df["Attendance (%)"].mean():.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Avg Study Hours</div>
        <div class="metric-value">
        {df["Suggested Daily Study Hours"].mean():.2f}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# STUDENT SELECTION
# =========================================================

st.header("👨‍🎓 Select Student")


student_ids = df["Student ID"].astype(str).tolist()

selected_id = st.selectbox(
    "Select Student ID",
    student_ids
)


selected_student = df[
    df["Student ID"].astype(str) == selected_id
].iloc[0]


# =========================================================
# STUDENT PROFILE
# =========================================================

st.subheader("📋 Student Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(f"**Name**\n\n{selected_student['Student Name']}")

with col2:
    st.info(f"**Branch**\n\n{selected_student['Branch']}")

with col3:
    st.info(f"**Year**\n\n{selected_student['Year']}")

with col4:
    st.info(f"**CGPA**\n\n{selected_student['CGPA']}")


col1, col2, col3 = st.columns(3)

with col1:
    st.write("### 🎯 Career Goal")
    st.success(selected_student["Career Goal"])

with col2:
    st.write("### 📚 Learning History")
    st.info(selected_student["Learning History"])

with col3:
    st.write("### ⚠️ Skill Gap")
    st.warning(selected_student["Skill Gap"])


# =========================================================
# ML INPUT DATA
# =========================================================

input_data = pd.DataFrame([{
    "College": selected_student["College"],
    "Branch": selected_student["Branch"],
    "Year": selected_student["Year"],
    "CGPA": selected_student["CGPA"],
    "Learning History": selected_student["Learning History"],
    "Skill Gap": selected_student["Skill Gap"],
    "Career Goal": selected_student["Career Goal"],
    "Assessment Score": selected_student["Assessment Score"],
    "Time Available (hrs/day)": selected_student["Time Available (hrs/day)"],
    "Attendance (%)": selected_student["Attendance (%)"],
    "Requirement Type": selected_student["Requirement Type"],
    "Current Level": selected_student["Current Level"]
}])


# =========================================================
# PREDICTION
# =========================================================

predicted_course = course_model.predict(input_data)[0]

predicted_project = project_model.predict(input_data)[0]


# =========================================================
# SKILL AREA
# =========================================================

career = str(selected_student["Career Goal"]).lower()
skill_gap = str(selected_student["Skill Gap"]).lower()


if "data scientist" in career:
    skill_area = "Python, Statistics, Machine Learning and Data Analysis"

elif "data analyst" in career:
    skill_area = "SQL, Excel, Python, Power BI and Data Visualization"

elif "ai" in career or "machine learning" in career:
    skill_area = "Python, Machine Learning, Deep Learning and AI"

elif "software" in career or "developer" in career:
    skill_area = "Programming, DSA, Git and Software Development"

elif "cyber" in career:
    skill_area = "Networking, Linux, Cybersecurity and Ethical Hacking"

else:
    skill_area = str(selected_student["Recommended Skill Area"])


# =========================================================
# GENERATE LEARNING PATH
# =========================================================

current_level = str(
    selected_student["Current Level"]
).lower()


if "beginner" in current_level:

    path = [
        (
            "Foundation",
            "Learn programming fundamentals and basic computer concepts."
        ),
        (
            "Core Skills",
            skill_area
        ),
        (
            "Recommended Course",
            predicted_course
        ),
        (
            "Practice",
            "Solve beginner exercises and small coding problems."
        ),
        (
            "Recommended Project",
            predicted_project
        ),
        (
            "Advanced Learning",
            "Build an advanced project and prepare for certification."
        )
    ]

elif "intermediate" in current_level:

    path = [
        (
            "Skill Gap Improvement",
            skill_area
        ),
        (
            "Recommended Course",
            predicted_course
        ),
        (
            "Hands-on Practice",
            "Practice real-world datasets and industry problems."
        ),
        (
            "Recommended Project",
            predicted_project
        ),
        (
            "Advanced Skills",
            "Learn advanced ML techniques and model optimization."
        ),
        (
            "Career Preparation",
            "Build portfolio projects and prepare for interviews."
        )
    ]

else:

    path = [
        (
            "Advanced Skill Gap",
            skill_area
        ),
        (
            "Advanced Course",
            predicted_course
        ),
        (
            "Advanced Practice",
            "Work with real-world datasets and complex problems."
        ),
        (
            "Major Project",
            predicted_project
        ),
        (
            "Portfolio",
            "Create GitHub projects and a professional portfolio."
        ),
        (
            "Career Preparation",
            "Prepare for interviews, internships and certifications."
        )
    ]


# =========================================================
# MAIN RECOMMENDATION
# =========================================================

st.header("🤖 AI / ML Personalized Recommendation")


st.markdown(
    f"""
    <div class="recommendation">

    <h3>🎯 Recommended Learning Direction</h3>

    <p><b>Career Goal:</b> {selected_student["Career Goal"]}</p>

    <p><b>Current Level:</b> {selected_student["Current Level"]}</p>

    <p><b>Skill Area:</b> {skill_area}</p>

    <p><b>Recommended Course:</b> {predicted_course}</p>

    <p><b>Recommended Project:</b> {predicted_project}</p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LEARNING PATH
# =========================================================

st.header("🗺️ Your Personalized Learning Path")


for i, (title, description) in enumerate(path, start=1):

    st.markdown(
        f"""
        <div class="path-card">

        <div class="path-number">
        STEP {i}
        </div>

        <h3>{title}</h3>

        <p>{description}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# STUDY PLAN
# =========================================================

st.header("⏰ Personalized Study Plan")


available_hours = float(
    selected_student["Time Available (hrs/day)"]
)


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Available Time",
        f"{available_hours:.1f} hrs/day"
    )


with col2:
    recommended_hours = min(
        available_hours,
        3.0
    )

    st.metric(
        "Recommended Study",
        f"{recommended_hours:.1f} hrs/day"
    )


with col3:

    weekly_hours = recommended_hours * 7

    st.metric(
        "Weekly Learning",
        f"{weekly_hours:.1f} hrs"
    )


# =========================================================
# WEEKLY PLAN
# =========================================================

st.subheader("📅 Weekly Practice Plan")


weekly_plan = pd.DataFrame({
    "Day": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ],

    "Activity": [
        "Learn concepts",
        "Watch course + notes",
        "Coding practice",
        "Solve problems",
        "Mini project",
        "Project development",
        "Revision + assessment"
    ],

    "Focus": [
        skill_area,
        predicted_course,
        "Hands-on coding",
        "Problem solving",
        predicted_project,
        predicted_project,
        "Revision"
    ]
})


st.dataframe(
    weekly_plan,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ORIGINAL DATA
# =========================================================

with st.expander("📊 View Student Data"):

    st.dataframe(
        pd.DataFrame([selected_student]),
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <center>
    <b>Personalized Learning Recommendation System</b><br>
    Machine Learning Based Student Learning Path
    </center>
    """,
    unsafe_allow_html=True
)
