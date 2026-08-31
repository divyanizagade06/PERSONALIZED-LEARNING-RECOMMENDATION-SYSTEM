```python
import streamlit as st
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Personalized Learning Recommendation",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Personalized Learning Recommendation System")
st.write(
    "Get personalized course, project and study recommendations "
    "based on your learning profile."
)

# -----------------------------
# Load Excel Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("PLR 3 FINAL(5).xlsx")

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔎 Find Student")

student_id = st.sidebar.selectbox(
    "Select Student ID",
    df["Student ID"].dropna().unique()
)

# Get selected student
student = df[df["Student ID"] == student_id].iloc[0]

# -----------------------------
# Student Profile
# -----------------------------
st.header("👤 Student Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Student Name", student["Student Name"])

with col2:
    st.metric("Branch", student["Branch"])

with col3:
    st.metric("Year", student["Year"])

with col4:
    st.metric("CGPA", student["CGPA"])

# -----------------------------
# Academic Information
# -----------------------------
st.subheader("📊 Academic & Learning Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Learning History**")
    st.info(student["Learning History"])

with col2:
    st.write("**Skill Gap**")
    st.warning(student["Skill Gap"])

with col3:
    st.write("**Career Goal**")
    st.success(student["Career Goal"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Assessment Score", student["Assessment Score"])

with col2:
    st.metric("Attendance", f'{student["Attendance (%)"]}%')

with col3:
    st.metric(
        "Available Time",
        f'{student["Time Available (hrs/day)"]} hrs/day'
    )

# -----------------------------
# Recommendation Section
# -----------------------------
st.header("🎯 Personalized Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Recommended Course")
    st.success(student["Recommended Course"])

    st.subheader("💡 Recommended Skill Area")
    st.info(student["Recommended Skill Area"])

with col2:
    st.subheader("🚀 Recommended Project")
    st.success(student["Recommended Project"])

    st.subheader("⏰ Suggested Daily Study")
    st.info(
        f'{student["Suggested Daily Study Hours"]} hours/day'
    )

# -----------------------------
# Requirement Type
# -----------------------------
st.subheader("🧠 Learning Profile")

st.write(
    f'**Requirement Type:** {student["Requirement Type"]}'
)

st.write(
    f'**Current Level:** {student["Current Level"]}'
)

st.write(
    f'**Learning Profile Cluster:** '
    f'{student["Learning_Profile_Cluster"]}'
)

# -----------------------------
# Learning Roadmap
# -----------------------------
st.header("🗺️ Suggested Learning Roadmap")

st.write("### Step 1 — Build Skills")
st.write(
    f'Focus on **{student["Recommended Skill Area"]}**.'
)

st.write("### Step 2 — Complete Course")
st.write(
    f'Complete **{student["Recommended Course"]}**.'
)

st.write("### Step 3 — Build Project")
st.write(
    f'Work on **{student["Recommended Project"]}**.'
)

st.write("### Step 4 — Daily Practice")
st.write(
    f'Spend approximately '
    f'**{student["Suggested Daily Study Hours"]} hours/day** '
    f'on learning and practice.'
)

# -----------------------------
# Dataset Preview
# -----------------------------
with st.expander("📋 View Dataset"):
    st.dataframe(df, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Personalized Learning Recommendation System | "
    "AI/ML Based Student Learning Support"
)
```
