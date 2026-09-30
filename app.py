"""
app.py
------
Streamlit dashboard for the AI Resume Analyzer + Job Matcher.

Run with:  streamlit run app.py
"""

from html import escape as html_escape

import streamlit as st

from resume_analyzer import (
    MODEL,
    ResumeAnalyzerError,
    analyze_resume,
    clean_text,
    extract_resume_text,
    format_report,
    generate_resume_tips,
)

# ----------------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .main-title { font-size: 40px; font-weight: 700; margin-bottom: 0; }
    .subtitle   { font-size: 17px; opacity: 0.7; margin-bottom: 20px; }
    .score-box  { padding: 22px; border-radius: 14px; text-align: center;
                  border: 2px solid var(--c); }
    .score-num  { font-size: 56px; font-weight: 800; line-height: 1.1; color: var(--c); }
    .score-lbl  { font-size: 14px; letter-spacing: 1px; opacity: 0.75; }
    .chip       { display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
                  border-radius: 999px; font-size: 14px; border: 1px solid; }
    .chip-ok    { border-color: #2e9e5b; color: #2e9e5b; }
    .chip-miss  { border-color: #d64545; color: #d64545; }
    .chip-part  { border-color: #d99a1e; color: #d99a1e; }
    .chip-info  { border-color: #4a7bd0; color: #4a7bd0; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_JD = """Trainee Software Engineer

We are looking for a Trainee Software Engineer with knowledge of Python and Java.

Requirements:
- Strong programming fundamentals
- Python
- Java
- Object Oriented Programming
- SQL
- MySQL
- Data Structures and Algorithms
- Git
- REST APIs
- Problem solving
- Good communication skills

Responsibilities:
- Develop software applications
- Write clean and maintainable code
- Work with databases
- Develop REST APIs
- Debug applications
- Participate in code reviews
- Work with the development team

Good to have:
- Spring Boot
- FastAPI
- Docker
- AWS
"""

if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""


def load_sample_jd():
    st.session_state["jd_text"] = SAMPLE_JD


def show_chips(items, css_class, empty_message):
    if not items:
        st.write(empty_message)
        return
    html = "".join(f'<span class="chip {css_class}">{html_escape(i)}</span>' for i in items)
    st.markdown(html, unsafe_allow_html=True)


def score_colour(score: int) -> str:
    if score >= 75:
        return "#2e9e5b"
    if score >= 50:
        return "#d99a1e"
    return "#d64545"


def score_label(score: int) -> str:
    if score >= 80:
        return "Excellent match"
    if score >= 65:
        return "Good match"
    if score >= 50:
        return "Moderate match"
    return "Weak match"


# ----------------------------------------------------------------------------
# Header + sidebar
# ----------------------------------------------------------------------------
st.markdown('<div class="main-title">📄 AI Resume Analyzer & Job Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Compare your resume with a job description using Generative AI.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "Upload a text-based PDF resume, paste a job description, and get an "
        "ATS-style score, skill gaps, recommendations and interview questions."
    )
    st.divider()
    st.subheader("Pipeline")
    st.markdown(
        "1. PDF text extraction\n"
        "2. Resume cleaning\n"
        "3. Prompt engineering\n"
        "4. Groq LLM\n"
        "5. Structured analysis\n"
        "6. Streamlit dashboard"
    )
    st.divider()
    st.caption(f"Model: `{MODEL}`")

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Resume (PDF only)", type=["pdf"])

with right:
    st.subheader("2. Job Description")
    st.text_area(
        "Paste the job description",
        key="jd_text",
        height=260,
        placeholder="Paste the complete job description here...",
    )
    st.button("Use sample job description", on_click=load_sample_jd)

st.divider()
analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Run analysis
# ----------------------------------------------------------------------------
if analyze_clicked:
    jd = st.session_state["jd_text"]
    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")
    elif not jd.strip():
        st.warning("Please enter a job description.")
    else:
        try:
            with st.spinner("Extracting resume text..."):
                resume_text = clean_text(extract_resume_text(uploaded_file))
            with st.spinner("Analyzing resume against the job description..."):
                result = analyze_resume(resume_text, jd)
            st.session_state["analysis"] = result
            st.session_state["resume_text"] = resume_text
            st.session_state.pop("tips", None)
        except ResumeAnalyzerError as exc:
            st.error(str(exc))
        except Exception as exc:  # last-resort guard so the app never crashes
            st.error(f"Unexpected error: {exc}")

# ----------------------------------------------------------------------------
# Results dashboard
# ----------------------------------------------------------------------------
if "analysis" in st.session_state:
    result = st.session_state["analysis"]
    score = result["ats_score"]
    colour = score_colour(score)

    st.header("Resume Analysis Report")

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        st.markdown(
            f'<div class="score-box" style="--c:{colour}">'
            f'<div class="score-lbl">ATS SCORE</div>'
            f'<div class="score-num">{score}%</div>'
            f'<div class="score-lbl">{score_label(score)}</div></div>',
            unsafe_allow_html=True,
        )
    c2.metric("Matched Skills", len(result["matched_skills"]))
    c3.metric("Missing Skills", len(result["missing_skills"]))
    c4.metric("Partial Matches", len(result["partial_match_skills"]))
    st.progress(score / 100)

    if result["candidate_summary"]:
        st.subheader("Candidate Summary")
        st.write(result["candidate_summary"])

    tab_skills, tab_profile, tab_sw, tab_improve, tab_interview, tab_export = st.tabs(
        ["Skills", "Experience & Education", "Strengths & Weaknesses",
         "Recommendations", "Interview Questions", "Export"]
    )

    with tab_skills:
        a, b = st.columns(2)
        with a:
            st.subheader("✓ Matched Skills")
            show_chips(result["matched_skills"], "chip-ok", "No strong matches identified.")
        with b:
            st.subheader("• Missing Skills")
            show_chips(result["missing_skills"], "chip-miss", "No major gaps identified.")
        st.subheader("~ Partial Match Skills")
        show_chips(result["partial_match_skills"], "chip-part", "None.")
        st.subheader("Skills Found in Resume")
        show_chips(result["candidate_skills"], "chip-info", "None extracted.")

    with tab_profile:
        st.subheader("Experience Analysis")
        st.write(result["experience_analysis"] or "Not available.")
        st.subheader("Education Analysis")
        st.write(result["education_analysis"] or "Not available.")
        st.subheader("Project Match")
        st.write(result["project_match"] or "Not available.")

    with tab_sw:
        a, b = st.columns(2)
        with a:
            st.subheader("Strengths")
            for item in result["strengths"]:
                st.write(f"✓ {item}")
        with b:
            st.subheader("Weaknesses")
            for item in result["weaknesses"]:
                st.write(f"• {item}")

    with tab_improve:
        if result["recommendation"]:
            st.info(f"**Recommendation:** {result['recommendation']}")
        st.subheader("Resume Improvement Recommendations")
        for n, item in enumerate(result["resume_improvements"], 1):
            st.write(f"{n}. {item}")

        with st.expander("General resume tips (independent of the job description)"):
            if st.button("Generate tips"):
                try:
                    with st.spinner("Generating tips..."):
                        st.session_state["tips"] = generate_resume_tips(st.session_state["resume_text"])
                except ResumeAnalyzerError as exc:
                    st.error(str(exc))
            if "tips" in st.session_state:
                st.write(st.session_state["tips"])

    with tab_interview:
        st.subheader("AI-Generated Interview Questions")
        for n, q in enumerate(result["interview_questions"], 1):
            st.write(f"{n}. {q}")

    with tab_export:
        report = format_report(result)
        st.text(report)
        st.download_button(
            "⬇️ Download report (.txt)",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
        )

st.divider()
st.caption("AI Resume Analyzer | Python + Streamlit + Groq")
