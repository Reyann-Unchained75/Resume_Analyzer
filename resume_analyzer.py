"""
resume_analyzer.py
------------------
Core logic for the AI Resume Analyzer + Job Matcher.

Pipeline:
    Resume PDF -> text extraction -> cleaning -> (+ job description)
    -> prompt engineering -> Groq LLM -> structured JSON analysis
"""

import io
import json
import os
import re
import unicodedata

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b"
MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

MAX_RESUME_CHARS = 12000        # keeps the prompt inside the model's context
MAX_JD_CHARS = 6000
MIN_RESUME_CHARS = 100          # below this the PDF is probably scanned/image-only

LIST_FIELDS = [
    "candidate_skills",
    "matched_skills",
    "missing_skills",
    "partial_match_skills",
    "strengths",
    "weaknesses",
    "resume_improvements",
    "interview_questions",
]
TEXT_FIELDS = [
    "candidate_summary",
    "experience_analysis",
    "education_analysis",
    "project_match",
    "recommendation",
]


class ResumeAnalyzerError(Exception):
    """Raised for any user-facing failure (bad PDF, missing key, bad LLM output)."""


# ----------------------------------------------------------------------------
# 1. PDF text extraction
# ----------------------------------------------------------------------------
def extract_resume_text(uploaded_file) -> str:
    """
    Extract text from a PDF (file path, file-like object, or Streamlit upload).

    Uses pdfplumber first (better layout handling) and falls back to pypdf.
    """
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        with open(uploaded_file, "rb") as f:
            data = f.read()

    text = ""

    # --- primary: pdfplumber -------------------------------------------------
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
        text = "\n".join(pages)
    except Exception:
        text = ""

    # --- fallback: pypdf -----------------------------------------------------
    if len(text.strip()) < MIN_RESUME_CHARS:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            pages = [(p.extract_text() or "") for p in reader.pages]
            fallback = "\n".join(p for p in pages if p.strip())
            if len(fallback.strip()) > len(text.strip()):
                text = fallback
        except Exception as exc:
            if not text.strip():
                raise ResumeAnalyzerError(f"Unable to read the PDF: {exc}") from exc

    return text


# ----------------------------------------------------------------------------
# 2. Resume cleaning
# ----------------------------------------------------------------------------
BULLET_CHARS = "•●▪■◦▫➢➤►✓✔–—·"


def clean_text(text: str) -> str:
    """Normalise extracted text while keeping line structure (helps the LLM)."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)   # control chars
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                    # de-hyphenate line breaks
    text = re.sub(rf"^[ \t]*[{re.escape(BULLET_CHARS)}][ \t]*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)                             # collapse spaces
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)                          # collapse blank lines
    return text.strip()


# ----------------------------------------------------------------------------
# 3. Prompt engineering
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert technical recruiter and ATS (Applicant Tracking System) "
    "resume analyzer. You compare resumes to job descriptions objectively and "
    "return valid JSON only."
)

JSON_SCHEMA = """{
  "ats_score": 0,
  "candidate_summary": "",
  "candidate_skills": [],
  "matched_skills": [],
  "missing_skills": [],
  "partial_match_skills": [],
  "experience_analysis": "",
  "education_analysis": "",
  "project_match": "",
  "strengths": [],
  "weaknesses": [],
  "resume_improvements": [],
  "recommendation": "",
  "interview_questions": []
}"""


def build_analysis_prompt(resume_text: str, job_description: str) -> str:
    return f"""Analyze the candidate's resume against the job description.

Return ONLY a JSON object with exactly this structure:
{JSON_SCHEMA}

Field rules:
1. ats_score: integer 0-100. Weigh required skills (about 50%), experience and
   projects relevance (about 25%), education (about 10%), and keyword/format
   quality (about 15%). Be realistic; do not inflate the score.
2. candidate_summary: 2-3 sentences describing the candidate.
3. candidate_skills: all technical and soft skills explicitly present in the resume.
4. matched_skills: skills required or preferred by the job description that are clearly present in the resume.
5. missing_skills: important job requirements that are absent from the resume.
6. partial_match_skills: skills that are related or only weakly demonstrated.
7. experience_analysis: how the candidate's work experience, internships and
   projects compare with the role (2-4 sentences).
8. education_analysis: how the candidate's education compares with the role (1-3 sentences).
9. project_match: how relevant the candidate's projects are to the role (1-3 sentences).
10. strengths: 3-6 concise points.
11. weaknesses: 3-6 concise points.
12. resume_improvements: 5-8 specific, actionable recommendations.
13. recommendation: one short paragraph summarising the most important next step.
14. interview_questions: exactly 10 questions tailored to this resume and this job.

Strict rules:
- Do NOT invent skills, employers, degrees or experience that are not in the resume.
- Use only the resume and job description below.
- Use short skill names (e.g. "Python", "REST API", "Docker").
- A skill must never appear in more than one of matched / missing / partial lists.
- Return JSON only. No markdown fences and no commentary.

RESUME:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{job_description[:MAX_JD_CHARS]}
\"\"\"
"""


# ----------------------------------------------------------------------------
# 4. Groq LLM
# ----------------------------------------------------------------------------
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ResumeAnalyzerError(
            "GROQ_API_KEY is missing. Add your key to the .env file "
            "(get one at https://console.groq.com)."
        )
    return Groq(api_key=api_key)


def _chat(client: Groq, messages, temperature=0.2, json_mode=False) -> str:
    kwargs = dict(model=MODEL, messages=messages, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Some models don't support JSON mode - retry once without it.
        if json_mode:
            kwargs.pop("response_format", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc2:
                raise ResumeAnalyzerError(f"Groq API error: {exc2}") from exc2
        else:
            raise ResumeAnalyzerError(f"Groq API error: {exc}") from exc
    return response.choices[0].message.content or ""


# ----------------------------------------------------------------------------
# 5. Structured analysis (parse + validate)
# ----------------------------------------------------------------------------
def parse_json_response(content: str) -> dict:
    """Extract a JSON object from the model output, tolerating fences/extra text."""
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ResumeAnalyzerError("The AI returned an invalid response. Please try again.")


def _as_str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out, seen = [], set()
    for item in value:
        if isinstance(item, dict):                       # e.g. {"question": "..."}
            item = next(iter(item.values()), "")
        item = str(item).strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def normalize_result(raw: dict) -> dict:
    """Coerce the model output into a predictable shape."""
    result = {}

    try:
        score = int(round(float(raw.get("ats_score", 0))))
    except (TypeError, ValueError):
        score = 0
    result["ats_score"] = max(0, min(100, score))

    for key in TEXT_FIELDS:
        result[key] = str(raw.get(key, "") or "").strip()
    for key in LIST_FIELDS:
        result[key] = _as_str_list(raw.get(key))

    # Backwards-compatible aliases the model sometimes uses.
    if not result["experience_analysis"]:
        result["experience_analysis"] = str(raw.get("experience_match", "") or "").strip()
    if not result["education_analysis"]:
        result["education_analysis"] = str(raw.get("education_match", "") or "").strip()

    # A skill should live in exactly one bucket (matched > partial > missing).
    matched = {s.lower() for s in result["matched_skills"]}
    result["partial_match_skills"] = [
        s for s in result["partial_match_skills"] if s.lower() not in matched
    ]
    taken = matched | {s.lower() for s in result["partial_match_skills"]}
    result["missing_skills"] = [s for s in result["missing_skills"] if s.lower() not in taken]

    return result


def analyze_resume(resume_text: str, job_description: str) -> dict:
    """Send resume + JD to Groq and return the structured analysis."""
    if len(resume_text.strip()) < MIN_RESUME_CHARS:
        raise ResumeAnalyzerError(
            "Very little text was extracted from the PDF. "
            "Please upload a text-based (not scanned) PDF."
        )
    if not job_description.strip():
        raise ResumeAnalyzerError("Please provide a job description.")

    client = get_client()
    content = _chat(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_analysis_prompt(resume_text, job_description)},
        ],
        temperature=0.2,
        json_mode=True,
    )
    return normalize_result(parse_json_response(content))


def generate_resume_tips(resume_text: str) -> str:
    """Optional extra: general (JD-independent) resume improvement tips."""
    client = get_client()
    prompt = f"""Review this resume as a professional technical recruiter.
Give 8 practical recommendations covering: ATS compatibility, technical skills,
project descriptions, achievement statements, keywords, formatting,
quantifiable results and the professional summary.
Do not invent details that are not in the resume. Return only a numbered list.

RESUME:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"
"""
    return _chat(
        client,
        [
            {"role": "system", "content": "You are an expert resume coach."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )


# ----------------------------------------------------------------------------
# 6. Plain-text report (for display / download)
# ----------------------------------------------------------------------------
def format_report(result: dict) -> str:
    def block(title, items, bullet="•"):
        if not items:
            return f"{title}\n  (none)\n"
        return f"{title}\n" + "\n".join(f"{bullet} {i}" for i in items) + "\n"

    lines = [
        f"ATS SCORE: {result['ats_score']}%",
        "",
        block("MATCHED SKILLS", result["matched_skills"], "✓"),
        block("MISSING SKILLS", result["missing_skills"]),
        block("PARTIAL MATCH SKILLS", result["partial_match_skills"], "~"),
        block("STRENGTHS", result["strengths"]),
        block("WEAKNESSES", result["weaknesses"]),
        "EXPERIENCE ANALYSIS\n" + (result["experience_analysis"] or "N/A") + "\n",
        "EDUCATION ANALYSIS\n" + (result["education_analysis"] or "N/A") + "\n",
        "RECOMMENDATION\n" + (result["recommendation"] or "N/A") + "\n",
        "RESUME IMPROVEMENTS\n"
        + "\n".join(f"{n}. {i}" for n, i in enumerate(result["resume_improvements"], 1))
        + "\n",
        "INTERVIEW QUESTIONS\n"
        + "\n".join(f"{n}. {q}" for n, q in enumerate(result["interview_questions"], 1)),
    ]
    return "\n".join(lines)
