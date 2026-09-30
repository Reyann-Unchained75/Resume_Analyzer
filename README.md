# AI Resume Analyzer + Job Matcher

Upload a PDF resume, paste a job description, and get an ATS-style match score,
skill gaps, strengths/weaknesses, improvement tips and tailored interview
questions — powered by a Groq-hosted LLM and a Streamlit dashboard.

## Workflow

```
Resume PDF
    ↓
PDF Text Extraction   (pdfplumber, pypdf fallback)
    ↓
Resume Cleaning
    ↓
Job Description
    ↓
Prompt Engineering
    ↓
Groq LLM
    ↓
Structured Analysis   (validated JSON)
    ↓
Streamlit Dashboard
```

## Features

- Resume PDF upload and job description input
- Resume text extraction and cleaning
- Skills extraction (skills found in the resume)
- Experience, education and project analysis
- Matched, missing and partial-match skills
- ATS-style score (0–100)
- Strengths and weaknesses
- Improvement recommendations
- 10 tailored interview questions
- Downloadable text report
- Optional general resume tips

## Example output

```
ATS SCORE: 82%

MATCHED SKILLS
✓ Python
✓ SQL
✓ REST API
✓ Git

MISSING SKILLS
• Docker
• AWS

STRENGTHS
• Strong Python knowledge
• Good project experience

RECOMMENDATION
Add cloud deployment and Docker projects.
```

## Project structure

```
resume_analyzer/
├── app.py                # Streamlit UI
├── resume_analyzer.py    # extraction, cleaning, prompt, Groq call, validation
├── requirements.txt
├── .env                  # your API key (never commit)
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Get a free API key from the [Groq Console](https://console.groq.com) and put it in `.env`:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

Optionally set `GROQ_MODEL` in `.env` to use a different Groq model
(DEFAULT_MODEL = "openai/gpt-oss-120b").

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a **text-based** PDF resume, paste a job
description (or click *Use sample job description*) and press **Analyze Resume**.

## Notes

- Scanned/image-only PDFs have no extractable text; export your resume as a text PDF.
- The ATS score is an AI estimate of resume–JD fit, not the output of a real ATS.
- Resume and JD text are sent to the Groq API for analysis.
- Never commit `.env` to GitHub (it is listed in `.gitignore`).

## Ideas for the next version

DOCX resume support · multiple resume comparison · PDF report export · resume
rewriting · candidate ranking · MySQL storage and resume history · login/signup ·
interview chatbot · RAG over job/company information · Streamlit Cloud deployment.
