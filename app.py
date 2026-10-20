

import json
import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import streamlit as st

# Load API key from environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# LangChain LLM Setup
def get_llm():
    # Uses gemini-2.0-flash model via Google Generative AI
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4,
    )
    return llm


# Extract text from uploaded PDF resume
@st.cache_data(show_spinner=False)
def extract_pdf_text(file):
    try:
        reader = PdfReader(file)
        text = " ".join(
            [
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            ]
        )
        if not text:
            raise ValueError("No text could be extracted from the PDF file.")
        return text
    except Exception as e:
        raise ValueError(f"PDF Extraction Error: {e}")


# Create LangChain PromptTemplate
def build_prompt():
    return PromptTemplate.from_template(
        """
        Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
        - Technical fields
        - Software engineering
        - Data science
        - Data analysis
        - Big data engineering
        
        Evaluate the following resume against the job description. Consider that the job market 
        is highly competitive. Provide detailed feedback for resume improvement.
    
        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Give response in ONLY this JSON format:
        {{
            "JD Match": "percentage between 0-100",
            "MissingKeywords": ["keyword1", "keyword2"],
            "Profile Summary": "detailed analysis of the match and specific improvement suggestions"
        }}
        """
    )


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Smart ATS Resume Analyzer", layout="centered")
st.title("📄 Smart ATS Resume Analyzer")
st.subheader("Optimize Your Resume for ATS")

with st.sidebar:
    st.title("🎯 Resume Analyzer")
    st.markdown("""
    This smart ATS tool helps you:
    - Match your resume against any job description
    - Identify missing technical and soft keywords
    - Receive expert actionable feedback for improvement
    """)

job_description = st.text_area(
    "Job Description", placeholder="Paste the job description here..."
)
resume_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")

if st.button("🔍 Analyze Resume"):
    if not GOOGLE_API_KEY:
        st.error(
            "API key missing. Please set GOOGLE_API_KEY in your .env file."
        )
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    elif not resume_file:
        st.warning("Please upload a resume in PDF format.")
    else:
        with st.spinner("Analyzing resume with Gemini..."):
            try:
                # 1. Extract text
                resume_text = extract_pdf_text(resume_file)

                # 2. Build prompt & chain
                prompt = build_prompt()
                chain = prompt | get_llm()

                # 3. Invoke LLM
                response = chain.invoke(
                    {
                        "resume_text": resume_text,
                        "job_description": job_description,
                    }
                )

                # 4. Extract raw text from response.content
                # In newer LangChain/Google GenAI versions, content can be a list of parts
                raw_content = response.content
                if isinstance(raw_content, list):
                    # Join all text parts from the list
                    raw_content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in raw_content
                    )

                # 5. Safely parse JSON response
                try:
                    response_json = json.loads(raw_content)
                except Exception:
                    # Fallback: strip markdown code fences and use regex extraction
                    cleaned = re.sub(r"```(?:json)?", "", raw_content).strip("`").strip()
                    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    response_json = json.loads(
                        match.group() if match else "{}"
                    )

                # 5. Render outputs
                st.success("✅ Analysis Complete!")

                st.metric(
                    label="Match Score",
                    value=response_json.get("JD Match", "N/A"),
                )

                st.subheader("🔑 Missing Keywords")
                keywords = response_json.get("MissingKeywords", [])
                if keywords:
                    st.write(", ".join(f"`{kw}`" for kw in keywords))
                else:
                    st.write("No critical missing keywords detected!")

                st.subheader("📝 Profile Summary & Feedback")
                st.write(
                    response_json.get(
                        "Profile Summary", "No summary provided."
                    )
                )

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")