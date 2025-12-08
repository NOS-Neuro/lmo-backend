import os
import json
import requests
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # adjust if you prefer a different model

if not OPENAI_API_KEY:
    # Render will set this; locally you'll set it in your shell
    print("WARNING: OPENAI_API_KEY is not set. The /run_scan endpoint will fail until it is configured.")

# --- FastAPI app setup ---
app = FastAPI(title="LuminAI Scan API")

# CORS: in production, you can restrict to your HF Space URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://your-space-name.hf.space"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---
class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # e.g. ["chatgpt", "claude", "gemini"]


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    findings: List[str]


# --- Helper: call OpenAI for LMO-style analysis ---
def run_lmo_analysis(business_name: str, website: str, models: List[str]) -> ScanResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    models_str = ", ".join(models) if models else "not specified"

    prompt = f"""
You are an expert in Language Model Optimization (LMO).

A business has requested an LMO-style scan.

Business name: {business_name}
Website: {website}
Models selected: {models_str}

You are NOT actually browsing the web; instead, you are estimating how a typical set of current LLMs would likely see this business based on the name + domain alone.

Return a single JSON object with this exact structure:

{{
  "discovery_score": <integer 0-100>,
  "accuracy_score": <integer 0-100>,
  "authority_score": <integer 0-100>,
  "findings": [
    "<short bullet about discovery>",
    "<short bullet about accuracy>",
    "<short bullet about authority>",
    "<optional extra insight>"
  ]
}}

Rules:
- Scores must be integers between 0 and 100.
- findings must be a non-empty list of short, readable bullets.
- Do not include any surrounding text, explanation, or markdown. Return ONLY JSON.
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=40,
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {resp.status_code} {resp.text}",
        )

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM JSON: {str(e)} | content={content}",
        )

    return ScanResponse(
        discovery_score=int(parsed.get("discovery_score", 50)),
        accuracy_score=int(parsed.get("accuracy_score", 50)),
        authority_score=int(parsed.get("authority_score", 50)),
        findings=parsed.get("findings", ["No findings returned."]),
    )


# --- API endpoint ---
@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest):
    try:
        return run_lmo_analysis(
          business_name=payload.businessName,
          website=str(payload.website),
          models=payload.models,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
