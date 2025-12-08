import os
import json
import requests
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # or any chat-capable model

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

# --- FastAPI setup ---
app = FastAPI(title="LuminAI Scan API")

# In production, you can restrict this to your site origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://jneuro-lmo-website.hf.space"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # ["chatgpt", "claude", "gemini"] etc.


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    findings: List[str]


# --- Helper: call OpenAI and get structured JSON ---
def run_lmo_analysis(business_name: str, website: str, models: List[str]) -> ScanResponse:
    prompt = f"""
You are an expert in Language Model Optimization (LMO).

A business has requested an LMO scan.

Business name: {business_name}
Website: {website}
Models selected: {", ".join(models) if models else "not specified"}

You are NOT actually browsing the web; instead, you are estimating how a typical set of current LLMs would likely see this business based on the name + domain.

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
        # Ask for JSON-formatted output
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

    # Basic validation / defaults
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

