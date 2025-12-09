import os
import json
import requests
import httpx
import asyncio
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# --- Config: OpenAI + Resend ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")  # optional

if not OPENAI_API_KEY:
    print("[VizAI] WARNING: OPENAI_API_KEY is not set. /run_scan will fail.")

if RESEND_API_KEY and NOTIFY_EMAIL_TO:
    print("[VizAI] Email notifications are enabled (Resend).")
else:
    print("[VizAI] Email notifications are DISABLED (missing RESEND_API_KEY or NOTIFY_EMAIL_TO).")

# --- FastAPI app setup ---

app = FastAPI(title="VizAI Scan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod you can restrict to https://vizai.io
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # kept for future, currently unused


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    findings: List[str]


# --- Core LMO analysis using OpenAI ---

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


# --- Package recommendation logic for notifications ---

def recommend_package(avg_score: int) -> str:
    if avg_score >= 80:
        return "Basic"
    if avg_score >= 40:
        return "Standard"
    return "Standard + Add-Ons"


# --- Resend email helper ---

def send_notification_email(
    business_name: str,
    website: str,
    scores: ScanResponse,
) -> None:
    """Send scan summary to your inbox via Resend."""
    if not (RESEND_API_KEY and NOTIFY_EMAIL_TO):
        print("[VizAI] Skipping notification email (missing RESEND_API_KEY or NOTIFY_EMAIL_TO).")
        return

    avg_score = (scores.discovery_score + scores.accuracy_score + scores.authority_score) // 3
    recommended_tier = recommend_package(avg_score)

    subject = f"[VizAI Scan] {business_name} – Avg {avg_score}"

    html_body = f"""
    <h2>New VizAI Scan Result</h2>
    <p><strong>Business:</strong> {business_name}</p>
    <p><strong>Website:</strong> {website}</p>
    <p><strong>Scores:</strong></p>
    <ul>
      <li>Discovery: {scores.discovery_score}</li>
      <li>Accuracy: {scores.accuracy_score}</li>
      <li>Authority: {scores.authority_score}</li>
    </ul>
    <p><strong>Average score:</strong> {avg_score}</p>
    <p><strong>Suggested package:</strong> {recommended_tier}</p>
    <p><strong>Findings:</strong></p>
    <ul>
      {''.join(f'<li>{f}</li>' for f in scores.findings)}
    </ul>
    <hr/>
    <p>This notification was sent automatically by VizAI Scan.</p>
    """

    from_address = NOTIFY_EMAIL_FROM or "VizAI Scan <onboarding@resend.dev>"

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "from": from_address,
        "to": [NOTIFY_EMAIL_TO],
        "subject": subject,
        "html": html_body,
    }

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        print("[VizAI] Notification email sent via Resend.")
    except Exception as e:
        print(f"[VizAI] Failed to send notification email via Resend: {e}")


# --- API endpoints ---

@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest):
    try:
        result = run_lmo_analysis(
            business_name=payload.businessName,
            website=str(payload.website),
            models=payload.models,
        )

        # fire-and-forget notification
        try:
            send_notification_email(
                business_name=payload.businessName,
                website=str(payload.website),
                scores=result,
            )
        except Exception as e:
            # don't break the scan if email fails
            print(f"[VizAI] Error during notification send (non-fatal): {e}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/test_email")
def test_email():
    """Manual test endpoint to confirm Resend + env vars work."""
    dummy_scores = ScanResponse(
        discovery_score=72,
        accuracy_score=68,
        authority_score=61,
        findings=[
            "Test run only – no real scan performed.",
            "This verifies that Resend + notification pipeline works.",
        ],
    )

    try:
        send_notification_email(
            business_name="VizAI Test Business",
            website="https://vizai.io",
            scores=dummy_scores,
        )
        return {"status": "notification_attempted"}
    except Exception as e:
        # Don't expose internals, just log them
        print(f"[VizAI] /test_email error: {e}")
        raise HTTPException(status_code=500, detail="Failed to attempt notification.")




