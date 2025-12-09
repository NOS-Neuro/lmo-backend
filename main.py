import os
import json
import requests
import smtplib
from email.message import EmailMessage
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# --- Config: OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. /run_scan will fail until it is configured.")

# --- Config: Email notifications ---
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")

def notifications_configured() -> bool:
    return all([
        SMTP_HOST,
        SMTP_PORT,
        SMTP_USER,
        SMTP_PASSWORD,
        NOTIFY_EMAIL_FROM,
        NOTIFY_EMAIL_TO,
    ])

if not notifications_configured():
    print("[VizAI] Email notifications are NOT fully configured yet.")
else:
    print("[VizAI] Email notifications are enabled.")

# --- FastAPI app setup ---
app = FastAPI(title="VizAI Scan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to https://vizai.io later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # kept for future use


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    findings: List[str]
    recommended_package: str
    strategy: str


# --- OpenAI call ---
def run_lmo_analysis(business_name: str, website: str, models: List[str]) -> ScanResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    models_str = ", ".join(models) if models else "ChatGPT (simulated only)"

    prompt = f"""
You are an expert in Language Model Optimization (LMO) and AI visibility diagnostics.

A business has requested a VizAI visibility scan.

Business name: {business_name}
Website: {website}
Models selected (for context only): {models_str}

You are NOT actually browsing the web; you are estimating how a typical set of current LLMs would see this business based on the name and domain alone.

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

    discovery = int(parsed.get("discovery_score", 50))
    accuracy = int(parsed.get("accuracy_score", 50))
    authority = int(parsed.get("authority_score", 50))
    findings = parsed.get("findings", ["No findings returned."])

    # Package suggestion logic
    avg_score = (discovery + accuracy + authority) / 3.0

    if avg_score >= 80:
        recommended = "Basic"
        strategy = (
            "You have strong AI visibility today. The VizAI Basic package focuses on keeping your "
            "facts current, monitoring for drift, and making light updates so your scores stay high "
            "as models evolve."
        )
    elif avg_score >= 40:
        recommended = "Standard"
        strategy = (
            "Your AI visibility is uneven. The VizAI Standard package helps stabilize how models "
            "describe you by improving structured data, tightening your core narrative, and adding "
            "regular drift monitoring."
        )
    else:
        recommended = "Standard + Add-Ons"
        strategy = (
            "Your scores suggest that AI models either can’t find you reliably or don’t understand "
            "your offer. We recommend VizAI Standard plus add-ons like API Documentation Publishing, "
            "Schema Deployment, and Competitor Deep Dive to rebuild your presence from the ground up."
        )

    return ScanResponse(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        findings=findings,
        recommended_package=recommended,
        strategy=strategy,
    )


# --- Email notification helper ---
def send_scan_notification(req: ScanRequest, res: ScanResponse) -> None:
    if not notifications_configured():
        print("[VizAI] Skipping email notification – SMTP / NOTIFY env vars incomplete.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = f"[VizAI Scan] {req.businessName} – Avg score {round((res.discovery_score + res.accuracy_score + res.authority_score)/3)}"
        msg["From"] = NOTIFY_EMAIL_FROM
        msg["To"] = NOTIFY_EMAIL_TO

        avg_score = (res.discovery_score + res.accuracy_score + res.authority_score) / 3.0

        body = f"""New VizAI scan submitted:

Business: {req.businessName}
Website: {req.website}

Scores:
- Discovery: {res.discovery_score}
- Accuracy:  {res.accuracy_score}
- Authority: {res.authority_score}
- Average:   {avg_score:.1f}

Recommended package: {res.recommended_package}

Strategy:
{res.strategy}

Findings:
- """ + "\n- ".join(res.findings) + "\n"

        msg.set_content(body)

        print("[VizAI] Attempting to send notification email...")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print("[VizAI] Notification email sent successfully.")
    except Exception as e:
        print(f"[VizAI] Failed to send notification email: {e}")


# --- API endpoints ---

@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest):
    try:
        result = run_lmo_analysis(
            business_name=payload.businessName,
            website=str(payload.website),
            models=payload.models,
        )

        # Fire-and-forget notification
        send_scan_notification(payload, result)

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/test_email")
def test_email():
    """Quick health-check for notification setup."""
    if not notifications_configured():
        return {"status": "notifications_not_configured"}

    dummy_req = ScanRequest(
        businessName="VizAI Test Business",
        website="https://example.com",
        models=["chatgpt"],
    )
    dummy_res = ScanResponse(
        discovery_score=70,
        accuracy_score=65,
        authority_score=60,
        findings=["Test finding 1", "Test finding 2"],
        recommended_package="Standard",
        strategy="This is a test strategy message from /test_email.",
    )

    send_scan_notification(dummy_req, dummy_res)
    return {"status": "notification_attempted"}



