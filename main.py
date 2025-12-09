import os
import json
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, EmailStr

# -------------------------------------------------------------------
# Config / Environment
# -------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")  # e.g. "scan@vizai.io"
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")      # e.g. "you@yourmail.com"

EMAIL_NOTIFICATIONS_ENABLED = bool(
    RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO
)

if not OPENAI_API_KEY:
    print("[VizAI] WARNING: OPENAI_API_KEY is not set. /run_scan will fail.")

if EMAIL_NOTIFICATIONS_ENABLED:
    print("[VizAI] Email notifications via Resend are ENABLED.")
else:
    print("[VizAI] Email notifications are DISABLED (missing env vars).")

# -------------------------------------------------------------------
# FastAPI app setup
# -------------------------------------------------------------------

app = FastAPI(title="VizAI Scan API")

# In production you can restrict this to your actual frontend origin
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # kept for future use, currently informational only
    contactEmail: Optional[EmailStr] = None  # optional user email


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int
    package_recommendation: str
    package_explanation: str
    strategy_summary: str
    findings: List[str]


# -------------------------------------------------------------------
# Helper: package recommendation logic
# -------------------------------------------------------------------


def derive_recommendation(
    discovery: int, accuracy: int, authority: int
) -> ScanResponse:
    overall = int(round((discovery + accuracy + authority) / 3))

    # Default texts; refined per band below
    package = "Standard LMO"
    explanation = ""
    strategy = ""

    if overall >= 80:
        package = "Basic LMO"
        explanation = (
            "Your AI visibility is strong. The Basic package focuses on "
            "monitoring and light adjustments so scores stay high as models evolve."
        )
        strategy = (
            "Lock in a clean Truth File, enable monthly score checks, and correct "
            "any small drifts before they become visible to customers."
        )
    elif overall >= 40:
        package = "Standard LMO"
        explanation = (
            "Your AI profile is partially correct but has noticeable gaps or "
            "inconsistencies. Standard is designed to close those gaps and raise "
            "confidence across models."
        )
        strategy = (
            "Prioritize fixing core facts (who you are, what you do, where you "
            "operate), then publish structured data and registry entries to push "
            "scores into the 70–80 range."
        )
    else:
        package = "Standard LMO + Add-Ons"
        explanation = (
            "AI currently has a weak or fragmented view of your business. "
            "You’ll need a deeper correction pass plus targeted add-ons to "
            "build a strong presence."
        )
        strategy = (
            "Start with Standard LMO to establish a clean Truth File and "
            "baseline visibility, then layer add-ons such as Schema Deployment, "
            "Competitor Deep Dive, and Dataset Creation to quickly improve how "
            "models discover and describe you."
        )

    return overall, package, explanation, strategy


# -------------------------------------------------------------------
# Helper: call OpenAI for LMO-style analysis
# -------------------------------------------------------------------


def run_lmo_analysis(business_name: str, website: str, models: List[str]) -> ScanResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY is not configured on the server."
        )

    models_str = ", ".join(models) if models else "ChatGPT (default)"

    prompt = f"""
You are an expert in Language Model Optimization (LMO).

A business has requested an AI visibility scan.

Business name: {business_name}
Website: {website}
Models selected (informational only): {models_str}

You are NOT actually browsing the web. Instead, estimate how a typical set of
current large language models would likely see this business based on the name
and domain alone.

Return a single JSON object with EXACTLY this structure (keys and types):

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

Guidelines:
- discovery_score: how likely models are to find / recognize the business.
- accuracy_score: how correctly they describe services, locations, and value.
- authority_score: how likely models are to rely on this business vs generic
  directories or competitors.
- Scores MUST be integers between 0 and 100.
- findings MUST be a non-empty list of short, readable bullets.
- Do NOT include any explanation outside the JSON. ONLY return JSON.
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

    overall, package, explanation, strategy = derive_recommendation(
        discovery, accuracy, authority
    )

    return ScanResponse(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        overall_score=overall,
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings,
    )


# -------------------------------------------------------------------
# Helper: send notification via Resend
# -------------------------------------------------------------------


def send_notification(request: ScanRequest, result: ScanResponse) -> bool:
    if not EMAIL_NOTIFICATIONS_ENABLED:
        print("[VizAI] Notifications not configured; skipping email.")
        return False

    subject = f"[VizAI Scan] {request.businessName} ({result.overall_score}/100)"

    models_str = ", ".join(request.models) if request.models else "ChatGPT (default)"

    findings_block = "\n".join(f"- {line}" for line in result.findings)

    body = f"""
New VizAI scan submitted.

Business: {request.businessName}
Website: {request.website}
Models (informational): {models_str}
Contact Email: {request.contactEmail or "n/a"}

Scores
- Discovery: {result.discovery_score}
- Accuracy: {result.accuracy_score}
- Authority: {result.authority_score}
- Overall: {result.overall_score}

Recommended Package: {result.package_recommendation}
Explanation: {result.package_explanation}

Strategy Summary:
{result.strategy_summary}

Findings:
{findings_block}
""".strip()

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "from": NOTIFY_EMAIL_FROM,
        "to": [NOTIFY_EMAIL_TO],
        "subject": subject,
        "text": body,
    }

    try:
        resp = requests.post(
            "https://api.resend.com/emails", headers=headers, json=data, timeout=20
        )
    except Exception as e:
        print(f"[VizAI] Error calling Resend: {e}")
        return False

    if 200 <= resp.status_code < 300:
        print("[VizAI] Notification email sent via Resend.")
        return True

    print(
        f"[VizAI] Failed to send email via Resend: "
        f"{resp.status_code} {resp.text}"
    )
    return False


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@app.get("/")
def root():
    return {"status": "ok", "service": "VizAI Scan API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest):
    """
    Main endpoint used by the VizAI frontend.
    """
    result = run_lmo_analysis(
        business_name=payload.businessName,
        website=str(payload.website),
        models=payload.models,
    )

    # Fire-and-forget style notification (errors are logged, not surfaced to user)
    try:
        sent = send_notification(payload, result)
        if sent:
            print("[VizAI] Scan notification attempted: success.")
        else:
            print("[VizAI] Scan notification attempted: not sent.")
    except Exception as e:
        print(f"[VizAI] Exception while sending notification: {e}")

    return result


@app.post("/test_email")
def test_email():
    """
    Simple endpoint to confirm email notifications are wired correctly.
    Uses dummy scores and business info.
    """
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "notifications_not_configured"}

    dummy_request = ScanRequest(
        businessName="Test Business",
        website="https://example.com",
        models=["chatgpt"],
        contactEmail="test@example.com",
    )

    dummy_result = ScanResponse(
        discovery_score=70,
        accuracy_score=65,
        authority_score=60,
        overall_score=65,
        package_recommendation="Standard LMO",
        package_explanation="Test email – standard tier.",
        strategy_summary="This is a test email from VizAI backend.",
        findings=["This is a test finding from /test_email."],
    )

    sent = send_notification(dummy_request, dummy_result)

    return {
        "status": "notification_attempted" if sent else "notification_failed",
    }



