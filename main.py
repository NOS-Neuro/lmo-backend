import os
import json
import smtplib
import requests
from typing import List
from email.mime.text import MIMEText

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# --- Config: OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. The /run_scan endpoint will fail until it is configured.")

# --- Config: Email notifications ---
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")

if not NOTIFY_EMAIL_TO:
    print("INFO: NOTIFY_EMAIL_TO is not set. Scan notifications will be disabled.")


# --- FastAPI app setup ---
app = FastAPI(title="VizAI Scan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down to your vizai.io domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---
class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # kept for future use, not required by frontend


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    findings: List[str]
    recommended_package: str
    strategy: str


# --- Helper: send notification email ---
def send_scan_notification(req: ScanRequest, res: ScanResponse) -> None:
    """Send an email with scan input + results. Fail silently if misconfigured."""
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO):
        # Notifications not configured; do nothing
        return

    try:
        avg_score = int(
            (res.discovery_score + res.accuracy_score + res.authority_score) / 3
        )

        subject = f"[VizAI] New Scan — {req.businessName} ({avg_score}/100)"

        body_lines = [
            "A new VizAI Scan has been run.",
            "",
            "=== Input ===",
            f"Business Name: {req.businessName}",
            f"Website:      {req.website}",
            f"Models:       {', '.join(req.models) if req.models else 'not specified'}",
            "",
            "=== Scores ===",
            f"Discovery: {res.discovery_score}",
            f"Accuracy:  {res.accuracy_score}",
            f"Authority: {res.authority_score}",
            f"Average:   {avg_score}",
            "",
            "=== Recommended Package ===",
            f"{res.recommended_package}",
            "",
            "=== Strategy ===",
            res.strategy,
            "",
            "=== Findings ===",
        ]

        for f in res.findings:
            body_lines.append(f"- {f}")

        body = "\n".join(body_lines)

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = NOTIFY_EMAIL_FROM
        msg["To"] = NOTIFY_EMAIL_TO

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"[VizAI] Notification email sent for {req.businessName}")

    except Exception as e:
        # Don't break the scan if email fails; just log
        print(f"[VizAI] Failed to send notification email: {e}")


# --- Helper: call OpenAI for LMO-style analysis ---
def run_lmo_analysis(business_name: str, website: str, models: List[str]) -> ScanResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    models_str = ", ".join(models) if models else "not specified"

    prompt = f"""
You are an expert in Language Model Optimization (LMO) and AI visibility diagnostics.

A business has requested a VizAI-style scan.

Business name: {business_name}
Website: {website}
Models selected: {models_str}

You are NOT actually browsing the web; instead, you are estimating how a typical set of current large language models would likely see this business based on the name + domain alone and your general knowledge of how AI models reason about businesses.

You must produce three visibility scores and insights:

- discovery_score (0–100): How easily AI systems would be expected to find or recognize this business for relevant queries.
- accuracy_score (0–100): How well AI systems would likely describe what this business actually does.
- authority_score (0–100): How confidently AI systems would be expected to rely on this business vs. other sources when answering questions in its space.

Then you must recommend a service package and short strategy based on the average of those three scores, using this logic:

1. Compute:
   average_score = (discovery_score + accuracy_score + authority_score) / 3

2. If average_score >= 80:
   recommended_package = "Basic"
   strategy = "Your AI visibility is strong. The goal now is to maintain this visibility as models evolve. VizAI Basic provides proactive monitoring and light monthly updates."

3. If 40 <= average_score < 80:
   recommended_package = "Standard"
   strategy = "AI understands your business partially but important gaps exist. VizAI Standard improves your visibility through structured optimization and authority reinforcement."

4. If average_score < 40:
   recommended_package = "Standard + Add-Ons"
   strategy = "Your AI visibility is critically low. AI may not be identifying or describing your business correctly. Immediate optimization is recommended, including truth-file rebuild, schema corrections, and ecosystem seeding."

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
  ],
  "recommended_package": "<Basic | Standard | Standard + Add-Ons>",
  "strategy": "<short plain-language explanation>"
}}

Rules:
- Scores must be integers between 0 and 100.
- findings must be a non-empty list of short, readable bullets.
- recommended_package must follow the rules above.
- strategy must be 1–3 sentences of practical language.
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
    recommended_package = parsed.get("recommended_package", "Standard")
    strategy = parsed.get("strategy", "AI visibility requires improvement. A structured optimization program is recommended.")

    return ScanResponse(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        findings=findings,
        recommended_package=recommended_package,
        strategy=strategy,
    )


# --- API endpoint ---
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


