import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

import requests
import psycopg2
from psycopg2.extras import Json
from fastapi import FastAPI, HTTPException, Request
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

# Render Postgres: set this env var in Render for your backend service
# Use the *Internal Database URL* (preferred) OR External if needed.
DATABASE_URL = os.getenv("DATABASE_URL")

EMAIL_NOTIFICATIONS_ENABLED = bool(
    RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO
)

if not OPENAI_API_KEY:
    print("[VizAI] WARNING: OPENAI_API_KEY is not set. /run_scan will fail.")

if EMAIL_NOTIFICATIONS_ENABLED:
    print("[VizAI] Email notifications via Resend are ENABLED.")
else:
    print("[VizAI] Email notifications are DISABLED (missing env vars).")

if DATABASE_URL:
    print("[VizAI] Database storage is ENABLED.")
else:
    print("[VizAI] Database storage is DISABLED (DATABASE_URL not set).")

# -------------------------------------------------------------------
# FastAPI app setup
# -------------------------------------------------------------------

app = FastAPI(title="VizAI Scan API")

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
    contactEmail: EmailStr
    requestContact: bool = False
    models: List[str] = []  # kept for backward compatibility


class ScanResponse(BaseModel):
    # DB metadata
    scan_id: Optional[str] = None
    created_at: Optional[str] = None

    # Scores
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int

    # Text
    package_recommendation: str
    package_explanation: str
    strategy_summary: str
    findings: List[str]


# -------------------------------------------------------------------
# DB Helpers
# -------------------------------------------------------------------


def db_conn():
    if not DATABASE_URL:
        return None
    # psycopg2 accepts postgresql://... or postgres://...
    return psycopg2.connect(DATABASE_URL)


def ensure_tables():
    if not DATABASE_URL:
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS vizai_scans (
      scan_id UUID PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL,
      business_name TEXT NOT NULL,
      website TEXT NOT NULL,
      contact_email TEXT NOT NULL,
      request_contact BOOLEAN NOT NULL DEFAULT FALSE,

      discovery_score INT NOT NULL,
      accuracy_score INT NOT NULL,
      authority_score INT NOT NULL,
      overall_score INT NOT NULL,

      package_recommendation TEXT NOT NULL,
      package_explanation TEXT NOT NULL,
      strategy_summary TEXT NOT NULL,

      findings JSONB NOT NULL,
      raw_llm JSONB,

      ip_address TEXT,
      user_agent TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_vizai_scans_created_at ON vizai_scans(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_contact_email ON vizai_scans(contact_email);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_business_name ON vizai_scans(business_name);
    """
    conn = None
    try:
        conn = db_conn()
        if not conn:
            return
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
        print("[VizAI] Database tables ensured.")
    except Exception as e:
        print(f"[VizAI] Failed ensuring tables: {e}")
    finally:
        if conn:
            conn.close()


@app.on_event("startup")
def on_startup():
    ensure_tables()


def insert_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    request: ScanRequest,
    result: ScanResponse,
    raw_llm: Optional[Dict[str, Any]],
    ip_address: Optional[str],
    user_agent: Optional[str],
) -> None:
    if not DATABASE_URL:
        return

    sql = """
    INSERT INTO vizai_scans (
      scan_id, created_at, business_name, website, contact_email, request_contact,
      discovery_score, accuracy_score, authority_score, overall_score,
      package_recommendation, package_explanation, strategy_summary,
      findings, raw_llm, ip_address, user_agent
    )
    VALUES (
      %s, %s, %s, %s, %s, %s,
      %s, %s, %s, %s,
      %s, %s, %s,
      %s, %s, %s, %s
    );
    """
    conn = None
    try:
        conn = db_conn()
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    str(scan_id),
                    created_at,
                    request.businessName,
                    str(request.website),
                    str(request.contactEmail),
                    bool(request.requestContact),

                    int(result.discovery_score),
                    int(result.accuracy_score),
                    int(result.authority_score),
                    int(result.overall_score),

                    result.package_recommendation,
                    result.package_explanation,
                    result.strategy_summary,

                    Json(result.findings),
                    Json(raw_llm) if raw_llm is not None else None,
                    ip_address,
                    user_agent,
                ),
            )
        conn.commit()
        print("[VizAI] Scan inserted into DB.")
    except Exception as e:
        print(f"[VizAI] Failed inserting scan into DB: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# -------------------------------------------------------------------
# Package recommendation logic
# -------------------------------------------------------------------


def derive_recommendation(discovery: int, accuracy: int, authority: int):
    overall = int(round((discovery + accuracy + authority) / 3))

    package = "Standard LMO"
    explanation = ""
    strategy = ""

    if overall >= 80:
        package = "Basic LMO"
        explanation = (
            "Your AI visibility is strong. Basic focuses on monitoring drift and "
            "small adjustments so your profile stays accurate as models evolve."
        )
        strategy = (
            "Lock in a canonical Truth File, validate schema/metadata, and run scheduled "
            "rechecks to catch drift early."
        )
    elif overall >= 40:
        package = "Standard LMO"
        explanation = (
            "Your AI profile is partially correct but has gaps or inconsistencies. "
            "Standard is designed to close gaps and strengthen reliable signals."
        )
        strategy = (
            "Fix core facts (who you are, what you do, where you operate), publish structured data, "
            "and seed authoritative profiles so AI answers become consistently correct."
        )
    else:
        package = "Standard LMO + Add-Ons"
        explanation = (
            "AI currently has a weak or fragmented view of your business. "
            "You’ll need deeper correction plus targeted add-ons to build trust and discoverability."
        )
        strategy = (
            "Start with a Truth File + schema deployment, then layer add-ons like authority seeding "
            "and competitor comparisons to correct the record quickly."
        )

    return overall, package, explanation, strategy


# -------------------------------------------------------------------
# OpenAI analysis (still AI-assisted, but honest)
# -------------------------------------------------------------------


def run_lmo_analysis(business_name: str, website: str, models: List[str]):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    models_str = ", ".join(models) if models else "default"

    prompt = f"""
You are VizAI Scan, an LMO-style business visibility diagnostic.

Business name: {business_name}
Website: {website}
Models (informational): {models_str}

Important constraints:
- Do NOT claim to have browsed the web.
- You may make reasonable inferences from the business name + domain shape only.
- Output must be practical: findings should be actionable and specific.

Return a single JSON object with EXACTLY this structure (keys and types):

{{
  "discovery_score": <integer 0-100>,
  "accuracy_score": <integer 0-100>,
  "authority_score": <integer 0-100>,
  "findings": [
    "<short actionable bullet about discovery>",
    "<short actionable bullet about accuracy>",
    "<short actionable bullet about authority>",
    "<optional extra actionable insight>"
  ]
}}

Guidelines:
- discovery_score: likelihood AI can find/recognize the business reliably.
- accuracy_score: likelihood AI describes services/locations/value correctly.
- authority_score: likelihood AI prefers this business vs directories/competitors.
- Keep findings short, concrete, and helpful.
- Return ONLY JSON (no prose).
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
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {resp.status_code} {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON: {str(e)} | content={content}")

    discovery = int(parsed.get("discovery_score", 50))
    accuracy = int(parsed.get("accuracy_score", 50))
    authority = int(parsed.get("authority_score", 50))
    findings = parsed.get("findings", ["No findings returned."])

    overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)

    result = ScanResponse(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        overall_score=overall,
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings,
    )

    return result, parsed


# -------------------------------------------------------------------
# Email helpers (Resend)
# -------------------------------------------------------------------


def resend_send_email(*, to_email: str, subject: str, text: str) -> bool:
    if not RESEND_API_KEY or not NOTIFY_EMAIL_FROM:
        return False

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "from": NOTIFY_EMAIL_FROM,
        "to": [to_email],
        "subject": subject,
        "text": text,
    }

    try:
        resp = requests.post("https://api.resend.com/emails", headers=headers, json=data, timeout=20)
    except Exception as e:
        print(f"[VizAI] Error calling Resend: {e}")
        return False

    if 200 <= resp.status_code < 300:
        return True

    print(f"[VizAI] Resend error: {resp.status_code} {resp.text}")
    return False


def format_report_text(req: ScanRequest, res: ScanResponse) -> str:
    findings_block = "\n".join(f"- {line}" for line in res.findings)
    return f"""
VizAI Scan Report

Business: {req.businessName}
Website: {req.website}

Scores
- Discovery: {res.discovery_score}
- Accuracy: {res.accuracy_score}
- Authority: {res.authority_score}
- Overall: {res.overall_score}/100

Recommended next step: {res.package_recommendation}
Why: {res.package_explanation}

Strategy summary:
{res.strategy_summary}

Key findings:
{findings_block}

Note: This report is AI-assisted and intended as a directional baseline. For a full multi-source audit and ongoing improvements, reply to this email.
""".strip()


def send_admin_notification(req: ScanRequest, res: ScanResponse, scan_id: Optional[str]) -> bool:
    if not EMAIL_NOTIFICATIONS_ENABLED:
        print("[VizAI] Admin notifications not configured; skipping email.")
        return False

    subject = f"[VizAI Scan] {req.businessName} ({res.overall_score}/100)"
    body = f"""
New VizAI scan submitted.

Scan ID: {scan_id or "n/a"}
Business: {req.businessName}
Website: {req.website}
Contact Email: {req.contactEmail}
Request Contact: {"YES" if req.requestContact else "no"}

Scores
- Discovery: {res.discovery_score}
- Accuracy: {res.accuracy_score}
- Authority: {res.authority_score}
- Overall: {res.overall_score}

Package: {res.package_recommendation}

Findings:
{chr(10).join("- " + f for f in res.findings)}
""".strip()

    return resend_send_email(to_email=NOTIFY_EMAIL_TO, subject=subject, text=body)


def send_user_report(req: ScanRequest, res: ScanResponse) -> bool:
    subject = f"Your VizAI Scan Report: {req.businessName} ({res.overall_score}/100)"
    body = format_report_text(req, res)
    return resend_send_email(to_email=str(req.contactEmail), subject=subject, text=body)


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
def run_scan(payload: ScanRequest, request: Request):
    """
    Main endpoint used by the VizAI frontend.
    - Runs scan
    - Stores in Postgres
    - Emails admin + user
    """
    result, raw_llm = run_lmo_analysis(
        business_name=payload.businessName,
        website=str(payload.website),
        models=payload.models,
    )

    scan_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    # Attach metadata to response (nice for debugging + later retrieval)
    result.scan_id = str(scan_id)
    result.created_at = created_at.isoformat()

    # Capture request context (best-effort)
    ip_address = None
    try:
        # If behind proxy/CDN you might later add X-Forwarded-For handling
        ip_address = request.client.host if request.client else None
    except Exception:
        ip_address = None

    user_agent = request.headers.get("user-agent")

    # Store in DB (best-effort)
    try:
        insert_scan(
            scan_id=scan_id,
            created_at=created_at,
            request=payload,
            result=result,
            raw_llm=raw_llm,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    except Exception as e:
        print(f"[VizAI] Exception while inserting scan: {e}")

    # Email admin + user (best-effort)
    try:
        send_admin_notification(payload, result, result.scan_id)
    except Exception as e:
        print(f"[VizAI] Exception while sending admin notification: {e}")

    try:
        send_user_report(payload, result)
    except Exception as e:
        print(f"[VizAI] Exception while sending user report: {e}")

    return result


@app.post("/test_email")
def test_email():
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "notifications_not_configured"}

    dummy_request = ScanRequest(
        businessName="Test Business",
        website="https://example.com",
        contactEmail="test@example.com",
        requestContact=True,
        models=["default"],
    )

    dummy_result = ScanResponse(
        scan_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        discovery_score=70,
        accuracy_score=65,
        authority_score=60,
        overall_score=65,
        package_recommendation="Standard LMO",
        package_explanation="Test email – standard tier.",
        strategy_summary="This is a test email from VizAI backend.",
        findings=["This is a test finding from /test_email."],
    )

    admin_sent = send_admin_notification(dummy_request, dummy_result, dummy_result.scan_id)
    user_sent = send_user_report(dummy_request, dummy_result)

    return {
        "status": "ok",
        "admin_sent": bool(admin_sent),
        "user_sent": bool(user_sent),
    }


