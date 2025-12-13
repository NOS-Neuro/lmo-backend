import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, HttpUrl, field_validator

import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("vizai")

# -------------------------------------------------------------------
# Config / Environment
# -------------------------------------------------------------------

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")  # e.g. scan@vizai.io
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")      # e.g. you@yourmail.com

DATABASE_URL = os.getenv("DATABASE_URL")  # Render Postgres connection string
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

EMAIL_NOTIFICATIONS_ENABLED = bool(RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO)

logger.info("Email notifications: %s", "ENABLED" if EMAIL_NOTIFICATIONS_ENABLED else "DISABLED")
logger.info("Database storage: %s", "ENABLED" if DATABASE_URL else "DISABLED")

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------

app = FastAPI(
    title="VizAI Scan API",
    version="2.0.0",
    description="VizAI: Real AI visibility scan (Perplexity web-backed evidence).",
)

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

    @field_validator("businessName")
    @classmethod
    def validate_business_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Business name too short")
        if len(v) > 200:
            raise ValueError("Business name too long")
        return v

    @field_validator("website")
    @classmethod
    def validate_website(cls, v: HttpUrl) -> HttpUrl:
        # basic SSRF reduction
        url_str = str(v).lower()
        blocked = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        if any(b in url_str for b in blocked):
            raise ValueError("Cannot scan localhost/internal addresses")
        if "192.168." in url_str or "10." in url_str or "172.16." in url_str:
            raise ValueError("Cannot scan private IP addresses")
        return v


class ScanResponse(BaseModel):
    scan_id: Optional[str] = None
    created_at: Optional[str] = None

    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int

    package_recommendation: str
    package_explanation: str
    strategy_summary: str

    findings: List[str]

    # New, optional (safe for existing frontend)
    recommendations: Optional[Dict[str, Any]] = None

    email_sent: Optional[bool] = None
    disclaimer: str = (
        "This scan is web-backed (Perplexity search) and stores citations as evidence. "
        "Accuracy is currently a proxy until a ground-truth Truth File comparison is added."
    )

# -------------------------------------------------------------------
# DB Pool
# -------------------------------------------------------------------

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "5"))

_db_pool: Optional[pool.ThreadedConnectionPool] = None


def init_db_pool() -> None:
    global _db_pool
    if not DATABASE_URL or _db_pool:
        return
    _db_pool = pool.ThreadedConnectionPool(minconn=DB_POOL_MIN, maxconn=DB_POOL_MAX, dsn=DATABASE_URL)
    logger.info("DB pool initialized (min=%s max=%s)", DB_POOL_MIN, DB_POOL_MAX)


def get_db_conn():
    if not _db_pool:
        return None
    try:
        return _db_pool.getconn()
    except Exception:
        logger.exception("Failed to get DB connection")
        return None


def return_db_conn(conn):
    if _db_pool and conn:
        try:
            _db_pool.putconn(conn)
        except Exception:
            logger.exception("Failed to return DB connection")


def ensure_tables_and_migrations() -> None:
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
      user_agent TEXT,

      email_sent BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_vizai_scans_created_at ON vizai_scans(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_contact_email ON vizai_scans(contact_email);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_business_name ON vizai_scans(business_name);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_overall_score ON vizai_scans(overall_score);
    """

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            return
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
        logger.info("DB ensured + indexes ready")
    except Exception:
        logger.exception("Failed ensuring tables/migrations")
    finally:
        return_db_conn(conn)


def insert_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    request_obj: ScanRequest,
    result: ScanResponse,
    raw_bundle: Optional[Dict[str, Any]],
    ip_address: Optional[str],
    user_agent: Optional[str],
    email_sent: bool,
) -> bool:
    if not DATABASE_URL:
        return False

    sql = """
    INSERT INTO vizai_scans (
      scan_id, created_at, business_name, website, contact_email, request_contact,
      discovery_score, accuracy_score, authority_score, overall_score,
      package_recommendation, package_explanation, strategy_summary,
      findings, raw_llm, ip_address, user_agent, email_sent
    )
    VALUES (
      %s, %s, %s, %s, %s, %s,
      %s, %s, %s, %s,
      %s, %s, %s,
      %s, %s, %s, %s, %s
    );
    """
    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            return False
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    str(scan_id),
                    created_at,
                    request_obj.businessName,
                    str(request_obj.website),
                    str(request_obj.contactEmail),
                    bool(request_obj.requestContact),
                    int(result.discovery_score),
                    int(result.accuracy_score),
                    int(result.authority_score),
                    int(result.overall_score),
                    result.package_recommendation,
                    result.package_explanation,
                    result.strategy_summary,
                    Json(result.findings),
                    Json(raw_bundle) if raw_bundle is not None else None,
                    ip_address,
                    user_agent,
                    bool(email_sent),
                ),
            )
        conn.commit()
        return True
    except Exception:
        logger.exception("Failed inserting scan into DB")
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return False
    finally:
        return_db_conn(conn)

# -------------------------------------------------------------------
# Email (Resend)
# -------------------------------------------------------------------

def resend_send_email(*, to_email: str, subject: str, text: str) -> bool:
    if not RESEND_API_KEY or not NOTIFY_EMAIL_FROM:
        return False
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}
    data = {"from": NOTIFY_EMAIL_FROM, "to": [to_email], "subject": subject, "text": text}
    try:
        r = requests.post("https://api.resend.com/emails", headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return True
    except Exception:
        logger.exception("Resend send failed")
        return False


def format_report_text(req: ScanRequest, res: ScanResponse) -> str:
    findings_block = "\n".join(f"• {line}" for line in res.findings)

    rec_block = ""
    if res.recommendations:
        fix_now = res.recommendations.get("fix_now") or []
        maintain = res.recommendations.get("maintain") or []
        focus = res.recommendations.get("next_scan_focus") or []

        def fmt(items):
            out = []
            for r in items[:5]:
                out.append(f"- {r.get('title')} ({r.get('priority')})")
                steps = r.get("action_steps") or []
                for s in steps[:3]:
                    out.append(f"    • {s}")
            return "\n".join(out)

        rec_block = f"""

RECOMMENDATIONS (Evidence → Actions)
==================================================
Fix now:
{fmt(fix_now) if fix_now else "- None"}

Maintain:
{fmt(maintain) if maintain else "- None"}

Next scan focus:
{', '.join(focus) if focus else 'None'}
""".rstrip()

    return f"""
VizAI Scan Report
==================================================

Business: {req.businessName}
Website: {req.website}

Scores
- Discovery: {res.discovery_score}/100
- Accuracy:  {res.accuracy_score}/100
- Authority: {res.authority_score}/100
- Overall:   {res.overall_score}/100

Recommended next step: {res.package_recommendation}
Why: {res.package_explanation}

Strategy summary:
{res.strategy_summary}

Key findings:
{findings_block}
{rec_block}

Note:
{res.disclaimer}
""".strip()


def send_admin_notification(req: ScanRequest, res: ScanResponse) -> bool:
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return False

    subject = f"[VizAI Scan] {req.businessName} ({res.overall_score}/100)"
    body = f"""
New VizAI scan submitted.

Business: {req.businessName}
Website: {req.website}
Contact Email: {req.contactEmail}
Request Contact: {"YES" if req.requestContact else "no"}

Scores: D={res.discovery_score} A={res.accuracy_score} Au={res.authority_score} Overall={res.overall_score}
Package: {res.package_recommendation}
""".strip()
    return resend_send_email(to_email=NOTIFY_EMAIL_TO, subject=subject, text=body)


def send_user_report(req: ScanRequest, res: ScanResponse) -> bool:
    subject = f"Your VizAI Scan Report: {req.businessName} ({res.overall_score}/100)"
    return resend_send_email(to_email=str(req.contactEmail), subject=subject, text=format_report_text(req, res))

# -------------------------------------------------------------------
# Startup / Shutdown
# -------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    logger.info("VizAI starting up...")
    if DATABASE_URL:
        init_db_pool()
        ensure_tables_and_migrations()
    logger.info("VizAI startup complete")


@app.on_event("shutdown")
def on_shutdown():
    global _db_pool
    logger.info("VizAI shutting down...")
    if _db_pool:
        try:
            _db_pool.closeall()
        except Exception:
            logger.exception("Error closing DB pool")
    _db_pool = None

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "VizAI Scan API", "version": "2.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "operational",
            "database": "operational" if DATABASE_URL else "not_configured",
            "email": "operational" if EMAIL_NOTIFICATIONS_ENABLED else "not_configured",
        },
    }


@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest, request: Request):
    # request context
    client_ip = None
    try:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
    except Exception:
        client_ip = None
    user_agent = request.headers.get("user-agent")

    logger.info("Scan started: business=%s ip=%s", payload.businessName, client_ip)

    # run real scan engine
    try:
        from scan_engine_real import run_real_scan_perplexity
        real_result, raw_bundle = run_real_scan_perplexity(
            business_name=payload.businessName,
            website=str(payload.website),
        )
    except Exception as e:
        logger.exception("Real scan engine failed: %s", e)
        raise HTTPException(status_code=503, detail="Scan engine unavailable. Please try again.")

    # map to API response
    result = ScanResponse(
        discovery_score=int(real_result.discovery_score),
        accuracy_score=int(real_result.accuracy_score),
        authority_score=int(real_result.authority_score),
        overall_score=int(real_result.overall_score),
        package_recommendation=str(real_result.package_recommendation),
        package_explanation=str(real_result.package_explanation),
        strategy_summary=str(real_result.strategy_summary),
        findings=list(real_result.findings or []),
        recommendations=(raw_bundle.get("recommendations") if isinstance(raw_bundle, dict) else None),
    )

    # metadata
    scan_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    result.scan_id = str(scan_id)
    result.created_at = created_at.isoformat()

    # emails (best-effort)
    email_sent = False
    try:
        if EMAIL_NOTIFICATIONS_ENABLED:
            try:
                send_admin_notification(payload, result)
            except Exception:
                logger.exception("Admin email failed (non-fatal)")
            try:
                email_sent = send_user_report(payload, result)
            except Exception:
                logger.exception("User email failed (non-fatal)")
        else:
            logger.info("Email not configured; skipping send")
    except Exception:
        logger.exception("Email block failed (non-fatal)")
    result.email_sent = bool(email_sent)

    # DB insert (best-effort)
    try:
        if DATABASE_URL:
            insert_scan(
                scan_id=scan_id,
                created_at=created_at,
                request_obj=payload,
                result=result,
                raw_bundle=raw_bundle if isinstance(raw_bundle, dict) else None,
                ip_address=client_ip,
                user_agent=user_agent,
                email_sent=bool(email_sent),
            )
    except Exception:
        logger.exception("DB insert failed (non-fatal)")

    logger.info("Scan complete: scan_id=%s overall=%s", scan_id, result.overall_score)
    return result


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "path": str(request.url)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})





