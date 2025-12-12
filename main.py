import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

import requests
import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, EmailStr, field_validator

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
    version="1.3.0",
    description="VizAI: Real scan (Perplexity-first) with audit evidence stored.",
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
        """
        Best-effort SSRF reduction:
        - block localhost
        - block common private ranges (string check; not perfect but a helpful guard)
        """
        url_str = str(v).lower()

        blocked = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        if any(b in url_str for b in blocked):
            raise ValueError("Cannot scan localhost/internal addresses")

        # basic private range checks (best-effort)
        if "192.168." in url_str or "10." in url_str or "172.16." in url_str:
            raise ValueError("Cannot scan private IP addresses")

        return v


class ScanResponse(BaseModel):
    # DB metadata (optional)
    scan_id: Optional[str] = None
    created_at: Optional[str] = None

    # Scores
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int

    # Recommendations
    package_recommendation: str
    package_explanation: str
    strategy_summary: str
    findings: List[str]

    # Operational extras (won't break your frontend)
    email_sent: Optional[bool] = None
    disclaimer: str = (
        "This scan is web-backed (Perplexity search_mode='web') and stores citations as audit evidence. "
        "Accuracy is a proxy until a Truth File compare is implemented."
    )

# -------------------------------------------------------------------
# DB Pool
# -------------------------------------------------------------------

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "5"))

_db_pool: Optional[pool.ThreadedConnectionPool] = None


def init_db_pool() -> None:
    global _db_pool
    if not DATABASE_URL:
        return
    if _db_pool:
        return

    _db_pool = pool.ThreadedConnectionPool(
        minconn=DB_POOL_MIN,
        maxconn=DB_POOL_MAX,
        dsn=DATABASE_URL,
    )
    logger.info("DB pool initialized (min=%s max=%s)", DB_POOL_MIN, DB_POOL_MAX)


def get_db_conn():
    if not _db_pool:
        return None
    try:
        return _db_pool.getconn()
    except Exception as e:
        logger.exception("Failed to get DB connection from pool: %s", e)
        return None


def return_db_conn(conn):
    if _db_pool and conn:
        try:
            _db_pool.putconn(conn)
        except Exception as e:
            logger.exception("Failed to return DB connection to pool: %s", e)


def ensure_tables_and_migrations() -> None:
    """
    Safe startup:
    - create table if not exists
    - add columns/indexes if missing
    """
    if not DATABASE_URL:
        return

    ddl_create = """
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
    """

    ddl_migrate = """
    ALTER TABLE vizai_scans
      ADD COLUMN IF NOT EXISTS email_sent BOOLEAN DEFAULT FALSE;

    CREATE INDEX IF NOT EXISTS idx_vizai_scans_created_at ON vizai_scans(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_contact_email ON vizai_scans(contact_email);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_business_name ON vizai_scans(business_name);
    CREATE INDEX IF NOT EXISTS idx_vizai_scans_overall_score ON vizai_scans(overall_score);
    """

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            logger.warning("DB not available during ensure_tables; skipping")
            return

        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl_create)
            cur.execute(ddl_migrate)

        logger.info("DB ensured + migrations applied")
    except Exception as e:
        logger.exception("Failed ensuring tables/migrations: %s", e)
    finally:
        return_db_conn(conn)


def insert_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    request_obj: ScanRequest,
    result: ScanResponse,
    raw_llm: Optional[Dict[str, Any]],
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
            logger.warning("DB unavailable; scan not stored")
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
                    Json(raw_llm) if raw_llm is not None else None,
                    ip_address,
                    user_agent,
                    bool(email_sent),
                ),
            )
        conn.commit()
        logger.info("Scan inserted into DB: %s", scan_id)
        return True
    except Exception as e:
        logger.exception("Failed inserting scan into DB: %s", e)
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return False
    finally:
        return_db_conn(conn)

# -------------------------------------------------------------------
# Package Logic
# -------------------------------------------------------------------

def derive_recommendation(discovery: int, accuracy: int, authority: int):
    overall = int(round((discovery + accuracy + authority) / 3))

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
# Email (Resend)
# -------------------------------------------------------------------

def resend_send_email(*, to_email: str, subject: str, text: str) -> bool:
    if not RESEND_API_KEY or not NOTIFY_EMAIL_FROM:
        return False

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"from": NOTIFY_EMAIL_FROM, "to": [to_email], "subject": subject, "text": text}

    try:
        resp = requests.post("https://api.resend.com/emails", headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.exception("Resend send failed: %s", e)
        return False


def format_report_text(req: ScanRequest, res: ScanResponse) -> str:
    findings_block = "\n".join(f"• {line}" for line in res.findings)
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

Note:
{res.disclaimer}

Reply to this email if you’d like a full multi-source audit + ongoing monitoring.
""".strip()


def send_admin_notification(req: ScanRequest, res: ScanResponse, scan_id: str) -> bool:
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return False

    subject = f"[VizAI Scan] {req.businessName} ({res.overall_score}/100)"
    body = f"""
New VizAI scan submitted.

Scan ID: {scan_id}
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
# Startup / Shutdown
# -------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    logger.info("VizAI starting up...")
    if DATABASE_URL:
        try:
            init_db_pool()
            ensure_tables_and_migrations()
        except Exception:
            logger.exception("DB init failed (non-fatal)")
    logger.info("VizAI startup complete")


@app.on_event("shutdown")
def on_shutdown():
    global _db_pool
    logger.info("VizAI shutting down...")
    if _db_pool:
        try:
            _db_pool.closeall()
            logger.info("DB pool closed")
        except Exception:
            logger.exception("Error closing DB pool")
        _db_pool = None

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "VizAI Scan API",
        "version": "1.3.0",
        "endpoints": {
            "scan": "/run_scan",
            "health": "/health",
            "docs": "/docs",
            "scan_by_id": "/scan/{scan_id}",
            "admin_stats": "/admin/stats",
        },
    }


@app.get("/health")
def health():
    services = {
        "api": "operational",
        "database": "not_configured" if not DATABASE_URL else "unknown",
        "email": "operational" if EMAIL_NOTIFICATIONS_ENABLED else "not_configured",
        "scan_engine": "unknown",
    }

    # DB check
    if DATABASE_URL:
        conn = None
        try:
            conn = get_db_conn()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                services["database"] = "operational"
            else:
                services["database"] = "unavailable"
        except Exception:
            services["database"] = "error"
        finally:
            return_db_conn(conn)

    # Scan engine import check
    try:
        from scan_engine_real import run_real_scan_perplexity  # noqa: F401
        services["scan_engine"] = "operational"
    except Exception:
        services["scan_engine"] = "error"

    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "services": services}


@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest, request: Request):
    """
    Main endpoint used by the VizAI frontend:
    - runs REAL scan via scan_engine_real.py (Perplexity web-backed)
    - stores scan in Postgres (best-effort)
    - emails admin + user (best-effort)
    """
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

    # run real scan
    try:
        from scan_engine_real import run_real_scan_perplexity
        real_result, raw_bundle = run_real_scan_perplexity(
            business_name=payload.businessName,
            website=str(payload.website),
        )
    except Exception as e:
        logger.exception("Real scan engine failed: %s", e)
        raise HTTPException(status_code=503, detail="Scan engine unavailable or failed. Try again shortly.")

    # map to ScanResponse (use engine's package/scores)
    result = ScanResponse(
        discovery_score=int(real_result.discovery_score),
        accuracy_score=int(real_result.accuracy_score),
        authority_score=int(real_result.authority_score),
        overall_score=int(real_result.overall_score),
        package_recommendation=str(real_result.package_recommendation),
        package_explanation=str(real_result.package_explanation),
        strategy_summary=str(real_result.strategy_summary),
        findings=list(real_result.findings or []),
    )

    raw_llm: Dict[str, Any] = raw_bundle

    # attach metadata
    scan_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    result.scan_id = str(scan_id)
    result.created_at = created_at.isoformat()

    # emails (best-effort)
    email_sent = False
    try:
        if EMAIL_NOTIFICATIONS_ENABLED:
            try:
                send_admin_notification(payload, result, result.scan_id)
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
                raw_llm=raw_llm,
                ip_address=client_ip,
                user_agent=user_agent,
                email_sent=bool(email_sent),
            )
    except Exception:
        logger.exception("DB insert failed (non-fatal)")

    logger.info("Scan complete: scan_id=%s overall=%s", scan_id, result.overall_score)
    return result


@app.get("/scan/{scan_id}")
def get_scan(scan_id: str):
    """Retrieve a previously run scan by ID (requires DB)."""
    if not DATABASE_URL:
        raise HTTPException(status_code=501, detail="Database not configured")

    try:
        scan_uuid = uuid.UUID(scan_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid scan ID format")

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise HTTPException(status_code=503, detail="Database unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    scan_id, created_at, business_name, website,
                    contact_email, request_contact,
                    discovery_score, accuracy_score, authority_score, overall_score,
                    package_recommendation, package_explanation, strategy_summary,
                    findings, email_sent
                FROM vizai_scans
                WHERE scan_id = %s
                """,
                (str(scan_uuid),),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Scan not found")

        return {
            "scan_id": str(row[0]),
            "created_at": row[1].isoformat() if row[1] else None,
            "business_name": row[2],
            "website": row[3],
            "contact_email": row[4],
            "request_contact": bool(row[5]),
            "discovery_score": row[6],
            "accuracy_score": row[7],
            "authority_score": row[8],
            "overall_score": row[9],
            "package_recommendation": row[10],
            "package_explanation": row[11],
            "strategy_summary": row[12],
            "findings": row[13],
            "email_sent": bool(row[14]) if row[14] is not None else None,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve scan: %s", scan_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve scan")
    finally:
        return_db_conn(conn)


@app.get("/admin/stats")
def admin_stats():
    """Basic aggregate stats (no auth yet)."""
    if not DATABASE_URL:
        return {"error": "Database not configured", "stats": None}

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise HTTPException(status_code=503, detail="Database unavailable")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM vizai_scans")
            total_scans = cur.fetchone()[0]

            cur.execute(
                """
                SELECT
                    COALESCE(AVG(discovery_score),0)::int,
                    COALESCE(AVG(accuracy_score),0)::int,
                    COALESCE(AVG(authority_score),0)::int,
                    COALESCE(AVG(overall_score),0)::int
                FROM vizai_scans
                """
            )
            avg = cur.fetchone()

            cur.execute("SELECT COUNT(*) FROM vizai_scans WHERE request_contact = true")
            contact_requests = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*) FROM vizai_scans
                WHERE created_at > NOW() - INTERVAL '24 hours'
                """
            )
            recent_24h = cur.fetchone()[0]

            cur.execute(
                """
                SELECT package_recommendation, COUNT(*)
                FROM vizai_scans
                GROUP BY package_recommendation
                """
            )
            pkg_dist = dict(cur.fetchall())

        conversion = round((contact_requests / total_scans) * 100, 1) if total_scans else 0.0

        return {
            "total_scans": total_scans,
            "recent_24h": recent_24h,
            "contact_requests": contact_requests,
            "conversion_rate": conversion,
            "average_scores": {
                "discovery": avg[0],
                "accuracy": avg[1],
                "authority": avg[2],
                "overall": avg[3],
            },
            "package_distribution": pkg_dist,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to compute admin stats")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
    finally:
        return_db_conn(conn)


@app.post("/test_email")
def test_email():
    """Send a test email (admin + user) to verify Resend configuration."""
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "notifications_not_configured"}

    dummy_req = ScanRequest(
        businessName="Test Business",
        website="https://example.com",
        contactEmail=NOTIFY_EMAIL_TO,  # send to admin address for test
        requestContact=True,
        models=["default"],
    )

    dummy_res = ScanResponse(
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

    admin_sent = send_admin_notification(dummy_req, dummy_res, dummy_res.scan_id)
    user_sent = send_user_report(dummy_req, dummy_res)

    return {"status": "ok", "admin_sent": bool(admin_sent), "user_sent": bool(user_sent)}

# -------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "path": str(request.url)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Unexpected error occurred."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)




