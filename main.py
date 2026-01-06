import uuid
import logging
import requests
import resend
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, EmailStr, field_validator, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from psycopg2 import pool
from psycopg2.extras import Json

from config import settings

# Configure Resend API
if settings.RESEND_API_KEY:
    resend.api_key = settings.RESEND_API_KEY

# -------------------------------------------------------------------
# Logging with Request Context
# -------------------------------------------------------------------

class RequestContextFilter(logging.Filter):
    """Add request context (request_id, scan_id) to log records"""

    def filter(self, record):
        # Add request_id if not present
        if not hasattr(record, 'request_id'):
            record.request_id = '-'
        if not hasattr(record, 'scan_id'):
            record.scan_id = '-'
        return True


# Configure logging with structured format
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [req:%(request_id)s] [scan:%(scan_id)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vizai")
logger.addFilter(RequestContextFilter())


def log_with_context(level: str, message: str, request_id: str = "-", scan_id: str = "-", **kwargs):
    """Log with request context for tracing"""
    log_func = getattr(logger, level.lower())
    log_func(message, extra={"request_id": request_id, "scan_id": scan_id}, **kwargs)


# -------------------------------------------------------------------
# Startup Info
# -------------------------------------------------------------------

logger.info("Database storage: %s", "ENABLED" if settings.database_enabled else "DISABLED")
logger.info("Email notifications: %s", "ENABLED" if settings.email_notifications_enabled else "DISABLED")
logger.info("Perplexity real scan: %s", "READY" if settings.PERPLEXITY_API_KEY else "NOT CONFIGURED")
logger.info("OpenAI fallback scan: %s", "READY" if settings.OPENAI_API_KEY else "NOT CONFIGURED")
logger.info("CORS origin: %s", settings.FRONTEND_ORIGIN)
logger.info("Rate limiting: ENABLED (10 req/min per IP on /run_scan)")

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

class CompetitorIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    website: HttpUrl

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Competitor name too short")
        return v


class QuestionIn(BaseModel):
    prompt_name: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=10, max_length=1000)


class ScanRequest(BaseModel):
    businessName: str = Field(..., min_length=2, max_length=200)
    industry: Optional[str] = Field(default=None, max_length=100)
    website: HttpUrl
    contactEmail: EmailStr
    requestContact: bool = False
    captchaToken: str = Field(..., min_length=10)
    models: List[str] = Field(default=[], max_length=5)
    competitors: List[CompetitorIn] = Field(default=[], max_length=10)
    questions: List[QuestionIn] = Field(default=[], max_length=20)


    @field_validator("businessName")
    @classmethod
    def validate_business_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Business name too short")
        return v


class QAPair(BaseModel):
    question: str
    answer: str
    prompt_name: Optional[str] = None


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
    qa_pairs: Optional[List[QAPair]] = None

    email_sent: Optional[bool] = None

    disclaimer: str = (
        "This scan is evidence-based when run in Real Scan mode. "
        "Fallback mode is an honest AI-assisted estimate."
    )

# -------------------------------------------------------------------
# DB Pool
# -------------------------------------------------------------------

_db_pool: Optional[pool.ThreadedConnectionPool] = None


def init_db_pool() -> None:
    global _db_pool
    if not settings.DATABASE_URL or _db_pool:
        return
    _db_pool = pool.ThreadedConnectionPool(
        minconn=settings.DB_POOL_MIN,
        maxconn=settings.DB_POOL_MAX,
        dsn=settings.DATABASE_URL,
    )
    logger.info("DB pool initialized (min=%s max=%s)", settings.DB_POOL_MIN, settings.DB_POOL_MAX)


def get_db_conn():
    if not _db_pool:
        return None
    try:
        return _db_pool.getconn()
    except Exception as e:
        logger.exception("Failed to get DB connection from pool: %s", e)
        return None


def return_db_conn(conn) -> None:
    if _db_pool and conn:
        try:
            _db_pool.putconn(conn)
        except Exception as e:
            logger.exception("Failed to return DB connection to pool: %s", e)

# -------------------------------------------------------------------
# DB Schema
# -------------------------------------------------------------------

def ensure_tables_and_migrations() -> None:
    if not settings.DATABASE_URL:
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS vizai_scans (
      scan_id UUID PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL,
      business_name TEXT NOT NULL,
      industry TEXT,
      website TEXT NOT NULL,
      contact_email TEXT NOT NULL,
      request_contact BOOLEAN NOT NULL,

      discovery_score INT NOT NULL,
      accuracy_score INT NOT NULL,
      authority_score INT NOT NULL,
      overall_score INT NOT NULL,

      package_recommendation TEXT NOT NULL,
      package_explanation TEXT NOT NULL,
      strategy_summary TEXT NOT NULL,

      findings JSONB NOT NULL,
      raw_llm JSONB,
      email_sent BOOLEAN DEFAULT FALSE,

      ip_address TEXT,
      user_agent TEXT
    );

    -- Migration: Add industry column if it doesn't exist
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                      WHERE table_name='vizai_scans' AND column_name='industry') THEN
            ALTER TABLE vizai_scans ADD COLUMN industry TEXT;
        END IF;
    END $$;

    CREATE INDEX IF NOT EXISTS idx_vizai_scans_created_at ON vizai_scans(created_at DESC);

    CREATE TABLE IF NOT EXISTS vizai_competitor_scans (
      id BIGSERIAL PRIMARY KEY,
      parent_scan_id UUID REFERENCES vizai_scans(scan_id) ON DELETE CASCADE,
      created_at TIMESTAMPTZ NOT NULL,

      competitor_name TEXT NOT NULL,
      competitor_website TEXT NOT NULL,

      discovery_score INT NOT NULL,
      accuracy_score INT NOT NULL,
      authority_score INT NOT NULL,
      overall_score INT NOT NULL,

      raw_bundle JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_vizai_comp_parent ON vizai_competitor_scans(parent_scan_id);
    """

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            logger.warning("DB unavailable; cannot ensure tables")
            return
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
        logger.info("DB tables ensured")
    except Exception as e:
        logger.exception("Failed ensuring tables/migrations: %s", e)
    finally:
        return_db_conn(conn)

# -------------------------------------------------------------------
# DB Inserts
# -------------------------------------------------------------------

def insert_main_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    payload: ScanRequest,
    result: ScanResponse,
    raw_llm: Optional[Dict[str, Any]],
    ip_address: Optional[str],
    user_agent: Optional[str],
) -> None:
    if not settings.DATABASE_URL:
        return

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise RuntimeError("DB unavailable (no connection from pool)")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vizai_scans (
                    scan_id, created_at, business_name, industry, website, contact_email, request_contact,
                    discovery_score, accuracy_score, authority_score, overall_score,
                    package_recommendation, package_explanation, strategy_summary,
                    findings, raw_llm, email_sent, ip_address, user_agent
                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,%s
                )
                """,
                (
                    str(scan_id),
                    created_at,
                    payload.businessName,
                    payload.industry,
                    str(payload.website),
                    str(payload.contactEmail),
                    bool(payload.requestContact),

                    int(result.discovery_score),
                    int(result.accuracy_score),
                    int(result.authority_score),
                    int(result.overall_score),

                    result.package_recommendation,
                    result.package_explanation,
                    result.strategy_summary,

                    Json(result.findings),
                    Json(raw_llm) if raw_llm is not None else None,
                    bool(result.email_sent) if result.email_sent is not None else False,

                    ip_address,
                    user_agent,
                ),
            )
        conn.commit()
        logger.info("Inserted main scan row: %s", str(scan_id))
    finally:
        return_db_conn(conn)


def insert_competitor_scan(
    *,
    parent_scan_id: uuid.UUID,
    created_at: datetime,
    competitor_name: str,
    competitor_website: str,
    scores: Dict[str, int],
    raw_bundle: Dict[str, Any],
) -> None:
    if not settings.DATABASE_URL:
        return

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise RuntimeError("DB unavailable (no connection from pool)")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vizai_competitor_scans (
                    parent_scan_id, created_at,
                    competitor_name, competitor_website,
                    discovery_score, accuracy_score, authority_score, overall_score,
                    raw_bundle
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    str(parent_scan_id),
                    created_at,
                    competitor_name,
                    competitor_website,
                    int(scores.get("discovery", 0)),
                    int(scores.get("accuracy", 0)),
                    int(scores.get("authority", 0)),
                    int(scores.get("overall", 0)),
                    Json(raw_bundle),
                ),
            )
        conn.commit()
        logger.info("Inserted competitor scan row: parent=%s name=%s", str(parent_scan_id), competitor_name)
    finally:
        return_db_conn(conn)

# -------------------------------------------------------------------
# Lifespan Context Manager
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)"""
    # Startup
    logger.info("VizAI API starting… DATABASE_URL present=%s", bool(settings.DATABASE_URL))
    if settings.DATABASE_URL:
        init_db_pool()
        ensure_tables_and_migrations()
    logger.info("VizAI API started")

    yield

    # Shutdown
    global _db_pool
    if _db_pool:
        logger.info("Closing database connection pool...")
        _db_pool.closeall()
        logger.info("Database pool closed")

# -------------------------------------------------------------------
# Rate Limiting
# -------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------

app = FastAPI(
    title="VizAI Scan API",
    version="1.4.2",
    description="VizAI: evidence-based LLM visibility scanning with competitor baselines.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -------------------------------------------------------------------
# Request ID Middleware
# -------------------------------------------------------------------

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to all requests for tracing"""
    request_id = str(uuid.uuid4())[:8]  # Short ID for readability
    request.state.request_id = request_id

    # Add to response headers for client-side debugging
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.vizai.io",
        "https://vizai.io",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS (fixes preflight)
    allow_headers=["*"],
)

def verify_turnstile(token: str, remote_ip: Optional[str] = None) -> bool:
    # Fail closed if secret missing (recommended for production)
    if not settings.TURNSTILE_SECRET_KEY:
        logger.error("TURNSTILE_SECRET_KEY not configured - captcha validation will fail")
        return False

    if not token:
        logger.warning("Turnstile verification failed: empty token provided")
        return False

    try:
        resp = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={
                "secret": settings.TURNSTILE_SECRET_KEY,
                "response": token,
                **({"remoteip": remote_ip} if remote_ip else {}),
            },
            timeout=settings.TURNSTILE_TIMEOUT,
        )
        data = resp.json()
        success = bool(data.get("success"))

        if not success:
            error_codes = data.get("error-codes", [])
            logger.warning(
                "Turnstile verification failed: success=%s, error_codes=%s, remote_ip=%s",
                success,
                error_codes,
                remote_ip
            )
        else:
            logger.debug("Turnstile verification successful for IP: %s", remote_ip)

        return success
    except requests.exceptions.Timeout:
        logger.error("Turnstile verification timeout after %s seconds", settings.TURNSTILE_TIMEOUT)
        return False
    except requests.exceptions.RequestException as e:
        logger.error("Turnstile verification request failed: %s", str(e))
        return False
    except Exception as e:
        logger.exception("Turnstile verification unexpected error: %s", e)
        return False


# -------------------------------------------------------------------
# Email Functions
# -------------------------------------------------------------------

def send_scan_results_email(
    to_email: str,
    business_name: str,
    scan_id: str,
    discovery_score: int,
    accuracy_score: int,
    authority_score: int,
    overall_score: int,
    findings: List[str],
    package_recommendation: str,
    strategy_summary: str,
    request_id: str = "-"
) -> bool:
    """Send scan results to the user via Resend"""

    if not settings.email_notifications_enabled:
        logger.debug("Email notifications disabled, skipping scan results email", extra={"request_id": request_id})
        return False

    try:
        # Create findings HTML
        findings_html = ""
        if findings:
            findings_items = "".join([f"<li style='margin-bottom: 8px;'>{finding}</li>" for finding in findings[:5]])
            findings_html = f"""
            <div style="margin: 24px 0;">
                <h3 style="color: #b89cff; margin-bottom: 12px;">Key Findings:</h3>
                <ul style="padding-left: 20px; line-height: 1.6;">
                    {findings_items}
                </ul>
            </div>
            """

        # Determine score badge color
        if overall_score >= 80:
            score_color = "#4ade80"
            score_label = "Strong Visibility"
        elif overall_score >= 40:
            score_color = "#fbbf24"
            score_label = "Partial Visibility"
        else:
            score_color = "#f87171"
            score_label = "High-Risk Visibility"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b0d13; color: #e4e6f1; margin: 0; padding: 0;">
            <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 32px;">
                    <h1 style="color: #b89cff; margin: 0; font-size: 28px;">VizAI Scan Results</h1>
                    <p style="color: #a0a3b1; margin: 8px 0 0;">Your AI Visibility Report for {business_name}</p>
                </div>

                <!-- Scores Card -->
                <div style="background: #14161f; border: 1px solid #1e2029; border-radius: 12px; padding: 24px; margin-bottom: 24px;">
                    <div style="text-align: center; margin-bottom: 24px;">
                        <div style="display: inline-block; background: {score_color}; color: #0b0d13; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-bottom: 12px;">
                            {score_label}
                        </div>
                        <h2 style="font-size: 48px; margin: 8px 0; color: #e4e6f1;">{overall_score}/100</h2>
                        <p style="color: #a0a3b1; margin: 0; font-size: 14px;">Overall AI Visibility Score</p>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 24px; padding-top: 24px; border-top: 1px solid #1e2029;">
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #b89cff;">{discovery_score}</div>
                            <div style="font-size: 12px; color: #a0a3b1; margin-top: 4px;">Discovery</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #b89cff;">{accuracy_score}</div>
                            <div style="font-size: 12px; color: #a0a3b1; margin-top: 4px;">Accuracy</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #b89cff;">{authority_score}</div>
                            <div style="font-size: 12px; color: #a0a3b1; margin-top: 4px;">Authority</div>
                        </div>
                    </div>
                </div>

                {findings_html}

                <!-- Recommendation -->
                <div style="background: rgba(123, 92, 255, 0.1); border: 1px solid rgba(123, 92, 255, 0.2); border-radius: 12px; padding: 20px; margin: 24px 0;">
                    <h3 style="color: #b89cff; margin: 0 0 12px;">Recommended Next Step:</h3>
                    <p style="margin: 0; line-height: 1.6; color: #e4e6f1; font-weight: 500;">{package_recommendation}</p>
                    <p style="margin: 12px 0 0; line-height: 1.6; color: #a0a3b1; font-size: 14px;">{strategy_summary}</p>
                </div>

                <!-- CTA -->
                <div style="text-align: center; margin: 32px 0;">
                    <a href="https://vizai.app/packages.html" style="display: inline-block; background: linear-gradient(120deg, #7b5cff, #b39cff); color: #0b0d13; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 16px;">
                        View Pricing & Packages
                    </a>
                </div>

                <!-- Footer -->
                <div style="margin-top: 40px; padding-top: 24px; border-top: 1px solid #1e2029; text-align: center; font-size: 14px; color: #a0a3b1;">
                    <p style="margin: 8px 0;">Questions about your results? Reply to this email or contact us at <a href="mailto:hello@vizai.io" style="color: #b89cff; text-decoration: none;">hello@vizai.io</a></p>
                    <p style="margin: 8px 0;">Visit us at <a href="https://vizai.app" style="color: #b89cff; text-decoration: none;">vizai.app</a></p>
                    <p style="margin: 16px 0 0; font-size: 12px; color: #6b6e7f;">Scan ID: {scan_id}</p>
                </div>
            </div>
        </body>
        </html>
        """

        params = {
            "from": settings.NOTIFY_EMAIL_FROM,
            "to": [to_email],
            "subject": f"Your VizAI Scan Results - {business_name} ({overall_score}/100)",
            "html": html_content,
        }

        response = resend.Emails.send(params)
        logger.info(
            "Scan results email sent successfully to %s (resend_id: %s)",
            to_email,
            response.get("id"),
            extra={"request_id": request_id}
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to send scan results email to %s: %s",
            to_email,
            str(e),
            extra={"request_id": request_id}
        )
        return False


def send_contact_request_notification(
    business_name: str,
    contact_email: str,
    website: str,
    industry: Optional[str],
    scan_id: str,
    overall_score: int,
    request_id: str = "-"
) -> bool:
    """Notify admin team when someone requests to be contacted"""

    if not settings.email_notifications_enabled:
        logger.debug("Email notifications disabled, skipping contact request notification", extra={"request_id": request_id})
        return False

    try:
        industry_text = industry if industry else "Not specified"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 32px;">
                <h2 style="color: #7b5cff; margin: 0 0 24px;">🔔 New Contact Request from Scan</h2>

                <div style="background: #f8f9fa; border-left: 4px solid #7b5cff; padding: 16px; margin-bottom: 24px;">
                    <p style="margin: 0; font-weight: bold; color: #333;">Someone wants to discuss improving their results</p>
                </div>

                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Business Name:</td>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; color: #333;">{business_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Contact Email:</td>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0;"><a href="mailto:{contact_email}" style="color: #7b5cff;">{contact_email}</a></td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Website:</td>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0;"><a href="{website}" style="color: #7b5cff;">{website}</a></td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Industry:</td>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; color: #333;">{industry_text}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Overall Score:</td>
                        <td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; color: #333;"><strong>{overall_score}/100</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: bold; color: #666;">Scan ID:</td>
                        <td style="padding: 12px 0; color: #999; font-family: monospace; font-size: 12px;">{scan_id}</td>
                    </tr>
                </table>

                <div style="margin-top: 32px; padding: 16px; background: #f0f7ff; border-radius: 6px;">
                    <p style="margin: 0; color: #555; font-size: 14px;">
                        💡 <strong>Next step:</strong> Reply to {contact_email} within 1 business day to discuss their AI visibility needs.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        params = {
            "from": settings.NOTIFY_EMAIL_FROM,
            "to": [settings.NOTIFY_EMAIL_TO],
            "subject": f"🔔 Contact Request: {business_name} (Score: {overall_score}/100)",
            "html": html_content,
            "reply_to": contact_email,
        }

        response = resend.Emails.send(params)
        logger.info(
            "Contact request notification sent to admin (resend_id: %s)",
            response.get("id"),
            extra={"request_id": request_id}
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to send contact request notification: %s",
            str(e),
            extra={"request_id": request_id}
        )
        return False


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "VizAI"}


@app.get("/health")
def health():
    services = {
        "api": "operational",
        "database": "not_configured" if not settings.DATABASE_URL else "unknown",
        "email": "operational" if settings.email_notifications_enabled else "not_configured",
        "perplexity": "operational" if settings.PERPLEXITY_API_KEY else "not_configured",
        "openai_fallback": "operational" if settings.OPENAI_API_KEY else "not_configured",
    }

    db_identity = None
    if settings.DATABASE_URL:
        conn = None
        try:
            conn = get_db_conn()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute("SELECT current_database(), current_user")
                    db_identity = cur.fetchone()
                services["database"] = "operational"
            else:
                services["database"] = "unavailable"
        except Exception:
            services["database"] = "error"
        finally:
            return_db_conn(conn)

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
        "db_url_present": bool(settings.DATABASE_URL),
        "db_identity": {"database": db_identity[0], "user": db_identity[1]} if db_identity else None,
    }


# -----------------------
# RUN SCAN
# -----------------------

@app.post("/run_scan", response_model=ScanResponse)
@limiter.limit("10/minute")
def run_scan(payload: ScanRequest, request: Request):
    # Get request ID from middleware
    request_id = getattr(request.state, "request_id", "-")

    # Generate scan ID
    scan_id = uuid.uuid4()
    scan_id_str = str(scan_id)[:8]  # Short ID for logging
    created_at = datetime.now(timezone.utc)

    # request metadata (Render/proxies safe)
    client_ip = None
    try:
        forwarded = request.headers.get("x-forwarded-for")
        real_ip = request.headers.get("x-real-ip")
        cf_ip = request.headers.get("cf-connecting-ip")

        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        elif real_ip:
            client_ip = real_ip.strip()
        elif cf_ip:
            client_ip = cf_ip.strip()
        elif request.client:
            client_ip = request.client.host
    except Exception:
        client_ip = None

    user_agent = request.headers.get("user-agent")

    # --- CAPTCHA enforcement (anti-abuse)
    captcha_valid = verify_turnstile(payload.captchaToken, client_ip)
    if not captcha_valid:
        logger.warning(
            "Scan rejected due to failed captcha: biz=%s ip=%s",
            payload.businessName,
            client_ip,
            extra={"request_id": request_id, "scan_id": scan_id_str}
        )
        raise HTTPException(
            status_code=400,
            detail="CAPTCHA verification failed. Please refresh the page and try again."
        )


    logger.info(
        "Scan start: biz=%s db_url_present=%s ip=%s ua_present=%s competitors=%s questions=%s",
        payload.businessName,
        bool(settings.DATABASE_URL),
        client_ip,
        bool(user_agent),
        len(payload.competitors),
        len(payload.questions),
        extra={"request_id": request_id, "scan_id": scan_id_str}
    )

    from scan_engine_real import run_real_scan_perplexity

    raw_llm: Optional[Dict[str, Any]] = None

    # --- Main scan
    try:
        custom_questions: Optional[List[Tuple[str, str]]] = None
        if payload.questions:
            custom_questions = [(q.prompt_name, q.question) for q in payload.questions]

        result_obj, raw_bundle = run_real_scan_perplexity(
            business_name=payload.businessName,
            website=str(payload.website),
            industry=payload.industry,
            questions=custom_questions,
            competitors=[{"name": c.name, "website": str(c.website)} for c in (payload.competitors or [])],
        )

        raw_llm = raw_bundle

        # Extract Q&A pairs from raw_bundle
        qa_pairs = []
        for run in raw_bundle.get("runs", []):
            qa_pairs.append(QAPair(
                question=run.get("question", ""),
                answer=run.get("answer_text", ""),
                prompt_name=run.get("prompt_name")
            ))

        result = ScanResponse(
            scan_id=str(scan_id),
            created_at=created_at.isoformat(),
            discovery_score=int(result_obj.discovery_score),
            accuracy_score=int(result_obj.accuracy_score),
            authority_score=int(result_obj.authority_score),
            overall_score=int(result_obj.overall_score),
            package_recommendation=str(result_obj.package_recommendation),
            package_explanation=str(result_obj.package_explanation),
            strategy_summary=str(result_obj.strategy_summary),
            findings=list(result_obj.findings or []),
            qa_pairs=qa_pairs,
            email_sent=False,
        )
    except Exception as e:
        logger.exception(
            "Real scan failed: %s",
            e,
            extra={"request_id": request_id, "scan_id": scan_id_str}
        )
        raise HTTPException(status_code=500, detail=str(e))

    # --- Send emails BEFORE DB insert (best-effort, never break the scan response)
    try:
        # Send scan results to user
        email_sent = send_scan_results_email(
            to_email=payload.contactEmail,
            business_name=payload.businessName,
            scan_id=str(scan_id),
            discovery_score=result.discovery_score,
            accuracy_score=result.accuracy_score,
            authority_score=result.authority_score,
            overall_score=result.overall_score,
            findings=result.findings,
            package_recommendation=result.package_recommendation,
            strategy_summary=result.strategy_summary,
            request_id=request_id
        )

        # Update result with email status
        result.email_sent = email_sent

        # If user requested contact, notify admin
        if payload.requestContact:
            send_contact_request_notification(
                business_name=payload.businessName,
                contact_email=payload.contactEmail,
                website=str(payload.website),
                industry=payload.industry,
                scan_id=str(scan_id),
                overall_score=result.overall_score,
                request_id=request_id
            )
    except Exception as e:
        logger.error(
            "Email sending failed (non-fatal): %s",
            str(e),
            extra={"request_id": request_id, "scan_id": scan_id_str}
        )
        # Don't break the scan if email fails
        result.email_sent = False

    # --- Store main scan (fail loud if DB configured)
    if settings.DATABASE_URL:
        try:
            insert_main_scan(
                scan_id=scan_id,
                created_at=created_at,
                payload=payload,
                result=result,
                raw_llm=raw_llm,
                ip_address=client_ip,
                user_agent=user_agent,
            )
        except Exception as e:
            logger.exception(
                "DB insert failed: %s",
                e,
                extra={"request_id": request_id, "scan_id": scan_id_str}
            )
            raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

        # --- Competitor baseline scans (best-effort; never break main scan)
        # Run competitor scans in parallel for better performance
        competitors_list = payload.competitors or []

        if competitors_list:
            def scan_single_competitor(competitor_data):
                """Scan a single competitor and return results"""
                try:
                    comp_obj, comp_bundle = run_real_scan_perplexity(
                        business_name=competitor_data.name,
                        website=str(competitor_data.website),
                    )
                    return {
                        "success": True,
                        "name": competitor_data.name,
                        "website": str(competitor_data.website),
                        "comp_obj": comp_obj,
                        "comp_bundle": comp_bundle,
                    }
                except Exception as e:
                    logger.warning(
                        "Competitor scan failed (non-fatal) for %s: %s",
                        competitor_data.name,
                        str(e)
                    )
                    return {
                        "success": False,
                        "name": competitor_data.name,
                        "error": str(e),
                    }

            # Use ThreadPoolExecutor for parallel scanning
            # max_workers configurable via settings to avoid overwhelming the API
            logger.info(
                "Starting parallel scan of %d competitors with %d workers",
                len(competitors_list),
                settings.MAX_COMPETITOR_SCAN_WORKERS,
                extra={"request_id": request_id, "scan_id": scan_id_str}
            )
            with ThreadPoolExecutor(max_workers=settings.MAX_COMPETITOR_SCAN_WORKERS) as executor:
                # Submit all competitor scans
                future_to_competitor = {
                    executor.submit(scan_single_competitor, comp): comp
                    for comp in competitors_list
                }

                # Process results as they complete
                for future in as_completed(future_to_competitor):
                    result_data = future.result()

                    if result_data["success"]:
                        try:
                            insert_competitor_scan(
                                parent_scan_id=scan_id,
                                created_at=created_at,
                                competitor_name=result_data["name"],
                                competitor_website=result_data["website"],
                                scores={
                                    "discovery": int(result_data["comp_obj"].discovery_score),
                                    "accuracy": int(result_data["comp_obj"].accuracy_score),
                                    "authority": int(result_data["comp_obj"].authority_score),
                                    "overall": int(result_data["comp_obj"].overall_score),
                                },
                                raw_bundle=result_data["comp_bundle"],
                            )
                            logger.info("Successfully inserted competitor scan: %s", result_data["name"])
                        except Exception as e:
                            logger.exception(
                                "Failed to insert competitor scan for %s: %s",
                                result_data["name"],
                                e
                            )

            logger.info(
                "Completed parallel competitor scanning",
                extra={"request_id": request_id, "scan_id": scan_id_str}
            )

    return result


# -----------------------
# GET COMPETITOR BASELINE
# -----------------------

@app.get("/scan/{scan_id}/competitors")
def get_scan_competitors(scan_id: str):
    if not settings.DATABASE_URL:
        raise HTTPException(status_code=501, detail="Database not configured")

    try:
        scan_uuid = uuid.UUID(scan_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid scan ID")

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise HTTPException(status_code=503, detail="Database unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  competitor_name,
                  competitor_website,
                  discovery_score,
                  accuracy_score,
                  authority_score,
                  overall_score,
                  created_at
                FROM vizai_competitor_scans
                WHERE parent_scan_id = %s
                ORDER BY overall_score DESC
                """,
                (str(scan_uuid),),
            )
            rows = cur.fetchall()

        # Build competitor list
        competitors = []
        for r in rows:
            competitors.append({
                "name": r[0],
                "website": r[1],
                "scores": {
                    "discovery": r[2],
                    "accuracy": r[3],
                    "authority": r[4],
                    "overall": r[5],
                },
                "created_at": r[6].isoformat() if r[6] else None,
            })

        result = {
            "scan_id": scan_id,
            "count": len(rows),
            "competitors": competitors,
        }
    finally:
        return_db_conn(conn)

    return result


# -------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
