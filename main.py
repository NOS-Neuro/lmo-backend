import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, EmailStr, field_validator

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")

DATABASE_URL = os.getenv("DATABASE_URL")

EMAIL_NOTIFICATIONS_ENABLED = bool(RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

logger.info("Database storage: %s", "ENABLED" if DATABASE_URL else "DISABLED")
logger.info("Email notifications: %s", "ENABLED" if EMAIL_NOTIFICATIONS_ENABLED else "DISABLED")
logger.info("Perplexity real scan: %s", "READY" if PERPLEXITY_API_KEY else "NOT CONFIGURED")
logger.info("OpenAI fallback scan: %s", "READY" if OPENAI_API_KEY else "NOT CONFIGURED")

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------

app = FastAPI(
    title="VizAI Scan API",
    version="1.4.1",
    description="VizAI: evidence-based LLM visibility scanning with competitor baselines.",
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

class CompetitorIn(BaseModel):
    name: str
    website: HttpUrl

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Competitor name too short")
        return v


class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    contactEmail: EmailStr
    requestContact: bool = False
    models: List[str] = []
    competitors: List[CompetitorIn] = []

    @field_validator("businessName")
    @classmethod
    def validate_business_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Business name too short")
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

    email_sent: Optional[bool] = None

    disclaimer: str = (
        "This scan is evidence-based when run in Real Scan mode. "
        "Fallback mode is an honest AI-assisted estimate."
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
    if not DATABASE_URL:
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS vizai_scans (
      scan_id UUID PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL,
      business_name TEXT NOT NULL,
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
    if not DATABASE_URL:
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
                    scan_id, created_at, business_name, website, contact_email, request_contact,
                    discovery_score, accuracy_score, authority_score, overall_score,
                    package_recommendation, package_explanation, strategy_summary,
                    findings, raw_llm, email_sent, ip_address, user_agent
                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,%s
                )
                """,
                (
                    str(scan_id),
                    created_at,
                    payload.businessName,
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
        logger.info("Inserted scan row: %s", str(scan_id))
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
    if not DATABASE_URL:
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
    finally:
        return_db_conn(conn)


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    logger.info("VizAI API starting… DATABASE_URL present=%s", bool(DATABASE_URL))
    if DATABASE_URL:
        init_db_pool()
        ensure_tables_and_migrations()
    logger.info("VizAI API started")


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
        "database": "not_configured" if not DATABASE_URL else "unknown",
        "email": "operational" if EMAIL_NOTIFICATIONS_ENABLED else "not_configured",
        "perplexity": "operational" if PERPLEXITY_API_KEY else "not_configured",
        "openai_fallback": "operational" if OPENAI_API_KEY else "not_configured",
    }

    db_identity = None
    if DATABASE_URL:
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
        "db_url_present": bool(DATABASE_URL),
        "db_identity": {"database": db_identity[0], "user": db_identity[1]} if db_identity else None,
    }


# -----------------------
# RUN SCAN
# -----------------------

@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest, request: Request):

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

    logger.info("Scan start: biz=%s db_url_present=%s ip=%s ua_present=%s competitors=%s",
                payload.businessName, bool(DATABASE_URL), client_ip, bool(user_agent), len(payload.competitors))

    from scan_engine_real import run_real_scan_perplexity

    scan_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    raw_llm: Optional[Dict[str, Any]] = None

    # Main scan (real scan only for now)
    try:
        result_obj, raw_bundle = run_real_scan_perplexity(
            business_name=payload.businessName,
            website=str(payload.website),
            competitors=[{"name": c.name, "website": str(c.website)} for c in (payload.competitors or [])],
        )
        raw_llm = raw_bundle

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
            email_sent=False,
        )

    except Exception as e:
        logger.exception("Real scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Store main scan (NOT best-effort while debugging: fail loud if DB configured)
    if DATABASE_URL:
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
            logger.exception("DB insert failed: %s", e)
            raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    return result


# -----------------------
# GET COMPETITOR BASELINE
# -----------------------

@app.get("/scan/{scan_id}/competitors")
def get_scan_competitors(scan_id: str):
    if not DATABASE_URL:
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

        return {
            "scan_id": scan_id,
            "count": len(rows),
            "competitors": [
                {
                    "name": r[0],
                    "website": r[1],
                    "scores": {
                        "discovery": r[2],
                        "accuracy": r[3],
                        "authority": r[4],
                        "overall": r[5],
                    },
                    "created_at": r[6].isoformat() if r[6] else None,
                }
                for r in rows
            ],
        }
    finally:
        return_db_conn(conn)


# -------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})










