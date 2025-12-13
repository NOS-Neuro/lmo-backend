import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

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

EMAIL_NOTIFICATIONS_ENABLED = bool(
    RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO
)

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
    version="1.4.0",
    description="VizAI: Evidence-based LLM Reality Alignment scanning.",
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
    models: List[str] = []

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

    email_sent: Optional[bool] = None
    disclaimer: str


# -------------------------------------------------------------------
# DB Pool
# -------------------------------------------------------------------

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "5"))

_db_pool: Optional[pool.ThreadedConnectionPool] = None


def init_db_pool():
    global _db_pool
    if not DATABASE_URL or _db_pool:
        return
    _db_pool = pool.ThreadedConnectionPool(
        minconn=DB_POOL_MIN,
        maxconn=DB_POOL_MAX,
        dsn=DATABASE_URL,
    )
    logger.info("DB pool initialized")


def get_db_conn():
    if not _db_pool:
        return None
    try:
        return _db_pool.getconn()
    except Exception:
        return None


def return_db_conn(conn):
    if _db_pool and conn:
        _db_pool.putconn(conn)


# -------------------------------------------------------------------
# Startup / Shutdown
# -------------------------------------------------------------------

@app.on_event("startup")
def startup():
    logger.info("VizAI starting up...")
    if DATABASE_URL:
        init_db_pool()
    logger.info("VizAI ready")


@app.on_event("shutdown")
def shutdown():
    global _db_pool
    if _db_pool:
        _db_pool.closeall()
        _db_pool = None


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "VizAI Scan API"}


@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest, request: Request):
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    scan_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    try:
        from scan_engine_real import run_real_scan_perplexity

        real_result, raw_bundle = run_real_scan_perplexity(
            business_name=payload.businessName,
            website=str(payload.website),
        )

        result = ScanResponse(
            scan_id=str(scan_id),
            created_at=created_at.isoformat(),
            discovery_score=real_result.discovery_score,
            accuracy_score=real_result.accuracy_score,
            authority_score=real_result.authority_score,
            overall_score=real_result.overall_score,
            package_recommendation=real_result.package_recommendation,
            package_explanation=real_result.package_explanation,
            strategy_summary=real_result.strategy_summary,
            findings=real_result.findings,
            disclaimer="Real Scan mode: web-backed answers with captured citations.",
        )

        raw_llm = raw_bundle

    except Exception as e:
        logger.exception("Real scan failed")
        raise HTTPException(status_code=503, detail="Scan failed")

    result.email_sent = False
    return result


@app.exception_handler(Exception)
async def error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )







