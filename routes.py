import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from slowapi import Limiter

from config import settings
from db import get_db_conn, return_db_conn
from schemas import ScanRequest, ScanResponse
from scan_service import (
    execute_run_scan,
    get_client_ip_from_request,
    get_rate_limit_key,
    verify_turnstile,
)


logger = logging.getLogger("vizai")
router = APIRouter()
limiter = Limiter(key_func=get_rate_limit_key)


@router.get("/")
def root():
    return {"status": "ok", "service": "VizAI"}


@router.get("/health")
def health():
    services = {
        "api": "operational",
        "database": "not_configured" if not settings.DATABASE_URL else "unknown",
        "email": "operational" if settings.email_notifications_enabled else "not_configured",
        "perplexity": "operational" if settings.PERPLEXITY_API_KEY else "not_configured",
        "openai_fallback": "operational" if settings.OPENAI_API_KEY else "not_configured",
    }

    if settings.DATABASE_URL:
        conn = None
        try:
            conn = get_db_conn()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute("SELECT current_database(), current_user")
                    cur.fetchone()
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
    }


@router.post("/run_scan", response_model=ScanResponse)
@limiter.limit("10/minute")
def run_scan(payload: ScanRequest, request: Request, background_tasks: BackgroundTasks):
    request_id = getattr(request.state, "request_id", "-")

    try:
        client_ip = get_client_ip_from_request(request)
    except Exception:
        client_ip = None

    user_agent = request.headers.get("user-agent")

    captcha_valid = verify_turnstile(payload.captchaToken, client_ip)
    if not captcha_valid:
        logger.warning(
            "Scan rejected due to failed captcha: biz=%s ip=%s",
            payload.businessName,
            client_ip,
            extra={"request_id": request_id, "scan_id": "-"},
        )
        raise HTTPException(
            status_code=400,
            detail="CAPTCHA verification failed. Please refresh the page and try again.",
        )

    return execute_run_scan(
        payload=payload,
        background_tasks=background_tasks,
        request_id=request_id,
        client_ip=client_ip,
        user_agent=user_agent,
    )


@router.get("/scan/{scan_id}/competitors")
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

        competitors = [
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
        ]

        return {"scan_id": scan_id, "count": len(rows), "competitors": competitors}
    finally:
        return_db_conn(conn)
