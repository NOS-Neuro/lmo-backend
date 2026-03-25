import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator
from slowapi import Limiter

from config import settings
from db import get_db_conn, insert_main_scan, return_db_conn
from scan_service import (
    get_client_ip_from_request,
    get_rate_limit_key,
    process_competitor_scans_in_background,
    send_scan_emails_in_background,
    verify_turnstile,
)


logger = logging.getLogger("vizai")
router = APIRouter()
limiter = Limiter(key_func=get_rate_limit_key)


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
    entity_status: Optional[str] = None
    entity_confidence: Optional[int] = None
    warnings: Optional[List[str]] = None
    disclaimer: str = (
        "This scan is evidence-based when run in Real Scan mode. "
        "Fallback mode is an honest AI-assisted estimate."
    )


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

    scan_id = uuid.uuid4()
    scan_id_str = str(scan_id)[:8]
    created_at = datetime.now(timezone.utc)

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
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
        raise HTTPException(
            status_code=400,
            detail="CAPTCHA verification failed. Please refresh the page and try again.",
        )

    logger.info(
        "Scan start: biz=%s db_url_present=%s ip=%s ua_present=%s competitors=%s questions=%s",
        payload.businessName,
        bool(settings.DATABASE_URL),
        client_ip,
        bool(user_agent),
        len(payload.competitors),
        len(payload.questions),
        extra={"request_id": request_id, "scan_id": scan_id_str},
    )

    from scan_engine_real import run_real_scan_perplexity

    raw_llm: Optional[Dict[str, Any]] = None
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
        qa_pairs = [
            QAPair(
                question=run.get("question", ""),
                answer=run.get("answer_text", ""),
                prompt_name=run.get("prompt_name"),
            )
            for run in raw_bundle.get("runs", [])
        ]

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
            entity_status=result_obj.entity_status,
            entity_confidence=result_obj.entity_confidence,
            warnings=result_obj.warnings,
        )
    except Exception as e:
        logger.exception(
            "Real scan failed: %s",
            e,
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
        raise HTTPException(status_code=500, detail="Scan processing failed. Please try again.")

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
                extra={"request_id": request_id, "scan_id": scan_id_str},
            )
            raise HTTPException(status_code=500, detail="Unable to store scan results. Please try again.")

    if settings.email_notifications_enabled:
        background_tasks.add_task(
            send_scan_emails_in_background,
            payload=payload,
            result=result,
            scan_id=scan_id,
            request_id=request_id,
        )

    competitors_list = payload.competitors or []
    if settings.DATABASE_URL and competitors_list:
        background_tasks.add_task(
            process_competitor_scans_in_background,
            parent_scan_id=scan_id,
            created_at=created_at,
            competitors_list=competitors_list,
            request_id=request_id,
            scan_id_str=scan_id_str,
        )

    return result


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
