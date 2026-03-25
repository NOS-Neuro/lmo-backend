import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, HTTPException
import requests
from fastapi import Request
from slowapi.util import get_remote_address

from config import settings
from db import (
    create_pending_scan,
    insert_competitor_scan,
    insert_main_scan,
    mark_scan_failed,
    update_main_scan_result,
)
from email_service import send_contact_request_notification, send_scan_results_email
from schemas import QAPair, ScanRequest, ScanResponse


logger = logging.getLogger("vizai")


def _duration_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def get_client_ip_from_request(request: Request) -> str:
    """Extract the best-effort client IP, accounting for common proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    real_ip = request.headers.get("x-real-ip")
    cf_ip = request.headers.get("cf-connecting-ip")

    if forwarded:
        return forwarded.split(",")[0].strip()
    if real_ip:
        return real_ip.strip()
    if cf_ip:
        return cf_ip.strip()
    client = getattr(request, "client", None)
    if client and client.host:
        return client.host
    return get_remote_address(request) or "unknown"


def get_rate_limit_key(request: Request) -> str:
    """Use the same client IP extraction logic for rate limiting and request logging."""
    return get_client_ip_from_request(request)


def verify_turnstile(token: str, remote_ip: Optional[str] = None) -> bool:
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
            logger.warning(
                "Turnstile verification failed: success=%s, error_codes=%s, remote_ip=%s",
                success,
                data.get("error-codes", []),
                remote_ip,
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


def send_scan_emails_in_background(
    *,
    payload: Any,
    result: Any,
    scan_id: uuid.UUID,
    request_id: str,
) -> None:
    """Run non-critical email side effects after the response is returned."""
    try:
        email_sent = send_scan_results_email(
            to_email=str(payload.contactEmail),
            business_name=payload.businessName,
            scan_id=str(scan_id),
            discovery_score=result.discovery_score,
            accuracy_score=result.accuracy_score,
            authority_score=result.authority_score,
            overall_score=result.overall_score,
            findings=result.findings,
            package_recommendation=result.package_recommendation,
            strategy_summary=result.strategy_summary,
            request_id=request_id,
        )

        if payload.requestContact:
            send_contact_request_notification(
                business_name=payload.businessName,
                contact_email=str(payload.contactEmail),
                website=str(payload.website),
                industry=payload.industry,
                scan_id=str(scan_id),
                overall_score=result.overall_score,
                request_id=request_id,
            )

        logger.info(
            "Background email processing completed: sent=%s",
            email_sent,
            extra={"request_id": request_id, "scan_id": str(scan_id)[:8]},
        )
    except Exception as e:
        logger.error(
            "Email sending failed (non-fatal): %s",
            str(e),
            extra={"request_id": request_id, "scan_id": str(scan_id)[:8]},
        )


def process_competitor_scans_in_background(
    *,
    parent_scan_id: uuid.UUID,
    created_at: datetime,
    competitors_list: List[Any],
    request_id: str,
    scan_id_str: str,
) -> None:
    """Persist competitor baselines without blocking the main response."""
    if not competitors_list:
        return

    from scan_engine_real import run_real_scan_perplexity

    def scan_single_competitor(competitor_data):
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
                str(e),
                extra={"request_id": request_id, "scan_id": scan_id_str},
            )
            return {"success": False, "name": competitor_data.name, "error": str(e)}

    logger.info(
        "Starting parallel scan of %d competitors with %d workers",
        len(competitors_list),
        settings.MAX_COMPETITOR_SCAN_WORKERS,
        extra={"request_id": request_id, "scan_id": scan_id_str},
    )
    with ThreadPoolExecutor(max_workers=settings.MAX_COMPETITOR_SCAN_WORKERS) as executor:
        future_to_competitor = {executor.submit(scan_single_competitor, comp): comp for comp in competitors_list}

        for future in as_completed(future_to_competitor):
            result_data = future.result()
            if result_data["success"]:
                try:
                    insert_competitor_scan(
                        parent_scan_id=parent_scan_id,
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
                    logger.info(
                        "Successfully inserted competitor scan: %s",
                        result_data["name"],
                        extra={"request_id": request_id, "scan_id": scan_id_str},
                    )
                except Exception as e:
                    logger.exception(
                        "Failed to insert competitor scan for %s: %s",
                        result_data["name"],
                        e,
                        extra={"request_id": request_id, "scan_id": scan_id_str},
                    )

    logger.info(
        "Completed parallel competitor scanning",
        extra={"request_id": request_id, "scan_id": scan_id_str},
    )


def execute_run_scan(
    *,
    payload: ScanRequest,
    background_tasks: BackgroundTasks,
    request_id: str,
    client_ip: Optional[str],
    user_agent: Optional[str],
    async_mode: bool = False,
) -> ScanResponse:
    """Run the scan, persist results, and enqueue non-critical follow-up work."""
    scan_id = uuid.uuid4()
    scan_id_str = str(scan_id)[:8]
    created_at = datetime.now(timezone.utc)
    request_started = time.perf_counter()

    logger.info(
        "Scan start: biz=%s db_url_present=%s ip=%s ua_present=%s competitors=%s questions=%s async_mode=%s",
        payload.businessName,
        bool(settings.DATABASE_URL),
        client_ip,
        bool(user_agent),
        len(payload.competitors),
        len(payload.questions),
        async_mode,
        extra={"request_id": request_id, "scan_id": scan_id_str},
    )

    if async_mode:
        if not settings.SCAN_ASYNC_ENABLED:
            raise HTTPException(status_code=503, detail="Async scan mode is currently unavailable.")
        if not settings.DATABASE_URL:
            raise HTTPException(status_code=501, detail="Async scan mode requires database support.")

        try:
            create_pending_scan(
                scan_id=scan_id,
                created_at=created_at,
                payload=payload,
                ip_address=client_ip,
                user_agent=user_agent,
            )
        except Exception as e:
            logger.exception(
                "Pending scan insert failed: %s",
                e,
                extra={"request_id": request_id, "scan_id": scan_id_str},
            )
            raise HTTPException(status_code=500, detail="Unable to create scan job. Please try again.")

        background_tasks.add_task(
            run_scan_job_in_background,
            scan_id=scan_id,
            created_at=created_at,
            payload=payload,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        logger.info(
            "Async scan accepted in %sms",
            _duration_ms(request_started),
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
        return {
            "scan_id": str(scan_id),
            "status": "processing",
            "created_at": created_at.isoformat(),
        }

    try:
        result, raw_llm = _perform_scan(
            scan_id=scan_id,
            created_at=created_at,
            payload=payload,
            request_id=request_id,
            scan_id_str=scan_id_str,
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
                completed_at=datetime.now(timezone.utc),
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

    logger.info(
        "Scan request completed in %sms",
        _duration_ms(request_started),
        extra={"request_id": request_id, "scan_id": scan_id_str},
    )
    return result


def _perform_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    payload: ScanRequest,
    request_id: str,
    scan_id_str: str,
) -> Tuple[ScanResponse, Optional[Dict[str, Any]]]:
    from scan_engine_real import run_real_scan_perplexity

    engine_started = time.perf_counter()
    raw_llm: Optional[Dict[str, Any]] = None

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

    logger.info(
        "Core scan completed in %sms",
        _duration_ms(engine_started),
        extra={"request_id": request_id, "scan_id": scan_id_str},
    )
    return result, raw_llm


def run_scan_job_in_background(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    payload: ScanRequest,
    request_id: str,
    client_ip: Optional[str],
    user_agent: Optional[str],
) -> None:
    scan_id_str = str(scan_id)[:8]
    job_started = time.perf_counter()

    try:
        result, raw_llm = _perform_scan(
            scan_id=scan_id,
            created_at=created_at,
            payload=payload,
            request_id=request_id,
            scan_id_str=scan_id_str,
        )
        update_main_scan_result(
            scan_id=scan_id,
            completed_at=datetime.now(timezone.utc),
            result=result,
            raw_llm=raw_llm,
        )

        if settings.email_notifications_enabled:
            send_scan_emails_in_background(
                payload=payload,
                result=result,
                scan_id=scan_id,
                request_id=request_id,
            )

        competitors_list = payload.competitors or []
        if settings.DATABASE_URL and competitors_list:
            process_competitor_scans_in_background(
                parent_scan_id=scan_id,
                created_at=created_at,
                competitors_list=competitors_list,
                request_id=request_id,
                scan_id_str=scan_id_str,
            )

        logger.info(
            "Async scan job completed in %sms",
            _duration_ms(job_started),
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
    except Exception as e:
        logger.exception(
            "Async scan job failed: %s",
            e,
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
        try:
            mark_scan_failed(
                scan_id=scan_id,
                completed_at=datetime.now(timezone.utc),
                failure_message="Scan processing failed. Please try again.",
            )
        except Exception:
            logger.exception(
                "Failed to persist async scan failure",
                extra={"request_id": request_id, "scan_id": scan_id_str},
            )
