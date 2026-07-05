"""
Forward completed free scans to the Vizai-discovery platform (WP-22, decision D1).

Vizai-discovery Postgres is the single source of truth for leads and scan
history. lmo-backend remains the scan execution worker; after a scan completes
it forwards the lead + public result snapshot to the platform intake endpoint
(POST {VIZAI_PLATFORM_URL}/api/free-scan/intake).

Design rules:
- Fire-and-forget: forwarding failures are logged and NEVER affect the
  user-facing scan response, persistence, or email delivery.
- Disabled by default: activates only when VIZAI_PLATFORM_URL and
  VIZAI_PLATFORM_API_KEY are both set.
- Lean payload: public result fields only; raw LLM output is not forwarded.
- Idempotent receiver: the platform dedupes on externalScanId, so retries
  and duplicate deliveries are safe.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Union

import requests

from config import settings
from schemas import ScanRequest, ScanResponse

logger = logging.getLogger("vizai")

INTAKE_PATH = "/api/free-scan/intake"

# Public-summary keys copied from the scan result when present.
_PUBLIC_SUMMARY_KEYS = (
    "visibility_status",
    "confidence_level",
    "evidence_summary",
    "verified_facts",
    "unclear_facts",
    "missing_signals",
    "limitations",
    "proof_signals",
)


def build_intake_payload(
    *,
    payload: ScanRequest,
    result: Union[ScanResponse, Dict[str, Any]],
    scan_id: uuid.UUID,
    created_at: datetime,
) -> Dict[str, Any]:
    """Build the lean lead + result snapshot sent to the platform."""
    data: Dict[str, Any] = result if isinstance(result, dict) else result.model_dump()

    return {
        "externalScanId": str(scan_id),
        "source": "lmo-backend",
        "businessName": payload.businessName,
        "website": str(payload.website),
        "industry": payload.industry,
        "email": payload.contactEmail,
        "requestContact": bool(payload.requestContact),
        "createdAt": created_at.isoformat(),
        "modelsUsed": [settings.PERPLEXITY_MODEL],
        "scores": {
            "discovery": int(data.get("discovery_score") or 0),
            "accuracy": int(data.get("accuracy_score") or 0),
            "authority": int(data.get("authority_score") or 0),
            "overall": int(data.get("overall_score") or 0),
        },
        "packageRecommendation": str(data.get("package_recommendation") or ""),
        "strategySummary": str(data.get("strategy_summary") or ""),
        "findings": list(data.get("findings") or []),
        "publicSummary": {
            key: data.get(key)
            for key in _PUBLIC_SUMMARY_KEYS
            if data.get(key) is not None
        },
    }


def forward_scan_to_platform(
    *,
    payload: ScanRequest,
    result: Union[ScanResponse, Dict[str, Any]],
    scan_id: uuid.UUID,
    created_at: datetime,
    request_id: str = "-",
) -> bool:
    """POST the completed scan to the platform intake endpoint. Never raises."""
    scan_id_str = str(scan_id)[:8]

    if not settings.platform_forward_enabled:
        logger.debug(
            "Platform forward skipped (not configured)",
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
        return False

    url = settings.VIZAI_PLATFORM_URL.rstrip("/") + INTAKE_PATH
    headers = {
        "Authorization": f"Bearer {settings.VIZAI_PLATFORM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        body = build_intake_payload(
            payload=payload, result=result, scan_id=scan_id, created_at=created_at
        )
    except Exception as e:  # defensive — malformed result must not break the caller
        logger.warning(
            "Platform forward payload build failed: %s",
            e,
            extra={"request_id": request_id, "scan_id": scan_id_str},
        )
        return False

    # One retry on network errors / 5xx; 4xx responses are not retried.
    for attempt in (1, 2):
        try:
            resp = requests.post(
                url, json=body, headers=headers, timeout=settings.PLATFORM_FORWARD_TIMEOUT
            )
            if resp.status_code in (200, 201):
                logger.info(
                    "Platform forward ok: status=%s attempt=%s",
                    resp.status_code,
                    attempt,
                    extra={"request_id": request_id, "scan_id": scan_id_str},
                )
                return True

            logger.warning(
                "Platform forward rejected: status=%s attempt=%s body=%s",
                resp.status_code,
                attempt,
                (resp.text or "")[:200],
                extra={"request_id": request_id, "scan_id": scan_id_str},
            )
            if resp.status_code < 500:
                return False
        except requests.RequestException as e:
            logger.warning(
                "Platform forward attempt %s failed: %s",
                attempt,
                e,
                extra={"request_id": request_id, "scan_id": scan_id_str},
            )

    return False
