"""Tests for platform_forward (WP-22, D1 — forward scans to Vizai-discovery)."""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import requests as requests_lib

from config import settings
from platform_forward import INTAKE_PATH, build_intake_payload, forward_scan_to_platform
from schemas import ScanRequest


def make_scan_request(**overrides):
    data = {
        "businessName": "Acme Plumbing",
        "industry": "plumbing",
        "website": "https://acme.example",
        "contactEmail": "owner@acme.example",
        "requestContact": True,
        "captchaToken": "x" * 12,
    }
    data.update(overrides)
    return ScanRequest(**data)


SAMPLE_RESULT = {
    "discovery_score": 61,
    "accuracy_score": 74,
    "authority_score": 48,
    "overall_score": 62,
    "visibility_status": "partially_visible",
    "confidence_level": "medium",
    "evidence_summary": "Some evidence found.",
    "verified_facts": ["Registered business"],
    "package_recommendation": "foundation",
    "package_explanation": "Because reasons.",
    "strategy_summary": "Improve citations.",
    "findings": ["Low authority signals"],
    "qa_pairs": [{"question": "q", "answer": "a"}],
}

SCAN_ID = uuid.uuid4()
CREATED_AT = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _configure(monkeypatch, url="https://app.vizai.io", key="test-service-key"):
    monkeypatch.setattr(settings, "VIZAI_PLATFORM_URL", url)
    monkeypatch.setattr(settings, "VIZAI_PLATFORM_API_KEY", key)


def _deconfigure(monkeypatch):
    monkeypatch.setattr(settings, "VIZAI_PLATFORM_URL", None)
    monkeypatch.setattr(settings, "VIZAI_PLATFORM_API_KEY", None)


# ── build_intake_payload ──────────────────────────────────────────────────────


def test_build_intake_payload_shape():
    payload = make_scan_request()
    body = build_intake_payload(
        payload=payload, result=SAMPLE_RESULT, scan_id=SCAN_ID, created_at=CREATED_AT
    )

    assert body["externalScanId"] == str(SCAN_ID)
    assert body["source"] == "lmo-backend"
    assert body["businessName"] == "Acme Plumbing"
    assert body["email"] == "owner@acme.example"
    assert body["requestContact"] is True
    assert body["scores"] == {
        "discovery": 61,
        "accuracy": 74,
        "authority": 48,
        "overall": 62,
    }
    assert body["packageRecommendation"] == "foundation"
    assert body["findings"] == ["Low authority signals"]
    assert body["publicSummary"]["visibility_status"] == "partially_visible"
    # Lean payload: raw LLM output and QA pairs must never be forwarded.
    assert "raw_llm" not in body
    assert "qa_pairs" not in body


def test_build_intake_payload_defaults_missing_scores():
    payload = make_scan_request()
    body = build_intake_payload(
        payload=payload, result={}, scan_id=SCAN_ID, created_at=CREATED_AT
    )
    assert body["scores"] == {"discovery": 0, "accuracy": 0, "authority": 0, "overall": 0}
    assert body["publicSummary"] == {}


# ── forward_scan_to_platform ──────────────────────────────────────────────────


def test_forward_disabled_without_config(monkeypatch):
    _deconfigure(monkeypatch)
    with patch("platform_forward.requests.post") as mock_post:
        ok = forward_scan_to_platform(
            payload=make_scan_request(),
            result=SAMPLE_RESULT,
            scan_id=SCAN_ID,
            created_at=CREATED_AT,
        )
    assert ok is False
    mock_post.assert_not_called()


def test_forward_posts_with_auth_header(monkeypatch):
    _configure(monkeypatch)
    response = MagicMock(status_code=201, text="")
    with patch("platform_forward.requests.post", return_value=response) as mock_post:
        ok = forward_scan_to_platform(
            payload=make_scan_request(),
            result=SAMPLE_RESULT,
            scan_id=SCAN_ID,
            created_at=CREATED_AT,
        )

    assert ok is True
    assert mock_post.call_count == 1
    args, kwargs = mock_post.call_args
    assert args[0].endswith(INTAKE_PATH)
    assert kwargs["headers"]["Authorization"] == "Bearer test-service-key"
    assert kwargs["json"]["businessName"] == "Acme Plumbing"


def test_forward_never_raises_on_network_error(monkeypatch):
    _configure(monkeypatch)
    with patch(
        "platform_forward.requests.post",
        side_effect=requests_lib.ConnectionError("boom"),
    ) as mock_post:
        ok = forward_scan_to_platform(
            payload=make_scan_request(),
            result=SAMPLE_RESULT,
            scan_id=SCAN_ID,
            created_at=CREATED_AT,
        )
    assert ok is False
    assert mock_post.call_count == 2  # one retry on network errors


def test_forward_no_retry_on_4xx(monkeypatch):
    _configure(monkeypatch)
    response = MagicMock(status_code=422, text="validation failed")
    with patch("platform_forward.requests.post", return_value=response) as mock_post:
        ok = forward_scan_to_platform(
            payload=make_scan_request(),
            result=SAMPLE_RESULT,
            scan_id=SCAN_ID,
            created_at=CREATED_AT,
        )
    assert ok is False
    assert mock_post.call_count == 1


def test_forward_retries_on_5xx(monkeypatch):
    _configure(monkeypatch)
    bad = MagicMock(status_code=503, text="unavailable")
    good = MagicMock(status_code=200, text="")
    with patch("platform_forward.requests.post", side_effect=[bad, good]) as mock_post:
        ok = forward_scan_to_platform(
            payload=make_scan_request(),
            result=SAMPLE_RESULT,
            scan_id=SCAN_ID,
            created_at=CREATED_AT,
        )
    assert ok is True
    assert mock_post.call_count == 2
