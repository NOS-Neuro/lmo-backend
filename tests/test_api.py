import asyncio
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

sys.modules.setdefault(
    "resend",
    SimpleNamespace(api_key=None, Emails=SimpleNamespace(send=lambda params: {"id": "stub"})),
)

import main
import routes
import db
import scan_service


class ASGIResponse:
    def __init__(self, status_code, headers, body):
        self.status_code = status_code
        self.headers = headers
        self.body = body

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def request_json(app, method, path, payload=None, headers=None):
    async def _run():
        body = b""
        raw_headers = []
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            raw_headers.append((b"content-type", b"application/json"))

        for key, value in (headers or {}).items():
            raw_headers.append((key.lower().encode("utf-8"), value.encode("utf-8")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": raw_headers,
            "client": ("testclient", 123),
            "server": ("testserver", 80),
        }

        messages = []
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)

        status_code = None
        response_headers = {}
        response_body = b""
        for message in messages:
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = {
                    key.decode("utf-8"): value.decode("utf-8")
                    for key, value in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")

        return ASGIResponse(status_code, response_headers, response_body)

    return asyncio.run(_run())


class FakeCursor:
    def __init__(self, rows=None, fetchone_value=None):
        self.rows = rows or []
        self.fetchone_value = fetchone_value
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.fetchone_value


class FakeConnection:
    def __init__(self, rows=None, fetchone_value=None):
        self.cursor_obj = FakeCursor(rows=rows, fetchone_value=fetchone_value)

    def cursor(self):
        return self.cursor_obj


def make_scan_result():
    return SimpleNamespace(
        discovery_score=81,
        accuracy_score=79,
        authority_score=77,
        overall_score=79,
        package_recommendation="Standard LMO",
        package_explanation="Explanation",
        strategy_summary="Strategy",
        findings=["Finding 1"],
        entity_status="CONFIRMED",
        entity_confidence=92,
        warnings=None,
    )


def make_payload(**overrides):
    payload = {
        "businessName": "Acme Corp",
        "industry": "Software",
        "website": "https://acme.com",
        "contactEmail": "owner@acme.com",
        "requestContact": False,
        "captchaToken": "valid-captcha-token",
        "models": [],
        "competitors": [],
        "questions": [],
    }
    payload.update(overrides)
    return payload


def reset_limiter():
    storage = getattr(routes.limiter, "_storage", None)
    if storage and hasattr(storage, "reset"):
        storage.reset()


def configure_scan_success(monkeypatch, result=None, raw_bundle=None):
    scan_result = result or make_scan_result()
    bundle = raw_bundle or {"runs": []}
    fake_module = SimpleNamespace(
        run_real_scan_perplexity=lambda **kwargs: (scan_result, bundle)
    )
    monkeypatch.setitem(sys.modules, "scan_engine_real", fake_module)


def test_run_scan_persists_before_sending_email(monkeypatch):
    reset_limiter()
    configure_scan_success(monkeypatch)

    events = []

    monkeypatch.setattr(routes, "verify_turnstile", lambda token, remote_ip=None: True)
    monkeypatch.setattr(main.settings, "DATABASE_URL", "postgres://db")
    monkeypatch.setattr(main.settings, "RESEND_API_KEY", "resend-key")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_FROM", "from@example.com")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_TO", "to@example.com")
    monkeypatch.setattr(routes, "insert_main_scan", lambda **kwargs: events.append("insert"))
    monkeypatch.setattr(scan_service, "send_scan_results_email", lambda **kwargs: events.append("email") or True)
    monkeypatch.setattr(scan_service, "send_contact_request_notification", lambda **kwargs: events.append("contact") or True)

    response = request_json(main.app, "POST", "/run_scan", payload=make_payload())

    assert response.status_code == 200
    assert events == ["insert", "email"]
    assert response.json()["email_sent"] is False


def test_run_scan_db_failure_returns_generic_error_and_skips_email(monkeypatch):
    reset_limiter()
    configure_scan_success(monkeypatch)

    email_calls = []

    monkeypatch.setattr(routes, "verify_turnstile", lambda token, remote_ip=None: True)
    monkeypatch.setattr(main.settings, "DATABASE_URL", "postgres://db")
    monkeypatch.setattr(main.settings, "RESEND_API_KEY", "resend-key")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_FROM", "from@example.com")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_TO", "to@example.com")
    monkeypatch.setattr(routes, "insert_main_scan", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("secret db failure")))
    monkeypatch.setattr(scan_service, "send_scan_results_email", lambda **kwargs: email_calls.append("email") or True)

    response = request_json(main.app, "POST", "/run_scan", payload=make_payload())

    assert response.status_code == 500
    assert response.json() == {"error": "Unable to store scan results. Please try again."}
    assert email_calls == []


def test_run_scan_scan_failure_returns_generic_error(monkeypatch):
    reset_limiter()
    fake_module = SimpleNamespace(
        run_real_scan_perplexity=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider secret"))
    )
    monkeypatch.setitem(sys.modules, "scan_engine_real", fake_module)
    monkeypatch.setattr(routes, "verify_turnstile", lambda token, remote_ip=None: True)
    monkeypatch.setattr(main.settings, "DATABASE_URL", None)
    monkeypatch.setattr(main.settings, "RESEND_API_KEY", None)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_FROM", None)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_TO", None)

    response = request_json(main.app, "POST", "/run_scan", payload=make_payload())

    assert response.status_code == 500
    assert response.json() == {"error": "Scan processing failed. Please try again."}


def test_run_scan_rate_limit_uses_forwarded_for(monkeypatch):
    reset_limiter()
    configure_scan_success(monkeypatch)

    monkeypatch.setattr(routes, "verify_turnstile", lambda token, remote_ip=None: True)
    monkeypatch.setattr(main.settings, "DATABASE_URL", None)
    monkeypatch.setattr(main.settings, "RESEND_API_KEY", None)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_FROM", None)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_TO", None)

    headers_one = {"x-forwarded-for": "1.1.1.1"}
    headers_two = {"x-forwarded-for": "2.2.2.2"}

    for _ in range(10):
        response = request_json(main.app, "POST", "/run_scan", payload=make_payload(), headers=headers_one)
        assert response.status_code == 200

    limited = request_json(main.app, "POST", "/run_scan", payload=make_payload(), headers=headers_one)
    separate_ip = request_json(main.app, "POST", "/run_scan", payload=make_payload(), headers=headers_two)

    assert limited.status_code == 429
    assert separate_ip.status_code == 200


def test_health_hides_db_identity(monkeypatch):
    monkeypatch.setattr(main.settings, "DATABASE_URL", "postgres://db")
    monkeypatch.setattr(routes, "get_db_conn", lambda: FakeConnection(fetchone_value=("vizai", "app_user")))
    monkeypatch.setattr(routes, "return_db_conn", lambda conn: None)

    response = request_json(main.app, "GET", "/health")

    assert response.status_code == 200
    body = response.json()
    assert body["services"]["database"] == "operational"
    assert "db_identity" not in body


def test_get_scan_competitors_returns_rows(monkeypatch):
    created_at = datetime.now(timezone.utc)
    rows = [
        ("Competitor A", "https://comp-a.com", 75, 70, 65, 72, created_at),
        ("Competitor B", "https://comp-b.com", 60, 61, 62, 63, created_at),
    ]

    monkeypatch.setattr(main.settings, "DATABASE_URL", "postgres://db")
    monkeypatch.setattr(routes, "get_db_conn", lambda: FakeConnection(rows=rows))
    monkeypatch.setattr(routes, "return_db_conn", lambda conn: None)

    scan_id = str(uuid4())
    response = request_json(main.app, "GET", f"/scan/{scan_id}/competitors")

    assert response.status_code == 200
    body = response.json()
    assert body["scan_id"] == scan_id
    assert body["count"] == 2
    assert body["competitors"][0]["name"] == "Competitor A"
    assert body["competitors"][0]["scores"]["overall"] == 72


def test_cors_origins_follow_settings(monkeypatch):
    monkeypatch.setattr(main.settings, "FRONTEND_ORIGIN", "https://vizai.io, https://www.vizai.io")

    assert main.settings.cors_allowed_origins == ["https://vizai.io", "https://www.vizai.io"]
