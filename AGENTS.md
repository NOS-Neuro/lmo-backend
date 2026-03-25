# AGENTS.md

## Project Overview
- VizAI backend: a FastAPI API for evidence-based LLM visibility scans, competitor baselines, scoring, and recommendation output.
- App entrypoint: [`main.py`](./main.py)
- HTTP routes: [`routes.py`](./routes.py)
- Route schemas: [`schemas.py`](./schemas.py)
- Scan orchestration / request-side helpers: [`scan_service.py`](./scan_service.py)
- Database lifecycle / persistence: [`db.py`](./db.py)
- Email delivery: [`email_service.py`](./email_service.py)
- Scan engine: [`scan_engine_real.py`](./scan_engine_real.py)
- Scoring / reporting helpers: [`core/ras`](./core/ras)
- Tests: [`tests`](./tests)
- Deployment notes: [`DEPLOYMENT.md`](./DEPLOYMENT.md)

## Local Workflow
- Install dependencies:
  - `pip install -r requirements.txt`
- Run the app:
  - `uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 180`
  - TODO: the repo documents the deployment/start command above, but does not define a separate local-dev command.
- Run tests:
  - `pytest -q`
  - `pytest -q tests\test_api.py`
- Coverage output is configured via `pytest.ini` and writes HTML to [`htmlcov`](./htmlcov).
- Pytest always runs with coverage and strict markers because those flags are set in `pytest.ini`.
- Linting:
  - TODO: no lint config or lint command is checked into this repo.
- Type checking:
  - TODO: no mypy/pyright config or typecheck command is checked into this repo.
- Formatting:
  - TODO: no formatter config or formatting command is checked into this repo.
- Docker / compose:
  - TODO: no Dockerfile or docker-compose file is checked into this repo.
- Make / scripts:
  - TODO: no Makefile, `scripts/`, or `bin/` workflow is checked into this repo.
- CI:
  - TODO: no checked-in CI workflow/config was found in this repo.

## Engineering Rules
- Prefer minimal, reviewable diffs.
- Preserve existing API contracts, response bodies, and status codes unless the change explicitly requires otherwise.
- Do not expose secrets, DB identity, stack traces, raw SQL errors, provider errors, or raw internal exceptions to clients.
- Add or update tests for behavioral changes.
- Mock external providers in tests where appropriate:
  - Perplexity / OpenAI
  - Cloudflare Turnstile
  - Resend
  - PostgreSQL access
- Avoid unrelated refactors during bug-fix or reliability passes.
- Keep behavior fixes, structural refactors, and text/encoding cleanups in separate diffs unless explicitly requested.
- For follow-up cleanup, prefer one concern per diff.
- Keep security and reliability fixes ahead of maintainability refactors.

## Project-Specific Guidance
- Keep routes thin where practical.
- Prefer service-layer orchestration over dense route handlers.
- Current module boundaries to preserve:
  - `routes.py`: request validation, HTTP wiring, response shaping
  - `scan_service.py`: scan execution, CAPTCHA/IP helpers, background follow-up orchestration
  - `db.py`: DB lifecycle and persistence
  - `email_service.py`: email rendering and delivery
  - `schemas.py`: route-facing Pydantic models
- Existing request/response conventions to preserve:
  - request IDs are added in middleware and returned via `X-Request-ID`
  - `/run_scan` is rate-limited to `10/minute` per client IP
  - HTTP error responses are shaped as `{"error": ...}`
  - `/run_scan` rejects failed CAPTCHA verification with HTTP 400
- External integrations used by this repo:
  - Perplexity API for web-backed scan queries
  - OpenAI API for optional validation layer
  - Cloudflare Turnstile for CAPTCHA verification
  - Resend for transactional email
  - PostgreSQL via `psycopg2`
- `DEPLOYMENT.md` documents long-running request expectations. Scans can take roughly 90-140 seconds end-to-end, so avoid changes that reintroduce fragile synchronous side effects or shorten timeouts without reason.
- The `/run_scan` path is sensitive. Keep side effects ordered safely:
  - persist scan results before non-critical user-facing side effects
  - keep user-facing errors generic
- Existing tests use lightweight in-process API invocation and monkeypatching rather than adding extra test dependencies.

## Definition Of Done
- Changed behavior is covered by tests.
- Relevant tests pass.
- Relevant lint/typecheck commands pass if the repo defines them.
- No accidental API contract changes.
- User-facing errors remain safe and generic.
- Final summary includes:
  - files changed
  - key decisions
  - risks / follow-ups

## Working Style For Codex
- Plan first for multi-step, risky, or cross-module changes.
- Ask for clarification only when blocked by missing critical information.
- Prefer the smallest safe implementation first.
- Leave clear TODOs for follow-up work that should be separate.
- If a workflow is not explicit in the repo, do not invent it; add a short TODO instead.
