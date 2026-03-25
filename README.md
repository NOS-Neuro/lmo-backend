# lmo-backend

FastAPI backend for VizAI. The codebase implements evidence-based LLM visibility scans, competitor baseline storage, scoring, and recommendation/report helpers.

## Key Files And Directories
- [`main.py`](./main.py): FastAPI app entrypoint and middleware wiring
- [`routes.py`](./routes.py): HTTP routes
- [`schemas.py`](./schemas.py): route request/response models
- [`scan_service.py`](./scan_service.py): scan orchestration, CAPTCHA, rate-limit IP helpers, background follow-up work
- [`db.py`](./db.py): PostgreSQL pool and persistence helpers
- [`email_service.py`](./email_service.py): transactional email helpers
- [`scan_engine_real.py`](./scan_engine_real.py): provider-backed scan engine
- [`core/ras`](./core/ras): scoring/reporting helpers
- [`tests`](./tests): test suite
- [`requirements.txt`](./requirements.txt): Python dependencies
- [`pytest.ini`](./pytest.ini): test and coverage configuration
- [`DEPLOYMENT.md`](./DEPLOYMENT.md): deployment notes and documented run command

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run The App
Documented in the repo:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 180
```

Local development run command: Not documented in repo.

## Run Tests
```bash
pytest -q
```

Example targeted API test run:

```bash
pytest -q tests\test_api.py
```

`pytest.ini` config enables coverage reporting and writes HTML output to [`htmlcov`](./htmlcov).

## Integrations
- Perplexity API
- OpenAI API
- Cloudflare Turnstile
- Resend
- PostgreSQL via `psycopg2`

## Notes
- No `pyproject.toml`, `package.json`, `Makefile`, Dockerfile, docker-compose file, or checked-in CI workflow was found in the repo.
- No lint, format, or typecheck command is documented in the repo.
