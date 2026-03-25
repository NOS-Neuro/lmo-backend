import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from db import close_db_pool, ensure_tables_and_migrations, init_db_pool
from routes import limiter, router


class RequestContextFilter(logging.Filter):
    """Add request context (request_id, scan_id) to log records."""

    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "scan_id"):
            record.scan_id = "-"
        return True


logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [req:%(request_id)s] [scan:%(scan_id)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vizai")
logger.addFilter(RequestContextFilter())

logger.info("Database storage: %s", "ENABLED" if settings.database_enabled else "DISABLED")
logger.info("Email notifications: %s", "ENABLED" if settings.email_notifications_enabled else "DISABLED")
logger.info("Perplexity real scan: %s", "READY" if settings.PERPLEXITY_API_KEY else "NOT CONFIGURED")
logger.info("OpenAI fallback scan: %s", "READY" if settings.OPENAI_API_KEY else "NOT CONFIGURED")
logger.info("CORS origin: %s", settings.FRONTEND_ORIGIN)
logger.info("Rate limiting: ENABLED (10 req/min per IP on /run_scan)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VizAI API starting... DATABASE_URL present=%s", bool(settings.DATABASE_URL))
    if settings.DATABASE_URL:
        init_db_pool()
        ensure_tables_and_migrations()
    logger.info("VizAI API started")

    yield

    close_db_pool()


app = FastAPI(
    title="VizAI Scan API",
    version="1.4.2",
    description="VizAI: evidence-based LLM visibility scanning with competitor baselines.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": exc.detail},
    )
