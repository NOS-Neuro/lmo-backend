import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from psycopg2 import pool
from psycopg2.extras import Json

from config import settings


logger = logging.getLogger("vizai")

_db_pool: Optional[pool.ThreadedConnectionPool] = None


def init_db_pool() -> None:
    global _db_pool
    if not settings.DATABASE_URL or _db_pool:
        return
    _db_pool = pool.ThreadedConnectionPool(
        minconn=settings.DB_POOL_MIN,
        maxconn=settings.DB_POOL_MAX,
        dsn=settings.DATABASE_URL,
    )
    logger.info("DB pool initialized (min=%s max=%s)", settings.DB_POOL_MIN, settings.DB_POOL_MAX)


def get_db_conn():
    if not _db_pool:
        return None
    try:
        return _db_pool.getconn()
    except Exception as e:
        logger.exception("Failed to get DB connection from pool: %s", e)
        return None


def return_db_conn(conn) -> None:
    if _db_pool and conn:
        try:
            _db_pool.putconn(conn)
        except Exception as e:
            logger.exception("Failed to return DB connection to pool: %s", e)


def ensure_tables_and_migrations() -> None:
    if not settings.DATABASE_URL:
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS vizai_scans (
      scan_id UUID PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL,
      business_name TEXT NOT NULL,
      industry TEXT,
      website TEXT NOT NULL,
      contact_email TEXT NOT NULL,
      request_contact BOOLEAN NOT NULL,

      discovery_score INT NOT NULL,
      accuracy_score INT NOT NULL,
      authority_score INT NOT NULL,
      overall_score INT NOT NULL,

      package_recommendation TEXT NOT NULL,
      package_explanation TEXT NOT NULL,
      strategy_summary TEXT NOT NULL,

      findings JSONB NOT NULL,
      raw_llm JSONB,
      email_sent BOOLEAN DEFAULT FALSE,

      ip_address TEXT,
      user_agent TEXT
    );

    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                      WHERE table_name='vizai_scans' AND column_name='industry') THEN
            ALTER TABLE vizai_scans ADD COLUMN industry TEXT;
        END IF;
    END $$;

    CREATE INDEX IF NOT EXISTS idx_vizai_scans_created_at ON vizai_scans(created_at DESC);

    CREATE TABLE IF NOT EXISTS vizai_competitor_scans (
      id BIGSERIAL PRIMARY KEY,
      parent_scan_id UUID REFERENCES vizai_scans(scan_id) ON DELETE CASCADE,
      created_at TIMESTAMPTZ NOT NULL,

      competitor_name TEXT NOT NULL,
      competitor_website TEXT NOT NULL,

      discovery_score INT NOT NULL,
      accuracy_score INT NOT NULL,
      authority_score INT NOT NULL,
      overall_score INT NOT NULL,

      raw_bundle JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_vizai_comp_parent ON vizai_competitor_scans(parent_scan_id);
    """

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            logger.warning("DB unavailable; cannot ensure tables")
            return
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
        logger.info("DB tables ensured")
    except Exception as e:
        logger.exception("Failed ensuring tables/migrations: %s", e)
    finally:
        return_db_conn(conn)


def close_db_pool() -> None:
    global _db_pool
    if _db_pool:
        logger.info("Closing database connection pool...")
        _db_pool.closeall()
        _db_pool = None
        logger.info("Database pool closed")


def insert_main_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    payload: Any,
    result: Any,
    raw_llm: Optional[Dict[str, Any]],
    ip_address: Optional[str],
    user_agent: Optional[str],
) -> None:
    if not settings.DATABASE_URL:
        return

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise RuntimeError("DB unavailable (no connection from pool)")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vizai_scans (
                    scan_id, created_at, business_name, industry, website, contact_email, request_contact,
                    discovery_score, accuracy_score, authority_score, overall_score,
                    package_recommendation, package_explanation, strategy_summary,
                    findings, raw_llm, email_sent, ip_address, user_agent
                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,%s
                )
                """,
                (
                    str(scan_id),
                    created_at,
                    payload.businessName,
                    payload.industry,
                    str(payload.website),
                    str(payload.contactEmail),
                    bool(payload.requestContact),
                    int(result.discovery_score),
                    int(result.accuracy_score),
                    int(result.authority_score),
                    int(result.overall_score),
                    result.package_recommendation,
                    result.package_explanation,
                    result.strategy_summary,
                    Json(result.findings),
                    Json(raw_llm) if raw_llm is not None else None,
                    bool(result.email_sent) if result.email_sent is not None else False,
                    ip_address,
                    user_agent,
                ),
            )
        conn.commit()
        logger.info("Inserted main scan row: %s", str(scan_id))
    finally:
        return_db_conn(conn)


def insert_competitor_scan(
    *,
    parent_scan_id: uuid.UUID,
    created_at: datetime,
    competitor_name: str,
    competitor_website: str,
    scores: Dict[str, int],
    raw_bundle: Dict[str, Any],
) -> None:
    if not settings.DATABASE_URL:
        return

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            raise RuntimeError("DB unavailable (no connection from pool)")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vizai_competitor_scans (
                    parent_scan_id, created_at,
                    competitor_name, competitor_website,
                    discovery_score, accuracy_score, authority_score, overall_score,
                    raw_bundle
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    str(parent_scan_id),
                    created_at,
                    competitor_name,
                    competitor_website,
                    int(scores.get("discovery", 0)),
                    int(scores.get("accuracy", 0)),
                    int(scores.get("authority", 0)),
                    int(scores.get("overall", 0)),
                    Json(raw_bundle),
                ),
            )
        conn.commit()
        logger.info("Inserted competitor scan row: parent=%s name=%s", str(parent_scan_id), competitor_name)
    finally:
        return_db_conn(conn)
