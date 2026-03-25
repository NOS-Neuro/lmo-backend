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
      scan_status TEXT NOT NULL DEFAULT 'completed',
      completed_at TIMESTAMPTZ,
      failure_message TEXT,

      ip_address TEXT,
      user_agent TEXT
    );

    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                      WHERE table_name='vizai_scans' AND column_name='industry') THEN
            ALTER TABLE vizai_scans ADD COLUMN industry TEXT;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                      WHERE table_name='vizai_scans' AND column_name='scan_status') THEN
            ALTER TABLE vizai_scans ADD COLUMN scan_status TEXT NOT NULL DEFAULT 'completed';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                      WHERE table_name='vizai_scans' AND column_name='completed_at') THEN
            ALTER TABLE vizai_scans ADD COLUMN completed_at TIMESTAMPTZ;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                      WHERE table_name='vizai_scans' AND column_name='failure_message') THEN
            ALTER TABLE vizai_scans ADD COLUMN failure_message TEXT;
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
    completed_at: Optional[datetime],
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
                    findings, raw_llm, email_sent, scan_status, completed_at, failure_message, ip_address, user_agent
                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s
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
                    "completed",
                    completed_at,
                    None,
                    ip_address,
                    user_agent,
                ),
            )
        conn.commit()
        logger.info("Inserted main scan row: %s", str(scan_id))
    finally:
        return_db_conn(conn)


def create_pending_scan(
    *,
    scan_id: uuid.UUID,
    created_at: datetime,
    payload: Any,
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
                    findings, raw_llm, email_sent, scan_status, completed_at, failure_message, ip_address, user_agent
                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s
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
                    0,
                    0,
                    0,
                    0,
                    "Pending",
                    "Scan processing has started.",
                    "Scan is running in the background.",
                    Json([]),
                    None,
                    False,
                    "processing",
                    None,
                    None,
                    ip_address,
                    user_agent,
                ),
            )
        conn.commit()
        logger.info("Inserted pending scan row: %s", str(scan_id))
    finally:
        return_db_conn(conn)


def update_main_scan_result(
    *,
    scan_id: uuid.UUID,
    completed_at: datetime,
    result: Any,
    raw_llm: Optional[Dict[str, Any]],
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
                UPDATE vizai_scans
                SET discovery_score = %s,
                    accuracy_score = %s,
                    authority_score = %s,
                    overall_score = %s,
                    package_recommendation = %s,
                    package_explanation = %s,
                    strategy_summary = %s,
                    findings = %s,
                    raw_llm = %s,
                    email_sent = %s,
                    scan_status = %s,
                    completed_at = %s,
                    failure_message = %s
                WHERE scan_id = %s
                """,
                (
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
                    "completed",
                    completed_at,
                    None,
                    str(scan_id),
                ),
            )
        conn.commit()
        logger.info("Updated completed scan row: %s", str(scan_id))
    finally:
        return_db_conn(conn)


def mark_scan_failed(
    *,
    scan_id: uuid.UUID,
    completed_at: datetime,
    failure_message: str,
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
                UPDATE vizai_scans
                SET scan_status = %s,
                    completed_at = %s,
                    failure_message = %s
                WHERE scan_id = %s
                """,
                ("failed", completed_at, failure_message, str(scan_id)),
            )
        conn.commit()
        logger.info("Marked scan failed: %s", str(scan_id))
    finally:
        return_db_conn(conn)


def get_scan_record(scan_id: uuid.UUID) -> Optional[Dict[str, Any]]:
    if not settings.DATABASE_URL:
        return None

    conn = None
    try:
        conn = get_db_conn()
        if not conn:
            return None

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    scan_id,
                    created_at,
                    discovery_score,
                    accuracy_score,
                    authority_score,
                    overall_score,
                    package_recommendation,
                    package_explanation,
                    strategy_summary,
                    findings,
                    email_sent,
                    scan_status,
                    completed_at,
                    failure_message
                FROM vizai_scans
                WHERE scan_id = %s
                """,
                (str(scan_id),),
            )
            row = cur.fetchone()

        if not row:
            return None

        return {
            "scan_id": str(row[0]),
            "created_at": row[1],
            "discovery_score": row[2],
            "accuracy_score": row[3],
            "authority_score": row[4],
            "overall_score": row[5],
            "package_recommendation": row[6],
            "package_explanation": row[7],
            "strategy_summary": row[8],
            "findings": row[9],
            "email_sent": row[10],
            "scan_status": row[11],
            "completed_at": row[12],
            "failure_message": row[13],
        }
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
