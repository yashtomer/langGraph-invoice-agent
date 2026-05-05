"""SQLite engine + invoice_history table for idempotency.

Three concerns share the same SQLite file:
  * langgraph_checkpoints — owned by SqliteSaver
  * apscheduler_jobs      — owned by APScheduler's SQLAlchemyJobStore
  * invoice_history       — owned by this module
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from .config import Settings, get_settings


_SCHEMA = """
CREATE TABLE IF NOT EXISTS invoice_history (
    month         TEXT NOT NULL,
    project_name  TEXT,
    pdf_path      TEXT,
    status        TEXT NOT NULL,           -- 'started' | 'sent' | 'cancelled' | 'errored'
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (month)
);
CREATE INDEX IF NOT EXISTS idx_invoice_history_status ON invoice_history(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(settings: Optional[Settings] = None) -> None:
    s = settings or get_settings()
    with sqlite3.connect(str(s.db_path)) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


@contextmanager
def connect(settings: Optional[Settings] = None) -> Iterator[sqlite3.Connection]:
    s = settings or get_settings()
    conn = sqlite3.connect(str(s.db_path))
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def get_status(month: str, *, settings: Optional[Settings] = None) -> Optional[str]:
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT status FROM invoice_history WHERE month = ?", (month,)
        ).fetchone()
        return row["status"] if row else None


def already_sent(month: str, *, settings: Optional[Settings] = None) -> bool:
    return get_status(month, settings=settings) == "sent"


def get_active_month(*, settings: Optional[Settings] = None) -> Optional[str]:
    """Most recent month with status='started' (waiting at an interrupt), or None.

    Used to route inbound WhatsApp replies to the right thread when the user
    has been triggered for a future month (e.g. testing June flow on May 4th).
    """
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT month FROM invoice_history WHERE status = 'started' "
            "ORDER BY month DESC LIMIT 1"
        ).fetchone()
        return row["month"] if row else None


def get_last_sent(*, settings: Optional[Settings] = None) -> Optional[sqlite3.Row]:
    """Most recent invoice_history row with status='sent', or None."""
    with connect(settings) as conn:
        return conn.execute(
            "SELECT month, project_name, pdf_path, created_at, updated_at "
            "FROM invoice_history WHERE status = 'sent' "
            "ORDER BY month DESC LIMIT 1"
        ).fetchone()


def mark_started(month: str, *, settings: Optional[Settings] = None) -> None:
    now = _now()
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO invoice_history (month, status, created_at, updated_at)
            VALUES (?, 'started', ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                status = CASE WHEN invoice_history.status = 'sent' THEN invoice_history.status ELSE 'started' END,
                updated_at = excluded.updated_at
            """,
            (month, now, now),
        )
        conn.commit()


def mark_sent(
    month: str,
    *,
    project_name: Optional[str] = None,
    pdf_path: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> None:
    now = _now()
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO invoice_history (month, project_name, pdf_path, status, created_at, updated_at)
            VALUES (?, ?, ?, 'sent', ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                status = 'sent',
                project_name = excluded.project_name,
                pdf_path = excluded.pdf_path,
                updated_at = excluded.updated_at
            """,
            (month, project_name, pdf_path, now, now),
        )
        conn.commit()


def mark_status(month: str, status: str, *, settings: Optional[Settings] = None) -> None:
    now = _now()
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO invoice_history (month, status, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(month) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at
            """,
            (month, status, now, now),
        )
        conn.commit()
