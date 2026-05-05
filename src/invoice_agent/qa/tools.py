"""LangChain @tool functions for the Q&A agent.

The docstrings on each @tool are load-bearing — the LLM uses them to choose
which tool to call. Edit with care.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ..config import Settings, get_settings
from ..db import connect
from ..logging_setup import get_logger
from .util import normalize_target_month

log = get_logger(__name__)


def _get_invoice_impl(month: str, *, settings: Optional[Settings] = None) -> dict:
    s = settings or get_settings()
    resolved = normalize_target_month(month, s.timezone)
    with connect(s) as conn:
        row = conn.execute(
            "SELECT month, project_name, pdf_path, amount_inr, attendance_days, "
            "       invoice_number, sent_at, status FROM invoice_history "
            "WHERE month = ?",
            (resolved,),
        ).fetchone()
    if row is None:
        return {"month": resolved, "status": "not_found"}
    return {
        "month": row["month"],
        "project_name": row["project_name"],
        "amount_inr": row["amount_inr"],
        "attendance_days": row["attendance_days"],
        "invoice_number": row["invoice_number"],
        "pdf_path": row["pdf_path"],
        "sent_at": row["sent_at"],
        "status": row["status"],
    }


@tool
def get_invoice(month: str) -> dict:
    """Look up THIS user's invoice for one month. Use this for any question
    about *their* invoices (amount, project, status, when sent, etc.).

    `month` accepts:
      - 'current' or 'this month'
      - 'previous' or 'last month'
      - 'YYYY-MM' (e.g. '2026-03')
      - a month name ('may', 'may 2026')

    Returns a dict with month, project_name, amount_inr, attendance_days,
    invoice_number, pdf_path, sent_at, status. If no record exists for that
    month, returns {month, status: 'not_found'} — say so plainly.
    """
    return _get_invoice_impl(month)
