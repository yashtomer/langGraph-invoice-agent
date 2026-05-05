"""LangChain @tool functions for the Q&A agent.

The docstrings on each @tool are load-bearing — the LLM uses them to choose
which tool to call. Edit with care.
"""
from __future__ import annotations

from contextvars import ContextVar

import httpx
from langchain_core.tools import tool

from ..config import Settings, get_settings
from ..db import connect
from ..logging_setup import get_logger
from .util import normalize_target_month

log = get_logger(__name__)


def _get_invoice_impl(month: str, *, settings: Settings | None = None) -> dict:
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


@tool
def compare_invoices() -> dict:
    """Compare THIS user's current month vs previous month invoices.
    Use for questions like 'is that more than last month?', 'higher',
    'difference', 'same as last month'. No arguments.

    Returns:
      {current: {...same shape as get_invoice...},
       previous: {...},
       amount_diff_inr: current.amount_inr - previous.amount_inr (None if either missing),
       same_project: bool (False if either missing)}
    """
    s = get_settings()
    cur = _get_invoice_impl("current", settings=s)
    prev = _get_invoice_impl("previous", settings=s)
    cur_amt = cur.get("amount_inr") if cur.get("status") != "not_found" else None
    prev_amt = prev.get("amount_inr") if prev.get("status") != "not_found" else None
    diff = (cur_amt - prev_amt) if (cur_amt is not None and prev_amt is not None) else None
    same_project = (
        cur.get("status") != "not_found"
        and prev.get("status") != "not_found"
        and cur.get("project_name") == prev.get("project_name")
        and cur.get("project_name") is not None
    )
    return {
        "current": cur,
        "previous": prev,
        "amount_diff_inr": diff,
        "same_project": same_project,
    }


# The counter lives inside a mutable single-element list so that mutations
# made inside tool.invoke() (which langchain runs in a copied context via
# copy_context()) still propagate back to the caller. The list object is
# shared by reference even though the ContextVar binding isn't.
_web_search_count: ContextVar[list[int] | None] = ContextVar("_web_search_count", default=None)


def reset_web_search_budget() -> None:
    """Reset the per-turn web-search counter. Called by qa.agent.answer at the
    start of every turn before agent.invoke."""
    _web_search_count.set([0])


@tool
def web_search(query: str) -> list[dict] | dict:
    """Search the web for general knowledge. Use ONLY for questions that are
    NOT about THIS user's own invoices — e.g. tax rates, GST rules,
    regulatory questions, definitions, current events.

    NEVER call this to look up the user's invoice data — use get_invoice or
    compare_invoices for that. Returns up to 3 results: list of
    {title, url, snippet}. On failure returns {error: 'search_unavailable'}
    or {error: 'search_budget_exceeded'}; if you see those, tell the user
    you couldn't look it up — don't guess.
    """
    s = get_settings()

    # Per-turn budget. The counter lives inside a mutable list so increments
    # made inside a copied context (langchain wraps tool execution in
    # copy_context().run(...)) are visible to the caller.
    counter = _web_search_count.get()
    if counter is None:
        # Caller didn't reset; initialize lazily.
        counter = [0]
        _web_search_count.set(counter)
    if counter[0] >= s.qa_web_search_max_calls_per_turn:
        log.warning("qa.tool_failed", tool="web_search", err="budget_exceeded")
        return {"error": "search_budget_exceeded"}
    counter[0] += 1

    api_key = s.tavily_api_key.get_secret_value()
    if not api_key:
        log.warning("qa.tool_failed", tool="web_search", err="no_api_key")
        return {"error": "search_unavailable"}

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": 3,
                    "include_raw_content": False,
                },
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("qa.tool_failed", tool="web_search", err=str(e)[:120])
        return {"error": "search_unavailable"}

    results = data.get("results") or []
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
        for x in results[:3]
    ]
