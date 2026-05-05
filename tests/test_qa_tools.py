"""qa/tools.py — invoice lookup, comparison, web search."""
from __future__ import annotations

from invoice_agent.db import init_db, mark_sent
from invoice_agent.qa.tools import get_invoice


def _invoke(tool, **kwargs):
    """Tools decorated with @tool expose .invoke({}) — call that."""
    return tool.invoke(kwargs)


def test_get_invoice_yyyy_mm_returns_full_shape(tmp_settings):
    init_db(tmp_settings)
    mark_sent(
        "2026-04",
        project_name="Madabranding",
        pdf_path="/tmp/x.pdf",
        amount_inr=200000,
        attendance_days=30,
        invoice_number="INV-2026-04-001",
        settings=tmp_settings,
    )
    out = _invoke(get_invoice, month="2026-04")
    assert out["month"] == "2026-04"
    assert out["project_name"] == "Madabranding"
    assert out["amount_inr"] == 200000
    assert out["attendance_days"] == 30
    assert out["invoice_number"] == "INV-2026-04-001"
    assert out["pdf_path"] == "/tmp/x.pdf"
    assert out["status"] == "sent"
    assert out["sent_at"] is not None


def test_get_invoice_relative_token_resolves(tmp_settings):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    init_db(tmp_settings)
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    cur_month = today.strftime("%Y-%m")
    mark_sent(cur_month, project_name="ThisMonth", amount_inr=150000, settings=tmp_settings)
    out = _invoke(get_invoice, month="current")
    assert out["month"] == cur_month
    assert out["project_name"] == "ThisMonth"


def test_get_invoice_missing_returns_not_found(tmp_settings):
    init_db(tmp_settings)
    out = _invoke(get_invoice, month="2030-01")
    assert out == {"month": "2030-01", "status": "not_found"}


def test_get_invoice_does_not_raise_on_unknown_token(tmp_settings):
    init_db(tmp_settings)
    # Unknown tokens fall back to today's month per normalize_target_month.
    out = _invoke(get_invoice, month="garbage")
    assert "status" in out  # either 'not_found' or a real status, never raises
