"""Shared month-token normaliser used by webhook/query and qa/tools."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from invoice_agent.qa.util import normalize_target_month


def test_yyyy_mm_passthrough():
    assert normalize_target_month("2026-05", "Asia/Kolkata") == "2026-05"


def test_relative_phrases():
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    assert normalize_target_month("current", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert normalize_target_month("this month", "Asia/Kolkata") == today.strftime("%Y-%m")


def test_previous_wraps_year():
    # Sanity: previous of 2026-01 is 2025-12. Test what we can without freezegun.
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    out = normalize_target_month("previous", "Asia/Kolkata")
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    assert out == f"{y:04d}-{m:02d}"


def test_month_name_with_year():
    assert normalize_target_month("may 2026", "Asia/Kolkata") == "2026-05"


def test_empty_falls_back_to_today():
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    assert normalize_target_month("", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert normalize_target_month(None, "Asia/Kolkata") == today.strftime("%Y-%m")
