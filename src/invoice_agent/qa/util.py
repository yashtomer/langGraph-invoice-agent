"""Shared utilities for the Q&A agent."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from ..logging_setup import get_logger

log = get_logger(__name__)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _today(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))


def normalize_target_month(token: Optional[str], tz: str) -> str:
    """Resolve a free-form month token to ``YYYY-MM``.

    Accepts: ``YYYY-MM``, ``"may"`` / ``"may 2026"``, ``"this month"``,
    ``"current"``, ``"previous month"`` / ``"last month"``, ``"next month"``.
    Falls back to today's month when the token is empty or unrecognised.
    """
    today = _today(tz)
    if not token:
        return today.strftime("%Y-%m")
    t = token.strip().lower()

    if len(t) == 7 and t[4] == "-" and t[:4].isdigit() and t[5:].isdigit():
        return t

    if t in ("current", "this month", "this", "now"):
        return today.strftime("%Y-%m")
    if t in ("previous month", "last month", "previous", "last"):
        y, m = today.year, today.month - 1
        if m == 0:
            y, m = y - 1, 12
        return f"{y:04d}-{m:02d}"
    if t in ("next month", "next"):
        y, m = today.year, today.month + 1
        if m == 13:
            y, m = y + 1, 1
        return f"{y:04d}-{m:02d}"

    parts = t.split()
    name = parts[0]
    if name in _MONTH_NAMES:
        mon = _MONTH_NAMES[name]
        year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else today.year
        return f"{year:04d}-{mon:02d}"

    log.warning("qa.normalize_target_month.unrecognised", token=token)
    return today.strftime("%Y-%m")
