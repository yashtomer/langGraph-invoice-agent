"""Free-form WhatsApp query handler.

Uses the LLM intent classifier (``tools.llm.parse_query_intent``) to route
inbound messages. Deterministic handlers cover ``last_invoice_amount`` and
``greeting``; ``start_invoice`` triggers a fresh flow for the requested month;
``generic_question`` falls back to ``chat_reply`` for open-ended chit-chat.

Active-flow guard: ``generic_question`` is suppressed when any month is
mid-conversation (``status == 'started'``) so a project-name or approval reply
that the LLM mis-classifies as a question doesn't get eaten by the chat
fallback.
"""
from __future__ import annotations

import calendar
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from ..config import Settings, get_settings
from ..db import get_active_month, get_last_sent
from ..logging_setup import get_logger
from ..tools.llm import QueryIntent, chat_reply, parse_query_intent
from ..tools.pdf import _month_label, fmt_inr
from ..tools.whatsapp import WhatsAppClient

log = get_logger(__name__)

IntentHandler = Callable[[str, QueryIntent, Settings], Optional[str]]

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _today(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))


def _normalize_target_month(token: Optional[str], tz: str) -> str:
    """Resolve a free-form month token to ``YYYY-MM``.

    Accepts: ``YYYY-MM``, ``"may"`` / ``"may 2026"``, ``"this month"``,
    ``"current"``, ``"previous month"`` / ``"last month"``, ``"next month"``.
    Falls back to today's month when the token is empty or unrecognised.
    """
    today = _today(tz)
    if not token:
        return today.strftime("%Y-%m")
    t = token.strip().lower()

    # YYYY-MM literal
    if len(t) == 7 and t[4] == "-" and t[:4].isdigit() and t[5:].isdigit():
        return t

    # Relative phrases
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

    # "may" or "may 2026"
    parts = t.split()
    name = parts[0]
    if name in _MONTH_NAMES:
        mon = _MONTH_NAMES[name]
        year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else today.year
        return f"{year:04d}-{mon:02d}"

    log.warning("query.normalize_target_month.unrecognised", token=token)
    return today.strftime("%Y-%m")


def _last_invoice_amount(_text: str, _intent: QueryIntent, s: Settings) -> str:
    row = get_last_sent(settings=s)
    if row is None:
        return "You don't have any sent invoices yet."
    amount = f"INR {fmt_inr(s.invoice_amount_inr)}"
    project = row["project_name"] or "(no project name)"
    return f"Your last invoice was {amount} for {project} ({row['month']})."


def _greeting(_text: str, _intent: QueryIntent, s: Settings) -> str:
    return (
        "Hi! I'm your invoice assistant. You can ask me things like:\n"
        "  • \"what is my last invoice amount\"\n"
        "  • \"send invoice for may\"\n"
        "  • \"hi\" for help"
    )


def _generic_question(text: str, _intent: QueryIntent, s: Settings) -> str:
    return chat_reply(text, settings=s)


def _start_invoice(_text: str, intent: QueryIntent, s: Settings) -> str:
    """Trigger a fresh invoice flow for the user-requested month.

    Sends a confirmation reply, then runs the graph through ``ask_project_name``
    (which sends the WhatsApp template message). ``force=True`` so we re-run
    even if the month was previously marked 'sent'.
    """
    from ..runner import start_for_month  # local import to avoid cycles

    month = _normalize_target_month(intent.target_month, s.timezone)
    label = _month_label(month)
    log.info("query.start_invoice", month=month, target_token=intent.target_month)

    # Confirmation goes out before we trigger so the user sees the ack first;
    # the graph's ask_project_name will follow with the template message.
    user_phone = s.user_whatsapp_number
    with WhatsAppClient(s) as wa:
        wa.send_text(
            to=user_phone,
            body=f"Starting invoice flow for {label}. I'll ask for the project name next.",
        )

    try:
        start_for_month(month, force=True, settings=s)
    except Exception as e:  # noqa: BLE001
        log.warning("query.start_invoice.failed", err=str(e), month=month)
        return f"Couldn't start the {label} invoice flow: {e}"

    # Flow's ask_project_name has already sent the template; no extra reply needed.
    return ""


_INTENT_HANDLERS: dict[str, IntentHandler] = {
    "last_invoice_amount": _last_invoice_amount,
    "greeting": _greeting,
    "start_invoice": _start_invoice,
    "generic_question": _generic_question,
}


def try_answer(text: str, *, settings: Optional[Settings] = None) -> Optional[str]:
    """Return a reply string, an empty string (matched but reply already sent),
    or ``None`` (no intent matched — webhook should fall through to flow router).
    """
    if not text or not text.strip():
        return None
    s = settings or get_settings()
    intent = parse_query_intent(text)
    log.info(
        "query.intent_classified",
        intent=intent.intent,
        target_month=intent.target_month,
        text_preview=text[:80],
    )

    if intent.intent == "generic_question":
        active = get_active_month(settings=s)
        if active is not None:
            log.info("query.suppress_chat_during_active_flow", active_month=active)
            return None

    handler = _INTENT_HANDLERS.get(intent.intent)
    return handler(text, intent, s) if handler else None
