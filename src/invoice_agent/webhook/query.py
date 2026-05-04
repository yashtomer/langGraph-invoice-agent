"""Free-form WhatsApp query handler.

Uses the LLM intent classifier (``tools.llm.parse_query_intent``) to route
inbound messages. Three intents have deterministic handlers; ``generic_question``
falls back to a free-form ``chat_reply`` LLM call for open-ended chit-chat.

Active-flow guard: ``generic_question`` is suppressed when the current month's
invoice flow is mid-conversation (``status == 'started'``), so a project-name
or approval reply that the LLM mis-classifies as a question doesn't get eaten
by the chat fallback.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from ..config import Settings, get_settings
from ..db import get_last_sent, get_status
from ..logging_setup import get_logger
from ..tools.llm import chat_reply, parse_query_intent

log = get_logger(__name__)

IntentHandler = Callable[[str, Settings], str]


def _last_invoice_amount(_text: str, s: Settings) -> str:
    row = get_last_sent(settings=s)
    if row is None:
        return "You don't have any sent invoices yet."
    amount = f"INR {s.invoice_amount_inr:,}"
    project = row["project_name"] or "(no project name)"
    return f"Your last invoice was {amount} for {project} ({row['month']})."


def _greeting(_text: str, s: Settings) -> str:
    return (
        "Hi! I'm your invoice assistant. You can ask me things like:\n"
        "  • \"what is my last invoice amount\"\n"
        "  • \"kitna tha pichla invoice\"\n"
        "Or wait for the monthly invoice nudge."
    )


def _generic_question(text: str, s: Settings) -> str:
    return chat_reply(text, settings=s)


_INTENT_HANDLERS: dict[str, IntentHandler] = {
    "last_invoice_amount": _last_invoice_amount,
    "greeting": _greeting,
    "generic_question": _generic_question,
}


def _current_month(tz: str) -> str:
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m")


def try_answer(text: str, *, settings: Optional[Settings] = None) -> Optional[str]:
    if not text or not text.strip():
        return None
    s = settings or get_settings()
    intent = parse_query_intent(text).intent
    log.info("query.intent_classified", intent=intent, text_preview=text[:80])

    if intent == "generic_question":
        # Don't free-chat when an invoice flow is mid-conversation — the user's
        # message is more likely a project-name or approval reply that the graph
        # needs to consume. Let the webhook fall through to resume_with_reply.
        status = get_status(_current_month(s.timezone), settings=s)
        if status == "started":
            log.info("query.suppress_chat_during_active_flow", status=status)
            return None

    handler = _INTENT_HANDLERS.get(intent)
    return handler(text, s) if handler else None
