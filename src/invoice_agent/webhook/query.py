"""Free-form WhatsApp query handler.

Uses the LLM intent classifier (``tools.llm.parse_query_intent``) to route
inbound messages. Deterministic handlers cover ``last_invoice_amount`` and
``greeting``; ``start_invoice`` triggers a fresh flow for the requested month;
``generic_question`` is delegated to the Q&A tool-calling agent
(``qa.answer``) for open-ended invoice-history questions.

Active-flow guard: ``generic_question`` is suppressed when any month is
mid-conversation (``status == 'started'``) so a project-name or approval reply
that the LLM mis-classifies as a question doesn't get eaten by the chat
fallback.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..config import Settings, get_settings
from ..db import get_active_month, get_last_sent
from ..logging_setup import get_logger
from ..qa.util import normalize_target_month as _normalize_target_month  # re-export
from ..tools.llm import QueryIntent, parse_query_intent
from ..tools.pdf import _month_label, fmt_inr
from ..tools.whatsapp import WhatsAppClient

log = get_logger(__name__)

IntentHandler = Callable[..., Optional[str]]


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


def _generic_question(text: str, _intent: QueryIntent, s: Settings,
                     *, user_phone: str) -> str:
    from ..qa import answer as qa_answer  # local import: avoid cycle
    return qa_answer(text, user_phone, settings=s)


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


def try_answer(
    text: str,
    *,
    user_phone: str = "",
    settings: Optional[Settings] = None,
) -> Optional[str]:
    """Return a reply string, an empty string (matched but reply already sent),
    or ``None`` (no intent matched — webhook should fall through to flow router)."""
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
    if handler is None:
        return None
    if intent.intent == "generic_question":
        return handler(text, intent, s, user_phone=user_phone)
    return handler(text, intent, s)
