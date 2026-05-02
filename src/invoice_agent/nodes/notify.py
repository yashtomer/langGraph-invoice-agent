"""Terminal-state nodes: confirm to the user (success) or notify cancelled."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.whatsapp import WhatsAppClient

log = get_logger(__name__)


def _now_local(tz: str) -> str:
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M %Z")


def confirm_to_user(
    state: InvoiceState,
    *,
    settings: Optional[Settings] = None,
    wa: Optional[WhatsAppClient] = None,
) -> InvoiceState:
    s = settings or get_settings()
    msg = (
        f"Sent to accounts at {_now_local(s.timezone)}.\n"
        f"Invoice #: {state.get('invoice_number')}\n"
        f"Project: {state.get('project_name')}"
    )
    client = wa or WhatsAppClient(s)
    try:
        client.send_text(to=state["user_phone"], body=msg)
    finally:
        if wa is None:
            client.close()
    log.info("notify.confirm", to=state["user_phone"])
    return state


def notify_cancelled(
    state: InvoiceState,
    *,
    settings: Optional[Settings] = None,
    wa: Optional[WhatsAppClient] = None,
) -> InvoiceState:
    s = settings or get_settings()
    msg = "Cancelled. POST /trigger or wait for next month to retry."
    client = wa or WhatsAppClient(s)
    try:
        client.send_text(to=state["user_phone"], body=msg)
    finally:
        if wa is None:
            client.close()
    log.info("notify.cancelled", to=state["user_phone"])
    return state
