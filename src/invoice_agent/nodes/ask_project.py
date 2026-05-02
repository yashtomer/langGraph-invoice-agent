"""Node: ask the user for the project name on WhatsApp.

This is the entry node. It runs once per invoice month and posts the opening
message via the approved Meta template (templates are required outside the 24h
service window). The graph is compiled with ``interrupt_after`` set on this
node, so execution pauses here until the FastAPI webhook resumes it with the
user's reply written into state.
"""
from __future__ import annotations

from typing import Optional

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.whatsapp import WhatsAppClient

log = get_logger(__name__)


def ask_project_name(
    state: InvoiceState,
    *,
    settings: Optional[Settings] = None,
    wa: Optional[WhatsAppClient] = None,
) -> InvoiceState:
    s = settings or get_settings()
    month = state["invoice_month"]
    to = state.get("user_phone") or s.user_whatsapp_number

    log.info("node.ask_project_name", month=month, to=to)

    client = wa or WhatsAppClient(s)
    try:
        # Use template message — required to *open* a conversation outside 24h window.
        client.send_template(
            to=to,
            template_name=s.approved_template_name,
            language="en",
            body_params=[month],
        )
    finally:
        if wa is None:
            client.close()

    # Reset reply-related fields so a re-run doesn't reuse the previous reply.
    return {
        **state,
        "user_phone": to,
        "user_reply_raw": None,
        "project_name": None,
        "new_project_name": None,
        "approval_status": None,
        "error": None,
    }
