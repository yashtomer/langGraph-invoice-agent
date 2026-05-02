"""Node: send the PDF preview to the user on WhatsApp and pause for approval."""
from __future__ import annotations

from typing import Optional

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.whatsapp import WhatsAppClient

log = get_logger(__name__)


_CAPTION = (
    "Preview ready for {month}.\n"
    "Project: {project}\n"
    "Invoice #: {invoice_number}\n\n"
    "Reply YES to send to accounts, NO to cancel, "
    "or 'change to <name>' to update the project name."
)


def send_preview(
    state: InvoiceState,
    *,
    settings: Optional[Settings] = None,
    wa: Optional[WhatsAppClient] = None,
) -> InvoiceState:
    s = settings or get_settings()
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        log.error("send_preview.no_pdf")
        return {**state, "error": "no_pdf"}

    caption = _CAPTION.format(
        month=state["invoice_month"],
        project=state.get("project_name") or "?",
        invoice_number=state.get("invoice_number") or "?",
    )

    client = wa or WhatsAppClient(s)
    try:
        client.send_document(to=state["user_phone"], path=pdf_path, caption=caption)
    finally:
        if wa is None:
            client.close()

    log.info("send_preview.sent", to=state["user_phone"], path=pdf_path)
    return {**state, "user_reply_raw": None, "error": None}
