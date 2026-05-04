"""Node: email the approved invoice PDF to accounts (and CC the user)."""
from __future__ import annotations

from typing import Callable, Optional

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.mailer import send_invoice_email

log = get_logger(__name__)


def email_accounts(
    state: InvoiceState,
    *,
    settings: Optional[Settings] = None,
    sender: Optional[Callable] = None,
) -> InvoiceState:
    s = settings or get_settings()
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        log.error("email_accounts.no_pdf")
        return {**state, "error": "no_pdf"}

    project = state.get("project_name") or "Unknown"
    month = state["invoice_month"]
    invoice_no = state.get("invoice_number") or "?"

    subject = f"Invoice {invoice_no} — {project} — {month}"
    body = (
        f"Hi team,\n\n"
        f"Please find attached the invoice for {project} ({month}).\n"
        f"Invoice #: {invoice_no}\n\n"
        f"Regards,\n{s.company_name}\n"
    )

    fn = sender or send_invoice_email
    fn(
        subject=subject,
        body=body,
        pdf_path=pdf_path,
        to=s.accounts_recipients(),
        cc=s.cc_recipients(),
        settings=s,
    )

    log.info("email_accounts.sent", subject=subject, to=s.accounts_recipients())
    return {**state, "accounts_email_sent": True, "error": None}
