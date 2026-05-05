"""Node: WhatsApp the draft-invoice summary for user confirmation.

The graph compiles with ``interrupt_after`` on this node, so execution pauses
once the summary message is sent. The webhook resumes with the user's reply
written into ``user_reply_raw``; ``parse_summary`` consumes it next.

Auto-fills any per-invoice field that's not yet in state (so the user only
has to provide ``project_name`` upstream — everything else is suggested).
"""
from __future__ import annotations

import calendar
from typing import Optional

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.pdf import _invoice_date_label, _month_label, fmt_inr, invoice_number_for
from ..tools.whatsapp import WhatsAppClient

log = get_logger(__name__)


def _format_summary(
    *,
    month: str,
    project_name: str,
    amount_inr: int,
    attendance_days: int,
    invoice_number: str,
) -> str:
    return (
        f"Draft for {_month_label(month)}:\n"
        f"• Project: {project_name}\n"
        f"• Amount: INR {fmt_inr(amount_inr)}\n"
        f"• Attendance: {attendance_days} days\n"
        f"• Invoice No: {invoice_number}\n"
        f"• Date: {_invoice_date_label(month)}\n"
        "\n"
        "Reply *approve* to generate, or change fields inline\n"
        "(e.g. \"amount 200000\", \"attendance 30\", \"approve with attendance 28\")."
    )


def send_summary(
    state: InvoiceState,
    *,
    settings: Optional[Settings] = None,
    wa: Optional[WhatsAppClient] = None,
) -> InvoiceState:
    s = settings or get_settings()
    month = state["invoice_month"]
    project = state.get("project_name") or "(unknown project)"
    to = state.get("user_phone") or s.user_whatsapp_number

    year, mon = (int(p) for p in month.split("-"))
    amount = state.get("invoice_amount_inr") or s.invoice_amount_inr
    attendance = state.get("attendance_days") or calendar.monthrange(year, mon)[1]
    invoice_number = state.get("invoice_number") or invoice_number_for(month)

    body = _format_summary(
        month=month,
        project_name=project,
        amount_inr=amount,
        attendance_days=attendance,
        invoice_number=invoice_number,
    )

    log.info(
        "node.send_summary",
        month=month,
        project=project,
        amount=amount,
        attendance=attendance,
        invoice_number=invoice_number,
    )

    client = wa or WhatsAppClient(s)
    try:
        client.send_text(to=to, body=body)
    finally:
        if wa is None:
            client.close()

    return {
        **state,
        "invoice_amount_inr": amount,
        "attendance_days": attendance,
        "invoice_number": invoice_number,
        "summary_status": "pending",
        "user_reply_raw": None,
    }
