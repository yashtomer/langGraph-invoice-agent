"""LangGraph state definition for the invoice flow."""
from __future__ import annotations

from typing import Literal, Optional, TypedDict


ApprovalStatus = Literal["approved", "rejected", "change_requested"]
SummaryStatus = Literal["pending", "approved", "change_requested"]


class InvoiceState(TypedDict, total=False):
    # Identity
    invoice_month: str          # "YYYY-MM"
    user_phone: str             # WhatsApp ID of the user

    # Conversation inputs
    user_reply_raw: Optional[str]

    # Parsed values
    project_name: Optional[str]
    new_project_name: Optional[str]
    approval_status: Optional[ApprovalStatus]

    # Per-invoice fields (auto-defaulted from settings/month; user can override
    # in the summary-confirmation step before PDF generation).
    invoice_amount_inr: Optional[int]
    attendance_days: Optional[int]
    summary_status: Optional[SummaryStatus]

    # Artifacts
    pdf_path: Optional[str]
    invoice_number: Optional[str]

    # Outcome flags
    accounts_email_sent: bool
    error: Optional[str]


def initial_state(invoice_month: str, user_phone: str) -> InvoiceState:
    return InvoiceState(
        invoice_month=invoice_month,
        user_phone=user_phone,
        user_reply_raw=None,
        project_name=None,
        new_project_name=None,
        approval_status=None,
        invoice_amount_inr=None,
        attendance_days=None,
        summary_status=None,
        pdf_path=None,
        invoice_number=None,
        accounts_email_sent=False,
        error=None,
    )
