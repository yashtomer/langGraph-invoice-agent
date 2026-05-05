"""Node: render the invoice PDF for the current state.

If a ``new_project_name`` was set (change-requested loop), it overrides
``project_name`` and is then cleared so the loop is single-shot.
"""
from __future__ import annotations

from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.pdf import invoice_number_for, render_invoice_pdf

log = get_logger(__name__)


def generate_pdf(state: InvoiceState) -> InvoiceState:
    new_name = state.get("new_project_name")
    project = new_name or state.get("project_name")
    if not project:
        log.error("generate_pdf.no_project_name")
        return {**state, "error": "no_project_name"}

    month = state["invoice_month"]
    pdf_path = render_invoice_pdf(
        project_name=project,
        month=month,
        amount_inr=state.get("invoice_amount_inr"),
        attendance_days=state.get("attendance_days"),
    )

    log.info("generate_pdf.done", path=str(pdf_path), project=project)
    return {
        **state,
        "project_name": project,
        "new_project_name": None,  # consumed
        "pdf_path": str(pdf_path),
        "invoice_number": invoice_number_for(month),
        "error": None,
    }
