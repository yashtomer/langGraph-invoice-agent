"""Node: parse the user's reply to the draft-invoice summary.

Pulls a structured ``SummaryReply`` from the LLM and applies any inline
overrides (amount, attendance, project name) onto the graph state. The
``summary_status`` field controls the conditional edge afterwards:

  * ``approved``         -> proceed to ``generate_pdf``
  * ``change_requested`` -> loop back to ``send_summary`` with new values
"""
from __future__ import annotations

from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.llm import parse_summary_reply

log = get_logger(__name__)


def parse_summary(state: InvoiceState) -> InvoiceState:
    raw = (state.get("user_reply_raw") or "").strip()
    if not raw:
        log.warning("parse_summary.empty_reply")
        return {**state, "error": "empty_reply"}

    parsed = parse_summary_reply(raw)
    log.info(
        "parse_summary.parsed",
        status=parsed.status,
        amount=parsed.amount_inr,
        attendance=parsed.attendance_days,
        project=parsed.project_name,
        raw=raw,
    )

    update: dict = {
        "summary_status": parsed.status,
        "user_reply_raw": None,
        "error": None,
    }
    if parsed.amount_inr is not None:
        update["invoice_amount_inr"] = parsed.amount_inr
    if parsed.attendance_days is not None:
        update["attendance_days"] = parsed.attendance_days
    if parsed.project_name:
        update["project_name"] = parsed.project_name

    return {**state, **update}
