"""Node: parse the user's reply to the preview into approved/rejected/change."""
from __future__ import annotations

from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.llm import parse_approval_reply

log = get_logger(__name__)


def parse_approval(state: InvoiceState) -> InvoiceState:
    raw = (state.get("user_reply_raw") or "").strip()
    if not raw:
        # Conservative default — if we somehow resumed without a reply, treat as rejected.
        log.warning("parse_approval.empty_reply.defaulting_rejected")
        return {**state, "approval_status": "rejected", "error": "empty_reply"}

    decision = parse_approval_reply(raw)
    log.info(
        "parse_approval.parsed",
        status=decision.status,
        new_project=decision.new_project_name,
        raw=raw,
    )

    return {
        **state,
        "approval_status": decision.status,
        "new_project_name": decision.new_project_name,
        "user_reply_raw": None,  # consumed
        "error": None,
    }
