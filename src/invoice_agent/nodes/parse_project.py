"""Node: parse the user's free-form reply into a structured project name."""
from __future__ import annotations

from ..logging_setup import get_logger
from ..state import InvoiceState
from ..tools.llm import parse_project_name

log = get_logger(__name__)


def parse_project_reply(state: InvoiceState) -> InvoiceState:
    raw = (state.get("user_reply_raw") or "").strip()
    if not raw:
        log.warning("parse_project.empty_reply")
        return {**state, "error": "empty_reply"}

    parsed = parse_project_name(raw)
    log.info("parse_project.parsed", project=parsed.project_name, raw=raw)
    return {
        **state,
        "project_name": parsed.project_name,
        "user_reply_raw": None,  # consumed
        "error": None,
    }
