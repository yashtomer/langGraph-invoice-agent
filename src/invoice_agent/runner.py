"""Helpers to start and resume the LangGraph invoice flow.

Two entry points are needed:

  * ``start_for_month`` — kick the flow off on the 25th (or via ``/trigger``).
    The graph runs through ``ask_project_name`` and then pauses (interrupt_after).

  * ``resume_with_reply`` — called from the webhook when the user replies.
    Writes the raw text into state and resumes the thread; the graph either
    pauses again (after ``send_preview``) or runs to END.
"""
from __future__ import annotations

from typing import Optional

from .config import Settings, get_settings
from .db import already_sent, mark_sent, mark_started, mark_status
from .graph import compile_graph, open_checkpointer, thread_config
from .logging_setup import get_logger
from .state import initial_state

log = get_logger(__name__)


def _final_status(snapshot_values: dict) -> str:
    if snapshot_values.get("accounts_email_sent"):
        return "sent"
    if snapshot_values.get("approval_status") == "rejected":
        return "cancelled"
    return "started"


def start_for_month(month: str, *, settings: Optional[Settings] = None) -> dict:
    """Begin (or resume from the start) the invoice flow for ``month``.

    Returns the post-invoke state values dict. Idempotent: if the month is
    already marked 'sent', this is a no-op.
    """
    s = settings or get_settings()
    if already_sent(month, settings=s):
        log.info("runner.skip.already_sent", month=month)
        return {"skipped": True, "month": month}

    mark_started(month, settings=s)
    cfg = thread_config(month)

    with open_checkpointer(s) as saver:
        graph = compile_graph(saver)
        # Seed the state if this is a fresh thread; if it already exists,
        # invoke(None, ...) will resume.
        existing = graph.get_state(cfg)
        if not existing.values:
            init = initial_state(invoice_month=month, user_phone=s.user_whatsapp_number)
            log.info("runner.start", month=month)
            graph.invoke(init, config=cfg)
        else:
            log.info("runner.resume_existing", month=month)
            graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)

    mark_status(month, _final_status(snap.values), settings=s)
    return dict(snap.values)


def resume_with_reply(month: str, reply_text: str, *, settings: Optional[Settings] = None) -> dict:
    """Resume the thread for ``month`` after a user WhatsApp reply."""
    s = settings or get_settings()
    cfg = thread_config(month)

    with open_checkpointer(s) as saver:
        graph = compile_graph(saver)
        # Write the reply into state, then continue.
        graph.update_state(cfg, {"user_reply_raw": reply_text})
        log.info("runner.resume_with_reply", month=month, reply_preview=reply_text[:80])
        graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)

    if snap.values.get("accounts_email_sent"):
        mark_sent(
            month,
            project_name=snap.values.get("project_name"),
            pdf_path=snap.values.get("pdf_path"),
            settings=s,
        )
    else:
        mark_status(month, _final_status(snap.values), settings=s)

    return dict(snap.values)
