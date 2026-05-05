"""Helpers to start and resume the LangGraph invoice flow.

Two entry points are needed:

  * ``start_for_month`` — kick the flow off on the 25th (or via ``/trigger``).
    The graph runs through ``ask_project_name`` and then pauses (interrupt_after).

  * ``resume_with_reply`` — called from the webhook when the user replies.
    Writes the raw text into state and resumes the thread; the graph either
    pauses again (after ``send_preview``) or runs to END.
"""
from __future__ import annotations

import sqlite3
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


def _reset_thread_state(month: str, *, settings: Settings) -> None:
    """Wipe LangGraph checkpoints + invoice_history row for a month.

    Used when ``start_for_month`` is called with ``force=True`` so a re-trigger
    starts from scratch instead of resuming a finished thread.
    """
    thread_id = f"invoice-{month}"
    with sqlite3.connect(str(settings.db_path)) as con:
        con.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        con.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        con.execute("DELETE FROM invoice_history WHERE month = ?", (month,))
        con.commit()
    log.info("runner.thread_reset", month=month, thread_id=thread_id)


def start_for_month(
    month: str,
    *,
    force: bool = False,
    settings: Optional[Settings] = None,
) -> dict:
    """Begin (or resume from the start) the invoice flow for ``month``.

    With ``force=False`` (the default, used by the scheduler), this is
    idempotent: a month already marked 'sent' is a no-op so a misfired cron
    can't double-send.

    With ``force=True`` (used by manual ``/trigger``), the existing thread
    state is wiped and the flow restarts from scratch, even if the month was
    already sent.

    Returns the post-invoke state values dict.
    """
    s = settings or get_settings()
    if force:
        _reset_thread_state(month, settings=s)
    elif already_sent(month, settings=s):
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
            amount_inr=snap.values.get("invoice_amount_inr"),
            attendance_days=snap.values.get("attendance_days"),
            invoice_number=snap.values.get("invoice_number"),
            settings=s,
        )
    else:
        mark_status(month, _final_status(snap.values), settings=s)

    return dict(snap.values)
