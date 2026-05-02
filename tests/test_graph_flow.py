"""End-to-end graph flow with mocked WhatsApp, mocked LLM, mocked PDF, mocked SMTP.

This is the "make demo" guarantee — the full state machine resumes correctly
through the interrupts and routes through approve / change_requested / reject
with no real network or filesystem-heavy dependencies.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invoice_agent.graph import compile_graph, thread_config
from invoice_agent.state import initial_state
from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import ApprovalDecision, ProjectReply


# --------- helpers ---------


def _stub_llm(monkeypatch, *, project: str, decision: ApprovalDecision):
    """Stub make_chat so both parsers return canned responses regardless of input."""
    project_reply = ProjectReply(project_name=project)
    state = {"calls": 0}

    class _Stub:
        def with_structured_output(self, schema):
            class _Inner:
                def invoke(self, _messages):
                    state["calls"] += 1
                    if schema is ProjectReply:
                        return project_reply
                    return decision
            return _Inner()

    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: _Stub())


def _stub_io(monkeypatch, tmp_settings):
    """Stub WhatsApp client, PDF renderer, and SMTP send across all node modules."""
    fake_wa = MagicMock()
    monkeypatch.setattr(
        "invoice_agent.nodes.ask_project.WhatsAppClient",
        lambda *a, **kw: fake_wa,
    )
    monkeypatch.setattr(
        "invoice_agent.nodes.send_preview.WhatsAppClient",
        lambda *a, **kw: fake_wa,
    )
    monkeypatch.setattr(
        "invoice_agent.nodes.notify.WhatsAppClient",
        lambda *a, **kw: fake_wa,
    )

    def _fake_render(project_name, month, settings=None):
        p = tmp_settings.out_dir / f"invoice_{month}_{project_name.replace(' ', '_').lower()}.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        return p

    monkeypatch.setattr("invoice_agent.nodes.generate_pdf.render_invoice_pdf", _fake_render)

    sent_emails = []

    def _fake_send_email(**kwargs):
        sent_emails.append(kwargs)

    monkeypatch.setattr("invoice_agent.nodes.email_accounts.send_invoice_email", _fake_send_email)

    return fake_wa, sent_emails


# --------- happy path ---------


def test_graph_happy_path(monkeypatch, tmp_settings):
    """ask -> reply project -> generate -> preview -> approve -> email -> confirm -> END."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    _stub_llm(
        monkeypatch,
        project="Birla Opus",
        decision=ApprovalDecision(status="approved", new_project_name=None),
    )
    fake_wa, sent_emails = _stub_io(monkeypatch, tmp_settings)

    monkeypatch.setattr("invoice_agent.nodes.ask_project.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("invoice_agent.nodes.send_preview.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("invoice_agent.nodes.notify.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("invoice_agent.nodes.email_accounts.get_settings", lambda: tmp_settings)

    cfg = thread_config("2026-05")
    with SqliteSaver.from_conn_string(str(tmp_settings.db_path)) as saver:
        graph = compile_graph(saver)

        # Start: runs ask_project_name then interrupts.
        graph.invoke(initial_state("2026-05", "919999999999"), config=cfg)
        snap = graph.get_state(cfg)
        assert snap.next == ("parse_project_reply",)

        # User replies with project name. Resume.
        graph.update_state(cfg, {"user_reply_raw": "Birla Opus"})
        graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)
        assert snap.values.get("project_name") == "Birla Opus"
        assert snap.values.get("pdf_path") is not None
        # After send_preview we should be paused before parse_approval.
        assert snap.next == ("parse_approval",)

        # User approves. Resume.
        graph.update_state(cfg, {"user_reply_raw": "haan bhej do"})
        graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)
        assert snap.values.get("approval_status") == "approved"
        assert snap.values.get("accounts_email_sent") is True
        assert snap.next == ()  # END

    # Verify side effects
    assert fake_wa.send_template.called
    assert fake_wa.send_document.called
    # confirm_to_user
    assert fake_wa.send_text.called
    assert len(sent_emails) == 1
    assert "Birla Opus" in sent_emails[0]["subject"]


def test_graph_change_requested_then_approved(monkeypatch, tmp_settings):
    """ask -> reply project -> preview -> change-to-X -> generate again -> preview -> approve."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    # First parse_approval returns change_requested; second returns approved.
    decisions = [
        ApprovalDecision(status="change_requested", new_project_name="DLF Camellias"),
        ApprovalDecision(status="approved", new_project_name=None),
    ]

    class _Stub:
        def with_structured_output(self, schema):
            class _Inner:
                def invoke(_self, _messages):
                    if schema is ProjectReply:
                        return ProjectReply(project_name="Birla Opus")
                    return decisions.pop(0)
            return _Inner()

    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: _Stub())
    fake_wa, sent_emails = _stub_io(monkeypatch, tmp_settings)
    for mod in ("ask_project", "send_preview", "notify", "email_accounts"):
        monkeypatch.setattr(f"invoice_agent.nodes.{mod}.get_settings", lambda: tmp_settings)

    cfg = thread_config("2026-05")
    with SqliteSaver.from_conn_string(str(tmp_settings.db_path)) as saver:
        graph = compile_graph(saver)
        graph.invoke(initial_state("2026-05", "919999999999"), config=cfg)

        graph.update_state(cfg, {"user_reply_raw": "Birla Opus"})
        graph.invoke(None, config=cfg)

        # change_requested branch
        graph.update_state(cfg, {"user_reply_raw": "change to DLF Camellias"})
        graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)
        # Loop landed back on send_preview interrupt
        assert snap.next == ("parse_approval",)
        assert snap.values.get("project_name") == "DLF Camellias"

        graph.update_state(cfg, {"user_reply_raw": "ok bhejo"})
        graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)
        assert snap.values.get("accounts_email_sent") is True

    assert len(sent_emails) == 1
    assert "DLF Camellias" in sent_emails[0]["subject"]


def test_graph_rejected(monkeypatch, tmp_settings):
    """User rejects -> notify_cancelled -> END, no email."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    _stub_llm(
        monkeypatch,
        project="Birla Opus",
        decision=ApprovalDecision(status="rejected", new_project_name=None),
    )
    fake_wa, sent_emails = _stub_io(monkeypatch, tmp_settings)
    for mod in ("ask_project", "send_preview", "notify", "email_accounts"):
        monkeypatch.setattr(f"invoice_agent.nodes.{mod}.get_settings", lambda: tmp_settings)

    cfg = thread_config("2026-05")
    with SqliteSaver.from_conn_string(str(tmp_settings.db_path)) as saver:
        graph = compile_graph(saver)
        graph.invoke(initial_state("2026-05", "919999999999"), config=cfg)
        graph.update_state(cfg, {"user_reply_raw": "Birla Opus"})
        graph.invoke(None, config=cfg)
        graph.update_state(cfg, {"user_reply_raw": "nahi ruk"})
        graph.invoke(None, config=cfg)
        snap = graph.get_state(cfg)
        assert snap.values.get("approval_status") == "rejected"
        assert snap.values.get("accounts_email_sent") is False
        assert snap.next == ()

    assert sent_emails == []


def test_graph_persists_across_restart(monkeypatch, tmp_settings):
    """Kill mid-flow, reopen the SqliteSaver, resume the same thread."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    _stub_llm(
        monkeypatch,
        project="Birla Opus",
        decision=ApprovalDecision(status="approved", new_project_name=None),
    )
    _stub_io(monkeypatch, tmp_settings)
    for mod in ("ask_project", "send_preview", "notify", "email_accounts"):
        monkeypatch.setattr(f"invoice_agent.nodes.{mod}.get_settings", lambda: tmp_settings)

    cfg = thread_config("2026-05")
    with SqliteSaver.from_conn_string(str(tmp_settings.db_path)) as saver:
        graph = compile_graph(saver)
        graph.invoke(initial_state("2026-05", "919999999999"), config=cfg)
        graph.update_state(cfg, {"user_reply_raw": "Birla Opus"})
        graph.invoke(None, config=cfg)
        snap_before = graph.get_state(cfg)
        assert snap_before.next == ("parse_approval",)

    # Process "restarts" — fresh saver, same DB file
    with SqliteSaver.from_conn_string(str(tmp_settings.db_path)) as saver2:
        graph2 = compile_graph(saver2)
        snap_after = graph2.get_state(cfg)
        assert snap_after.values.get("project_name") == "Birla Opus"
        assert snap_after.next == ("parse_approval",)
        graph2.update_state(cfg, {"user_reply_raw": "yes"})
        graph2.invoke(None, config=cfg)
        snap_final = graph2.get_state(cfg)
        assert snap_final.values.get("accounts_email_sent") is True
