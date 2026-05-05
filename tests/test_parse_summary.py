"""Tests for parse_summary_reply — LLM (mocked) + heuristic fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import (
    SummaryReply,
    _heuristic_summary,
    parse_summary_reply,
)


def _fake_llm_returning(reply: SummaryReply):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = reply
    fake.with_structured_output.return_value = structured
    return fake


def test_llm_path_approve(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "make_chat", lambda **_: _fake_llm_returning(SummaryReply(status="approved"))
    )
    out = parse_summary_reply("approve")
    assert out.status == "approved"
    assert out.amount_inr is None
    assert out.attendance_days is None


def test_llm_path_with_overrides(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "make_chat",
        lambda **_: _fake_llm_returning(
            SummaryReply(status="approved", amount_inr=200000, attendance_days=28)
        ),
    )
    out = parse_summary_reply("approve with attendance 28 and amount 200000")
    assert out.status == "approved"
    assert out.amount_inr == 200000
    assert out.attendance_days == 28


def test_falls_back_to_heuristic_on_double_failure(monkeypatch):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("ollama down")
    fake.with_structured_output.return_value = structured
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)

    out = parse_summary_reply("approve with amount 200000")
    assert out.status == "approved"
    assert out.amount_inr == 200000


def test_heuristic_approval_only():
    assert _heuristic_summary("approve").status == "approved"
    assert _heuristic_summary("yes go ahead").status == "approved"
    assert _heuristic_summary("haan bhej do").status == "approved"
    assert _heuristic_summary("ok generate it").status == "approved"


def test_heuristic_change_only():
    out = _heuristic_summary("amount 50000")
    assert out.status == "change_requested"
    assert out.amount_inr == 50000


def test_heuristic_amount_with_suffix():
    assert _heuristic_summary("amount 200k").amount_inr == 200_000
    assert _heuristic_summary("change to 2 lakh").amount_inr == 200_000
    assert _heuristic_summary("set amount 1.5 lac").amount_inr is not None  # captured


def test_heuristic_attendance():
    out = _heuristic_summary("attendance 28")
    assert out.attendance_days == 28
    out = _heuristic_summary("change to 30 days")
    assert out.attendance_days == 30


def test_heuristic_approve_with_overrides():
    out = _heuristic_summary("approve with attendance 28")
    assert out.status == "approved"
    assert out.attendance_days == 28
    out = _heuristic_summary("yes amount 200000")
    assert out.status == "approved"
    assert out.amount_inr == 200000


def test_heuristic_project_change():
    out = _heuristic_summary("change project to Birla Opus")
    assert out.status == "change_requested"
    assert out.project_name == "Birla Opus"
