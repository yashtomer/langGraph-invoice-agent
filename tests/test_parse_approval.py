"""Tests for parse_approval_reply — exercise mock LLM and heuristic fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import (
    ApprovalDecision,
    _heuristic_approval,
    parse_approval_reply,
)


def _fake_llm_returning(decision: ApprovalDecision):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = decision
    fake.with_structured_output.return_value = structured
    return fake


@pytest.mark.parametrize(
    "text,expected_status,new_name",
    [
        ("haan bhej do", "approved", None),
        ("send it please", "approved", None),
        ("nahi ruk", "rejected", None),
        ("change to Birla Opus", "change_requested", "Birla Opus"),
    ],
)
def test_parse_approval_paths_mocked(monkeypatch, text, expected_status, new_name):
    fake = _fake_llm_returning(
        ApprovalDecision(status=expected_status, new_project_name=new_name)
    )
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)

    out = parse_approval_reply(text)
    assert out.status == expected_status
    assert out.new_project_name == new_name


def test_parse_approval_falls_back_to_heuristic(monkeypatch):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("nope")
    fake.with_structured_output.return_value = structured
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)

    assert parse_approval_reply("yes").status == "approved"
    assert parse_approval_reply("nahi cancel kar do").status == "rejected"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", "approved"),
        ("ok bhejo", "approved"),
        ("haan", "approved"),
        ("ji bhej do", "approved"),
        ("no", "rejected"),
        ("nahi", "rejected"),
        ("ruk", "rejected"),
        ("cancel", "rejected"),
        ("change to Birla Opus", "change_requested"),
    ],
)
def test_heuristic_approval_direct(text, expected):
    out = _heuristic_approval(text)
    assert out.status == expected
    if expected == "change_requested":
        assert out.new_project_name == "Birla Opus"
