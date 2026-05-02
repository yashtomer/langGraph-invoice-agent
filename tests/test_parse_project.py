"""Tests for parse_project_name — both LLM path (mocked) and heuristic fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import (
    ProjectReply,
    _heuristic_project,
    parse_project_name,
)


def _fake_llm_returning(reply: ProjectReply):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = reply
    fake.with_structured_output.return_value = structured
    return fake


def test_parse_project_english_simple(monkeypatch):
    fake = _fake_llm_returning(ProjectReply(project_name="Birla Opus"))
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)
    out = parse_project_name("Birla Opus")
    assert out.project_name == "Birla Opus"


def test_parse_project_hinglish(monkeypatch):
    fake = _fake_llm_returning(ProjectReply(project_name="DLF Camellias"))
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)
    out = parse_project_name("project ka naam DLF Camellias hai bhai")
    assert out.project_name == "DLF Camellias"


def test_parse_project_falls_back_to_heuristic_on_double_failure(monkeypatch):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("ollama down")
    fake.with_structured_output.return_value = structured
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)

    out = parse_project_name("project naam Tata Steel hai")
    # Heuristic strips filler words; should contain Tata Steel
    assert "Tata Steel" in out.project_name


def test_heuristic_project_directly():
    assert _heuristic_project("Birla Opus").project_name == "Birla Opus"
    assert "DLF Camellias" in _heuristic_project("the project naam DLF Camellias hai").project_name
