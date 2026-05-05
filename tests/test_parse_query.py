"""Tests for parse_query_intent — both LLM path (mocked) and heuristic fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import (
    QueryIntent,
    _heuristic_query,
    parse_query_intent,
)


def _fake_llm_returning(intent: str):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = QueryIntent(intent=intent)  # type: ignore[arg-type]
    fake.with_structured_output.return_value = structured
    return fake


def test_llm_path_last_invoice_amount(monkeypatch):
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: _fake_llm_returning("last_invoice_amount"))
    out = parse_query_intent("how much was my last invoice")
    assert out.intent == "last_invoice_amount"


def test_llm_path_none(monkeypatch):
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: _fake_llm_returning("none"))
    out = parse_query_intent("Birla Opus")
    assert out.intent == "none"


def test_falls_back_to_heuristic_on_double_failure(monkeypatch):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("ollama down")
    fake.with_structured_output.return_value = structured
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)

    assert parse_query_intent("what is my last invoice amount").intent == "last_invoice_amount"
    assert parse_query_intent("kitna tha pichla invoice").intent == "last_invoice_amount"
    assert parse_query_intent("Birla Opus").intent == "none"


def test_heuristic_directly():
    assert _heuristic_query("last invoice amount?").intent == "last_invoice_amount"
    assert _heuristic_query("how much was the previous invoice").intent == "last_invoice_amount"
    assert _heuristic_query("amount on last bill").intent == "last_invoice_amount"
    assert _heuristic_query("kitna tha pichla invoice").intent == "last_invoice_amount"
    assert _heuristic_query("Birla Opus").intent == "none"
    assert _heuristic_query("yes send it").intent == "none"


def test_heuristic_start_invoice():
    out = _heuristic_query("send invoice for may")
    assert out.intent == "start_invoice"
    assert out.target_month == "may"
    out = _heuristic_query("i want to send invoice for june 2026")
    assert out.intent == "start_invoice"
    assert out.target_month == "june 2026"
    out = _heuristic_query("trigger this month invoice")
    assert out.intent == "start_invoice"
    assert out.target_month == "this month"
    out = _heuristic_query("invoice bana do")
    assert out.intent == "start_invoice"
    assert out.target_month == "current"
    out = _heuristic_query("create previous month invoice")
    assert out.intent == "start_invoice"
    assert out.target_month == "previous month"
    # Approval-shaped phrases without 'invoice'/'bill' must NOT trip start_invoice.
    assert _heuristic_query("haan bhej do").intent == "none"
    assert _heuristic_query("yes send it").intent == "none"


def test_heuristic_greetings():
    for g in ["hi", "Hii", "hello", "Hey!", "namaste", "good morning", "kaise ho"]:
        assert _heuristic_query(g).intent == "greeting", g
    # Greeting + real intent should NOT be classified as greeting (heuristic checks
    # last-invoice patterns first).
    assert _heuristic_query("hi what is my last invoice amount").intent == "last_invoice_amount"
    # Project names that start with a greeting-like word should not match.
    assert _heuristic_query("Hi-Tech Pvt Ltd").intent == "none"
