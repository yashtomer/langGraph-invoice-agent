"""Free-form WhatsApp query handler — LLM-driven intent + deterministic answer."""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from invoice_agent.db import init_db, mark_sent, mark_started
from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import QueryIntent
from invoice_agent.webhook import query as query_mod
from invoice_agent.webhook.query import try_answer
from invoice_agent.webhook.server import create_app


def _payload(from_phone: str, body: str) -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": from_phone, "type": "text", "text": {"body": body}}
                            ]
                        }
                    }
                ]
            }
        ]
    }


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _fake_llm_returning(intent: str):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = QueryIntent(intent=intent)  # type: ignore[arg-type]
    fake.with_structured_output.return_value = structured
    return fake


@pytest.fixture
def llm_says_none(monkeypatch):
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: _fake_llm_returning("none"))


@pytest.fixture
def llm_says_last_invoice_amount(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "make_chat", lambda **_: _fake_llm_returning("last_invoice_amount")
    )


@pytest.fixture
def llm_says_greeting(monkeypatch):
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: _fake_llm_returning("greeting"))


@pytest.fixture
def llm_says_generic_question(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "make_chat", lambda **_: _fake_llm_returning("generic_question")
    )
    # Stub chat_reply so we don't actually need a live LLM for the chat fallback.
    monkeypatch.setattr(query_mod, "chat_reply", lambda text, settings=None: f"chat:{text}")


def test_empty_input_short_circuits_without_llm(tmp_settings, monkeypatch):
    """Empty/whitespace messages shouldn't even reach the LLM."""
    sentinel = MagicMock(side_effect=AssertionError("LLM should not be called"))
    monkeypatch.setattr(llm_mod, "make_chat", sentinel)
    assert try_answer("", settings=tmp_settings) is None
    assert try_answer("   ", settings=tmp_settings) is None


def test_intent_none_returns_none(tmp_settings, llm_says_none):
    init_db(tmp_settings)
    assert try_answer("Acme Q1 Project", settings=tmp_settings) is None
    assert try_answer("yes send it", settings=tmp_settings) is None


def test_no_history(tmp_settings, llm_says_last_invoice_amount):
    init_db(tmp_settings)
    answer = try_answer("what is my last invoice amount", settings=tmp_settings)
    assert answer == "You don't have any sent invoices yet."


def test_greeting_returns_help_text(tmp_settings, llm_says_greeting):
    init_db(tmp_settings)
    answer = try_answer("hi", settings=tmp_settings)
    assert answer is not None
    assert "invoice assistant" in answer.lower()
    assert "last invoice" in answer.lower()


def test_returns_amount_and_project(tmp_settings, llm_says_last_invoice_amount):
    init_db(tmp_settings)
    mark_sent("2026-04", project_name="Madabranding", pdf_path="/tmp/x.pdf", settings=tmp_settings)
    mark_sent("2026-03", project_name="Older", pdf_path="/tmp/y.pdf", settings=tmp_settings)
    answer = try_answer("how much was my last invoice?", settings=tmp_settings)
    assert "Madabranding" in answer
    assert "2026-04" in answer
    assert "150,000" in answer  # tmp_settings sets INVOICE_AMOUNT_INR=150000


def test_generic_question_uses_chat_reply(tmp_settings, llm_says_generic_question):
    init_db(tmp_settings)
    answer = try_answer("what can you do", settings=tmp_settings)
    assert answer == "chat:what can you do"


def test_generic_question_suppressed_during_active_flow(
    tmp_settings, llm_says_generic_question
):
    """Mid-flow, a misclassified message must NOT be eaten by chat — it has to
    fall through to resume_with_reply so the graph can consume it."""
    init_db(tmp_settings)
    # Mark current month as actively waiting for input (e.g. project name).
    from datetime import datetime
    from zoneinfo import ZoneInfo
    month = datetime.now(ZoneInfo(tmp_settings.timezone)).strftime("%Y-%m")
    mark_started(month, settings=tmp_settings)
    assert try_answer("what can you do", settings=tmp_settings) is None


@respx.mock
def test_webhook_answers_query_via_whatsapp(
    tmp_settings, llm_says_last_invoice_amount
):
    """Full path: signed POST → LLM intent → DB lookup → outbound send_text."""
    init_db(tmp_settings)
    mark_sent("2026-04", project_name="Madabranding", pdf_path="/tmp/x.pdf", settings=tmp_settings)

    app = create_app(tmp_settings)
    client = TestClient(app)

    sent_messages = respx.post(
        f"https://graph.facebook.com/v21.0/{tmp_settings.meta_wa_phone_number_id}/messages"
    ).mock(return_value=Response(200, json={"messages": [{"id": "wamid.test"}]}))

    body = json.dumps(_payload("919999999999", "kitna tha pichla invoice")).encode()
    sig = _sign(body, tmp_settings.meta_wa_app_secret.get_secret_value())

    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sig})

    assert r.status_code == 200
    assert r.json() == {"ok": True, "answered": True}
    assert sent_messages.called
    sent_body = json.loads(sent_messages.calls.last.request.content)
    assert sent_body["type"] == "text"
    assert "Madabranding" in sent_body["text"]["body"]
