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
    return _fake_llm_returning_intent(intent)


def _fake_llm_returning_intent(intent: str, **extra):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = QueryIntent(intent=intent, **extra)  # type: ignore[arg-type]
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
def llm_says_start_invoice_for(monkeypatch):
    """Factory fixture: parameterised by target_month token."""
    def _setup(token: str):
        monkeypatch.setattr(
            llm_mod,
            "make_chat",
            lambda **_: _fake_llm_returning_intent("start_invoice", target_month=token),
        )
    return _setup


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
    assert "1,50,000" in answer  # Indian-format from INVOICE_AMOUNT_INR=150000


def test_normalize_target_month():
    from invoice_agent.webhook.query import _normalize_target_month
    # Explicit YYYY-MM passes through.
    assert _normalize_target_month("2026-05", "Asia/Kolkata") == "2026-05"
    # Month name without year defaults to current year.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    assert _normalize_target_month("may", "Asia/Kolkata") == f"{today.year:04d}-05"
    assert _normalize_target_month("june", "Asia/Kolkata") == f"{today.year:04d}-06"
    # Month name + year.
    assert _normalize_target_month("may 2026", "Asia/Kolkata") == "2026-05"
    # Relative phrases.
    assert _normalize_target_month("current", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert _normalize_target_month("this month", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert _normalize_target_month("", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert _normalize_target_month(None, "Asia/Kolkata") == today.strftime("%Y-%m")


def test_start_invoice_triggers_runner_and_returns_empty(tmp_settings, monkeypatch):
    """start_invoice must call start_for_month(force=True) and signal the
    webhook that the reply was already sent (empty-string sentinel)."""
    from invoice_agent.db import init_db, mark_sent
    from invoice_agent.webhook import query as query_mod
    init_db(tmp_settings)
    # Pre-existing 'sent' row for the target month — force=True must override it.
    mark_sent("2026-05", project_name="OldProject", pdf_path="/tmp/x.pdf", settings=tmp_settings)

    monkeypatch.setattr(
        llm_mod,
        "make_chat",
        lambda **_: _fake_llm_returning_intent("start_invoice", target_month="may"),
    )
    fake_wa = MagicMock()
    monkeypatch.setattr(query_mod, "WhatsAppClient", lambda *a, **kw: fake_wa)
    fake_wa.__enter__ = lambda self: self
    fake_wa.__exit__ = lambda *a: None

    triggered = {}

    def _fake_start(month, *, force=False, settings=None):
        triggered["month"] = month
        triggered["force"] = force
        return {"month": month}

    import invoice_agent.runner as runner_mod
    monkeypatch.setattr(runner_mod, "start_for_month", _fake_start)

    answer = try_answer("send invoice for may", settings=tmp_settings)
    assert answer == ""  # empty-string sentinel: handler already sent the ack
    assert triggered["force"] is True
    # target_month "may" with current year (test env determined; just check shape)
    assert triggered["month"].endswith("-05")
    fake_wa.send_text.assert_called_once()
    body = fake_wa.send_text.call_args.kwargs["body"]
    assert "May" in body and "Starting" in body


def test_generic_question_uses_chat_reply(tmp_settings, llm_says_generic_question):
    init_db(tmp_settings)
    answer = try_answer("what can you do", settings=tmp_settings)
    assert answer == "chat:what can you do"


def test_generic_question_suppressed_during_active_flow(
    tmp_settings, llm_says_generic_question
):
    """Mid-flow (any month with status='started'), a misclassified message must
    NOT be eaten by chat — it has to fall through to resume_with_reply."""
    init_db(tmp_settings)
    mark_started("2026-06", settings=tmp_settings)  # any month, current or future
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
