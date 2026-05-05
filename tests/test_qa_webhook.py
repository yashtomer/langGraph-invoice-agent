"""End-to-end webhook → QA agent path."""
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
from invoice_agent.webhook.server import create_app


def _payload(from_phone: str, body: str) -> dict:
    return {
        "entry": [
            {"changes": [{"value": {"messages": [
                {"from": from_phone, "type": "text", "text": {"body": body}}
            ]}}]}
        ]
    }


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _fake_llm_intent(intent: str):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = QueryIntent(intent=intent)
    fake.with_structured_output.return_value = structured
    return fake


@respx.mock
def test_webhook_routes_generic_to_qa_agent(tmp_settings, monkeypatch):
    init_db(tmp_settings)
    monkeypatch.setattr(llm_mod, "make_chat",
                        lambda **_: _fake_llm_intent("generic_question"))

    from invoice_agent.qa import agent as qa_agent_mod
    captured = {}
    def fake_answer(text, user_phone, settings=None):
        captured["text"] = text
        captured["user_phone"] = user_phone
        return f"reply: {text}"
    monkeypatch.setattr(qa_agent_mod, "answer", fake_answer)

    sent = respx.post(
        f"https://graph.facebook.com/v21.0/{tmp_settings.meta_wa_phone_number_id}/messages"
    ).mock(return_value=Response(200, json={"messages": [{"id": "wamid.test"}]}))

    app = create_app(tmp_settings)
    client = TestClient(app)
    body = json.dumps(_payload("919999999999", "what's up?")).encode()
    sig = _sign(body, tmp_settings.meta_wa_app_secret.get_secret_value())
    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sig})

    assert r.status_code == 200
    assert captured["text"] == "what's up?"
    assert captured["user_phone"] == "919999999999"
    out = json.loads(sent.calls.last.request.content)
    assert out["text"]["body"] == "reply: what's up?"


@respx.mock
def test_webhook_active_flow_does_not_invoke_qa_agent(tmp_settings, monkeypatch):
    """Active-flow guard: when invoice_history.status='started' for any month,
    inbound text routes to resume_with_reply, not to the QA agent."""
    init_db(tmp_settings)
    mark_started("2026-06", settings=tmp_settings)

    monkeypatch.setattr(llm_mod, "make_chat",
                        lambda **_: _fake_llm_intent("generic_question"))

    from invoice_agent.qa import agent as qa_agent_mod
    sentinel = MagicMock(side_effect=AssertionError("QA agent must NOT run mid-flow"))
    monkeypatch.setattr(qa_agent_mod, "answer", sentinel)

    # Stub resume_with_reply so we don't actually drive the graph.
    from invoice_agent.webhook import server as server_mod
    monkeypatch.setattr(
        server_mod, "resume_with_reply",
        lambda month, text, settings=None: {"approval_status": "pending"},
    )

    respx.post(
        f"https://graph.facebook.com/v21.0/{tmp_settings.meta_wa_phone_number_id}/messages"
    ).mock(return_value=Response(200, json={"messages": [{"id": "wamid.x"}]}))

    app = create_app(tmp_settings)
    client = TestClient(app)
    body = json.dumps(_payload("919999999999", "anything")).encode()
    sig = _sign(body, tmp_settings.meta_wa_app_secret.get_secret_value())
    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sig})

    assert r.status_code == 200
    sentinel.assert_not_called()


@respx.mock
def test_webhook_bad_signature_still_403(tmp_settings):
    """Regression: signature check unchanged."""
    init_db(tmp_settings)
    app = create_app(tmp_settings)
    client = TestClient(app)
    body = json.dumps(_payload("919999999999", "hi")).encode()
    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=bad"})
    assert r.status_code == 403
