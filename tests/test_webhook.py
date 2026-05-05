"""Webhook signature verification + extraction."""
from __future__ import annotations

import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from invoice_agent.config import get_settings
from invoice_agent.webhook.server import _extract_inbound_text, _verify_signature, create_app


def test_verify_signature_good():
    body = b'{"hello": "world"}'
    secret = "test-app-secret"
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert _verify_signature(body, sig, secret) is True


def test_verify_signature_bad():
    assert _verify_signature(b"x", "sha256=deadbeef", "test-app-secret") is False
    assert _verify_signature(b"x", None, "test-app-secret") is False
    assert _verify_signature(b"x", "no-prefix", "test-app-secret") is False


def test_extract_text_from_payload():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "919999999999",
                                    "type": "text",
                                    "text": {"body": "haan bhej do"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    from_phone, text = _extract_inbound_text(payload)
    assert from_phone == "919999999999"
    assert text == "haan bhej do"


def test_extract_text_status_update_returns_none():
    payload = {"entry": [{"changes": [{"value": {"statuses": []}}]}]}
    assert _extract_inbound_text(payload) == (None, None)


def test_webhook_rejects_unsigned_post(tmp_settings):
    app = create_app(tmp_settings)
    client = TestClient(app)
    r = client.post("/webhook", json={"entry": []})
    assert r.status_code == 403


def test_webhook_verify_handshake(tmp_settings):
    app = create_app(tmp_settings)
    client = TestClient(app)
    s = get_settings()
    r = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": s.meta_wa_verify_token.get_secret_value(),
            "hub.challenge": "12345",
        },
    )
    assert r.status_code == 200
    assert r.json() == 12345


def test_trigger_requires_shared_secret(tmp_settings):
    app = create_app(tmp_settings)
    client = TestClient(app)
    r = client.post("/trigger")
    assert r.status_code == 401


def test_webhook_sends_fallback_when_no_active_flow(tmp_settings, monkeypatch):
    """Authorized user texts something the bot doesn't understand and no flow
    is in progress — fallback reply must go out, never silently dropped."""
    import json
    from unittest.mock import MagicMock

    import respx
    from httpx import Response

    from invoice_agent.db import init_db
    from invoice_agent.tools import llm as llm_mod
    from invoice_agent.tools.llm import QueryIntent

    init_db(tmp_settings)

    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = QueryIntent(intent="none")
    fake.with_structured_output.return_value = structured
    monkeypatch.setattr(llm_mod, "make_chat", lambda **_: fake)

    app = create_app(tmp_settings)
    client = TestClient(app)

    with respx.mock() as mock:
        sent = mock.post(
            f"https://graph.facebook.com/v21.0/{tmp_settings.meta_wa_phone_number_id}/messages"
        ).mock(return_value=Response(200, json={"messages": [{"id": "wamid.test"}]}))

        body = json.dumps({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [
                            {"from": "919999999999", "type": "text", "text": {"body": "asdfqwer"}}
                        ]
                    }
                }]
            }]
        }).encode()
        sig = "sha256=" + hmac.new(
            tmp_settings.meta_wa_app_secret.get_secret_value().encode(), body, hashlib.sha256
        ).hexdigest()
        r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sig})

        assert r.status_code == 200
        assert r.json() == {"ok": True, "fallback_sent": True}
        assert sent.called
        sent_body = json.loads(sent.calls.last.request.content)
        assert sent_body["type"] == "text"
        assert "didn't catch" in sent_body["text"]["body"].lower()
