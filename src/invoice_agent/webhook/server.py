"""FastAPI webhook for Meta WhatsApp Cloud API + manual /trigger endpoint.

Endpoints:
  GET  /webhook   -> Meta verification challenge (hub.challenge echo).
  POST /webhook   -> Inbound message receiver. Validates X-Hub-Signature-256
                     and resumes the LangGraph thread for the current month.
  POST /trigger   -> Manual kick-off. Authenticated by the
                     ``X-Shared-Secret`` header.
  GET  /healthz   -> Liveness probe.

All inbound messages route to the *current* month's thread. If no flow is
in progress for the month, the message is logged and ignored — we don't
auto-start a flow from a random user message.
"""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request, status

from ..config import Settings, get_settings
from ..db import get_status
from ..logging_setup import configure_logging, get_logger
from ..runner import resume_with_reply, start_for_month
from ..tools.whatsapp import WhatsAppClient
from .query import try_answer

log = get_logger(__name__)


def _current_month(tz: str) -> str:
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m")


def _verify_signature(raw_body: bytes, signature_header: Optional[str], app_secret: str) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    received = signature_header.split("=", 1)[1]
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received, expected)


def _extract_inbound_text(payload: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (from_phone, text) from a Meta webhook payload, or (None, None) if not a text message."""
    try:
        entry = payload["entry"][0]
        change = entry["changes"][0]
        value = change["value"]
        messages = value.get("messages")
        if not messages:
            return None, None
        msg = messages[0]
        if msg.get("type") != "text":
            return msg.get("from"), None
        return msg["from"], msg["text"]["body"]
    except (KeyError, IndexError, TypeError):
        return None, None


def build_router(settings: Optional[Settings] = None) -> APIRouter:
    s = settings or get_settings()
    router = APIRouter()

    @router.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    # ----- Meta webhook verification (one-time handshake) -----
    @router.get("/webhook")
    def verify(
        hub_mode: str = Query(default="", alias="hub.mode"),
        hub_verify_token: str = Query(default="", alias="hub.verify_token"),
        hub_challenge: str = Query(default="", alias="hub.challenge"),
    ):
        expected = s.meta_wa_verify_token.get_secret_value()
        if hub_mode == "subscribe" and hub_verify_token == expected:
            return int(hub_challenge) if hub_challenge.isdigit() else hub_challenge
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="bad verify token")

    # ----- Inbound message receiver -----
    @router.post("/webhook")
    async def receive(
        request: Request,
        x_hub_signature_256: Optional[str] = Header(default=None),
    ):
        raw = await request.body()
        if not _verify_signature(raw, x_hub_signature_256, s.meta_wa_app_secret.get_secret_value()):
            log.warning("webhook.bad_signature")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="bad signature")

        payload = await request.json()
        from_phone, text = _extract_inbound_text(payload)
        log.info("webhook.recv", from_phone=from_phone, has_text=bool(text))

        if not text:
            return {"ok": True, "ignored": "non-text or status update"}

        # Only accept replies from the configured user.
        if s.user_whatsapp_number and from_phone and from_phone != s.user_whatsapp_number:
            log.warning("webhook.unauthorized_sender", from_phone=from_phone)
            return {"ok": True, "ignored": "unauthorized sender"}

        # Free-form question intercept (runs before flow routing so a query
        # mid-flow doesn't get consumed as a project-name / approval reply).
        answer = try_answer(text, settings=s)
        if answer is not None:
            with WhatsAppClient(s) as wa:
                wa.send_text(to=from_phone, body=answer)
            log.info("webhook.query_answered", text_preview=text[:80])
            return {"ok": True, "answered": True}

        month = _current_month(s.timezone)
        if get_status(month, settings=s) is None:
            log.info("webhook.no_active_flow", month=month)
            return {"ok": True, "ignored": "no active flow for current month"}

        result = resume_with_reply(month, text, settings=s)
        return {"ok": True, "month": month, "approval_status": result.get("approval_status")}

    # ----- Manual trigger -----
    @router.post("/trigger")
    def trigger(
        month: Optional[str] = None,
        x_shared_secret: Optional[str] = Header(default=None),
    ):
        expected = s.webhook_shared_secret.get_secret_value()
        if not expected or x_shared_secret != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad shared secret")
        month = month or _current_month(s.timezone)
        result = start_for_month(month, settings=s)
        return {"ok": True, "month": month, "result": result}

    return router


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    configure_logging()
    app = FastAPI(title="Invoice Agent", version="0.1.0")
    app.include_router(build_router(settings))
    from .legal import build_legal_router
    app.include_router(build_legal_router())
    return app
