"""Meta WhatsApp Cloud API client — minimal surface, sync httpx.

Two send paths matter for this agent:
  1. ``send_text`` — used inside the 24h conversation window.
  2. ``send_template`` — used to *open* the conversation (the monthly nudge).
  3. ``send_document`` — used to ship the PDF preview.

The client is intentionally tiny; we avoid pulling a SDK to keep dependencies
small and the on-the-wire contract obvious.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_GRAPH_VERSION = "v21.0"


class WhatsAppError(RuntimeError):
    pass


class WhatsAppClient:
    def __init__(self, settings: Optional[Settings] = None, *, http: Optional[httpx.Client] = None):
        self.settings = settings or get_settings()
        self._http = http or httpx.Client(timeout=30.0)
        self._owns_http = http is None

    # ---- low-level ----

    @property
    def _base(self) -> str:
        return f"https://graph.facebook.com/{_GRAPH_VERSION}/{self.settings.meta_wa_phone_number_id}"

    @property
    def _headers(self) -> dict[str, str]:
        token = self.settings.meta_wa_access_token.get_secret_value()
        return {"Authorization": f"Bearer {token}"}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}{path}"
        r = self._http.post(url, headers={**self._headers, "Content-Type": "application/json"}, json=payload)
        if r.status_code >= 400:
            log.error("whatsapp.error", status=r.status_code, body=r.text, path=path)
            raise WhatsAppError(f"{r.status_code}: {r.text}")
        return r.json()

    # ---- public API ----

    def send_text(self, to: str, body: str) -> dict[str, Any]:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body, "preview_url": False},
        }
        log.info("whatsapp.send_text", to=to, len=len(body))
        return self._post_json("/messages", payload)

    def send_template(
        self,
        to: str,
        template_name: str,
        language: str = "en",
        body_params: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        components = []
        if body_params:
            components.append(
                {
                    "type": "body",
                    "parameters": [{"type": "text", "text": p} for p in body_params],
                }
            )
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language},
                "components": components,
            },
        }
        log.info("whatsapp.send_template", to=to, template=template_name, params=body_params)
        return self._post_json("/messages", payload)

    def send_document(self, to: str, path: str | Path, caption: str = "") -> dict[str, Any]:
        media_id = self._upload_media(path)
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "document",
            "document": {
                "id": media_id,
                "caption": caption,
                "filename": Path(path).name,
            },
        }
        log.info("whatsapp.send_document", to=to, media_id=media_id, file=str(path))
        return self._post_json("/messages", payload)

    # ---- media upload ----

    def _upload_media(self, path: str | Path) -> str:
        p = Path(path)
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "application/pdf"
        url = f"{self._base}/media"
        with p.open("rb") as fh:
            files = {"file": (p.name, fh, mime)}
            data = {"messaging_product": "whatsapp", "type": mime}
            r = self._http.post(url, headers=self._headers, files=files, data=data)
        if r.status_code >= 400:
            log.error("whatsapp.upload_error", status=r.status_code, body=r.text)
            raise WhatsAppError(f"upload failed: {r.status_code}: {r.text}")
        return r.json()["id"]

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "WhatsAppClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
