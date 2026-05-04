"""Email sender using Microsoft Graph API (application / client-credentials auth)."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, Optional

import httpx
import msal

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class MailerError(RuntimeError):
    pass


def _acquire_token(s: Settings) -> str:
    if not s.azure_client_id or not s.azure_tenant_id:
        raise MailerError("AZURE_CLIENT_ID / AZURE_TENANT_ID not configured")
    secret = s.azure_client_secret.get_secret_value()
    if not secret:
        raise MailerError("AZURE_CLIENT_SECRET not configured")

    app = msal.ConfidentialClientApplication(
        client_id=s.azure_client_id,
        client_credential=secret,
        authority=f"https://login.microsoftonline.com/{s.azure_tenant_id}",
    )
    result = app.acquire_token_for_client(scopes=GRAPH_SCOPE)
    if "access_token" not in result:
        raise MailerError(
            f"token acquisition failed: {result.get('error_description') or result}"
        )
    return result["access_token"]


def send_invoice_email(
    *,
    subject: str,
    body: str,
    pdf_path: str | Path,
    to: Iterable[str],
    cc: Iterable[str] = (),
    settings: Optional[Settings] = None,
) -> None:
    s = settings or get_settings()

    to_list = [a for a in to if a]
    cc_list = [a for a in cc if a]
    if not to_list:
        raise MailerError("no 'to' recipients configured")
    if not s.azure_mail_user:
        raise MailerError("AZURE_MAIL_USER not configured")

    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise MailerError(f"PDF not found: {pdf}")
    pdf_b64 = base64.b64encode(pdf.read_bytes()).decode("ascii")

    message: dict = {
        "subject": subject,
        "body": {"contentType": "Text", "content": body},
        "toRecipients": [{"emailAddress": {"address": a}} for a in to_list],
        "attachments": [
            {
                "@odata.type": "#microsoft.graph.fileAttachment",
                "name": pdf.name,
                "contentType": "application/pdf",
                "contentBytes": pdf_b64,
            }
        ],
    }
    if cc_list:
        message["ccRecipients"] = [{"emailAddress": {"address": a}} for a in cc_list]

    token = _acquire_token(s)
    url = f"{GRAPH_BASE}/users/{s.azure_mail_user}/sendMail"
    log.info(
        "mail.send",
        to=to_list,
        cc=cc_list,
        subject=subject,
        attach=pdf.name,
        via="graph",
    )
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"message": message, "saveToSentItems": "true"},
        )
    if r.status_code not in (200, 202):
        raise MailerError(f"Graph sendMail failed: HTTP {r.status_code} {r.text}")
