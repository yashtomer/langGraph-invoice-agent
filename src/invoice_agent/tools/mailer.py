"""SMTP mailer over Gmail (stdlib only)."""
from __future__ import annotations

import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)


class MailerError(RuntimeError):
    pass


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
    if not s.smtp_user or not s.smtp_app_password.get_secret_value():
        raise MailerError("SMTP credentials not configured")

    msg = EmailMessage()
    msg["From"] = s.smtp_user
    msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.set_content(body)

    pdf = Path(pdf_path)
    with pdf.open("rb") as fh:
        msg.add_attachment(
            fh.read(),
            maintype="application",
            subtype="pdf",
            filename=pdf.name,
        )

    log.info("mail.send", to=to_list, cc=cc_list, subject=subject, attach=pdf.name)
    with smtplib.SMTP(s.smtp_host, s.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(s.smtp_user, s.smtp_app_password.get_secret_value())
        smtp.send_message(msg, to_addrs=to_list + cc_list)
