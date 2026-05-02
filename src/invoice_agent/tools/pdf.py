"""WeasyPrint PDF rendering."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip()).strip("_")
    return s.lower() or "project"


def invoice_number_for(month: str) -> str:
    """Stable, deterministic invoice number for a given month string ('YYYY-MM')."""
    yyyymm = month.replace("-", "")
    return f"INV-{yyyymm}-001"


def render_invoice_pdf(
    project_name: str,
    month: str,
    settings: Optional[Settings] = None,
) -> Path:
    """Render templates/invoice.html with project + month and write a PDF.

    Returns the absolute path to the written PDF.
    """
    s = settings or get_settings()
    # WeasyPrint is heavy — import lazily so unrelated unit tests don't pay for it.
    from weasyprint import HTML  # noqa: WPS433

    env = Environment(
        loader=FileSystemLoader(str(s.template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("invoice.html")

    context = {
        "company_name": s.company_name,
        "project_name": project_name,
        "month": month,
        "invoice_number": invoice_number_for(month),
        "amount_inr": s.invoice_amount_inr,
        "issue_date": date.today().isoformat(),
    }
    html_str = template.render(**context)

    out_path = s.out_dir / f"invoice_{month}_{_slug(project_name)}.pdf"
    HTML(string=html_str, base_url=str(s.template_dir)).write_pdf(str(out_path))
    log.info("pdf.rendered", path=str(out_path), project=project_name, month=month)
    return out_path
