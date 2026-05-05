"""Invoice PDF rendering — docxtpl + LibreOffice headless conversion.

Pipeline: ``templates/invoice.docx`` (Jinja-templated docx) → docxtpl render
→ filled .docx → ``soffice --headless --convert-to pdf`` → final PDF.

LibreOffice must be installed on the host (``soffice`` in $PATH). On Ubuntu:
``sudo apt install libreoffice-writer --no-install-recommends``.
"""
from __future__ import annotations

import calendar
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from docxtpl import DocxTemplate

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class PDFRenderError(RuntimeError):
    pass


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip()).strip("_")
    return s.lower() or "project"


def fmt_inr(n: int) -> str:
    """Indian-style integer formatting: 145000 -> '1,45,000'."""
    s = str(int(n))
    if len(s) <= 3:
        return s
    last3, rest = s[-3:], s[:-3]
    groups: list[str] = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.append(rest)
    return ",".join(reversed(groups)) + "," + last3


def invoice_number_for(month: str) -> str:
    """Fiscal-year invoice number, format ``YT/<FY-start-yy>-<FY-end-yy>/<seq:02d>``.

    Indian fiscal year runs April–March. April is sequence 01.
    e.g. 2026-04 -> YT/26-27/01, 2026-05 -> YT/26-27/02, 2027-03 -> YT/26-27/12.
    """
    year, mon = (int(p) for p in month.split("-"))
    if mon >= 4:
        fy_start, fy_end = year, year + 1
        seq = mon - 3
    else:
        fy_start, fy_end = year - 1, year
        seq = mon + 9
    return f"YT/{fy_start % 100:02d}-{fy_end % 100:02d}/{seq:02d}"


def _month_label(month: str) -> str:
    """'2026-05' -> 'May 2026'."""
    year, mon = (int(p) for p in month.split("-"))
    return f"{_MONTHS[mon - 1]} {year}"


def _invoice_date_label(month: str) -> str:
    """Last day of the month, e.g. '2026-05' -> '31 May 2026'."""
    year, mon = (int(p) for p in month.split("-"))
    last_day = calendar.monthrange(year, mon)[1]
    return f"{last_day} {_MONTHS[mon - 1]} {year}"


def _convert_docx_to_pdf(docx_path: Path, out_dir: Path) -> Path:
    """Convert a .docx to PDF via LibreOffice headless. Returns the PDF path."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise PDFRenderError(
            "LibreOffice not found. Install with: "
            "`sudo apt install libreoffice-writer --no-install-recommends`"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(docx_path)]
    log.info("pdf.libreoffice_invoke", cmd=cmd)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise PDFRenderError(f"LibreOffice timeout: {e}") from e
    if proc.returncode != 0:
        raise PDFRenderError(
            f"LibreOffice failed (rc={proc.returncode}): {proc.stderr or proc.stdout}"
        )
    pdf_path = out_dir / (docx_path.stem + ".pdf")
    if not pdf_path.is_file():
        raise PDFRenderError(f"LibreOffice produced no output at {pdf_path}")
    return pdf_path


def render_invoice_pdf(
    project_name: str,
    month: str,
    *,
    amount_inr: Optional[int] = None,
    attendance_days: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> Path:
    """Render templates/invoice.docx with state and write a PDF.

    ``amount_inr`` and ``attendance_days`` override the defaults derived from
    settings / calendar. Returns the absolute path to the written PDF.
    """
    s = settings or get_settings()
    template_path = s.template_dir / "invoice.docx"
    if not template_path.is_file():
        raise PDFRenderError(f"template not found: {template_path}")

    year, mon = (int(p) for p in month.split("-"))
    amount = amount_inr if amount_inr is not None else s.invoice_amount_inr
    attendance = attendance_days if attendance_days is not None else calendar.monthrange(year, mon)[1]
    context = {
        "project_name": project_name,
        "invoice_number": invoice_number_for(month),
        "invoice_month_label": _month_label(month),
        "invoice_date_label": _invoice_date_label(month),
        "months_billed": 1,
        "attendance_days": attendance,
        "amount_formatted": fmt_inr(amount),
    }

    docx_out = s.out_dir / f"invoice_{month}_{_slug(project_name)}.docx"
    tpl = DocxTemplate(str(template_path))
    tpl.render(context)
    tpl.save(str(docx_out))

    pdf_path = _convert_docx_to_pdf(docx_out, s.out_dir)
    log.info("pdf.rendered", path=str(pdf_path), project=project_name, month=month)
    return pdf_path
