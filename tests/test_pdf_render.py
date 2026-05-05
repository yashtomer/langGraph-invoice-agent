"""Renderer tests — pure helpers always run; full PDF render needs LibreOffice."""
from __future__ import annotations

import shutil

import pytest

from invoice_agent.tools.pdf import (
    _invoice_date_label,
    _month_label,
    fmt_inr,
    invoice_number_for,
)


def test_invoice_number_fiscal_year():
    assert invoice_number_for("2026-04") == "YT/26-27/01"
    assert invoice_number_for("2026-05") == "YT/26-27/02"
    assert invoice_number_for("2026-12") == "YT/26-27/09"
    assert invoice_number_for("2027-01") == "YT/26-27/10"
    assert invoice_number_for("2027-03") == "YT/26-27/12"
    assert invoice_number_for("2027-04") == "YT/27-28/01"


def test_fmt_inr():
    assert fmt_inr(0) == "0"
    assert fmt_inr(50) == "50"
    assert fmt_inr(999) == "999"
    assert fmt_inr(1000) == "1,000"
    assert fmt_inr(99999) == "99,999"
    assert fmt_inr(145000) == "1,45,000"
    assert fmt_inr(245000) == "2,45,000"
    assert fmt_inr(10000000) == "1,00,00,000"  # 1 crore


def test_month_and_date_labels():
    assert _month_label("2026-05") == "May 2026"
    assert _month_label("2026-12") == "December 2026"
    assert _invoice_date_label("2026-04") == "30 April 2026"
    assert _invoice_date_label("2026-05") == "31 May 2026"
    assert _invoice_date_label("2024-02") == "29 February 2024"  # leap year
    assert _invoice_date_label("2026-02") == "28 February 2026"


def test_pdf_renders_full_pipeline(tmp_settings):
    """End-to-end: docxtpl + LibreOffice. Skipped when soffice isn't installed."""
    if not (shutil.which("soffice") or shutil.which("libreoffice")):
        pytest.skip("LibreOffice not installed")

    from invoice_agent.tools.pdf import render_invoice_pdf

    out = render_invoice_pdf(
        project_name="Birla Opus",
        month="2026-05",
        settings=tmp_settings,
    )
    assert out.exists()
    assert out.suffix == ".pdf"
    assert out.stat().st_size > 1000
    assert "birla_opus" in out.name.lower()
