"""Smoke test for the WeasyPrint renderer.

Skipped automatically if WeasyPrint's native deps (libcairo etc.) aren't
available in the test env — the test asserts shape and project-name presence,
not pixel rendering.
"""
from __future__ import annotations

import pytest


def test_pdf_renders_with_project_name(tmp_settings):
    pytest.importorskip("weasyprint")

    from invoice_agent.tools.pdf import render_invoice_pdf

    out = render_invoice_pdf(
        project_name="Birla Opus",
        month="2026-05",
        settings=tmp_settings,
    )
    assert out.exists()
    assert out.stat().st_size > 1000
    assert "birla_opus" in out.name.lower()
