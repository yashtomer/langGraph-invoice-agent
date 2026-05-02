"""invoice_history idempotency."""
from __future__ import annotations

from invoice_agent import db


def test_db_lifecycle(tmp_settings):
    db.init_db(tmp_settings)
    assert db.get_status("2026-05", settings=tmp_settings) is None
    assert db.already_sent("2026-05", settings=tmp_settings) is False

    db.mark_started("2026-05", settings=tmp_settings)
    assert db.get_status("2026-05", settings=tmp_settings) == "started"

    db.mark_sent("2026-05", project_name="Birla Opus", pdf_path="/tmp/x.pdf", settings=tmp_settings)
    assert db.already_sent("2026-05", settings=tmp_settings) is True

    # mark_started after sent is a no-op (sent stays sent — protects idempotency)
    db.mark_started("2026-05", settings=tmp_settings)
    assert db.already_sent("2026-05", settings=tmp_settings) is True
