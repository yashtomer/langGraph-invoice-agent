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


def test_invoice_history_extended_columns(tmp_settings):
    from invoice_agent import db
    db.init_db(tmp_settings)
    db.mark_sent(
        "2026-04",
        project_name="Acme",
        pdf_path="/tmp/x.pdf",
        amount_inr=200000,
        attendance_days=30,
        invoice_number="INV-2026-04-001",
        settings=tmp_settings,
    )
    with db.connect(tmp_settings) as conn:
        row = conn.execute(
            "SELECT amount_inr, attendance_days, invoice_number, sent_at "
            "FROM invoice_history WHERE month = ?", ("2026-04",)
        ).fetchone()
    assert row["amount_inr"] == 200000
    assert row["attendance_days"] == 30
    assert row["invoice_number"] == "INV-2026-04-001"
    assert row["sent_at"] is not None  # set when status flips to 'sent'


def test_chat_memory_table_exists(tmp_settings):
    from invoice_agent import db
    db.init_db(tmp_settings)
    with db.connect(tmp_settings) as conn:
        cols = conn.execute("PRAGMA table_info(chat_memory)").fetchall()
    col_names = {c["name"] for c in cols}
    assert col_names == {"user_phone", "turn_idx", "role", "content", "created_at"}
