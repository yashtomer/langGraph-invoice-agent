"""Shared pytest fixtures."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

# Ensure we're running with deterministic env *before* config is imported anywhere.
os.environ.setdefault("META_WA_PHONE_NUMBER_ID", "test-phone-id")
os.environ.setdefault("META_WA_ACCESS_TOKEN", "test-token")
os.environ.setdefault("META_WA_VERIFY_TOKEN", "test-verify")
os.environ.setdefault("META_WA_APP_SECRET", "test-app-secret")
os.environ.setdefault("USER_WHATSAPP_NUMBER", "919999999999")
os.environ.setdefault("APPROVED_TEMPLATE_NAME", "invoice_monthly_prompt")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b-instruct")
os.environ.setdefault("AZURE_CLIENT_ID", "test-client-id")
os.environ.setdefault("AZURE_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("AZURE_TENANT_ID", "test-tenant-id")
os.environ.setdefault("AZURE_MAIL_USER", "test@example.com")
os.environ.setdefault("ACCOUNTS_EMAIL", "accounts@example.com")
os.environ.setdefault("CC_EMAIL", "yash@example.com")
os.environ.setdefault("COMPANY_NAME", "Test Co.")
os.environ.setdefault("INVOICE_AMOUNT_INR", "150000")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("WEBHOOK_SHARED_SECRET", "shh")


@pytest.fixture
def tmp_settings(monkeypatch) -> Iterator:
    """Provide a Settings instance pointed at an isolated tmp dir."""
    from invoice_agent import config as config_mod

    tmpdir = Path(tempfile.mkdtemp(prefix="invoice_agent_test_"))
    db_path = tmpdir / "db.sqlite"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))

    config_mod.get_settings.cache_clear()
    s = config_mod.get_settings()
    # Override paths to be inside tmpdir
    monkeypatch.setattr(type(s), "out_dir", property(lambda self: tmpdir / "out"))
    monkeypatch.setattr(type(s), "db_path", property(lambda self: db_path))
    (tmpdir / "out").mkdir(parents=True, exist_ok=True)

    yield s

    config_mod.get_settings.cache_clear()
    shutil.rmtree(tmpdir, ignore_errors=True)
