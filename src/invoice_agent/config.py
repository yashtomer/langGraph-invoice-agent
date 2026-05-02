"""Typed config loaded from .env via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # WhatsApp (Meta Cloud API)
    meta_wa_phone_number_id: str = ""
    meta_wa_access_token: SecretStr = SecretStr("")
    meta_wa_verify_token: SecretStr = SecretStr("")
    meta_wa_app_secret: SecretStr = SecretStr("")
    user_whatsapp_number: str = ""
    approved_template_name: str = "invoice_monthly_prompt"

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_app_password: SecretStr = SecretStr("")
    accounts_email: str = ""
    cc_email: str = ""

    # Invoice
    company_name: str = "Aeologic Technologies Ltd."
    invoice_amount_inr: int = 0
    invoice_day_of_month: int = 25
    invoice_time_local: str = "10:30"
    timezone: str = "Asia/Kolkata"

    # Webhook
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8000
    webhook_shared_secret: SecretStr = SecretStr("")

    # DB
    sqlite_path: str = "./data/invoice_agent.db"

    # Paths (computed)
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def db_path(self) -> Path:
        p = self.project_root / self.sqlite_path.lstrip("./")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def out_dir(self) -> Path:
        p = self.project_root / "out"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def template_dir(self) -> Path:
        return self.project_root / "templates"

    def accounts_recipients(self) -> list[str]:
        return [e.strip() for e in self.accounts_email.split(",") if e.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
