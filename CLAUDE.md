# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

All commands assume `uv` is installed. The Makefile is the canonical entry point.

```bash
make dev                            # uv sync --extra dev (installs test/lint deps)
make run                            # boots FastAPI + APScheduler on :8000 (uv run invoice-agent)
make test                           # full pytest suite with coverage
uv run pytest tests/test_graph_flow.py::test_happy_path   # single test
uv run pytest -k webhook            # subset by keyword
make lint                           # ruff check src tests
make demo                           # end-to-end LangGraph replay, mocked I/O (approve)
make demo-change                    # change-request loop scenario
make demo-reject                    # rejection scenario
make clean                          # nukes out/*.pdf, data/*.sqlite, caches
```

The `invoice-agent` script is wired in `pyproject.toml` to `invoice_agent.main:run`. Coverage and `pythonpath = ["src"]` are configured under `[tool.pytest.ini_options]`, so `uv run pytest` works from the repo root with no extra flags.

WeasyPrint requires native libs (`brew install pango cairo gdk-pixbuf libffi`). `tests/test_pdf_render.py` auto-skips when those aren't present, but the `generate_pdf` node will fail at runtime without them.

## Architecture

This is a human-in-the-loop LangGraph agent that runs once a month, drives a WhatsApp conversation, generates a PDF invoice, and emails it on user approval. The non-obvious parts:

### Three concerns, one SQLite file

`data/invoice_agent.db` is shared by **three independent owners** that must not collide:

1. `langgraph_checkpoints` — owned by `SqliteSaver` (LangGraph state per thread).
2. `apscheduler_jobs` — owned by APScheduler's `SQLAlchemyJobStore`.
3. `invoice_history` — owned by `db.py` for idempotency (`started` / `sent` / `cancelled` / `errored` per month).

When touching schema or migrations, remember the file is multi-owner — only mess with the `invoice_history` DDL in `db.py:_SCHEMA`.

### Interrupt-after, not interrupt-inside

The graph (`src/invoice_agent/graph.py`) compiles with `interrupt_after=["ask_project_name", "send_preview"]`. Each of those nodes' *last action* is to send a WhatsApp message — pausing AFTER them is what makes the nodes pure and lets the webhook resume cleanly. Don't add `interrupt()` calls inside nodes; the existing pattern is deliberate.

### One thread per month

`thread_id = f"invoice-{YYYY-MM}"` (`graph.thread_config`). Three call sites must agree on this key:

- `runner.start_for_month(month)` — kicked by APScheduler cron *and* `/trigger`.
- `runner.resume_with_reply(month, text)` — kicked by `/webhook` POST.
- `webhook/server.py` always uses the *current* month from `s.timezone`.

If you introduce a new entry point, route it through `runner.py` — that's the single place where graph compilation, checkpointer lifecycle, and `invoice_history` idempotency intersect. Open one `SqliteSaver` per call via `open_checkpointer()` and let the context manager close it.

### Idempotency contract

`start_for_month` is a no-op when `invoice_history.status == 'sent'` for the month. This is what makes APScheduler's `misfire_grace_time=86_400` safe: a missed firing on a sleeping Mac fires when the Mac wakes, but won't double-send if the month already completed.

### LLM is for parsing only — never for financial fields

`tools/llm.py` parses inbound replies into two Pydantic schemas (`ProjectReply`, `ApprovalDecision`) via `with_structured_output`. Both calls have a **two-stage retry** then a regex/keyword **heuristic fallback** so the graph never wedges on a bad LLM response. The heuristic for `parse_approval` defaults to `rejected` on ambiguous input — that's intentional fail-safe so we never email an invoice we shouldn't have.

Amount, recipients, invoice number, and template content are all deterministic from `Settings` + state. Don't add LLM calls anywhere in `nodes/email_accounts.py`, `tools/mailer.py`, or `tools/pdf.py`.

### Webhook security

`/webhook` POST validates `X-Hub-Signature-256` (HMAC-SHA256 of raw body with `META_WA_APP_SECRET`) — read `request.body()` once, before parsing JSON, or the HMAC won't match. The handler also rejects messages whose `from` ≠ `USER_WHATSAPP_NUMBER` and ignores inbound messages when no `invoice_history` row exists for the current month (no auto-start from random user texts). `/trigger` uses a separate `X-Shared-Secret` header.

The GET `/webhook` verification handshake must echo `hub.challenge` *without* HMAC checking — that's a one-time GET, not a signed POST.

### Config

`config.py` uses `pydantic-settings` with `@lru_cache` on `get_settings()`. Tests must call `config_mod.get_settings.cache_clear()` after monkeypatching env (see `tests/conftest.py:tmp_settings`). All secrets are `SecretStr` — call `.get_secret_value()` at the boundary.

`Settings.project_root` derives from `__file__` (two parents up from `config.py`), so the package layout `src/invoice_agent/config.py` is load-bearing — don't flatten it.

## Testing notes

- `tests/conftest.py` pre-seeds env vars at import time so `config.Settings` won't fail on missing required values. Add new required settings there too.
- The `tmp_settings` fixture isolates `db_path` and `out_dir` per test by overriding the `Settings` properties via `monkeypatch.setattr(type(s), ...)`. Use it for any test that touches the DB or PDF rendering.
- `respx` is the HTTP mocking library used for Meta API calls.
- `scripts/demo.py` stubs out `WhatsAppClient`, `render_invoice_pdf`, `send_invoice_email`, and `make_chat` by monkey-patching the node modules — useful as a reference for writing integration tests without external services.

## Things to leave alone

- `interrupt_after` list in `graph.compile_graph` — moving where the graph pauses breaks the webhook resume contract.
- Module-level `WhatsAppClient` references in `nodes/ask_project.py`, `nodes/send_preview.py`, `nodes/notify.py` — the demo and tests rely on patching them at module scope.
- `SqliteSaver.from_conn_string(...)` is used as a context manager intentionally; do not hold the saver open across requests.
