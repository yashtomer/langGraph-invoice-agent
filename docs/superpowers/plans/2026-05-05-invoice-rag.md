# Invoice Q&A Tool-Calling Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph ReAct tool-calling agent that answers free-form WhatsApp questions about the user's current and previous month invoices, with web-search fallback for general-knowledge questions, conversational memory across turns, and a human-feeling tone.

**Architecture:** A new `src/invoice_agent/qa/` package exposes `qa_agent.answer(text, user_phone)`. The function loads recent chat turns from a new `chat_memory` SQLite table, runs `langgraph.prebuilt.create_react_agent` over `qwen2.5:7b-instruct` with three tools (`get_invoice`, `compare_invoices`, `web_search`), runs a post-generation amount-verification check, persists the turn, and returns the reply. The webhook routes the existing `generic_question` intent to this agent instead of the old `chat_reply`. A `ThreadPoolExecutor.result(timeout=...)` bounds wall-clock time so the FastAPI event loop never stalls.

**Tech Stack:**
- LangGraph `create_react_agent` (already installed via `langgraph-prebuilt 1.0.13`)
- `langchain-ollama` ChatOllama (existing `make_chat`)
- SQLite via `sqlite3` (existing `db.py`)
- Tavily Search via direct httpx POST (no new SDK; `httpx` already a dep)
- `concurrent.futures.ThreadPoolExecutor` for timeout
- `contextvars.ContextVar` for per-turn web-search budget

**Spec:** `docs/superpowers/specs/2026-05-05-invoice-rag-design.md` (commit `9b84ff6`).

**Implementation note (deviation from spec):** The spec proposes `get_invoice` reading `amount_inr` / `attendance_days` / `invoice_number` from the LangGraph checkpoint. That makes testing painful (need to populate checkpoint state) and is fragile (checkpoint shape can change). This plan extends `invoice_history` with three nullable columns (`amount_inr`, `attendance_days`, `invoice_number`) and persists them in `mark_sent`, populated from the snapshot in `runner.resume_with_reply`. The tool's public return shape is identical to the spec; the source changes from "checkpoint" to "denormalized in invoice_history". CLAUDE.md's "only mess with `invoice_history` DDL in `db.py:_SCHEMA`" rule is followed.

---

## File structure

**Create:**
- `src/invoice_agent/qa/__init__.py` — re-exports `answer`
- `src/invoice_agent/qa/util.py` — `_normalize_target_month` (extracted from webhook/query.py)
- `src/invoice_agent/qa/memory.py` — chat_memory CRUD
- `src/invoice_agent/qa/prompts.py` — `QA_SYSTEM` constant
- `src/invoice_agent/qa/tools.py` — `get_invoice`, `compare_invoices`, `web_search` `@tool` functions + budget ContextVar
- `src/invoice_agent/qa/agent.py` — `build_qa_agent`, `answer`, `_amounts_verified`, `_FALLBACK_STRING`
- `tests/test_qa_memory.py`
- `tests/test_qa_tools.py`
- `tests/test_qa_agent.py`
- `tests/test_qa_safety.py`
- `tests/test_qa_webhook.py`

**Modify:**
- `src/invoice_agent/db.py` — extend `invoice_history` schema (3 nullable cols) + add `chat_memory` table; extend `mark_sent` signature
- `src/invoice_agent/runner.py` — pass new fields through `mark_sent` call
- `src/invoice_agent/config.py` — add 5 settings (`tavily_api_key`, `qa_chat_memory_turns`, `qa_web_search_max_calls_per_turn`, `qa_web_search_daily_cap`, `qa_invoke_timeout_seconds`)
- `src/invoice_agent/webhook/query.py` — re-import `_normalize_target_month` from `qa.util`; thread `user_phone` through; swap `_generic_question` to call `qa_agent.answer`
- `src/invoice_agent/webhook/server.py` — pass `user_phone=from_phone` to `try_answer`; wrap call in `await asyncio.to_thread(...)`
- `tests/conftest.py` — pre-seed `TAVILY_API_KEY=test-tavily-key`
- `tests/test_webhook_query.py` — update/delete tests that depended on the old `chat_reply` path
- `CLAUDE.md` — list `chat_memory` as the fourth owner of the SQLite file

---

## Task 1: Add Q&A config settings

**Files:**
- Modify: `src/invoice_agent/config.py:11-53`
- Modify: `tests/conftest.py:12-29`
- Test: `tests/test_qa_config.py` (new)

- [ ] **Step 1: Pre-seed `TAVILY_API_KEY` in `tests/conftest.py`**

Add this line in the `os.environ.setdefault` block (alphabetically near the other test creds, around line 26):

```python
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_qa_config.py`:

```python
"""Settings additions for the QA agent."""
from __future__ import annotations

from invoice_agent.config import Settings


def test_qa_settings_defaults():
    s = Settings()
    assert s.qa_chat_memory_turns == 6
    assert s.qa_web_search_max_calls_per_turn == 5
    assert s.qa_web_search_daily_cap == 0
    assert s.qa_invoke_timeout_seconds == 30.0
    assert s.tavily_api_key.get_secret_value() == "test-tavily-key"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_config.py -v
```

Expected: FAIL — `Settings` has no attribute `qa_chat_memory_turns`.

- [ ] **Step 4: Add settings to `src/invoice_agent/config.py`**

Insert this block immediately after the `# Webhook` section (after line 50):

```python
    # QA agent
    tavily_api_key: SecretStr = SecretStr("")
    qa_chat_memory_turns: int = 6
    qa_web_search_max_calls_per_turn: int = 5
    qa_web_search_daily_cap: int = 0  # 0 = disabled
    qa_invoke_timeout_seconds: float = 30.0
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_qa_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the full suite to confirm no regressions**

```bash
uv run pytest -q
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/invoice_agent/config.py tests/conftest.py tests/test_qa_config.py
git commit -m "feat(qa): add Q&A agent config settings"
```

---

## Task 2: Extend invoice_history schema + add chat_memory table

**Files:**
- Modify: `src/invoice_agent/db.py:18-29` (`_SCHEMA`)
- Modify: `src/invoice_agent/db.py:106-127` (`mark_sent`)
- Test: `tests/test_db.py` (existing, will extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_db.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_db.py::test_invoice_history_extended_columns tests/test_db.py::test_chat_memory_table_exists -v
```

Expected: FAIL — columns don't exist; `mark_sent` rejects new kwargs.

- [ ] **Step 3: Replace `_SCHEMA` in `src/invoice_agent/db.py`**

Replace the existing `_SCHEMA` constant (lines 18-29) with:

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS invoice_history (
    month            TEXT NOT NULL,
    project_name     TEXT,
    pdf_path         TEXT,
    amount_inr       INTEGER,
    attendance_days  INTEGER,
    invoice_number   TEXT,
    sent_at          TEXT,
    status           TEXT NOT NULL,           -- 'started' | 'sent' | 'cancelled' | 'errored'
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (month)
);
CREATE INDEX IF NOT EXISTS idx_invoice_history_status ON invoice_history(status);

CREATE TABLE IF NOT EXISTS chat_memory (
    user_phone   TEXT NOT NULL,
    turn_idx     INTEGER NOT NULL,
    role         TEXT NOT NULL,            -- 'user' | 'assistant'
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (user_phone, turn_idx)
);
CREATE INDEX IF NOT EXISTS idx_chat_memory_phone_idx
    ON chat_memory(user_phone, turn_idx DESC);
"""
```

- [ ] **Step 4: Add a one-shot ALTER for existing DBs**

After the `_SCHEMA` constant, add:

```python
_MIGRATIONS = [
    "ALTER TABLE invoice_history ADD COLUMN amount_inr INTEGER",
    "ALTER TABLE invoice_history ADD COLUMN attendance_days INTEGER",
    "ALTER TABLE invoice_history ADD COLUMN invoice_number TEXT",
    "ALTER TABLE invoice_history ADD COLUMN sent_at TEXT",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Idempotent: SQLite raises 'duplicate column name' if already applied."""
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
    conn.commit()
```

Then update `init_db`:

```python
def init_db(settings: Optional[Settings] = None) -> None:
    s = settings or get_settings()
    with sqlite3.connect(str(s.db_path)) as conn:
        conn.executescript(_SCHEMA)
        _apply_migrations(conn)
        conn.commit()
```

- [ ] **Step 5: Replace `mark_sent` in `src/invoice_agent/db.py`**

Replace the existing `mark_sent` function (lines 106-127) with:

```python
def mark_sent(
    month: str,
    *,
    project_name: Optional[str] = None,
    pdf_path: Optional[str] = None,
    amount_inr: Optional[int] = None,
    attendance_days: Optional[int] = None,
    invoice_number: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> None:
    now = _now()
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO invoice_history (
                month, project_name, pdf_path,
                amount_inr, attendance_days, invoice_number,
                sent_at, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'sent', ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                status = 'sent',
                project_name = COALESCE(excluded.project_name, invoice_history.project_name),
                pdf_path = COALESCE(excluded.pdf_path, invoice_history.pdf_path),
                amount_inr = COALESCE(excluded.amount_inr, invoice_history.amount_inr),
                attendance_days = COALESCE(excluded.attendance_days, invoice_history.attendance_days),
                invoice_number = COALESCE(excluded.invoice_number, invoice_history.invoice_number),
                sent_at = excluded.sent_at,
                updated_at = excluded.updated_at
            """,
            (month, project_name, pdf_path, amount_inr, attendance_days,
             invoice_number, now, now, now),
        )
        conn.commit()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_db.py -v
```

Expected: PASS, including pre-existing `tests/test_db.py` tests.

- [ ] **Step 7: Run full suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/invoice_agent/db.py tests/test_db.py
git commit -m "feat(db): extend invoice_history with amount/attendance/invoice_number, add chat_memory table"
```

---

## Task 3: Plumb new fields through runner.resume_with_reply

**Files:**
- Modify: `src/invoice_agent/runner.py:108-114`
- Test: `tests/test_db.py` (existing; covered indirectly by integration tests later — no new test needed for plumbing itself)

- [ ] **Step 1: Update `mark_sent` call in `runner.py`**

Replace the `mark_sent(...)` call inside `resume_with_reply` (lines 108-114) with:

```python
    if snap.values.get("accounts_email_sent"):
        mark_sent(
            month,
            project_name=snap.values.get("project_name"),
            pdf_path=snap.values.get("pdf_path"),
            amount_inr=snap.values.get("invoice_amount_inr"),
            attendance_days=snap.values.get("attendance_days"),
            invoice_number=snap.values.get("invoice_number"),
            settings=s,
        )
```

- [ ] **Step 2: Run full suite**

```bash
uv run pytest -q
```

Expected: all tests pass (existing flow tests don't assert on the new columns, but they exercise the path).

- [ ] **Step 3: Commit**

```bash
git add src/invoice_agent/runner.py
git commit -m "feat(runner): persist amount/attendance/invoice_number on send"
```

---

## Task 4: Extract `_normalize_target_month` to `qa/util.py`

**Files:**
- Create: `src/invoice_agent/qa/__init__.py`
- Create: `src/invoice_agent/qa/util.py`
- Modify: `src/invoice_agent/webhook/query.py:31-81`
- Test: `tests/test_qa_util.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_util.py`:

```python
"""Shared month-token normaliser used by webhook/query and qa/tools."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from invoice_agent.qa.util import normalize_target_month


def test_yyyy_mm_passthrough():
    assert normalize_target_month("2026-05", "Asia/Kolkata") == "2026-05"


def test_relative_phrases():
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    assert normalize_target_month("current", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert normalize_target_month("this month", "Asia/Kolkata") == today.strftime("%Y-%m")


def test_previous_wraps_year():
    # Sanity: previous of 2026-01 is 2025-12. Test what we can without freezegun.
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    out = normalize_target_month("previous", "Asia/Kolkata")
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    assert out == f"{y:04d}-{m:02d}"


def test_month_name_with_year():
    assert normalize_target_month("may 2026", "Asia/Kolkata") == "2026-05"


def test_empty_falls_back_to_today():
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    assert normalize_target_month("", "Asia/Kolkata") == today.strftime("%Y-%m")
    assert normalize_target_month(None, "Asia/Kolkata") == today.strftime("%Y-%m")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_util.py -v
```

Expected: FAIL — `invoice_agent.qa` module not found.

- [ ] **Step 3: Create `src/invoice_agent/qa/__init__.py`**

```python
"""Q&A tool-calling agent package."""
from __future__ import annotations
```

- [ ] **Step 4: Create `src/invoice_agent/qa/util.py`**

```python
"""Shared utilities for the Q&A agent."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from ..logging_setup import get_logger

log = get_logger(__name__)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _today(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))


def normalize_target_month(token: Optional[str], tz: str) -> str:
    """Resolve a free-form month token to ``YYYY-MM``.

    Accepts: ``YYYY-MM``, ``"may"`` / ``"may 2026"``, ``"this month"``,
    ``"current"``, ``"previous month"`` / ``"last month"``, ``"next month"``.
    Falls back to today's month when the token is empty or unrecognised.
    """
    today = _today(tz)
    if not token:
        return today.strftime("%Y-%m")
    t = token.strip().lower()

    if len(t) == 7 and t[4] == "-" and t[:4].isdigit() and t[5:].isdigit():
        return t

    if t in ("current", "this month", "this", "now"):
        return today.strftime("%Y-%m")
    if t in ("previous month", "last month", "previous", "last"):
        y, m = today.year, today.month - 1
        if m == 0:
            y, m = y - 1, 12
        return f"{y:04d}-{m:02d}"
    if t in ("next month", "next"):
        y, m = today.year, today.month + 1
        if m == 13:
            y, m = y + 1, 1
        return f"{y:04d}-{m:02d}"

    parts = t.split()
    name = parts[0]
    if name in _MONTH_NAMES:
        mon = _MONTH_NAMES[name]
        year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else today.year
        return f"{year:04d}-{mon:02d}"

    log.warning("qa.normalize_target_month.unrecognised", token=token)
    return today.strftime("%Y-%m")
```

- [ ] **Step 5: Re-export from `webhook/query.py`**

In `src/invoice_agent/webhook/query.py`, replace lines 31-81 (the `_MONTH_NAMES`, `_today`, `_normalize_target_month` block) with:

```python
from ..qa.util import normalize_target_month as _normalize_target_month  # re-export
```

Remove now-unused imports: `from datetime import datetime`, `from zoneinfo import ZoneInfo`, `import calendar`. Keep the rest.

- [ ] **Step 6: Run tests to verify pass**

```bash
uv run pytest tests/test_qa_util.py tests/test_webhook_query.py -v
```

Expected: PASS — including `test_normalize_target_month` which still imports `_normalize_target_month` from `webhook.query`.

- [ ] **Step 7: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/invoice_agent/qa/__init__.py src/invoice_agent/qa/util.py \
        src/invoice_agent/webhook/query.py tests/test_qa_util.py
git commit -m "refactor(qa): extract normalize_target_month to qa.util"
```

---

## Task 5: chat_memory CRUD — `qa/memory.py`

**Files:**
- Create: `src/invoice_agent/qa/memory.py`
- Test: `tests/test_qa_memory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_memory.py`:

```python
"""chat_memory CRUD: append, load, trim, multi-user isolation."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from invoice_agent.db import init_db
from invoice_agent.qa.memory import append_turn, load_recent_turns, trim_old


def test_empty_load_returns_empty_list(tmp_settings):
    init_db(tmp_settings)
    assert load_recent_turns("91XXX", n=6, settings=tmp_settings) == []


def test_append_then_load_roundtrip(tmp_settings):
    init_db(tmp_settings)
    append_turn("91A", "hello", "hi there", settings=tmp_settings)
    msgs = load_recent_turns("91A", n=6, settings=tmp_settings)
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "hello"
    assert isinstance(msgs[1], AIMessage) and msgs[1].content == "hi there"


def test_load_returns_last_n_in_oldest_first_order(tmp_settings):
    init_db(tmp_settings)
    for i in range(8):
        append_turn("91A", f"u{i}", f"a{i}", settings=tmp_settings)
    msgs = load_recent_turns("91A", n=3, settings=tmp_settings)
    # 3 turns × 2 msgs = 6, oldest first → u5, a5, u6, a6, u7, a7
    assert [m.content for m in msgs] == ["u5", "a5", "u6", "a6", "u7", "a7"]


def test_multi_user_isolation(tmp_settings):
    init_db(tmp_settings)
    append_turn("91A", "a-user", "a-bot", settings=tmp_settings)
    append_turn("91B", "b-user", "b-bot", settings=tmp_settings)
    a = load_recent_turns("91A", n=6, settings=tmp_settings)
    b = load_recent_turns("91B", n=6, settings=tmp_settings)
    assert [m.content for m in a] == ["a-user", "a-bot"]
    assert [m.content for m in b] == ["b-user", "b-bot"]


def test_turn_idx_is_monotonic(tmp_settings):
    from invoice_agent.db import connect
    init_db(tmp_settings)
    for i in range(3):
        append_turn("91A", f"u{i}", f"a{i}", settings=tmp_settings)
    with connect(tmp_settings) as conn:
        idxs = [r["turn_idx"] for r in conn.execute(
            "SELECT turn_idx FROM chat_memory WHERE user_phone='91A' ORDER BY turn_idx"
        )]
    # 3 turns × 2 rows each = 6 monotonic indices starting at 0
    assert idxs == [0, 1, 2, 3, 4, 5]


def test_trim_keeps_newest_n(tmp_settings):
    from invoice_agent.db import connect
    init_db(tmp_settings)
    for i in range(25):
        append_turn("91A", f"u{i}", f"a{i}", settings=tmp_settings)
    trim_old("91A", keep=5, settings=tmp_settings)  # 5 turns = 10 rows
    with connect(tmp_settings) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM chat_memory WHERE user_phone='91A'"
        ).fetchone()["c"]
    assert n == 10
    # Verify the kept rows are the newest (turns 20..24)
    msgs = load_recent_turns("91A", n=5, settings=tmp_settings)
    assert [m.content for m in msgs] == [
        "u20", "a20", "u21", "a21", "u22", "a22", "u23", "a23", "u24", "a24"
    ]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_memory.py -v
```

Expected: FAIL — `invoice_agent.qa.memory` module not found.

- [ ] **Step 3: Implement `src/invoice_agent/qa/memory.py`**

```python
"""Chat memory persistence — fourth owner of data/invoice_agent.db.

Stores per-user-phone alternating Human/AI messages. Each turn is two rows
(role='user' then role='assistant') with a monotonic turn_idx scoped per user.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..config import Settings
from ..db import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_recent_turns(
    user_phone: str, *, n: int = 6, settings: Optional[Settings] = None
) -> list[BaseMessage]:
    """Return the last n turns (n*2 rows) as alternating HumanMessage/AIMessage,
    oldest first. n=6 → up to 12 messages."""
    limit = n * 2
    with connect(settings) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_memory "
            "WHERE user_phone = ? ORDER BY turn_idx DESC LIMIT ?",
            (user_phone, limit),
        ).fetchall()
    rows = list(reversed(rows))  # oldest first
    out: list[BaseMessage] = []
    for r in rows:
        if r["role"] == "user":
            out.append(HumanMessage(r["content"]))
        else:
            out.append(AIMessage(r["content"]))
    return out


def append_turn(
    user_phone: str,
    user_msg: str,
    assistant_msg: str,
    *,
    settings: Optional[Settings] = None,
) -> None:
    """Insert two rows (user + assistant) atomically. SQLite serialises writes
    so concurrent webhook calls won't collide on turn_idx."""
    now = _now()
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT COALESCE(MAX(turn_idx), -1) + 1 AS next FROM chat_memory "
            "WHERE user_phone = ?",
            (user_phone,),
        )
        next_idx = cur.fetchone()["next"]
        conn.executemany(
            "INSERT INTO chat_memory (user_phone, turn_idx, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (user_phone, next_idx, "user", user_msg, now),
                (user_phone, next_idx + 1, "assistant", assistant_msg, now),
            ],
        )
        conn.commit()


def trim_old(
    user_phone: str, *, keep: int = 20, settings: Optional[Settings] = None
) -> None:
    """Delete rows beyond the most recent `keep` turns (keep*2 rows)."""
    keep_rows = keep * 2
    with connect(settings) as conn:
        conn.execute(
            "DELETE FROM chat_memory WHERE user_phone = ? AND turn_idx NOT IN ("
            "  SELECT turn_idx FROM chat_memory WHERE user_phone = ? "
            "  ORDER BY turn_idx DESC LIMIT ?"
            ")",
            (user_phone, user_phone, keep_rows),
        )
        conn.commit()
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_memory.py -v
```

Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/qa/memory.py tests/test_qa_memory.py
git commit -m "feat(qa): add chat_memory CRUD"
```

---

## Task 6: `get_invoice` tool

**Files:**
- Create: `src/invoice_agent/qa/tools.py` (initial — get_invoice only)
- Test: `tests/test_qa_tools.py` (initial — get_invoice only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_tools.py`:

```python
"""qa/tools.py — invoice lookup, comparison, web search."""
from __future__ import annotations

from invoice_agent.db import init_db, mark_sent
from invoice_agent.qa.tools import get_invoice


def _invoke(tool, **kwargs):
    """Tools decorated with @tool expose .invoke({}) — call that."""
    return tool.invoke(kwargs)


def test_get_invoice_yyyy_mm_returns_full_shape(tmp_settings):
    init_db(tmp_settings)
    mark_sent(
        "2026-04",
        project_name="Madabranding",
        pdf_path="/tmp/x.pdf",
        amount_inr=200000,
        attendance_days=30,
        invoice_number="INV-2026-04-001",
        settings=tmp_settings,
    )
    out = _invoke(get_invoice, month="2026-04")
    assert out["month"] == "2026-04"
    assert out["project_name"] == "Madabranding"
    assert out["amount_inr"] == 200000
    assert out["attendance_days"] == 30
    assert out["invoice_number"] == "INV-2026-04-001"
    assert out["pdf_path"] == "/tmp/x.pdf"
    assert out["status"] == "sent"
    assert out["sent_at"] is not None


def test_get_invoice_relative_token_resolves(tmp_settings):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    init_db(tmp_settings)
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    cur_month = today.strftime("%Y-%m")
    mark_sent(cur_month, project_name="ThisMonth", amount_inr=150000, settings=tmp_settings)
    out = _invoke(get_invoice, month="current")
    assert out["month"] == cur_month
    assert out["project_name"] == "ThisMonth"


def test_get_invoice_missing_returns_not_found(tmp_settings):
    init_db(tmp_settings)
    out = _invoke(get_invoice, month="2030-01")
    assert out == {"month": "2030-01", "status": "not_found"}


def test_get_invoice_does_not_raise_on_unknown_token(tmp_settings):
    init_db(tmp_settings)
    # Unknown tokens fall back to today's month per normalize_target_month.
    out = _invoke(get_invoice, month="garbage")
    assert "status" in out  # either 'not_found' or a real status, never raises
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_tools.py -v
```

Expected: FAIL — `invoice_agent.qa.tools` module not found.

- [ ] **Step 3: Implement `src/invoice_agent/qa/tools.py` (initial)**

```python
"""LangChain @tool functions for the Q&A agent.

The docstrings on each @tool are load-bearing — the LLM uses them to choose
which tool to call. Edit with care.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ..config import Settings, get_settings
from ..db import connect
from ..logging_setup import get_logger
from .util import normalize_target_month

log = get_logger(__name__)


def _get_invoice_impl(month: str, *, settings: Optional[Settings] = None) -> dict:
    s = settings or get_settings()
    resolved = normalize_target_month(month, s.timezone)
    with connect(s) as conn:
        row = conn.execute(
            "SELECT month, project_name, pdf_path, amount_inr, attendance_days, "
            "       invoice_number, sent_at, status FROM invoice_history "
            "WHERE month = ?",
            (resolved,),
        ).fetchone()
    if row is None:
        return {"month": resolved, "status": "not_found"}
    return {
        "month": row["month"],
        "project_name": row["project_name"],
        "amount_inr": row["amount_inr"],
        "attendance_days": row["attendance_days"],
        "invoice_number": row["invoice_number"],
        "pdf_path": row["pdf_path"],
        "sent_at": row["sent_at"],
        "status": row["status"],
    }


@tool
def get_invoice(month: str) -> dict:
    """Look up THIS user's invoice for one month. Use this for any question
    about *their* invoices (amount, project, status, when sent, etc.).

    `month` accepts:
      - 'current' or 'this month'
      - 'previous' or 'last month'
      - 'YYYY-MM' (e.g. '2026-03')
      - a month name ('may', 'may 2026')

    Returns a dict with month, project_name, amount_inr, attendance_days,
    invoice_number, pdf_path, sent_at, status. If no record exists for that
    month, returns {month, status: 'not_found'} — say so plainly.
    """
    return _get_invoice_impl(month)
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_tools.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/qa/tools.py tests/test_qa_tools.py
git commit -m "feat(qa): add get_invoice tool"
```

---

## Task 7: `compare_invoices` tool

**Files:**
- Modify: `src/invoice_agent/qa/tools.py`
- Modify: `tests/test_qa_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qa_tools.py`:

```python
def test_compare_invoices_both_present(tmp_settings):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from invoice_agent.qa.tools import compare_invoices

    init_db(tmp_settings)
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    cur = today.strftime("%Y-%m")
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    prev = f"{y:04d}-{m:02d}"

    mark_sent(cur, project_name="A", amount_inr=200000, settings=tmp_settings)
    mark_sent(prev, project_name="A", amount_inr=150000, settings=tmp_settings)

    out = _invoke(compare_invoices)
    assert out["current"]["amount_inr"] == 200000
    assert out["previous"]["amount_inr"] == 150000
    assert out["amount_diff_inr"] == 50000
    assert out["same_project"] is True


def test_compare_invoices_one_missing(tmp_settings):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from invoice_agent.qa.tools import compare_invoices

    init_db(tmp_settings)
    today = datetime.now(ZoneInfo("Asia/Kolkata"))
    cur = today.strftime("%Y-%m")
    mark_sent(cur, project_name="A", amount_inr=200000, settings=tmp_settings)

    out = _invoke(compare_invoices)
    assert out["current"]["amount_inr"] == 200000
    assert out["previous"]["status"] == "not_found"
    assert out["amount_diff_inr"] is None
    assert out["same_project"] is False


def test_compare_invoices_neither_present(tmp_settings):
    from invoice_agent.qa.tools import compare_invoices
    init_db(tmp_settings)
    out = _invoke(compare_invoices)
    assert out["current"]["status"] == "not_found"
    assert out["previous"]["status"] == "not_found"
    assert out["amount_diff_inr"] is None
    assert out["same_project"] is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_tools.py::test_compare_invoices_both_present -v
```

Expected: FAIL — `compare_invoices` not defined.

- [ ] **Step 3: Add `compare_invoices` to `src/invoice_agent/qa/tools.py`**

Append below `get_invoice`:

```python
@tool
def compare_invoices() -> dict:
    """Compare THIS user's current month vs previous month invoices.
    Use for questions like 'is that more than last month?', 'higher',
    'difference', 'same as last month'. No arguments.

    Returns:
      {current: {...same shape as get_invoice...},
       previous: {...},
       amount_diff_inr: current.amount_inr - previous.amount_inr (None if either missing),
       same_project: bool (False if either missing)}
    """
    s = get_settings()
    cur = _get_invoice_impl("current", settings=s)
    prev = _get_invoice_impl("previous", settings=s)
    cur_amt = cur.get("amount_inr") if cur.get("status") != "not_found" else None
    prev_amt = prev.get("amount_inr") if prev.get("status") != "not_found" else None
    diff = (cur_amt - prev_amt) if (cur_amt is not None and prev_amt is not None) else None
    same_project = (
        cur.get("status") != "not_found"
        and prev.get("status") != "not_found"
        and cur.get("project_name") == prev.get("project_name")
        and cur.get("project_name") is not None
    )
    return {
        "current": cur,
        "previous": prev,
        "amount_diff_inr": diff,
        "same_project": same_project,
    }
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_tools.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/qa/tools.py tests/test_qa_tools.py
git commit -m "feat(qa): add compare_invoices tool"
```

---

## Task 8: `web_search` tool with per-turn budget ContextVar

**Files:**
- Modify: `src/invoice_agent/qa/tools.py`
- Modify: `tests/test_qa_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qa_tools.py`:

```python
import respx
from httpx import Response


@respx.mock
def test_web_search_parses_top_3_results(tmp_settings):
    from invoice_agent.qa.tools import (
        reset_web_search_budget,
        web_search,
    )
    reset_web_search_budget()

    respx.post("https://api.tavily.com/search").mock(
        return_value=Response(200, json={
            "results": [
                {"title": "T1", "url": "https://a", "content": "snippet 1"},
                {"title": "T2", "url": "https://b", "content": "snippet 2"},
                {"title": "T3", "url": "https://c", "content": "snippet 3"},
                {"title": "T4", "url": "https://d", "content": "snippet 4"},
            ]
        })
    )

    out = _invoke(web_search, query="GST IT services India")
    assert isinstance(out, list)
    assert len(out) == 3  # capped at top 3
    assert out[0] == {"title": "T1", "url": "https://a", "snippet": "snippet 1"}


@respx.mock
def test_web_search_network_failure_returns_error(tmp_settings):
    from invoice_agent.qa.tools import reset_web_search_budget, web_search
    reset_web_search_budget()
    respx.post("https://api.tavily.com/search").mock(return_value=Response(503))
    out = _invoke(web_search, query="anything")
    assert out == {"error": "search_unavailable"}


@respx.mock
def test_web_search_budget_blocks_after_5_calls(tmp_settings):
    from invoice_agent.qa.tools import reset_web_search_budget, web_search
    reset_web_search_budget()
    respx.post("https://api.tavily.com/search").mock(
        return_value=Response(200, json={"results": []})
    )
    for _ in range(5):
        _invoke(web_search, query="x")
    out = _invoke(web_search, query="x")  # 6th call
    assert out == {"error": "search_budget_exceeded"}


def test_web_search_no_api_key_returns_error(tmp_settings, monkeypatch):
    """When Tavily key is missing the tool short-circuits without a network call."""
    from invoice_agent.qa import tools as tools_mod
    from invoice_agent.qa.tools import reset_web_search_budget, web_search
    from pydantic import SecretStr
    reset_web_search_budget()

    # Patch the module-level get_settings used by web_search so we don't have
    # to mutate the pydantic Settings instance in place.
    class _NoKeySettings:
        tavily_api_key = SecretStr("")
        qa_web_search_max_calls_per_turn = 5
    monkeypatch.setattr(tools_mod, "get_settings", lambda: _NoKeySettings())

    out = _invoke(web_search, query="x")
    assert out == {"error": "search_unavailable"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_tools.py::test_web_search_parses_top_3_results -v
```

Expected: FAIL — `web_search` not defined.

- [ ] **Step 3: Add `web_search` and budget primitives to `src/invoice_agent/qa/tools.py`**

Append below `compare_invoices` (and add the imports at the top of the file):

At the top of `src/invoice_agent/qa/tools.py`, add to the imports:

```python
from contextvars import ContextVar
import httpx
```

Then below `compare_invoices`:

```python
_web_search_count: ContextVar[int] = ContextVar("_web_search_count", default=0)


def reset_web_search_budget() -> None:
    """Reset the per-turn web-search counter. Called by qa.agent.answer at the
    start of every turn before agent.invoke."""
    _web_search_count.set(0)


@tool
def web_search(query: str) -> list[dict] | dict:
    """Search the web for general knowledge. Use ONLY for questions that are
    NOT about THIS user's own invoices — e.g. tax rates, GST rules,
    regulatory questions, definitions, current events.

    NEVER call this to look up the user's invoice data — use get_invoice or
    compare_invoices for that. Returns up to 3 results: list of
    {title, url, snippet}. On failure returns {error: 'search_unavailable'}
    or {error: 'search_budget_exceeded'}; if you see those, tell the user
    you couldn't look it up — don't guess.
    """
    s = get_settings()

    # Per-turn budget
    cur = _web_search_count.get()
    if cur >= s.qa_web_search_max_calls_per_turn:
        log.warning("qa.tool_failed", tool="web_search", err="budget_exceeded")
        return {"error": "search_budget_exceeded"}
    _web_search_count.set(cur + 1)

    api_key = s.tavily_api_key.get_secret_value()
    if not api_key:
        log.warning("qa.tool_failed", tool="web_search", err="no_api_key")
        return {"error": "search_unavailable"}

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": 3,
                    "include_raw_content": False,
                },
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("qa.tool_failed", tool="web_search", err=str(e)[:120])
        return {"error": "search_unavailable"}

    results = data.get("results") or []
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
        for x in results[:3]
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_qa_tools.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/qa/tools.py tests/test_qa_tools.py
git commit -m "feat(qa): add web_search tool with per-turn budget"
```

---

## Task 9: Static `QA_SYSTEM` prompt

**Files:**
- Create: `src/invoice_agent/qa/prompts.py`
- Test: `tests/test_qa_prompts.py` (new — light sanity)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_prompts.py`:

```python
"""Sanity check that QA_SYSTEM contains the load-bearing rules."""
from invoice_agent.qa.prompts import QA_SYSTEM


def test_prompt_formats_with_company():
    rendered = QA_SYSTEM.format(company="Acme")
    assert "Acme" in rendered


def test_prompt_mentions_all_three_tools():
    assert "get_invoice" in QA_SYSTEM
    assert "compare_invoices" in QA_SYSTEM
    assert "web_search" in QA_SYSTEM


def test_prompt_has_financial_safety_rule():
    rendered = QA_SYSTEM.lower()
    # Some form of "quote exactly" + "never invent"
    assert "exactly" in rendered or "verbatim" in rendered
    assert "invent" in rendered or "make up" in rendered or "fabricate" in rendered


def test_prompt_says_dont_call_yourself_a_bot():
    assert "bot" in QA_SYSTEM.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_prompts.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create `src/invoice_agent/qa/prompts.py`**

```python
"""System prompt for the Q&A tool-calling agent.

Edits to QA_SYSTEM affect both the bot's tone (human-feel goal) and its
financial-safety guardrail. Keep both in mind."""
from __future__ import annotations

QA_SYSTEM = """You're {company}'s billing assistant on WhatsApp. You help one user — they own these invoices.

Talk like a person texting back: short, casual, contractions are fine. Match the user's language (English, Hindi, Hinglish — mirror what they sent). 1-3 short lines unless they explicitly ask for more.

Tools you have:
- get_invoice(month): for anything about THIS user's invoice for a given month. Use 'current', 'previous', or 'YYYY-MM'.
- compare_invoices(): when they're comparing this month vs last month ("more than", "higher", "diff", "same as").
- web_search(query): ONLY for general knowledge (tax rates, GST rules, definitions, news). NEVER use it to look up the user's own invoices.

Hard rules:
- Quote numbers EXACTLY as tools return them. Never invent or estimate amounts, dates, or invoice numbers.
- Never claim to be a bot, AI, or assistant unless the user directly asks.
- No bulleted help menus. No "I'd be happy to help" filler.
- If a tool returns {{"status": "not_found"}}, say plainly there's no record on file for that month.
- If web_search returns {{"error": "..."}}, tell the user you couldn't look it up right now — don't guess."""
```

Note: `{{` and `}}` are escaped braces because we'll `.format(company=...)` later — only `{company}` should expand.

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_prompts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/qa/prompts.py tests/test_qa_prompts.py
git commit -m "feat(qa): add system prompt"
```

---

## Task 10: `_amounts_verified` safety helper

**Files:**
- Create: `src/invoice_agent/qa/agent.py` (initial — only the helper)
- Test: `tests/test_qa_safety.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_safety.py`:

```python
"""Post-generation amount-verification: reply must not quote any INR amount
that didn't appear in get_invoice / compare_invoices tool output this turn."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from invoice_agent.qa.agent import _amounts_verified


def _msgs_with_tool_result(name: str, content_dict: dict):
    return [
        HumanMessage("hi"),
        AIMessage("calling tool"),
        ToolMessage(content=json.dumps(content_dict), tool_call_id="x", name=name),
    ]


def test_no_amounts_in_reply_passes():
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("hi there, all good", msgs) is True


def test_reply_amount_matches_tool_passes():
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("It was 200000 last month.", msgs) is True
    assert _amounts_verified("It was 2,00,000 last month.", msgs) is True  # comma form


def test_reply_amount_not_in_tool_fails():
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("It was 50000 last month.", msgs) is False


def test_lakh_phrase_in_reply_passes_when_tool_has_numeric():
    # Spec open question 1: accept either numeric or lakh form.
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("It was 2 lakh.", msgs) is True


def test_web_search_results_excluded_from_whitelist():
    msgs = [
        HumanMessage("hi"),
        AIMessage("searching"),
        ToolMessage(
            content=json.dumps([{"title": "T", "url": "u", "snippet": "rate is 18000"}]),
            tool_call_id="x",
            name="web_search",
        ),
    ]
    # 18000 is in web_search snippet but NOT in invoice tools — must fail.
    assert _amounts_verified("The rate is 18000.", msgs) is False


def test_no_tool_messages_passes_when_reply_has_no_amounts():
    msgs = [HumanMessage("hello"), AIMessage("hi!")]
    assert _amounts_verified("hi!", msgs) is True


def test_no_tool_messages_fails_when_reply_has_amount():
    # LLM hallucinated a number with no tool to back it up.
    msgs = [HumanMessage("hello"), AIMessage("you billed 50000")]
    assert _amounts_verified("you billed 50000", msgs) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_safety.py -v
```

Expected: FAIL — `invoice_agent.qa.agent` module not found.

- [ ] **Step 3: Create `src/invoice_agent/qa/agent.py` (helper only for now)**

```python
"""QA tool-calling agent — entrypoint and safety helpers."""
from __future__ import annotations

import re
from typing import Iterable

from langchain_core.messages import BaseMessage, ToolMessage

# Match large numbers with optional Indian/Western thousand separators.
# Examples matched: 200000, 2,00,000, 200,000, 1,45,000
_INR_NUMERIC_RE = re.compile(r"\b\d[\d,]{2,}\b")
# Indian-format magnitude phrases. Matched as their numeric equivalent below.
_LAKH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|crore|crores|cr)\b", re.IGNORECASE)

_INVOICE_TOOLS = {"get_invoice", "compare_invoices"}


def _normalize(token: str) -> str:
    """Strip commas so '2,00,000' and '200000' compare equal."""
    return token.replace(",", "")


def _expand_lakh(text: str) -> set[str]:
    """Expand 'X lakh' / 'X crore' phrases to their numeric equivalents."""
    out: set[str] = set()
    for m in _LAKH_RE.finditer(text):
        n = float(m.group(1))
        unit = m.group(2).lower()
        mult = 100_000 if unit.startswith("la") or unit.startswith("lac") else 10_000_000
        out.add(str(int(n * mult)))
    return out


def _extract_amounts(text: str) -> set[str]:
    """Extract candidate INR amounts from text. Returns normalized (no commas)
    digit strings."""
    if not text:
        return set()
    found = {_normalize(m) for m in _INR_NUMERIC_RE.findall(text)}
    found |= _expand_lakh(text)
    # Drop tiny numbers (sub-1000) — they're likely days, percentages, line refs.
    return {a for a in found if len(a) >= 4}


def _amounts_verified(reply: str, messages: Iterable[BaseMessage]) -> bool:
    """Return True iff every INR-shaped amount in `reply` also appears in the
    concatenated content of get_invoice / compare_invoices ToolMessages
    in `messages`. web_search ToolMessages are deliberately excluded — those
    snippets aren't authoritative number sources."""
    reply_amounts = _extract_amounts(reply)
    if not reply_amounts:
        return True
    whitelist_text = " ".join(
        m.content if isinstance(m.content, str) else str(m.content)
        for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "name", "") in _INVOICE_TOOLS
    )
    whitelist = _extract_amounts(whitelist_text)
    return reply_amounts.issubset(whitelist)
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_safety.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/qa/agent.py tests/test_qa_safety.py
git commit -m "feat(qa): add post-generation amount verification"
```

---

## Task 11: `build_qa_agent` + `answer` (happy path, no timeout yet)

**Files:**
- Modify: `src/invoice_agent/qa/agent.py`
- Test: `tests/test_qa_agent.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_agent.py`:

```python
"""Q&A agent — wires LLM (stubbed) + tools + memory + safety."""
from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from invoice_agent.db import init_db, mark_sent


class FakeChatLLM:
    """Mimics ChatOllama enough for create_react_agent to drive a turn.

    Scripted: each .invoke() returns the next AIMessage from `responses`.
    Exposes .bind_tools() (returns self) so create_react_agent can call it.
    """

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools, **_):
        return self

    def invoke(self, messages, config: Any = None, **_):
        self.calls.append(list(messages))
        if not self._responses:
            return AIMessage("done")
        return self._responses.pop(0)


def _ai_tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
    )


def test_answer_invokes_get_invoice_and_returns_reply(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    mark_sent(
        "2026-04",
        project_name="Madabranding",
        amount_inr=200000,
        attendance_days=30,
        invoice_number="INV-2026-04-001",
        settings=tmp_settings,
    )

    fake = FakeChatLLM([
        _ai_tool_call("get_invoice", {"month": "2026-04"}),
        AIMessage("April was 200000 for Madabranding."),
    ])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    reply = agent_mod.answer("what did i bill in april?", "91XXX", settings=tmp_settings)
    assert "200000" in reply
    assert "Madabranding" in reply


def test_answer_persists_turn_to_chat_memory(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.memory import load_recent_turns
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    fake = FakeChatLLM([AIMessage("hey")])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    agent_mod.answer("hi", "91XXX", settings=tmp_settings)
    msgs = load_recent_turns("91XXX", n=6, settings=tmp_settings)
    assert [m.content for m in msgs] == ["hi", "hey"]


def test_answer_passes_history_into_agent(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.memory import append_turn
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    append_turn("91XXX", "earlier-q", "earlier-a", settings=tmp_settings)
    fake = FakeChatLLM([AIMessage("got it")])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    agent_mod.answer("follow-up", "91XXX", settings=tmp_settings)
    # First call's input should include both prior turns + the new HumanMessage.
    first_call = fake.calls[0]
    contents = [m.content for m in first_call if isinstance(m, (HumanMessage, AIMessage))]
    assert "earlier-q" in contents
    assert "earlier-a" in contents
    assert "follow-up" in contents


def test_answer_returns_fallback_when_agent_raises(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)

    class BoomLLM:
        def bind_tools(self, *a, **kw):
            return self
        def invoke(self, *a, **kw):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: BoomLLM())
    reply = agent_mod.answer("hi", "91XXX", settings=tmp_settings)
    assert reply == agent_mod._FALLBACK_STRING


def test_answer_swaps_to_fallback_when_amount_unverified(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    # Stub: LLM emits a number with no tool call to back it up.
    fake = FakeChatLLM([AIMessage("Your last invoice was 999999.")])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    reply = agent_mod.answer("how much?", "91XXX", settings=tmp_settings)
    assert reply == agent_mod._FALLBACK_STRING
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_agent.py -v
```

Expected: FAIL — `agent.answer`, `agent.make_chat`, `agent._FALLBACK_STRING` not defined.

- [ ] **Step 3: Extend `src/invoice_agent/qa/agent.py`**

Append the following imports at the top of `qa/agent.py` (immediately after the existing `import re` block):

```python
import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..tools.llm import make_chat
from .memory import append_turn, load_recent_turns, trim_old
from .prompts import QA_SYSTEM
from .tools import compare_invoices, get_invoice, reset_web_search_budget, web_search

log = get_logger(__name__)

_FALLBACK_STRING = (
    "Sorry, something went wrong on my end. You can ask me "
    "'what was my last invoice amount?'"
)
_MAX_HISTORY_TRIM = 20  # rows kept per user after trim
```

Then append at the bottom of the file:

```python
def build_qa_agent(settings: Settings, *, llm=None):
    """Build a ReAct agent over the three QA tools.

    `llm` is injectable for tests; production passes None and gets ChatOllama.
    """
    chat = llm if llm is not None else make_chat(settings, temperature=0.4)
    tools = [get_invoice, compare_invoices, web_search]
    return create_react_agent(
        chat,
        tools=tools,
        prompt=QA_SYSTEM.format(company=settings.company_name),
    )


def answer(text: str, user_phone: str, *, settings: Optional[Settings] = None) -> str:
    """Synchronous entry point. Caller (webhook) should wrap in
    asyncio.to_thread() to keep the event loop responsive."""
    s = settings or get_settings()
    reset_web_search_budget()

    history = load_recent_turns(user_phone, n=s.qa_chat_memory_turns, settings=s)
    try:
        agent = build_qa_agent(s)
        result = agent.invoke(
            {"messages": history + [HumanMessage(text)]},
            config={"recursion_limit": 8},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("qa.invoke_failed", err=str(e), user_phone=user_phone)
        return _FALLBACK_STRING

    msgs = result.get("messages", [])
    reply_msg = msgs[-1] if msgs else None
    reply = reply_msg.content if reply_msg is not None else ""
    if not isinstance(reply, str):
        reply = str(reply)

    if not _amounts_verified(reply, msgs):
        log.warning("qa.amount_unverified", reply_preview=reply[:80])
        return _FALLBACK_STRING

    append_turn(user_phone, text, reply, settings=s)
    trim_old(user_phone, keep=_MAX_HISTORY_TRIM, settings=s)
    log.info("qa.turn_complete", user_phone=user_phone)
    return reply
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_agent.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/invoice_agent/qa/agent.py tests/test_qa_agent.py
git commit -m "feat(qa): add ReAct agent with chat memory + amount verification"
```

---

## Task 12: Bound `agent.invoke` with ThreadPoolExecutor timeout

**Files:**
- Modify: `src/invoice_agent/qa/agent.py:answer`
- Modify: `tests/test_qa_safety.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qa_safety.py`:

```python
import time

import pytest
from langchain_core.messages import AIMessage, BaseMessage


class SlowLLM:
    def bind_tools(self, *a, **kw):
        return self
    def invoke(self, messages: list[BaseMessage], config=None, **_):
        time.sleep(2.0)
        return AIMessage("eventually")


def test_answer_times_out_with_slow_llm(tmp_settings, monkeypatch):
    from invoice_agent.db import init_db
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: SlowLLM())
    # Force a tight timeout so the test finishes fast.
    monkeypatch.setattr(
        type(tmp_settings),
        "qa_invoke_timeout_seconds",
        property(lambda self: 0.1),
    )

    reply = agent_mod.answer("hi", "91XXX", settings=tmp_settings)
    assert reply == agent_mod._FALLBACK_STRING
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_safety.py::test_answer_times_out_with_slow_llm -v
```

Expected: FAIL — current `answer()` has no timeout (test waits 2s then asserts fallback, but LLM returns "eventually" which passes verification).

- [ ] **Step 3: Add timeout to `answer()` in `src/invoice_agent/qa/agent.py`**

At the top of `qa/agent.py`, add to the imports:

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
```

Replace the `try / except` block in `answer()` (the `agent.invoke` call) with:

```python
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        agent = build_qa_agent(s)
        future = ex.submit(
            agent.invoke,
            {"messages": history + [HumanMessage(text)]},
            config={"recursion_limit": 8},
        )
        result = future.result(timeout=s.qa_invoke_timeout_seconds)
    except FuturesTimeoutError:
        log.warning("qa.timeout", user_phone=user_phone)
        return _FALLBACK_STRING
    except Exception as e:  # noqa: BLE001
        log.warning("qa.invoke_failed", err=str(e), user_phone=user_phone)
        return _FALLBACK_STRING
    finally:
        # wait=False so a stuck LLM thread doesn't block our return.
        # Python can't actually kill the thread; it'll complete on its own
        # and the result is discarded harmlessly.
        ex.shutdown(wait=False)
```

Note: `ThreadPoolExecutor.submit(fn, *args, **kwargs)` forwards both, so passing `config={...}` as a kwarg works. `wait=False` on shutdown is required so a stuck LLM thread (sleep 2s in the test, infinite tool-loop in prod) doesn't block our return.

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_qa_safety.py -v
```

Expected: PASS, including the new timeout test.

- [ ] **Step 5: Run full QA test suite**

```bash
uv run pytest tests/test_qa_agent.py tests/test_qa_safety.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/invoice_agent/qa/agent.py tests/test_qa_safety.py
git commit -m "feat(qa): bound agent.invoke with ThreadPoolExecutor timeout"
```

---

## Task 13: Re-export `answer` from `qa/__init__.py`

**Files:**
- Modify: `src/invoice_agent/qa/__init__.py`

- [ ] **Step 1: Update `qa/__init__.py`**

Replace the file contents with:

```python
"""Q&A tool-calling agent package."""
from __future__ import annotations

from .agent import answer

__all__ = ["answer"]
```

- [ ] **Step 2: Verify the import works**

```bash
uv run python -c "from invoice_agent.qa import answer; print(answer.__module__)"
```

Expected: prints `invoice_agent.qa.agent`.

- [ ] **Step 3: Commit**

```bash
git add src/invoice_agent/qa/__init__.py
git commit -m "feat(qa): re-export answer from package root"
```

---

## Task 14: Wire QA agent into `webhook/query.py`

**Files:**
- Modify: `src/invoice_agent/webhook/query.py`
- Modify: `tests/test_webhook_query.py` (existing — update fixture and one test)

- [ ] **Step 1: Update `try_answer` and `_generic_question` signatures in `webhook/query.py`**

In `src/invoice_agent/webhook/query.py`, change `IntentHandler` typedef and the `_generic_question` function:

Replace:

```python
IntentHandler = Callable[[str, QueryIntent, Settings], Optional[str]]
```

with:

```python
IntentHandler = Callable[..., Optional[str]]
```

Replace `_generic_question` (currently calls `chat_reply`):

```python
def _generic_question(text: str, _intent: QueryIntent, s: Settings,
                     *, user_phone: str) -> str:
    from ..qa import answer as qa_answer  # local import: avoid cycle
    return qa_answer(text, user_phone, settings=s)
```

Update the `_INTENT_HANDLERS` dispatch and `try_answer`:

```python
def try_answer(
    text: str,
    *,
    user_phone: str = "",
    settings: Optional[Settings] = None,
) -> Optional[str]:
    """Return a reply string, an empty string (matched but reply already sent),
    or ``None`` (no intent matched — webhook should fall through to flow router)."""
    if not text or not text.strip():
        return None
    s = settings or get_settings()
    intent = parse_query_intent(text)
    log.info(
        "query.intent_classified",
        intent=intent.intent,
        target_month=intent.target_month,
        text_preview=text[:80],
    )

    if intent.intent == "generic_question":
        active = get_active_month(settings=s)
        if active is not None:
            log.info("query.suppress_chat_during_active_flow", active_month=active)
            return None

    handler = _INTENT_HANDLERS.get(intent.intent)
    if handler is None:
        return None
    if intent.intent == "generic_question":
        return handler(text, intent, s, user_phone=user_phone)
    return handler(text, intent, s)
```

(`generic_question` is the only handler that needs `user_phone`. The deterministic fast-paths don't.)

- [ ] **Step 2: Update one existing test in `tests/test_webhook_query.py`**

The fixture `llm_says_generic_question` (line 86-91) currently stubs `chat_reply`. Replace with stubbing `qa.answer`. Find:

```python
@pytest.fixture
def llm_says_generic_question(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "make_chat", lambda **_: _fake_llm_returning("generic_question")
    )
    # Stub chat_reply so we don't actually need a live LLM for the chat fallback.
    monkeypatch.setattr(query_mod, "chat_reply", lambda text, settings=None: f"chat:{text}")
```

Replace with:

```python
@pytest.fixture
def llm_says_generic_question(monkeypatch):
    monkeypatch.setattr(
        llm_mod, "make_chat", lambda **_: _fake_llm_returning("generic_question")
    )
    # Stub the QA agent's answer() so we don't run a real ReAct loop.
    from invoice_agent.qa import agent as qa_agent_mod
    monkeypatch.setattr(qa_agent_mod, "answer", lambda text, user_phone, settings=None: f"qa:{text}")
```

Find the test `test_generic_question_uses_chat_reply` (line 190) and replace its body:

```python
def test_generic_question_uses_qa_agent(tmp_settings, llm_says_generic_question):
    init_db(tmp_settings)
    answer = try_answer("what can you do", user_phone="91XXX", settings=tmp_settings)
    assert answer == "qa:what can you do"
```

(Rename the test as shown — it tests the new path.)

- [ ] **Step 3: Run tests to verify pass**

```bash
uv run pytest tests/test_webhook_query.py -v
```

Expected: all PASS, including the renamed test.

- [ ] **Step 4: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/invoice_agent/webhook/query.py tests/test_webhook_query.py
git commit -m "feat(qa): route generic_question intent to QA agent"
```

---

## Task 15: Wire `user_phone` + `asyncio.to_thread` into `webhook/server.py`

**Files:**
- Modify: `src/invoice_agent/webhook/server.py:107-118`
- Test: `tests/test_qa_webhook.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_webhook.py`:

```python
"""End-to-end webhook → QA agent path."""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from invoice_agent.db import init_db, mark_sent, mark_started
from invoice_agent.tools import llm as llm_mod
from invoice_agent.tools.llm import QueryIntent
from invoice_agent.webhook.server import create_app


def _payload(from_phone: str, body: str) -> dict:
    return {
        "entry": [
            {"changes": [{"value": {"messages": [
                {"from": from_phone, "type": "text", "text": {"body": body}}
            ]}}]}
        ]
    }


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _fake_llm_intent(intent: str):
    fake = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = QueryIntent(intent=intent)
    fake.with_structured_output.return_value = structured
    return fake


@respx.mock
def test_webhook_routes_generic_to_qa_agent(tmp_settings, monkeypatch):
    init_db(tmp_settings)
    monkeypatch.setattr(llm_mod, "make_chat",
                        lambda **_: _fake_llm_intent("generic_question"))

    from invoice_agent.qa import agent as qa_agent_mod
    captured = {}
    def fake_answer(text, user_phone, settings=None):
        captured["text"] = text
        captured["user_phone"] = user_phone
        return f"reply: {text}"
    monkeypatch.setattr(qa_agent_mod, "answer", fake_answer)

    sent = respx.post(
        f"https://graph.facebook.com/v21.0/{tmp_settings.meta_wa_phone_number_id}/messages"
    ).mock(return_value=Response(200, json={"messages": [{"id": "wamid.test"}]}))

    app = create_app(tmp_settings)
    client = TestClient(app)
    body = json.dumps(_payload("919999999999", "what's up?")).encode()
    sig = _sign(body, tmp_settings.meta_wa_app_secret.get_secret_value())
    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sig})

    assert r.status_code == 200
    assert captured["text"] == "what's up?"
    assert captured["user_phone"] == "919999999999"
    out = json.loads(sent.calls.last.request.content)
    assert out["text"]["body"] == "reply: what's up?"


@respx.mock
def test_webhook_active_flow_does_not_invoke_qa_agent(tmp_settings, monkeypatch):
    """Active-flow guard: when invoice_history.status='started' for any month,
    inbound text routes to resume_with_reply, not to the QA agent."""
    init_db(tmp_settings)
    mark_started("2026-06", settings=tmp_settings)

    monkeypatch.setattr(llm_mod, "make_chat",
                        lambda **_: _fake_llm_intent("generic_question"))

    from invoice_agent.qa import agent as qa_agent_mod
    sentinel = MagicMock(side_effect=AssertionError("QA agent must NOT run mid-flow"))
    monkeypatch.setattr(qa_agent_mod, "answer", sentinel)

    # Stub resume_with_reply so we don't actually drive the graph.
    from invoice_agent.webhook import server as server_mod
    monkeypatch.setattr(
        server_mod, "resume_with_reply",
        lambda month, text, settings=None: {"approval_status": "pending"},
    )

    respx.post(
        f"https://graph.facebook.com/v21.0/{tmp_settings.meta_wa_phone_number_id}/messages"
    ).mock(return_value=Response(200, json={"messages": [{"id": "wamid.x"}]}))

    app = create_app(tmp_settings)
    client = TestClient(app)
    body = json.dumps(_payload("919999999999", "anything")).encode()
    sig = _sign(body, tmp_settings.meta_wa_app_secret.get_secret_value())
    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sig})

    assert r.status_code == 200
    sentinel.assert_not_called()


@respx.mock
def test_webhook_bad_signature_still_403(tmp_settings):
    """Regression: signature check unchanged."""
    init_db(tmp_settings)
    app = create_app(tmp_settings)
    client = TestClient(app)
    body = json.dumps(_payload("919999999999", "hi")).encode()
    r = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=bad"})
    assert r.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_qa_webhook.py -v
```

Expected: FAIL — `try_answer` is called without `user_phone`, so the QA agent isn't invoked.

- [ ] **Step 3: Update `src/invoice_agent/webhook/server.py`**

In the imports section near the top, add:

```python
import asyncio
```

Replace the `try_answer(...)` call inside `receive()` (around line 112). Find:

```python
        # Free-form question intercept (runs before flow routing so a query
        # mid-flow doesn't get consumed as a project-name / approval reply).
        # Empty-string return means "intent matched, the handler already sent
        # any user-facing message itself" (e.g. start_invoice triggers the
        # graph which sends the template — no extra reply needed).
        answer = try_answer(text, settings=s)
```

Replace with:

```python
        # Free-form question intercept (runs before flow routing so a query
        # mid-flow doesn't get consumed as a project-name / approval reply).
        # Empty-string return means "intent matched, the handler already sent
        # any user-facing message itself" (e.g. start_invoice triggers the
        # graph which sends the template — no extra reply needed).
        # Run in a worker thread because the QA agent path can take seconds —
        # we don't want to stall FastAPI's event loop.
        answer = await asyncio.to_thread(
            try_answer, text, user_phone=from_phone or "", settings=s
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_qa_webhook.py tests/test_webhook_query.py tests/test_webhook.py -v
```

Expected: all PASS, including the existing webhook tests.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add src/invoice_agent/webhook/server.py tests/test_qa_webhook.py
git commit -m "feat(qa): route webhook to QA agent via asyncio.to_thread"
```

---

## Task 16: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` — three concerns / four concerns sentence

- [ ] **Step 1: Edit `CLAUDE.md`**

In `CLAUDE.md`, find the section header:

```
### Three concerns, one SQLite file

`data/invoice_agent.db` is shared by **three independent owners** that must not collide:

1. `langgraph_checkpoints` — owned by `SqliteSaver` (LangGraph state per thread).
2. `apscheduler_jobs` — owned by APScheduler's `SQLAlchemyJobStore`.
3. `invoice_history` — owned by `db.py` for idempotency (`started` / `sent` / `cancelled` / `errored` per month).

When touching schema or migrations, remember the file is multi-owner — only mess with the `invoice_history` DDL in `db.py:_SCHEMA`.
```

Replace with:

```
### Four concerns, one SQLite file

`data/invoice_agent.db` is shared by **four independent owners** that must not collide:

1. `langgraph_checkpoints` — owned by `SqliteSaver` (LangGraph state per thread).
2. `apscheduler_jobs` — owned by APScheduler's `SQLAlchemyJobStore`.
3. `invoice_history` — owned by `db.py` for idempotency (`started` / `sent` / `cancelled` / `errored` per month). Also stores `amount_inr` / `attendance_days` / `invoice_number` so the QA agent can answer questions without reading the LangGraph checkpoint.
4. `chat_memory` — owned by `qa/memory.py`. Per-user-phone alternating user/assistant turns (Q&A conversation history). Keyed on `(user_phone, turn_idx)`.

When touching schema or migrations, remember the file is multi-owner — only mess with the `invoice_history` and `chat_memory` DDL in `db.py:_SCHEMA`.
```

- [ ] **Step 2: Add a `### Q&A agent` section under `## Architecture`**

After the existing `### Email via Microsoft Graph (not SMTP)` section (and before `### Webhook security`), add:

```
### Q&A agent (free-form questions)

Generic question intents from `parse_query_intent` route to `qa.answer()` instead of the old `chat_reply`. The agent is `langgraph.prebuilt.create_react_agent` over three tools (`get_invoice`, `compare_invoices`, `web_search` via Tavily) backed by `qwen2.5:7b-instruct`. Per-user chat history is stored in `chat_memory`. The webhook calls `try_answer` via `asyncio.to_thread` because the agent can take seconds.

Three guardrails sit between the LLM and the WhatsApp send:
1. **`recursion_limit=8`** — caps tool-call loop iterations (≤ ~3 tool calls then answer).
2. **`ThreadPoolExecutor` timeout** — `qa_invoke_timeout_seconds` (default 30s) hard cap on the whole turn.
3. **`_amounts_verified`** — every INR-shaped amount in the reply must appear in the concatenated `get_invoice` / `compare_invoices` ToolMessage content. `web_search` snippets are deliberately excluded from the whitelist; they're not authoritative number sources.

If any guardrail fires, `answer()` returns `_FALLBACK_STRING` rather than risk quoting a fabricated amount.

The deterministic fast-paths in `webhook/query.py` (`last_invoice_amount`, `greeting`, `start_invoice`) are kept — they're cheaper and more predictable than a tool-calling round trip for the most common questions.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(qa): document QA agent + chat_memory as 4th DB owner"
```

---

## Task 17: Final integration sweep

**Files:** all (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: every test passes. If any QA-related test is failing, fix it inline (read the failure and address it before moving on; don't paper over with skip markers).

- [ ] **Step 2: Run linting**

```bash
uv run ruff check src tests
```

Expected: no errors. Fix any reported issues.

- [ ] **Step 3: Verify coverage on `qa/` module**

```bash
uv run pytest --cov=src/invoice_agent/qa --cov-report=term-missing tests/test_qa_*.py
```

Expected: ≥85% line coverage on `qa/`. If below, add focused tests for the missing lines (most likely the `web_search` no-api-key branch or rare error paths in `agent.answer`).

- [ ] **Step 4: Smoke test the demo (no real LLM/Tavily needed)**

```bash
uv run python -c "
from invoice_agent.qa import answer
from invoice_agent.qa.tools import get_invoice, compare_invoices, web_search
from invoice_agent.qa.memory import append_turn, load_recent_turns
from invoice_agent.qa.prompts import QA_SYSTEM
print('imports ok')
print('tools:', [t.name for t in (get_invoice, compare_invoices, web_search)])
print('answer signature:', answer.__doc__.splitlines()[0] if answer.__doc__ else '(no doc)')
"
```

Expected: prints `imports ok` and the tool names.

- [ ] **Step 5: Commit any fixups**

If the previous steps required fixes:

```bash
git add -A
git commit -m "chore(qa): final integration fixes"
```

If nothing changed, skip the commit.

---

## Summary

After all 17 tasks, the system has:

- A new `qa/` package (memory, tools, prompts, agent, util)
- Extended `invoice_history` (amount/attendance/invoice_number/sent_at) and a new `chat_memory` table
- `qa_*` config settings + `tavily_api_key`
- `webhook/query.py` routes `generic_question` to the QA agent
- `webhook/server.py` calls `try_answer` via `asyncio.to_thread` with the user's phone
- Three guardrails: `recursion_limit=8`, 30s ThreadPoolExecutor timeout, post-generation amount verification
- Updated CLAUDE.md documenting the new DB owner and architecture section
- Five new test files covering tools, memory, agent, safety, and end-to-end webhook integration
