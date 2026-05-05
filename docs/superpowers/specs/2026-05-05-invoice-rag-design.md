# Invoice Q&A — Tool-Calling Agent Design

**Status:** approved, ready for implementation plan
**Date:** 2026-05-05
**Owner:** yashdeep.tomer@gmail.com

## Problem

The WhatsApp invoice agent today can run the monthly invoice flow and answer
exactly one structured question (`last_invoice_amount`) from a hand-coded handler.
Anything else routes to `chat_reply`, which is system-prompted to refuse quoting
*any* number — so the bot literally cannot answer "what did I bill in March?"
or "what's the GST rate for IT services?".

Users want to ask basic, free-form questions about their current and previous
month invoices and have the bot answer naturally — without it feeling like a
scripted bot. When the answer requires general knowledge (tax rates,
regulations), the bot should be able to look it up on the web.

## Goals

- Answer free-form questions about the current and previous month's invoice
  using authoritative data from `invoice_history` and the LangGraph checkpoint
  state for that month.
- Fall back to web search when the question is general knowledge, not invoice
  data.
- Maintain conversational memory across turns within a user's session so
  pronouns and follow-ups work ("is that more than last month?").
- Sound like a person texting back — not a templated bot. Match the user's
  language (English / Hindi / Hinglish).
- Preserve the existing financial-safety guarantee: never quote a fabricated
  amount, date, or invoice number.

## Non-goals (v1)

- Extracting text from rendered PDF invoices (line items, GST breakup, HSN).
  The structured DB + LangGraph checkpoint state covers the "basic questions"
  use case.
- Cross-month aggregates beyond two-month comparison (e.g. "total billed this
  year"). Easy to add later as a new tool.
- Multi-user beyond the configured `USER_WHATSAPP_NUMBER` — the bot still only
  serves one user; chat memory is keyed by phone number for forward
  compatibility, not multi-tenancy.
- Real-time tone tuning / persona configuration. One system prompt for v1.

## High-level architecture

```
WhatsApp inbound → /webhook (POST)
                    │
                    ├─ HMAC + sender check (unchanged)
                    │
                    ├─ active flow router (unchanged)
                    │   if invoice_history.status == 'started' for any month:
                    │       → resume_with_reply()
                    │
                    └─ Q&A path
                          │
                          ├─ try_answer(text, user_phone) in webhook/query.py
                          │     parse_query_intent — kept; routes:
                          │       last_invoice_amount → deterministic (kept)
                          │       greeting           → deterministic (kept)
                          │       start_invoice      → deterministic (kept)
                          │       generic_question   → qa_agent.answer()  ← NEW
                          │       none               → fall through
                          │
                          └─ qa_agent.answer(text, user_phone)
                                ├─ load_recent_turns from chat_memory
                                ├─ ReAct agent (LangGraph create_react_agent)
                                │   tools: get_invoice, compare_invoices, web_search
                                │   model: ChatOllama qwen2.5:7b-instruct, bind_tools
                                │   recursion_limit=8, 30s timeout
                                ├─ post-generation amount-verification check
                                ├─ append_turn to chat_memory
                                └─ return reply text
```

The QA agent is a sibling of `tools/`, not a replacement for it. `tools/llm.py`'s
structured-output parsers (`parse_project_reply`, `parse_approval_reply`,
`parse_summary_reply`, `parse_query_intent`) are flow-internal and stay
untouched.

## Components

### `src/invoice_agent/qa/tools.py`

Three LangChain `@tool`-decorated functions. Tool docstrings are load-bearing —
the LLM uses them to choose which tool to call. They must steer the model
toward the right tool for invoice questions vs general knowledge.

```python
@tool
def get_invoice(month: str) -> dict:
    """Look up a single invoice. month is 'current', 'previous', or 'YYYY-MM'."""
    # Resolve token via shared util (extract _normalize_target_month from
    # webhook/query.py to qa/util.py — used by both modules).
    # Returns:
    #   {month, project_name, amount_inr, attendance_days, status, sent_at,
    #    invoice_number, pdf_path}
    # Source for status / sent_at / pdf_path / project_name: invoice_history row.
    # Source for amount_inr / attendance_days / invoice_number: LangGraph
    # checkpoint state for thread "invoice-{YYYY-MM}", read via
    # SqliteSaver.from_conn_string() as a context manager (per CLAUDE.md
    # convention — open per call, close immediately).
    # Missing month: {month, status: "not_found"} — never raise.

@tool
def compare_invoices() -> dict:
    """Side-by-side current vs previous month. Use for higher/lower/diff questions."""
    # Returns:
    #   {current: {...same shape as get_invoice...},
    #    previous: {...},
    #    amount_diff_inr,
    #    same_project: bool}
    # If either month missing, that side is {status: "not_found"}.

@tool
def web_search(query: str) -> list[dict] | dict:
    """Search the web. Use ONLY when the question is not about the user's own
    invoices (e.g. tax rates, GST rules, general knowledge)."""
    # Tavily, max_results=3, include_raw_content=False.
    # Returns: [{title, url, snippet}, ...]
    # Errors: {error: "search_unavailable"} on network/auth failure.
    # Budget: 5 calls/turn enforced via a ContextVar counter. answer() opens a
    # new context (counter=0) before agent.invoke and the tool reads/increments
    # it. Over-budget returns {error: "search_budget_exceeded"}.
```

### `src/invoice_agent/qa/memory.py`

Chat memory — fourth owner of `data/invoice_agent.db`.

```python
def load_recent_turns(user_phone: str, n: int = 6) -> list[BaseMessage]:
    """Return last n*2 rows (n turns × 2 messages) as alternating
    HumanMessage/AIMessage, oldest first."""

def append_turn(user_phone: str, user_msg: str, assistant_msg: str) -> None:
    """Insert two rows (user + assistant) with monotonic turn_idx.

    Atomicity: SELECT MAX(turn_idx)+1 and the two INSERTs run inside a single
    transaction (BEGIN ... COMMIT). SQLite's write serialisation + the unique
    PRIMARY KEY make concurrent appends from two webhook calls safe — the
    second call sees the first's commit before computing its turn_idx."""

def trim_old(user_phone: str, keep: int = 20) -> None:
    """Delete rows beyond the most recent `keep` turns. Called from append_turn
    when row count exceeds keep + 10 (cheap amortised cleanup)."""
```

Schema:

```sql
CREATE TABLE IF NOT EXISTS chat_memory (
    user_phone   TEXT NOT NULL,
    turn_idx     INTEGER NOT NULL,
    role         TEXT NOT NULL,         -- 'user' | 'assistant'
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (user_phone, turn_idx)
);
CREATE INDEX IF NOT EXISTS idx_chat_memory_phone_idx
    ON chat_memory(user_phone, turn_idx DESC);
```

DDL added to `db.py:_SCHEMA` (the only place schema for non-LangGraph,
non-APScheduler tables lives). CLAUDE.md updated to list `chat_memory` as the
fourth owner of the SQLite file.

### `src/invoice_agent/qa/prompts.py`

```python
QA_SYSTEM = """You're {company}'s billing assistant on WhatsApp. You help one
user — they own these invoices.

Talk like a person texting back: short, casual, contractions are fine. Match
the user's language (English, Hindi, Hinglish — mirror what they sent).

Tools you have:
- get_invoice(month): for anything about THIS user's invoice for a given month.
  Use 'current', 'previous', or 'YYYY-MM'.
- compare_invoices(): when they're comparing this month vs last month
  ("more than", "higher", "diff", "same as").
- web_search(query): ONLY for general knowledge (tax rates, GST rules,
  definitions, news). NEVER use it to look up the user's own invoices.

Hard rules:
- Quote numbers EXACTLY as tools return them. Never invent or estimate
  amounts, dates, or invoice numbers.
- Never claim to be a bot, AI, or assistant unless directly asked.
- No bulleted help menus. No "I'd be happy to help" filler.
- If a tool returns {status: "not_found"}, say plainly that there's no record
  on file for that month.
- If web_search returns {error: ...}, tell the user you couldn't look it up
  right now — don't guess.

Keep replies to 1-3 short lines unless they explicitly ask for detail."""
```

### `src/invoice_agent/qa/agent.py`

```python
def build_qa_agent(settings: Settings) -> CompiledGraph:
    llm = make_chat(settings, temperature=0.4)  # 0.4 for natural phrasing variety
    tools = [get_invoice, compare_invoices, web_search]
    return create_react_agent(
        llm.bind_tools(tools),
        tools=tools,
        state_modifier=QA_SYSTEM.format(company=settings.company_name),
    )

def answer(text: str, user_phone: str, *, settings: Settings | None = None) -> str:
    """Synchronous entry point. Bounded via concurrent.futures so it can be
    safely called from FastAPI's async handler via asyncio.to_thread()."""
    s = settings or get_settings()
    history = load_recent_turns(user_phone, n=s.qa_chat_memory_turns)
    agent = build_qa_agent(s)
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                agent.invoke,
                {"messages": history + [HumanMessage(text)]},
                config={"recursion_limit": 8},
            )
            result = future.result(timeout=s.qa_invoke_timeout_seconds)
        reply = result["messages"][-1].content
    except FuturesTimeoutError:
        log.warning("qa.timeout", user_phone=user_phone)
        return _FALLBACK_STRING
    except Exception as e:  # noqa: BLE001
        log.warning("qa.invoke_failed", err=str(e), user_phone=user_phone)
        return _FALLBACK_STRING

    if not _amounts_verified(reply, result["messages"]):
        log.warning("qa.amount_unverified", reply_preview=reply[:80])
        return _FALLBACK_STRING

    append_turn(user_phone, text, reply)
    return reply
```

`_amounts_verified` extracts INR-shaped tokens from `reply` and confirms each
appears verbatim in the concatenated `ToolMessage` content from this turn (not
from `web_search` outputs). Implementation note: build the whitelist set from
tool messages whose name is `get_invoice` or `compare_invoices`; the
`web_search` snippet content is excluded from the whitelist deliberately.

### `webhook/query.py` change

```python
def _generic_question(text: str, _intent: QueryIntent, s: Settings,
                     *, user_phone: str) -> str:
    return qa_agent.answer(text, user_phone, settings=s)
```

`try_answer` signature gains a `user_phone` parameter, threaded through from
`webhook/server.py` (which already has access to the inbound message's `from`
field).

### `webhook/server.py` change

`try_answer` today is called synchronously from the async `receive()` handler.
That's fine when `try_answer` returns in milliseconds (DB lookup), but the QA
agent can take seconds. To avoid blocking FastAPI's event loop:

```python
# was: answer = try_answer(text, settings=s)
answer = await asyncio.to_thread(try_answer, text, user_phone=msg["from"], settings=s)
```

The bounded timeout inside `qa_agent.answer` (via `ThreadPoolExecutor.result(timeout=...)`)
ensures the to_thread() call returns within `qa_invoke_timeout_seconds` even
on a stuck LLM.

### `config.py` additions

```python
tavily_api_key: SecretStr  # required when QA agent enabled
qa_chat_memory_turns: int = 6
qa_web_search_max_calls_per_turn: int = 5
qa_web_search_daily_cap: int = 0  # 0 = disabled
qa_invoke_timeout_seconds: float = 30.0
```

`tests/conftest.py` pre-seeds `tavily_api_key` so existing tests don't break.

## Data flow examples

### Example 1: structured lookup

```
user: "what did i bill in march?"
  → parse_query_intent: generic_question
  → qa_agent.answer
  → ReAct turn 1 (LLM): tool_call get_invoice(month="2026-03")
  → ReAct turn 2 (tool): {month:"2026-03", project_name:"madabranding",
                          amount_inr:200000, attendance_days:30,
                          status:"sent", sent_at:"2026-03-05T...",
                          invoice_number:"INV-2026-03-001",
                          pdf_path:"out/invoice_2026-03_madabranding.pdf"}
  → ReAct turn 3 (LLM): "March was 2 lakh for madabranding — sent on the 5th."
  → amount-verify: "2 lakh" not in tool result text? Check: "200000" is.
    "2 lakh" is ambiguous; verifier checks numeric forms. (See open question 1.)
  → append_turn, send via WhatsApp.
```

### Example 2: multi-turn comparison

```
prior history: [Human("what did i bill in march?"), AI("March was 2 lakh ...")]
user: "is that more than last month?"
  → ReAct turn 1 (LLM): tool_call compare_invoices()
  → ReAct turn 2 (tool): {current:{month:"2026-05", amount_inr:200000,...},
                          previous:{month:"2026-04", amount_inr:200000,...},
                          amount_diff_inr:0, same_project:true}
  → ReAct turn 3 (LLM): "Same actually — both months were 2 lakh."
```

### Example 3: web search fallback

```
user: "what's the GST rate for IT services in india?"
  → ReAct turn 1 (LLM): tool_call web_search("GST rate IT services India 2026")
  → ReAct turn 2 (tool): [{title:"GST on IT services", snippet:"18% on most ..."}, ...]
  → ReAct turn 3 (LLM): "It's 18% for most IT services in India."
  → amount-verify: "18%" — percentages aren't in the INR-token regex, so it
    passes through. (Verifier targets currency, not percentages.)
```

## Error handling

| Scenario | Behaviour |
|---|---|
| LLM emits invalid tool args | LangGraph tool node errors → ReAct retries once → if still bad, LLM falls back to text answer. |
| LLM never calls a tool, just answers | Allowed. For "hi" / chit-chat the agent answers directly. |
| `get_invoice` returns `not_found` | Tool returns `{status:"not_found"}` — LLM phrases politely. |
| Tavily down / no key | `web_search` returns `{error:"search_unavailable"}` — LLM tells user it can't search now, doesn't guess. |
| Tool-loop > 8 iterations | `recursion_limit=8` → returns last message, logs `qa.recursion_limit_hit`. |
| `agent.invoke` exceeds `qa_invoke_timeout_seconds` (default 30s) | `ThreadPoolExecutor.result(timeout=...)` raises `FuturesTimeoutError` → fallback string, logs `qa.timeout`. |
| Agent throws any other exception | Caught in `answer()`, logs `qa.invoke_failed`, returns fallback string. Webhook still returns 200. |
| Reply contains an INR amount not in tool output | `_amounts_verified` returns false → fallback string, logs `qa.amount_unverified`. |
| Active flow in progress | `try_answer` already bails before reaching QA agent (existing guard). |

`_FALLBACK_STRING = "Sorry, something went wrong on my end. You can ask me 'what was my last invoice amount?'"`

## Cost & rate limits

- **Per-turn:** `recursion_limit=8` ≈ ≤3 tool calls before answer.
- **Per-turn web search:** 5 calls max via in-tool counter.
- **Per-day web search (optional):** `qa_web_search_daily_cap` env-flag-gated;
  default 0 (disabled). When > 0, daily count tracked in a new
  `web_search_audit(user_phone, date, count)` table — added in a follow-up
  spec, not v1.
- **LLM:** local Ollama, no per-token cost. Hard timeout
  (`qa_invoke_timeout_seconds`, default 30s) via
  `ThreadPoolExecutor.result(timeout=...)` to bound webhook latency.

## Logging

All structlog, no raw user message bodies (truncate to 80 chars).

| Key | When |
|---|---|
| `qa.intent_routed` | `try_answer` hands off to QA agent |
| `qa.tool_called` | `{tool, args_preview, latency_ms}` |
| `qa.tool_failed` | `{tool, err}` |
| `qa.amount_unverified` | post-generation check tripped |
| `qa.recursion_limit_hit` | ReAct hit 8-iter cap |
| `qa.timeout` | 30s wait_for fired |
| `qa.invoke_failed` | agent threw |
| `qa.fallback_string` | sent the canned recovery string |
| `qa.turn_complete` | `{turns_in_history, tool_calls, total_latency_ms}` |

## Testing

Five new test files, all using existing repo conventions (`tmp_settings`
fixture, `respx` for HTTP, stubbed `ChatOllama`).

### `tests/test_qa_tools.py`
Pure unit, no LLM, no real network.
- `get_invoice("current")`, `("previous")`, `("2026-03")` returns expected
  shape when DB has the row.
- Returns `{status:"not_found"}` for absent month — never raises.
- `compare_invoices` with both / one / neither month present.
- `web_search` mocked via respx → Tavily endpoint; verifies request, parses
  top 3, error path returns `{error:"search_unavailable"}`.
- Budget cap: 6th call same turn returns `{error:"search_budget_exceeded"}`.

### `tests/test_qa_memory.py`
- `append_turn` + `load_recent_turns(n=6)` returns last 6 turns (12 rows) as
  alternating Human/AI, oldest first.
- `trim_old(keep=20)` keeps newest 20, deletes the rest.
- Multi-user isolation: phone A's turns don't leak into phone B.
- `turn_idx` monotonic per user.

### `tests/test_qa_agent.py`
LLM stubbed (scripted tool calls), tools real against `tmp_settings` DB.
- "what did I bill in march?" → stub emits `get_invoice("2026-03")` → reply
  contains DB amount.
- "is that more than last month?" with prior turn in `chat_memory` → stub emits
  `compare_invoices()` → assert history was passed to agent.
- "what's GST for IT services?" → stub emits `web_search` → respx mocks Tavily
  → reply uses snippet.
- Stub raises → `answer()` returns fallback string, doesn't propagate.

### `tests/test_qa_webhook.py`
Full POST → reply path.
- Inbound POST + valid HMAC + no active flow + "what was last month's amount?"
  → asserts `WhatsAppClient.send_text` called with reply containing DB amount.
- Active-flow guard: `invoice_history.status='started'` → generic-question
  text **not** routed to QA agent (assert agent factory not invoked).
- Bad HMAC / unauthorized sender paths re-asserted as regression.

### `tests/test_qa_safety.py`
- Stub LLM emits text with INR amount NOT in tool result → fallback string,
  logs `qa.amount_unverified`.
- Stub LLM emits text quoting amount that WAS in tool result → reply passes.
- Stub always tool-calls → hits `recursion_limit=8`, logs
  `qa.recursion_limit_hit`.
- Slow stub (sleep > timeout) → `ThreadPoolExecutor` timeout fires → fallback
  string, logs `qa.timeout`.

Coverage target: 85% lines on `qa/`, matching the `tools/` bar.

## What does NOT change

- HMAC validation, sender check, GET verification handshake.
- Graph compilation, `interrupt_after`, checkpointer lifecycle.
- `tools/llm.py` structured-output parsers.
- Idempotency contract on `start_for_month`.
- APScheduler wiring, cron, misfire grace.
- WeasyPrint / docx rendering pipeline.
- The deterministic fast-paths in `webhook/query.py` for `last_invoice_amount`,
  `greeting`, `start_invoice` — kept for predictability and so the most common
  question doesn't burn a tool-calling round trip.

## Open questions / follow-ups

1. **Amount-verification regex granularity.** Reply containing "2 lakh" vs
   tool result containing "200000" — should the verifier normalise? v1
   approach: regex matches both numeric (`\b\d[\d,]{2,}\b`) and Hindi-style
   ("lakh", "crore") tokens; require *either* form to appear in tool result.
   Refine empirically if false positives spike.

2. **Per-user daily search cap.** Out of v1 scope; deferred until we see real
   usage patterns. Schema sketch in spec but no code.

3. **PDF text extraction (question type C).** Out of v1. If users start
   asking about line items / GST breakup, add a `read_invoice_pdf(month)`
   tool that runs the docx through python-docx or extracts text from the
   rendered PDF.

4. **Active-flow off-topic handling.** Today, asking a question mid-flow gets
   silently re-prompted with the project-name template. Could be improved
   with a "let's finish this invoice first" reply. Out of scope here.
