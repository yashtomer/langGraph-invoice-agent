# Invoice Agent

A local-first WhatsApp invoice automation agent built on **LangGraph** + **Ollama**. On the 25th of every month it pings you on WhatsApp, parses your free-form (English / Hindi / Hinglish) reply into a structured project name, generates a docx-templated PDF invoice, sends it to you for preview, and on your approval emails it to your accounts team via Microsoft Graph.

It also answers free-form questions between invoices ("what did I bill in March?", "is that more than last month?", "what's the GST rate for IT services?") using a tool-calling Q&A agent over the same SQLite store, with optional Tavily web-search fallback for general-knowledge questions.

Runs on a Mac Mini or any Linux box with `uv` + Ollama + LibreOffice.

---

## Architecture at a glance

```
                   25th of month                 user replies on WhatsApp
                        |                                  |
                        v                                  v
 APScheduler  ----> start_for_month()           Meta webhook -> /webhook
       \                |                                  |
        \               v                                  v
         \--->  LangGraph StateGraph    <--  resume_with_reply() OR  try_answer()
                        |                            (Q&A intent)        |
   ask_project_name  ->  parse_project  ->  send_summary                 v
                                              |              qa.answer() -> ReAct agent
                                       parse_summary           tools: get_invoice,
                                              |                        compare_invoices,
                                       generate_pdf                    web_search (Tavily)
                                              |
                                        send_preview
                                              |
                                       parse_approval
                                       /     |     \
                                 approved  reject  change_requested
                                     |       |        |
                              email_accounts |       loop -> send_summary
                                     |       v
                              confirm_to_user  notify_cancelled
```

State is persisted in a single SQLite file via LangGraph's `SqliteSaver`, so a kill/restart resumes mid-conversation. APScheduler uses the same SQLite for its job store and a 24h `misfire_grace_time` so a sleeping host doesn't drop the monthly run. The Q&A agent persists per-user chat history in a fourth `chat_memory` table in the same DB.

---

## Prerequisites

- macOS 13+ on Apple Silicon, or Linux (Ubuntu 22.04+ tested)
- Python 3.11
- [`uv`](https://github.com/astral-sh/uv) (`brew install uv` / `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Ollama](https://ollama.com/) running locally
- [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) for the public webhook URL
- LibreOffice (`soffice`) for docx → PDF conversion
- A Meta developer account (free)
- An Azure tenant with `Mail.Send` Application permission (Microsoft Graph)
- *(optional)* A Tavily API key for the Q&A agent's web-search tool

---

## 1. Local setup

```bash
git clone <this-repo> invoice-agent
cd invoice-agent
uv sync --extra dev
cp .env.example .env
mkdir -p data out
```

Install LibreOffice for the docx → PDF step:

```bash
# macOS
brew install --cask libreoffice

# Ubuntu / Debian
sudo apt install libreoffice-writer --no-install-recommends
```

Drop your invoice template at `templates/invoice.docx` — it's a Jinja-templated `.docx` rendered by `docxtpl` (placeholders like `{{ project_name }}`, `{{ amount }}`, `{{ invoice_number }}`).

---

## 2. Pull the LLM

```bash
ollama serve            # in one terminal — leave running
ollama pull qwen2.5:7b-instruct
```

Smoke test:

```bash
ollama run qwen2.5:7b-instruct "say hi"
```

The 7B model handles structured output, Hindi/English/Hinglish parsing, and tool-calling for the Q&A agent on 24GB. If you have headroom, `qwen2.5:14b-instruct` is a clean upgrade. If you're tight, `qwen2.5:3b-instruct` still works for the structured-output paths but tool-calling reliability drops.

---

## 3. Meta WhatsApp Cloud API setup

1. Go to https://developers.facebook.com → **My Apps** → **Create App** → **Business**.
2. Add the **WhatsApp** product to the app.
3. In **WhatsApp → API setup**:
   - Note the **Phone number ID** → `META_WA_PHONE_NUMBER_ID`
   - Generate a **permanent access token** (System User token, recommended) → `META_WA_ACCESS_TOKEN`
   - Add your personal WhatsApp number as a "test recipient" until you submit for review.
4. In **App settings → Basic**:
   - Copy the **App Secret** → `META_WA_APP_SECRET`
5. In **WhatsApp → Configuration**:
   - **Callback URL**: `https://<your-cloudflared-hostname>/webhook` (set up in step 4 below)
   - **Verify token**: any random string you choose → also set as `META_WA_VERIFY_TOKEN`
   - Subscribe to the **`messages`** field.

`USER_WHATSAPP_NUMBER` must be the bare international format `91XXXXXXXXXX` — **no leading `+`**. Meta strips the `+` on inbound webhook payloads and the sender-equality check is exact-string.

### Approved message template

WhatsApp requires a pre-approved template to *open* a conversation outside the 24-hour service window. Create one named exactly `invoice_monthly_prompt`:

- **Category**: Utility
- **Language**: English
- **Body**: `It's the 25th — time for the {{1}} invoice. What's the project name?`

Submit for review. Approval typically takes a few minutes to a few hours.

---

## 4. Cloudflare Tunnel (free, stable URL)

```bash
cloudflared tunnel login                    # opens browser, picks a domain you control
cloudflared tunnel create invoice-agent
cloudflared tunnel route dns invoice-agent invoice.<yourdomain>.com
```

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: invoice-agent
credentials-file: /home/<you>/.cloudflared/<tunnel-uuid>.json

ingress:
  - hostname: invoice.<yourdomain>.com
    service: http://localhost:8000
  - service: http_status:404
```

Run it:

```bash
cloudflared tunnel run invoice-agent
```

Set Meta's **Callback URL** to `https://invoice.<yourdomain>.com/webhook`.

---

## 5. Microsoft Graph email setup

The agent sends invoice emails via the **Microsoft Graph `sendMail` endpoint** using the MSAL **client-credentials** flow (no SMTP, no Gmail).

1. In the Azure portal → **App registrations** → **New registration** (single tenant).
2. Note the **Application (client) ID** → `AZURE_CLIENT_ID`, **Directory (tenant) ID** → `AZURE_TENANT_ID`.
3. **Certificates & secrets** → new client secret → copy the *value* once → `AZURE_CLIENT_SECRET`.
4. **API permissions** → add `Microsoft Graph` → **Application permissions** → `Mail.Send`. Click **Grant admin consent**.
5. Set `AZURE_MAIL_USER` to the mailbox you want emails to come from (e.g. `you@yourdomain.com`).

By default `Mail.Send` Application permission grants "send-as anyone in the tenant". Restrict it to a single mailbox via Exchange Online's Application Access Policy:

```powershell
New-ApplicationAccessPolicy `
  -AppId <AZURE_CLIENT_ID> `
  -PolicyScopeGroupId <a security group containing only AZURE_MAIL_USER> `
  -AccessRight RestrictAccess `
  -Description "Invoice agent — send-as only the agent mailbox"
```

`ACCOUNTS_EMAIL` accepts comma-separated recipients. `CC_EMAIL` mirrors that for cc.

---

## 6. *(optional)* Tavily for the Q&A agent's web search

The Q&A agent has a `web_search` tool for general-knowledge questions ("what's the current GST rate for IT services?"). It's bypassed for invoice-data questions. Without a key the tool short-circuits to `{"error": "search_unavailable"}` and the agent tells the user it can't look it up.

1. Sign up at https://tavily.com → grab a key.
2. Set `TAVILY_API_KEY` in `.env`.

Per-turn budget defaults to 5 calls (`QA_WEB_SEARCH_MAX_CALLS_PER_TURN`).

---

## 7. Configure `.env`

Copy `.env.example` and fill in. Required minimum:

```
META_WA_PHONE_NUMBER_ID=...
META_WA_ACCESS_TOKEN=...
META_WA_VERIFY_TOKEN=...
META_WA_APP_SECRET=...
USER_WHATSAPP_NUMBER=91XXXXXXXXXX

AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_TENANT_ID=...
AZURE_MAIL_USER=you@yourdomain.com
ACCOUNTS_EMAIL=accounts@yourcompany.com
CC_EMAIL=you@yourcompany.com

COMPANY_NAME=Your Co.
INVOICE_AMOUNT_INR=150000
WEBHOOK_SHARED_SECRET=<long random string>

# Optional — only if you want web search in the Q&A agent
TAVILY_API_KEY=tvly-...
```

---

## 8. First run

```bash
make run    # = uv run invoice-agent
```

You should see a JSON log line `app.started`. Then in a second terminal:

```bash
curl http://localhost:8000/healthz
# {"ok":true}
```

Test the manual trigger (uses `WEBHOOK_SHARED_SECRET`):

```bash
curl -X POST http://localhost:8000/trigger \
     -H "x-shared-secret: <your secret>"
```

Your phone should buzz with the `invoice_monthly_prompt` template. Reply with a project name → review the draft summary → reply `yes` (or `haan bhej do`) → accounts gets the email → you get a confirmation.

---

## 9. Free-form Q&A

Between monthly runs you can ask the bot questions:

- *"what did I bill in march?"* → `get_invoice("2026-03")` → "March was 2 lakh for Madabranding."
- *"is that more than last month?"* → `compare_invoices()` → "Same actually — both months were 2 lakh."
- *"what's the GST rate for IT services in india?"* → `web_search(...)` → "It's 18% for most IT services in India."

The agent is `langgraph.prebuilt.create_react_agent` over three tools (`get_invoice`, `compare_invoices`, `web_search`) backed by `qwen2.5:7b-instruct`. Per-user chat history is stored in `chat_memory`. Three guardrails sit between the LLM and the WhatsApp send:

1. **`recursion_limit=8`** — caps tool-call iterations.
2. **`ThreadPoolExecutor` 30s timeout** — bounds total LLM wall-clock.
3. **Post-generation amount verification** — every INR-shaped amount in the reply must appear verbatim in the `get_invoice` / `compare_invoices` ToolMessage content. `web_search` snippets are excluded from the whitelist; if any guardrail trips, the agent returns a fallback string rather than risk a fabricated number.

The deterministic fast-paths in `webhook/query.py` (`last_invoice_amount`, `greeting`, `start_invoice`) are kept ahead of the agent — cheaper and more predictable than a tool-calling round trip for the most common questions.

---

## 10. Wake-from-sleep on the 25th (macOS only)

APScheduler's `misfire_grace_time=86400` already handles missed firings — the job runs as soon as the host comes back. To proactively wake a Mac:

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 10:24:00
```

This wakes daily at 10:24 (one minute before the 10:30 cron). To wake only on the 25th, use `pmset schedule` or a launchd plist. On Linux, leave the host running or use systemd timers.

---

## 11. Run the demo

The demo replays the entire LangGraph flow end-to-end with mocked WhatsApp + Microsoft Graph + LLM, printing each interrupt boundary:

```bash
make demo            # default: approve scenario
make demo-change     # change-requested loop, then approve
make demo-reject     # rejected
```

No external services required.

---

## 12. Run the tests

```bash
make test
# or
uv run pytest
```

`test_pdf_render.py` is auto-skipped if LibreOffice isn't installed.

---

## Project layout

```
invoice-agent/
├── pyproject.toml
├── Makefile
├── .env.example
├── README.md
├── CLAUDE.md
├── docs/superpowers/
│   ├── specs/2026-05-05-invoice-rag-design.md   # Q&A agent spec
│   └── plans/2026-05-05-invoice-rag.md          # 17-task TDD plan
├── scripts/
│   └── demo.py                 # `make demo`
├── src/invoice_agent/
│   ├── config.py               # pydantic-settings (incl. Azure + Tavily)
│   ├── logging_setup.py        # structlog JSON
│   ├── state.py                # InvoiceState TypedDict
│   ├── graph.py                # StateGraph wiring + SqliteSaver + interrupt_after
│   ├── runner.py               # start_for_month / resume_with_reply
│   ├── scheduler.py            # APScheduler cron job
│   ├── db.py                   # invoice_history + chat_memory + idempotent migrations
│   ├── main.py                 # FastAPI + uvicorn entry
│   ├── nodes/
│   │   ├── ask_project.py
│   │   ├── parse_project.py
│   │   ├── send_summary.py     # draft summary preview
│   │   ├── parse_summary.py    # parse summary-confirm reply
│   │   ├── generate_pdf.py
│   │   ├── send_preview.py
│   │   ├── parse_approval.py
│   │   ├── email_accounts.py
│   │   └── notify.py
│   ├── tools/
│   │   ├── whatsapp.py         # Meta Cloud API client
│   │   ├── pdf.py              # docxtpl + LibreOffice headless converter
│   │   ├── mailer.py           # MSAL client-credentials → Graph sendMail
│   │   └── llm.py              # ChatOllama + structured-output + heuristic fallback
│   ├── qa/                     # NEW: free-form Q&A tool-calling agent
│   │   ├── agent.py            # build_qa_agent + answer + amount verification
│   │   ├── tools.py            # @tool: get_invoice, compare_invoices, web_search
│   │   ├── memory.py           # chat_memory CRUD (per-user history)
│   │   ├── prompts.py          # QA_SYSTEM
│   │   └── util.py             # normalize_target_month
│   └── webhook/
│       ├── server.py           # FastAPI: /webhook, /trigger, /healthz
│       ├── query.py            # intent router (last_invoice_amount, greeting, generic_question, …)
│       └── legal.py            # /privacy, /terms, /data-deletion (Meta requires)
├── templates/
│   └── invoice.docx            # Jinja-templated docx
├── out/                        # generated PDFs/docx (gitignored)
├── data/                       # sqlite (4 owners: checkpoints + apscheduler + invoice_history + chat_memory)
└── tests/
    ├── conftest.py
    ├── test_db.py
    ├── test_parse_query.py
    ├── test_parse_summary.py
    ├── test_pdf_render.py
    ├── test_webhook.py
    ├── test_webhook_query.py
    ├── test_graph_flow.py      # full happy path + change loop + rejected + restart
    ├── test_qa_agent.py        # ReAct agent + chat_memory + safety
    ├── test_qa_config.py
    ├── test_qa_memory.py
    ├── test_qa_prompts.py
    ├── test_qa_safety.py       # amount verification + timeout
    ├── test_qa_tools.py        # get_invoice / compare_invoices / web_search
    ├── test_qa_util.py
    └── test_qa_webhook.py      # end-to-end webhook → QA agent
```

---

## How LangGraph drives the conversation

Three design choices are worth calling out:

1. **`interrupt_after=["ask_project_name", "send_summary", "send_preview"]`** at compile time, instead of `interrupt()` calls inside nodes. Each WhatsApp prompt is the *last* action of its node, so this is cleaner and keeps the nodes pure.
2. **One thread per month** (`thread_id = f"invoice-{YYYY-MM}"`). The webhook's `/webhook` always resumes the *current* month's thread; the manual `/trigger` and APScheduler agree on the same key. Idempotency lives in the `invoice_history` table — a second start for a month already marked `sent` is a no-op.
3. **Q&A agent runs in a worker thread.** `webhook/server.py` calls `try_answer` via `asyncio.to_thread` so the FastAPI event loop stays responsive while a multi-second LLM tool-call resolves. Inside the worker, `qa.answer` further bounds itself with a `ThreadPoolExecutor.result(timeout=…)` so a stuck Ollama can't pin the request.

`runner.py` is the one place where graph compilation and idempotency intersect. Open one `SqliteSaver` per call and close it; LangGraph's checkpointer is happy with that.

---

## Security

- `/webhook` validates `X-Hub-Signature-256` (HMAC-SHA256 with `META_WA_APP_SECRET`).
- `/trigger` requires `X-Shared-Secret` matching `WEBHOOK_SHARED_SECRET`.
- The webhook only accepts inbound text from the configured `USER_WHATSAPP_NUMBER`.
- The LLM never authors financial fields in the invoice itself — amount, recipients, invoice number, and template content are deterministic from config + state.
- The Q&A agent quotes numbers strictly from `get_invoice` / `compare_invoices` ToolMessage content. Post-generation regex extraction + whitelist subset check; `web_search` snippets are explicitly excluded from the whitelist. If verification fails, the agent returns the fallback string instead of forwarding a possibly-fabricated reply.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `webhook.bad_signature` 403 | `META_WA_APP_SECRET` doesn't match the App Secret in Meta dashboard | Re-copy from **App settings → Basic → Show** |
| Meta verification handshake fails | `META_WA_VERIFY_TOKEN` mismatch or webhook URL not reachable | `curl https://invoice.<yourdomain>.com/healthz` should return `{"ok": true}` |
| Template message rejected | Template not approved yet, or name mismatch | Check status in WhatsApp Manager; `APPROVED_TEMPLATE_NAME` must match |
| `webhook.unauthorized_sender` on every reply | `+` prefix in `USER_WHATSAPP_NUMBER` | Strip the `+` — Meta delivers the bare number |
| LLM parsing returns garbage | Ollama down, model not pulled | `ollama list` should show the model; `ollama serve` must be running. The heuristic fallback will keep the structured-output paths working. |
| LibreOffice / docx render error | `soffice` not on PATH | `which soffice`; install via Homebrew or apt. |
| Microsoft Graph 401 | `Mail.Send` permission missing or not admin-consented | Re-check **API permissions** → "Granted for tenant" |
| Microsoft Graph 403 | App Access Policy excluding `AZURE_MAIL_USER` | Add the mailbox to the policy's security group |
| `qa.amount_unverified` log spam | LLM is paraphrasing amounts in a form the regex doesn't cover (e.g. words instead of digits) | Tighten the QA system prompt or extend `_amounts_verified` to recognise the form |
| `qa.timeout` log spam | Ollama is overloaded or the question caused a tool-call loop | Bump `QA_INVOKE_TIMEOUT_SECONDS`, or check `recursion_limit` hits |
| Job didn't fire on the 25th | Mac was asleep and woke after `misfire_grace_time` | Check `pmset` schedule; tail logs for `scheduler.fire` |

---

## Out of scope (intentionally)

- Multi-tenant support (chat memory is keyed by phone for forward compat, but the bot serves one configured user)
- Web dashboard
- Multiple invoice templates
- Voice replies
- Auth beyond shared-secret on `/trigger`
- PDF text extraction in the Q&A agent (tool-calling reads structured DB columns; reading rendered PDFs for line-item / GST-breakup questions is a future tool)

---

## License

MIT.
