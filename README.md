# Invoice Agent

A local-first, fully open-source WhatsApp invoice automation agent built on **LangGraph** + **Ollama**. On the 25th of every month it pings you on WhatsApp, parses your free-form (English / Hindi / Hinglish) reply into a structured project name, generates a PDF invoice from a fixed template, sends it to you for preview, and on your approval emails it to your accounts team.

Zero paid APIs. Runs on a Mac Mini.

---

## Architecture at a glance

```
                   25th of month                 user replies on WhatsApp
                        |                                  |
                        v                                  v
 APScheduler  ----> start_for_month()           Meta webhook -> /webhook
       \                |                                  |
        \               v                                  v
         \--->  LangGraph StateGraph  <----  resume_with_reply() (writes user_reply_raw, invokes None)
                        |
   ask_project_name  ->  parse_project_reply  ->  generate_pdf  ->  send_preview
                                                                      |
                                                                parse_approval
                                                                /     |     \
                                                          approved  reject  change_requested
                                                              |       |        |
                                                       email_accounts |       loop -> generate_pdf
                                                              |       v
                                                       confirm_to_user  notify_cancelled
```

State is persisted in a single SQLite file via LangGraph's `SqliteSaver`, so a kill/restart resumes mid-conversation. APScheduler uses the same SQLite for its job store and a 24h `misfire_grace_time` so a sleeping Mac doesn't drop the monthly run.

---

## Prerequisites

- Mac Mini / MacBook on Apple Silicon, macOS 13+
- Python 3.11
- [`uv`](https://github.com/astral-sh/uv) (`brew install uv`)
- [Ollama](https://ollama.com/) (`brew install ollama` or download from site)
- [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) (`brew install cloudflared`)
- A Meta developer account (free)
- A Gmail account with 2FA + an app password

---

## 1. Local setup

```bash
git clone <this-repo> invoice-agent
cd invoice-agent
uv sync --extra dev
cp .env.example .env
mkdir -p data out
```

WeasyPrint needs Pango/Cairo on macOS:

```bash
brew install pango cairo gdk-pixbuf libffi
```

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

The 7B model handles structured output + Hindi/English/Hinglish parsing fine on 24GB. If you have headroom, `qwen2.5:14b-instruct` is a clean upgrade. If you're tight, `qwen2.5:3b-instruct` still works with the heuristic fallback.

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
credentials-file: /Users/<you>/.cloudflared/<tunnel-uuid>.json

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

## 5. Gmail app password

1. Enable 2FA on your Google account.
2. Visit https://myaccount.google.com/apppasswords
3. Generate a password named "invoice agent" → set as `SMTP_APP_PASSWORD` in `.env`.
4. `SMTP_USER` is the same Gmail address.

---

## 6. Configure `.env`

Fill in every section in `.env` (copy from `.env.example`). Required minimum:

```
META_WA_PHONE_NUMBER_ID=...
META_WA_ACCESS_TOKEN=...
META_WA_VERIFY_TOKEN=...
META_WA_APP_SECRET=...
USER_WHATSAPP_NUMBER=91XXXXXXXXXX
SMTP_USER=you@gmail.com
SMTP_APP_PASSWORD=...
ACCOUNTS_EMAIL=accounts@yourcompany.com
CC_EMAIL=you@yourcompany.com
INVOICE_AMOUNT_INR=150000
WEBHOOK_SHARED_SECRET=<long random string>
```

---

## 7. First run

```bash
make run    # = uv run invoice-agent
```

You should see a JSON log line `app.started`. Then in a second terminal:

```bash
curl -X POST http://localhost:8000/healthz
```

Test the manual trigger (uses `WEBHOOK_SHARED_SECRET`):

```bash
curl -X POST http://localhost:8000/trigger \
     -H "x-shared-secret: <your secret>"
```

Your phone should buzz with the `invoice_monthly_prompt` template. Reply with a project name; you'll get the PDF preview back; reply `yes` (or `haan bhej do`); accounts gets the email; you get a confirmation.

---

## 8. Wake-from-sleep on the 25th

APScheduler's `misfire_grace_time=86400` already handles missed firings — the job runs as soon as the Mac comes back. To proactively wake the Mac:

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 10:24:00
```

This wakes daily at 10:24 (one minute before the 10:30 cron). To wake only on the 25th, use `pmset schedule` or a launchd plist.

---

## 9. Run the demo

The demo replays the entire LangGraph flow end-to-end with mocked WhatsApp + SMTP + LLM, printing each interrupt boundary:

```bash
make demo            # default: approve scenario
make demo-change     # change-requested loop, then approve
make demo-reject     # rejected
```

No external services required.

---

## 10. Run the tests

```bash
make test
# or
uv run pytest
```

`test_pdf_render.py` is auto-skipped if WeasyPrint's native deps aren't installed.

---

## Project layout

```
invoice-agent/
├── pyproject.toml
├── docker-compose.yml          # optional: ollama + agent + cloudflared
├── Dockerfile
├── Makefile
├── .env.example
├── README.md
├── scripts/
│   └── demo.py                 # `make demo`
├── src/invoice_agent/
│   ├── config.py               # pydantic-settings
│   ├── logging_setup.py        # structlog JSON
│   ├── state.py                # InvoiceState TypedDict
│   ├── graph.py                # StateGraph wiring + SqliteSaver
│   ├── runner.py               # start_for_month / resume_with_reply
│   ├── scheduler.py            # APScheduler cron job
│   ├── db.py                   # invoice_history idempotency
│   ├── main.py                 # FastAPI + uvicorn entry
│   ├── nodes/
│   │   ├── ask_project.py
│   │   ├── parse_project.py
│   │   ├── generate_pdf.py
│   │   ├── send_preview.py
│   │   ├── parse_approval.py
│   │   ├── email_accounts.py
│   │   └── notify.py
│   ├── tools/
│   │   ├── whatsapp.py         # Meta Cloud API client
│   │   ├── pdf.py              # WeasyPrint renderer
│   │   ├── mailer.py           # smtplib over Gmail
│   │   └── llm.py              # ChatOllama + structured output + heuristic fallback
│   └── webhook/
│       └── server.py           # FastAPI: /webhook, /trigger, /healthz
├── templates/
│   └── invoice.html
├── out/                        # generated PDFs (gitignored)
├── data/                       # sqlite (checkpoints + apscheduler + invoice_history)
└── tests/
    ├── conftest.py
    ├── test_db.py
    ├── test_parse_project.py
    ├── test_parse_approval.py
    ├── test_pdf_render.py
    ├── test_webhook.py
    └── test_graph_flow.py      # full happy path + change loop + rejected + restart
```

---

## How LangGraph drives the conversation

Two design choices are worth calling out:

1. **`interrupt_after=["ask_project_name", "send_preview"]`** at compile time, instead of `interrupt()` calls inside nodes. Each WhatsApp prompt is the *last* action of its node, so this is cleaner and keeps the nodes pure.
2. **One thread per month** (`thread_id = f"invoice-{YYYY-MM}"`). The webhook's `/webhook` always resumes the *current* month's thread; the manual `/trigger` and APScheduler agree on the same key. Idempotency lives in the `invoice_history` table — a second start for a month already marked `sent` is a no-op.

`runner.py` is the one place where graph compilation and idempotency intersect — both `/trigger` and the scheduler call `start_for_month(month)`, and the webhook calls `resume_with_reply(month, text)`. Open one `SqliteSaver` per call and close it; LangGraph's checkpointer is happy with that.

---

## Security

- `/webhook` validates `X-Hub-Signature-256` (HMAC-SHA256 with `META_WA_APP_SECRET`).
- `/trigger` requires `X-Shared-Secret` matching `WEBHOOK_SHARED_SECRET`.
- The webhook only accepts inbound text from the configured `USER_WHATSAPP_NUMBER`.
- The LLM never authors financial fields. Amount, recipients, invoice number, and template content are all deterministic from config + state.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `webhook.bad_signature` 403 | `META_WA_APP_SECRET` doesn't match the App Secret in Meta dashboard | Re-copy from **App settings → Basic → Show** |
| Meta verification handshake fails | `META_WA_VERIFY_TOKEN` mismatch or webhook URL not reachable | `curl https://invoice.<yourdomain>.com/healthz` should return `{"ok": true}` |
| Template message rejected | Template not approved yet, or name mismatch | Check status in WhatsApp Manager; `APPROVED_TEMPLATE_NAME` must match |
| LLM parsing returns garbage | Ollama down, model not pulled | `ollama list` should show the model; `ollama serve` must be running. The heuristic fallback will keep things working. |
| WeasyPrint ImportError | Missing native deps | `brew install pango cairo gdk-pixbuf libffi` |
| Gmail SMTP auth fails | Using account password instead of app password | Generate at https://myaccount.google.com/apppasswords |
| Job didn't fire on the 25th | Mac was asleep and woke after `misfire_grace_time` | Check `pmset` schedule; tail logs for `scheduler.fire` |

---

## Out of scope (intentionally)

- Multi-tenant support
- Web dashboard
- Multiple invoice templates
- Vector DB / RAG over past invoices
- Voice replies
- Auth beyond shared-secret on `/trigger`

---

## License

MIT.
