"""QA tool-calling agent — entrypoint and safety helpers."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Iterable, Optional

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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
