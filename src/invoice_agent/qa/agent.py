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
