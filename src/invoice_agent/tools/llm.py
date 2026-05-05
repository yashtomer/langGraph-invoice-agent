"""ChatOllama factory + Pydantic structured-output helpers.

The agent only uses the LLM to *parse* user inputs into typed schemas — never
to generate financial fields. Two parsers are exposed here:

  * ``parse_project_name`` — turn a free-form reply into ``ProjectReply``.
  * ``parse_approval``     — turn a free-form reply into ``ApprovalDecision``.

Both run with temperature=0 and ``with_structured_output``. If structured
parsing fails twice, we fall back to a regex/keyword heuristic so the graph
never wedges on a single bad LLM response.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)


# --------- Schemas ---------


class ProjectReply(BaseModel):
    """User's reply parsed into a project name."""

    project_name: str = Field(
        ...,
        description="The project name the user mentioned. Strip filler words and quotes.",
    )


class QueryIntent(BaseModel):
    """Classifies a free-form WhatsApp message into a query intent."""

    intent: Literal[
        "last_invoice_amount",
        "greeting",
        "start_invoice",
        "generic_question",
        "none",
    ] = Field(
        ...,
        description=(
            "last_invoice_amount when the user is asking about the amount, total, "
            "or value of their last/previous/most-recent invoice. "
            "greeting when the user is just saying hi/hello/namaste with no other intent. "
            "start_invoice when the user wants to TRIGGER / GENERATE / SEND an invoice "
            "for some month (e.g. 'send invoice for may', 'create this month invoice', "
            "'invoice bhejo'). "
            "generic_question when the user is asking an open-ended question or having "
            "free-form chat that is NOT a project name and NOT an approval reply. "
            "none when the message looks like it belongs to an active invoice flow — "
            "a short project-name reply (one or two proper nouns) or an approval/"
            "rejection (yes/no/haan/nahi/cancel/send)."
        ),
    )
    target_month: Optional[str] = Field(
        default=None,
        description=(
            "For start_invoice only — the month the user mentioned, as a literal "
            "token (e.g. 'may', 'may 2026', 'june', 'this month', 'current', "
            "'previous month', 'last month'). None for any other intent."
        ),
    )


class ApprovalDecision(BaseModel):
    """User's reply parsed into an approval decision."""

    status: Literal["approved", "rejected", "change_requested"] = Field(
        ...,
        description=(
            "approved if user says yes/send/haan/bhejo/ok; "
            "rejected if user says no/cancel/nahi/ruk; "
            "change_requested if user wants the project name changed."
        ),
    )
    new_project_name: Optional[str] = Field(
        default=None,
        description="If status is change_requested, the new project name; otherwise None.",
    )


class SummaryReply(BaseModel):
    """User's reply to the draft-summary confirmation step.

    The user can either approve the draft as-is, or override one or more
    fields. Inline overrides like "approve with attendance 28" set
    ``status=approved`` AND populate the override field — the bot applies
    the change and proceeds to PDF generation in one shot.
    """

    status: Literal["approved", "change_requested"] = Field(
        ...,
        description=(
            "approved when the user wants the invoice generated as drafted "
            "(possibly with the inline overrides below). change_requested "
            "when they only want to update fields and re-see the summary."
        ),
    )
    amount_inr: Optional[int] = Field(
        default=None,
        description=(
            "Override the invoice amount in INR (rupees, integer). Set when "
            "the user mentions an amount/total/rate. None if not specified."
        ),
    )
    attendance_days: Optional[int] = Field(
        default=None,
        description=(
            "Override the attendance / present-days count for the month. Set "
            "when the user mentions attendance/present/days. None if not specified."
        ),
    )
    project_name: Optional[str] = Field(
        default=None,
        description=(
            "Override the project name. Set only when the user explicitly says "
            "to change the project. None if they don't mention it."
        ),
    )


# --------- LLM factory ---------


def make_chat(settings: Optional[Settings] = None, temperature: float = 0.0) -> ChatOllama:
    s = settings or get_settings()
    return ChatOllama(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=temperature,
    )


# --------- Prompts ---------

_PROJECT_SYSTEM = """You extract a project name from a free-form WhatsApp reply.

The user may reply in English, Hindi, or mixed (Hinglish). Examples:
  "Birla Opus" -> project_name="Birla Opus"
  "project ka naam Birla Opus hai" -> project_name="Birla Opus"
  "iska naam DLF Camellias likh do" -> project_name="DLF Camellias"
  "Tata Steel Plant" -> project_name="Tata Steel Plant"

Rules:
- Strip filler words ("project", "naam", "ka", "hai", "likh do", "for", "bhai").
- Preserve original casing for proper nouns when possible.
- If the user gave only one phrase, treat the whole phrase as the project name.
"""

_QUERY_SYSTEM = """You classify a WhatsApp message as one of:
  - last_invoice_amount  (asking about amount/total of last/previous/recent invoice)
  - greeting             (pure salutation — hi/hello/namaste/good morning/kaise ho)
  - start_invoice        (user wants to TRIGGER/GENERATE/SEND an invoice for some month)
  - generic_question     (open-ended question or chit-chat that is NOT a project name
                          and NOT an approval reply)
  - none                 (looks like a project-name reply or an approve/reject reply)

If intent=start_invoice, also fill ``target_month`` with the literal month token
the user mentioned: a month name ("may", "may 2026"), or a relative phrase
("this month", "current", "previous month", "last month"). If the user didn't
specify a month, set target_month="current".

The user may write in English, Hindi, or mixed (Hinglish). Examples:

  "what is my last invoice amount"        -> last_invoice_amount
  "kitna tha pichla invoice"              -> last_invoice_amount

  "hi" / "hello" / "namaste"              -> greeting
  "good morning" / "kaise ho"             -> greeting

  "send invoice for may"                  -> start_invoice, target_month="may"
  "i want to send invoice for may"        -> start_invoice, target_month="may"
  "create may 2026 invoice"               -> start_invoice, target_month="may 2026"
  "trigger this month invoice"            -> start_invoice, target_month="this month"
  "invoice bhejo"                         -> start_invoice, target_month="current"
  "send invoice"                          -> start_invoice, target_month="current"
  "generate previous month invoice"       -> start_invoice, target_month="previous month"
  "june ka invoice bana do"               -> start_invoice, target_month="june"

  "what can you do" / "tell me a joke"    -> generic_question
  "who built you" / "thanks!"             -> generic_question

  "Birla Opus" / "DLF Camellias"          -> none
  "project ka naam Tata Steel hai"        -> none
  "yes send it" / "haan bhej do"          -> none
  "no cancel" / "nahi"                    -> none
  "change to Birla Opus"                  -> none

Rules:
- "none" wins for short messages that are 1-3 proper nouns (likely a project name)
  or contain approve/reject words (yes/no/haan/nahi/send/cancel/bhejo) WITHOUT
  the words "invoice", "bill", or a month name.
- "start_invoice" requires a verb of generation (send/create/generate/trigger/
  invoice/bana do/bhejo) PAIRED with the word "invoice" or "bill". Just "send"
  or "bhejo" alone is "none" (it's an approval reply).
- Classify as "greeting" only if the message is *purely* a salutation.
"""


_CHAT_SYSTEM = """You are a friendly WhatsApp invoice assistant for {company}.
Reply in 1-2 short sentences, plain text, no markdown. Match the user's language
(English / Hindi / Hinglish).

You CAN:
  - Tell the user their last invoice amount (they ask, you answer from records).
  - Send the monthly invoice on the configured day.

You CANNOT and MUST NOT:
  - Quote any new amount, invoice number, date, or financial figure that wasn't in
    the user's question. If asked, say you don't have that info.
  - Promise to take any action outside the monthly invoice workflow.

If the question is outside scope, politely say so in one line and remind the user
they can ask "what is my last invoice amount".
"""


def chat_reply(text: str, *, settings: Optional[Settings] = None, llm: Optional[ChatOllama] = None) -> str:
    """Free-form conversational LLM reply. Used as a chat fallback for generic
    questions outside the structured invoice workflow."""
    s = settings or get_settings()
    chat = llm or make_chat(s, temperature=0.3)
    sys = _CHAT_SYSTEM.format(company=s.company_name)
    try:
        resp = chat.invoke([SystemMessage(sys), HumanMessage(text)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        return content if isinstance(content, str) else str(content)
    except Exception as e:  # noqa: BLE001
        log.warning("llm.chat_reply.failed", err=str(e))
        return "Sorry — I couldn't process that. You can ask me 'what is my last invoice amount'."


_SUMMARY_SYSTEM = """You parse a WhatsApp reply to a draft-invoice summary.

The user is shown a draft invoice (project, amount, attendance days, etc.) and
can either approve it for generation, or change one or more fields. They may
combine the two — e.g. "approve with attendance 28" means status=approved AND
attendance_days=28.

Set status:
  - approved          when the user clearly says approve/yes/ok/generate it/
                      bhejo/haan, possibly with inline overrides
  - change_requested  when the user only wants to change a field and see the
                      summary again, with no approval signal

Set the override fields ONLY when the user explicitly mentions them:
  - amount_inr        — integer rupees (parse "amount 200000", "200k" -> 200000,
                        "2 lakh" -> 200000, "1,45,000" -> 145000). None if absent.
  - attendance_days   — integer (parse "attendance 28", "28 days", "present 30").
                        None if absent.
  - project_name      — only when the user says to change the project name.
                        None otherwise.

Examples:
  "approve"                              -> approved
  "yes go ahead"                         -> approved
  "haan bhej do"                         -> approved
  "ok generate it"                       -> approved
  "approve with attendance 28"           -> approved, attendance_days=28
  "yes but amount 200000"                -> approved, amount_inr=200000
  "ok with 2 lakh"                       -> approved, amount_inr=200000
  "amount 50000"                         -> change_requested, amount_inr=50000
  "attendance 30 days"                   -> change_requested, attendance_days=30
  "change project to Birla Opus"         -> change_requested, project_name="Birla Opus"
  "change amount to 100000 and attendance 25" -> change_requested, amount_inr=100000, attendance_days=25

Default to status=change_requested when the user mentions any field but no clear approval.
"""


_APPROVAL_SYSTEM = """You classify a WhatsApp reply about an invoice preview into one of:
  - approved          (user wants it sent to accounts)
  - rejected          (user wants to cancel)
  - change_requested  (user wants the project name changed)

The user may reply in English, Hindi, or mixed (Hinglish). Examples:
  "yes" / "send it" / "ok bhej do" / "haan" / "ji" / "bhejo" / "go ahead" -> approved
  "no" / "cancel" / "nahi" / "ruk" / "stop" / "abort" -> rejected
  "change to Birla Opus" / "naam Birla Opus kar do" / "actually it's DLF Camellias"
       -> change_requested with new_project_name="Birla Opus" (or "DLF Camellias")

Rules:
- If the user says "change to <X>" or "<X> kar do" or "actually <X>", set
  status=change_requested AND extract new_project_name.
- Only set new_project_name when status=change_requested.
"""


# --------- Heuristic fallback ---------

_APPROVE_RE = re.compile(
    r"\b(yes|yep|yeah|ok|okay|sure|send( it)?|go ahead|approved?|haan|haa|ji|bhej(o|do)?|theek hai|ok bhejo)\b",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"\b(no|nope|cancel|stop|abort|nahi|nahin|ruk|mat|reject(ed)?)\b",
    re.IGNORECASE,
)
_CHANGE_RE = re.compile(
    r"(?:change to|actually|naam|change it to|replace with|update to)\s+(?:[\"']?)([A-Za-z0-9 .&'-]{2,80}?)(?:[\"']?)(?:\s+kar do|\s+kr do|\s*$)",
    re.IGNORECASE,
)


_QUERY_LAST_INVOICE_PATTERNS = (
    re.compile(r"\b(last|previous|pichla|pichli|recent)\s+(invoice|bill)\b.*\b(amount|total|how\s+much|paid|kitna|kitne)\b", re.I),
    re.compile(r"\b(amount|total|how\s+much|kitna|kitne)\b.*\b(last|previous|pichla|pichli|recent)\s+(invoice|bill)\b", re.I),
)


_GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hello+|hey+|namaste|namaskar|salaam|salam|good\s+(morning|afternoon|evening)|kaise\s+ho|kya\s+haal)[\s!.?,]*$",
    re.IGNORECASE,
)


_START_INVOICE_RE = re.compile(
    r"\b(send|create|generate|trigger|make|fire|bana\s*do|bhej(?:o|do)|kar\s*do)\b"
    r"[^\n]{0,40}?\b(invoice|bill)\b"
    r"|\b(invoice|bill)\b[^\n]{0,40}?\b(send|generate|trigger|create|bana\s*do|bhej(?:o|do))\b",
    re.IGNORECASE,
)
_TARGET_MONTH_RE = re.compile(
    r"\b(this\s+month|current(?:\s+month)?|previous\s+month|last\s+month|next\s+month)\b"
    r"|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"(?:\s+(\d{4}))?",
    re.IGNORECASE,
)


def _heuristic_query(text: str) -> QueryIntent:
    if any(p.search(text) for p in _QUERY_LAST_INVOICE_PATTERNS):
        return QueryIntent(intent="last_invoice_amount")
    if _START_INVOICE_RE.search(text):
        m = _TARGET_MONTH_RE.search(text)
        if m:
            if m.group(1):  # relative phrase
                target = m.group(1).lower()
            else:           # month name (+ optional year)
                target = m.group(2).lower() + (f" {m.group(3)}" if m.group(3) else "")
        else:
            target = "current"
        return QueryIntent(intent="start_invoice", target_month=target)
    if _GREETING_RE.match(text):
        return QueryIntent(intent="greeting")
    return QueryIntent(intent="none")


_SUMMARY_AMOUNT_RE = re.compile(
    r"\b(?:amount|total|rate|rs|inr|rupees?)\b[^\d]*(\d[\d,]*)\s*(k|lakh|lac|cr|crore)?",
    re.IGNORECASE,
)
_SUMMARY_BARE_AMOUNT_RE = re.compile(r"\b(\d[\d,]{3,})\b")  # standalone large number
_SUMMARY_SUFFIX_AMOUNT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(k|lakh|lac|cr|crore)\b",
    re.IGNORECASE,
)
_SUMMARY_ATTENDANCE_RE = re.compile(
    r"\b(?:attendance|present|days?)\b[^\d]*(\d{1,3})|\b(\d{1,3})\s*days?\b",
    re.IGNORECASE,
)
_SUMMARY_PROJECT_RE = re.compile(
    r"(?:change\s+(?:the\s+)?project(?:\s+name)?\s+to|project\s+(?:naam\s+)?(?:ko\s+)?)\s+([A-Za-z0-9 .&'-]{2,80}?)(?:\s*(?:kar do|kr do|hai|$))",
    re.IGNORECASE,
)


def _scale(num_str: str, suffix: Optional[str]) -> int:
    n = float(num_str.replace(",", ""))
    if not suffix:
        return int(n)
    s = suffix.lower()
    if s == "k":
        return int(n * 1_000)
    if s in ("lakh", "lac"):
        return int(n * 100_000)
    if s in ("cr", "crore"):
        return int(n * 10_000_000)
    return int(n)


def _heuristic_summary(text: str) -> SummaryReply:
    amount: Optional[int] = None
    attendance: Optional[int] = None
    project: Optional[str] = None

    m = _SUMMARY_AMOUNT_RE.search(text)
    if m:
        amount = _scale(m.group(1), m.group(2))
    elif (m := _SUMMARY_SUFFIX_AMOUNT_RE.search(text)):
        amount = _scale(m.group(1), m.group(2))
    elif (m := _SUMMARY_BARE_AMOUNT_RE.search(text)):
        amount = int(m.group(1).replace(",", ""))

    m = _SUMMARY_ATTENDANCE_RE.search(text)
    if m:
        attendance = int(m.group(1) or m.group(2))

    m = _SUMMARY_PROJECT_RE.search(text)
    if m:
        project = m.group(1).strip()

    has_approval = bool(re.search(
        r"\b(approve[d]?|yes|yep|yeah|ok|okay|sure|send( it)?|go ahead|generate( it)?|haan|haa|ji|bhej(o|do)?|theek hai)\b",
        text, re.IGNORECASE,
    ))
    status = "approved" if has_approval else "change_requested"
    return SummaryReply(
        status=status,
        amount_inr=amount,
        attendance_days=attendance,
        project_name=project,
    )


def _heuristic_approval(text: str) -> ApprovalDecision:
    m = _CHANGE_RE.search(text)
    if m:
        return ApprovalDecision(status="change_requested", new_project_name=m.group(1).strip())
    if _REJECT_RE.search(text):
        return ApprovalDecision(status="rejected")
    if _APPROVE_RE.search(text):
        return ApprovalDecision(status="approved")
    # Last-ditch: treat as rejected so we never email by accident.
    return ApprovalDecision(status="rejected")


def _heuristic_project(text: str) -> ProjectReply:
    cleaned = re.sub(
        r"\b(project|ka|naam|hai|is|the|for|please|pls|likh do|likhdo|kr do|kar do|bhai)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.\"'")
    return ProjectReply(project_name=cleaned or text.strip())


# --------- Public parsers ---------


def parse_project_name(text: str, *, llm: Optional[ChatOllama] = None) -> ProjectReply:
    chat = (llm or make_chat()).with_structured_output(ProjectReply)
    messages = [SystemMessage(_PROJECT_SYSTEM), HumanMessage(text)]
    try:
        return chat.invoke(messages)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_project.first_attempt_failed", err=str(e))
    try:
        strict = [
            SystemMessage(_PROJECT_SYSTEM + "\n\nReturn ONLY valid JSON for ProjectReply."),
            HumanMessage(text),
        ]
        return chat.invoke(strict)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_project.fallback_to_heuristic", err=str(e))
        return _heuristic_project(text)


def parse_query_intent(text: str, *, llm: Optional[ChatOllama] = None) -> QueryIntent:
    chat = (llm or make_chat()).with_structured_output(QueryIntent)
    messages = [SystemMessage(_QUERY_SYSTEM), HumanMessage(text)]
    try:
        return chat.invoke(messages)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_query.first_attempt_failed", err=str(e))
    try:
        strict = [
            SystemMessage(_QUERY_SYSTEM + "\n\nReturn ONLY valid JSON for QueryIntent."),
            HumanMessage(text),
        ]
        return chat.invoke(strict)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_query.fallback_to_heuristic", err=str(e))
        return _heuristic_query(text)


def parse_summary_reply(text: str, *, llm: Optional[ChatOllama] = None) -> SummaryReply:
    chat = (llm or make_chat()).with_structured_output(SummaryReply)
    messages = [SystemMessage(_SUMMARY_SYSTEM), HumanMessage(text)]
    try:
        return chat.invoke(messages)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_summary.first_attempt_failed", err=str(e))
    try:
        strict = [
            SystemMessage(_SUMMARY_SYSTEM + "\n\nReturn ONLY valid JSON for SummaryReply."),
            HumanMessage(text),
        ]
        return chat.invoke(strict)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_summary.fallback_to_heuristic", err=str(e))
        return _heuristic_summary(text)


def parse_approval_reply(text: str, *, llm: Optional[ChatOllama] = None) -> ApprovalDecision:
    chat = (llm or make_chat()).with_structured_output(ApprovalDecision)
    messages = [SystemMessage(_APPROVAL_SYSTEM), HumanMessage(text)]
    try:
        return chat.invoke(messages)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_approval.first_attempt_failed", err=str(e))
    try:
        strict = [
            SystemMessage(_APPROVAL_SYSTEM + "\n\nReturn ONLY valid JSON for ApprovalDecision."),
            HumanMessage(text),
        ]
        return chat.invoke(strict)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("llm.parse_approval.fallback_to_heuristic", err=str(e))
        return _heuristic_approval(text)
