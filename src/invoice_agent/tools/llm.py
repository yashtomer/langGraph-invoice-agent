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
