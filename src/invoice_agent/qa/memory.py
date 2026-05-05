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
