"""LangGraph StateGraph wiring for the invoice flow.

The graph encodes the full human-in-the-loop conversation:

    ask_project_name  --interrupt-->  parse_project_reply
                                              |
                                      generate_pdf
                                              |
                                       send_preview --interrupt--> parse_approval
                                                                       |
                       +-----------------------------------------------+
                       |                       |                       |
                  approved              rejected              change_requested
                       |                       |                       |
                email_accounts        notify_cancelled        generate_pdf  (loop)
                       |                       |
                confirm_to_user                END
                       |
                      END

Nodes are imported from ``invoice_agent.nodes`` so the graph wiring stays small
and the node implementations can be tested in isolation.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import Settings, get_settings
from .nodes.ask_project import ask_project_name
from .nodes.email_accounts import email_accounts
from .nodes.generate_pdf import generate_pdf
from .nodes.notify import confirm_to_user, notify_cancelled
from .nodes.parse_approval import parse_approval
from .nodes.parse_project import parse_project_reply
from .nodes.send_preview import send_preview
from .state import InvoiceState


def _route_after_approval(state: InvoiceState) -> str:
    status = state.get("approval_status")
    if status == "approved":
        return "email_accounts"
    if status == "change_requested":
        return "generate_pdf"
    # Default to rejected if missing or unknown — fail safe.
    return "notify_cancelled"


def build_graph_definition() -> StateGraph:
    """Build the StateGraph (without compiling) — useful for inspection/testing."""
    g: StateGraph = StateGraph(InvoiceState)

    g.add_node("ask_project_name", ask_project_name)
    g.add_node("parse_project_reply", parse_project_reply)
    g.add_node("generate_pdf", generate_pdf)
    g.add_node("send_preview", send_preview)
    g.add_node("parse_approval", parse_approval)
    g.add_node("email_accounts", email_accounts)
    g.add_node("confirm_to_user", confirm_to_user)
    g.add_node("notify_cancelled", notify_cancelled)

    g.set_entry_point("ask_project_name")

    # Linear edges
    g.add_edge("ask_project_name", "parse_project_reply")
    g.add_edge("parse_project_reply", "generate_pdf")
    g.add_edge("generate_pdf", "send_preview")
    g.add_edge("send_preview", "parse_approval")

    # Branch after parsing approval
    g.add_conditional_edges(
        "parse_approval",
        _route_after_approval,
        {
            "email_accounts": "email_accounts",
            "notify_cancelled": "notify_cancelled",
            "generate_pdf": "generate_pdf",
        },
    )

    g.add_edge("email_accounts", "confirm_to_user")
    g.add_edge("confirm_to_user", END)
    g.add_edge("notify_cancelled", END)

    return g


@contextmanager
def open_checkpointer(settings: Optional[Settings] = None) -> Iterator[SqliteSaver]:
    """Open a SqliteSaver bound to the configured DB path.

    Use as a context manager so the underlying connection is closed cleanly.
    """
    s = settings or get_settings()
    with SqliteSaver.from_conn_string(str(s.db_path)) as saver:
        yield saver


def compile_graph(checkpointer: SqliteSaver, *, interrupt_after: Optional[list[str]] = None):
    """Compile the graph with the given checkpointer.

    We interrupt AFTER ``ask_project_name`` and ``send_preview`` so the graph
    pauses, the WhatsApp prompt goes out, and the FastAPI webhook can resume
    the thread once the user replies.
    """
    interrupt_after = interrupt_after or ["ask_project_name", "send_preview"]
    g = build_graph_definition()
    return g.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def thread_config(invoice_month: str) -> dict:
    """Stable thread-id per invoice month so re-runs resume the same conversation."""
    return {"configurable": {"thread_id": f"invoice-{invoice_month}"}}
