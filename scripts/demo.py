"""make demo entry point.

Runs the full LangGraph flow end-to-end with mocked WhatsApp + SMTP + LLM,
printing each interrupt boundary so the state machine can be inspected without
any external services. Useful both as a smoke test and as a learning aid for
LangGraph's interrupt + checkpointer model.

Usage:
    python scripts/demo.py
    python scripts/demo.py --month 2026-05 --scenario approve
    python scripts/demo.py --scenario change       # change-request loop
    python scripts/demo.py --scenario reject
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Pre-seed env so config doesn't fail on missing real values.
os.environ.setdefault("META_WA_PHONE_NUMBER_ID", "demo-id")
os.environ.setdefault("META_WA_ACCESS_TOKEN", "demo-token")
os.environ.setdefault("META_WA_VERIFY_TOKEN", "demo-verify")
os.environ.setdefault("META_WA_APP_SECRET", "demo-secret")
os.environ.setdefault("USER_WHATSAPP_NUMBER", "919999999999")
os.environ.setdefault("ACCOUNTS_EMAIL", "accounts@example.com")
os.environ.setdefault("CC_EMAIL", "yash@example.com")
os.environ.setdefault("INVOICE_AMOUNT_INR", "150000")
os.environ.setdefault("WEBHOOK_SHARED_SECRET", "demo")
os.environ.setdefault("SMTP_USER", "demo@example.com")
os.environ.setdefault("SMTP_APP_PASSWORD", "demo")

# Add src to path so 'python scripts/demo.py' works without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from invoice_agent.config import get_settings  # noqa: E402
from invoice_agent.graph import compile_graph, thread_config  # noqa: E402
from invoice_agent.state import initial_state  # noqa: E402
from invoice_agent.tools import llm as llm_mod  # noqa: E402
from invoice_agent.tools.llm import ApprovalDecision, ProjectReply  # noqa: E402


def _stub_llm(decisions: list[ApprovalDecision], project: str = "Birla Opus") -> None:
    queue = list(decisions)

    class _Stub:
        def with_structured_output(self, schema):
            class _Inner:
                def invoke(_self, _messages):
                    if schema is ProjectReply:
                        return ProjectReply(project_name=project)
                    return queue.pop(0)
            return _Inner()

    llm_mod.make_chat = lambda **_: _Stub()  # type: ignore[assignment]


def _stub_io(out_dir: Path):
    fake_wa = MagicMock()

    import invoice_agent.nodes.ask_project as ap
    import invoice_agent.nodes.email_accounts as ea
    import invoice_agent.nodes.generate_pdf as gp
    import invoice_agent.nodes.notify as nt
    import invoice_agent.nodes.send_preview as sp

    ap.WhatsAppClient = lambda *a, **kw: fake_wa  # type: ignore[assignment]
    sp.WhatsAppClient = lambda *a, **kw: fake_wa  # type: ignore[assignment]
    nt.WhatsAppClient = lambda *a, **kw: fake_wa  # type: ignore[assignment]

    def _fake_render(project_name, month, settings=None):
        p = out_dir / f"invoice_{month}_{project_name.replace(' ', '_').lower()}.pdf"
        p.write_bytes(b"%PDF-1.4 demo")
        return p

    gp.render_invoice_pdf = _fake_render  # type: ignore[assignment]

    sent = []

    def _fake_email(**kwargs):
        sent.append(kwargs)
        print(f"  [email] -> {kwargs['to']}  subject={kwargs['subject']!r}  attach={Path(kwargs['pdf_path']).name}")

    ea.send_invoice_email = _fake_email  # type: ignore[assignment]
    return fake_wa, sent


def _print_event(label: str, fake_wa, snapshot) -> None:
    print(f"\n>>> {label}")
    if fake_wa.send_template.called:
        last = fake_wa.send_template.call_args
        print(f"  [whatsapp:template] -> {last.kwargs.get('to') or last.args[0]}  body_params={last.kwargs.get('body_params')}")
        fake_wa.send_template.reset_mock()
    if fake_wa.send_document.called:
        last = fake_wa.send_document.call_args
        print(f"  [whatsapp:document] -> caption preview={last.kwargs.get('caption', '')[:80]!r}")
        fake_wa.send_document.reset_mock()
    if fake_wa.send_text.called:
        last = fake_wa.send_text.call_args
        print(f"  [whatsapp:text]     -> {last.kwargs.get('body', '')[:120]!r}")
        fake_wa.send_text.reset_mock()
    print(f"  next={snapshot.next}  project={snapshot.values.get('project_name')!r}  status={snapshot.values.get('approval_status')!r}")


SCENARIOS = {
    "approve": [ApprovalDecision(status="approved", new_project_name=None)],
    "reject": [ApprovalDecision(status="rejected", new_project_name=None)],
    "change": [
        ApprovalDecision(status="change_requested", new_project_name="DLF Camellias"),
        ApprovalDecision(status="approved", new_project_name=None),
    ],
}

SCENARIO_REPLIES = {
    "approve": ["Birla Opus", "haan bhej do"],
    "reject": ["Birla Opus", "nahi ruk"],
    "change": ["Birla Opus", "change to DLF Camellias", "ok bhejo"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2026-05")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="approve")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="invoice_agent_demo_"))
    db = tmp / "demo.sqlite"
    out_dir = tmp / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    s = get_settings()
    # Point settings at tmp paths just for this run
    type(s).out_dir = property(lambda self: out_dir)
    type(s).db_path = property(lambda self: db)

    _stub_llm(SCENARIOS[args.scenario])
    fake_wa, sent_emails = _stub_io(out_dir)

    cfg = thread_config(args.month)
    print(f"=== invoice-agent demo  scenario={args.scenario}  month={args.month}  tmp={tmp} ===")

    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = compile_graph(saver)

        graph.invoke(initial_state(args.month, "919999999999"), config=cfg)
        _print_event("ask_project_name fired (interrupt)", fake_wa, graph.get_state(cfg))

        for reply in SCENARIO_REPLIES[args.scenario]:
            print(f"\n  <<< user replies: {reply!r}")
            graph.update_state(cfg, {"user_reply_raw": reply})
            graph.invoke(None, config=cfg)
            snap = graph.get_state(cfg)
            label = "END reached" if not snap.next else f"paused before {snap.next[0]}"
            _print_event(label, fake_wa, snap)

        final = graph.get_state(cfg)

    print("\n=== final ===")
    print(f"  accounts_email_sent={final.values.get('accounts_email_sent')}")
    print(f"  approval_status={final.values.get('approval_status')}")
    print(f"  project_name={final.values.get('project_name')}")
    print(f"  pdf_path={final.values.get('pdf_path')}")
    print(f"  emails_dispatched={len(sent_emails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
