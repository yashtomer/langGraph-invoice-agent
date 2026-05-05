"""Post-generation amount-verification: reply must not quote any INR amount
that didn't appear in get_invoice / compare_invoices tool output this turn."""
from __future__ import annotations

import json
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from invoice_agent.qa.agent import _amounts_verified


def _msgs_with_tool_result(name: str, content_dict: dict):
    return [
        HumanMessage("hi"),
        AIMessage("calling tool"),
        ToolMessage(content=json.dumps(content_dict), tool_call_id="x", name=name),
    ]


def test_no_amounts_in_reply_passes():
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("hi there, all good", msgs) is True


def test_reply_amount_matches_tool_passes():
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("It was 200000 last month.", msgs) is True
    assert _amounts_verified("It was 2,00,000 last month.", msgs) is True  # comma form


def test_reply_amount_not_in_tool_fails():
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("It was 50000 last month.", msgs) is False


def test_lakh_phrase_in_reply_passes_when_tool_has_numeric():
    # Spec open question 1: accept either numeric or lakh form.
    msgs = _msgs_with_tool_result("get_invoice", {"amount_inr": 200000})
    assert _amounts_verified("It was 2 lakh.", msgs) is True


def test_web_search_results_excluded_from_whitelist():
    msgs = [
        HumanMessage("hi"),
        AIMessage("searching"),
        ToolMessage(
            content=json.dumps([{"title": "T", "url": "u", "snippet": "rate is 18000"}]),
            tool_call_id="x",
            name="web_search",
        ),
    ]
    # 18000 is in web_search snippet but NOT in invoice tools — must fail.
    assert _amounts_verified("The rate is 18000.", msgs) is False


def test_no_tool_messages_passes_when_reply_has_no_amounts():
    msgs = [HumanMessage("hello"), AIMessage("hi!")]
    assert _amounts_verified("hi!", msgs) is True


def test_no_tool_messages_fails_when_reply_has_amount():
    # LLM hallucinated a number with no tool to back it up.
    msgs = [HumanMessage("hello"), AIMessage("you billed 50000")]
    assert _amounts_verified("you billed 50000", msgs) is False


class SlowLLM(RunnableLambda):
    def __init__(self):
        super().__init__(self._slow)
    def bind_tools(self, *a, **kw):
        return self
    def _slow(self, messages, **_):
        time.sleep(2.0)
        return AIMessage("eventually")


def test_answer_times_out_with_slow_llm(tmp_settings, monkeypatch):
    from invoice_agent.db import init_db
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: SlowLLM())
    # Force a tight timeout so the test finishes fast.
    monkeypatch.setattr(
        type(tmp_settings),
        "qa_invoke_timeout_seconds",
        property(lambda self: 0.1),
        raising=False,
    )

    reply = agent_mod.answer("hi", "91XXX", settings=tmp_settings)
    assert reply == agent_mod._FALLBACK_STRING
