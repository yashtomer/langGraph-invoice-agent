"""Post-generation amount-verification: reply must not quote any INR amount
that didn't appear in get_invoice / compare_invoices tool output this turn."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
