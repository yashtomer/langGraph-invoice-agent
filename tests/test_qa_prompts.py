"""Sanity check that QA_SYSTEM contains the load-bearing rules."""
from invoice_agent.qa.prompts import QA_SYSTEM


def test_prompt_formats_with_company():
    rendered = QA_SYSTEM.format(company="Acme")
    assert "Acme" in rendered


def test_prompt_mentions_all_three_tools():
    assert "get_invoice" in QA_SYSTEM
    assert "compare_invoices" in QA_SYSTEM
    assert "web_search" in QA_SYSTEM


def test_prompt_has_financial_safety_rule():
    rendered = QA_SYSTEM.lower()
    # Some form of "quote exactly" + "never invent"
    assert "exactly" in rendered or "verbatim" in rendered
    assert "invent" in rendered or "make up" in rendered or "fabricate" in rendered


def test_prompt_says_dont_call_yourself_a_bot():
    assert "bot" in QA_SYSTEM.lower()
