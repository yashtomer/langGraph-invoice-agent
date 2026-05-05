"""Settings additions for the QA agent."""
from __future__ import annotations

from invoice_agent.config import Settings


def test_qa_settings_defaults():
    s = Settings()
    assert s.qa_chat_memory_turns == 6
    assert s.qa_web_search_max_calls_per_turn == 5
    assert s.qa_web_search_daily_cap == 0
    assert s.qa_invoke_timeout_seconds == 30.0
    assert s.tavily_api_key.get_secret_value() == "test-tavily-key"
