"""Q&A agent — wires LLM (stubbed) + tools + memory + safety."""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from invoice_agent.db import init_db, mark_sent


class FakeChatLLM(RunnableLambda):
    """Mimics ChatOllama enough for create_react_agent to drive a turn.

    Inherits from RunnableLambda so it satisfies create_react_agent's
    `prompt | model` pipe requirement. .bind_tools returns self so the
    agent's tool-binding doesn't drop us back to a non-Runnable.
    """

    def __init__(self, responses: list[AIMessage]):
        super().__init__(self._invoke_impl)
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools, **_):
        return self

    def _invoke_impl(self, messages, **_):
        # When invoked via Runnable .invoke, messages may be a PromptValue
        # or a list[BaseMessage]; the agent calls our model with the post-
        # prompt messages list.
        msgs = messages.to_messages() if hasattr(messages, "to_messages") else list(messages)
        self.calls.append(msgs)
        if not self._responses:
            return AIMessage("done")
        return self._responses.pop(0)


def _ai_tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
    )


def test_answer_invokes_get_invoice_and_returns_reply(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    mark_sent(
        "2026-04",
        project_name="Madabranding",
        amount_inr=200000,
        attendance_days=30,
        invoice_number="INV-2026-04-001",
        settings=tmp_settings,
    )

    fake = FakeChatLLM([
        _ai_tool_call("get_invoice", {"month": "2026-04"}),
        AIMessage("April was 200000 for Madabranding."),
    ])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    reply = agent_mod.answer("what did i bill in april?", "91XXX", settings=tmp_settings)
    assert "200000" in reply
    assert "Madabranding" in reply


def test_answer_persists_turn_to_chat_memory(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.memory import load_recent_turns
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    fake = FakeChatLLM([AIMessage("hey")])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    agent_mod.answer("hi", "91XXX", settings=tmp_settings)
    msgs = load_recent_turns("91XXX", n=6, settings=tmp_settings)
    assert [m.content for m in msgs] == ["hi", "hey"]


def test_answer_passes_history_into_agent(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.memory import append_turn
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    append_turn("91XXX", "earlier-q", "earlier-a", settings=tmp_settings)
    fake = FakeChatLLM([AIMessage("got it")])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    agent_mod.answer("follow-up", "91XXX", settings=tmp_settings)
    # First call's input should include both prior turns + the new HumanMessage.
    first_call = fake.calls[0]
    contents = [m.content for m in first_call if isinstance(m, (HumanMessage, AIMessage))]
    assert "earlier-q" in contents
    assert "earlier-a" in contents
    assert "follow-up" in contents


def test_answer_returns_fallback_when_agent_raises(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)

    class BoomLLM(RunnableLambda):
        def __init__(self):
            super().__init__(self._boom)
        def bind_tools(self, *a, **kw):
            return self
        def _boom(self, *a, **kw):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: BoomLLM())
    reply = agent_mod.answer("hi", "91XXX", settings=tmp_settings)
    assert reply == agent_mod._FALLBACK_STRING


def test_answer_swaps_to_fallback_when_amount_unverified(tmp_settings, monkeypatch):
    from invoice_agent.qa import agent as agent_mod
    from invoice_agent.qa.tools import reset_web_search_budget
    reset_web_search_budget()

    init_db(tmp_settings)
    # Stub: LLM emits a number with no tool call to back it up.
    fake = FakeChatLLM([AIMessage("Your last invoice was 999999.")])
    monkeypatch.setattr(agent_mod, "make_chat", lambda *a, **kw: fake)

    reply = agent_mod.answer("how much?", "91XXX", settings=tmp_settings)
    assert reply == agent_mod._FALLBACK_STRING
