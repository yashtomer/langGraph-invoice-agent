"""chat_memory CRUD: append, load, trim, multi-user isolation."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from invoice_agent.db import init_db
from invoice_agent.qa.memory import append_turn, load_recent_turns, trim_old


def test_empty_load_returns_empty_list(tmp_settings):
    init_db(tmp_settings)
    assert load_recent_turns("91XXX", n=6, settings=tmp_settings) == []


def test_append_then_load_roundtrip(tmp_settings):
    init_db(tmp_settings)
    append_turn("91A", "hello", "hi there", settings=tmp_settings)
    msgs = load_recent_turns("91A", n=6, settings=tmp_settings)
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "hello"
    assert isinstance(msgs[1], AIMessage) and msgs[1].content == "hi there"


def test_load_returns_last_n_in_oldest_first_order(tmp_settings):
    init_db(tmp_settings)
    for i in range(8):
        append_turn("91A", f"u{i}", f"a{i}", settings=tmp_settings)
    msgs = load_recent_turns("91A", n=3, settings=tmp_settings)
    # 3 turns × 2 msgs = 6, oldest first → u5, a5, u6, a6, u7, a7
    assert [m.content for m in msgs] == ["u5", "a5", "u6", "a6", "u7", "a7"]


def test_multi_user_isolation(tmp_settings):
    init_db(tmp_settings)
    append_turn("91A", "a-user", "a-bot", settings=tmp_settings)
    append_turn("91B", "b-user", "b-bot", settings=tmp_settings)
    a = load_recent_turns("91A", n=6, settings=tmp_settings)
    b = load_recent_turns("91B", n=6, settings=tmp_settings)
    assert [m.content for m in a] == ["a-user", "a-bot"]
    assert [m.content for m in b] == ["b-user", "b-bot"]


def test_turn_idx_is_monotonic(tmp_settings):
    from invoice_agent.db import connect
    init_db(tmp_settings)
    for i in range(3):
        append_turn("91A", f"u{i}", f"a{i}", settings=tmp_settings)
    with connect(tmp_settings) as conn:
        idxs = [r["turn_idx"] for r in conn.execute(
            "SELECT turn_idx FROM chat_memory WHERE user_phone='91A' ORDER BY turn_idx"
        )]
    # 3 turns × 2 rows each = 6 monotonic indices starting at 0
    assert idxs == [0, 1, 2, 3, 4, 5]


def test_trim_keeps_newest_n(tmp_settings):
    from invoice_agent.db import connect
    init_db(tmp_settings)
    for i in range(25):
        append_turn("91A", f"u{i}", f"a{i}", settings=tmp_settings)
    trim_old("91A", keep=5, settings=tmp_settings)  # 5 turns = 10 rows
    with connect(tmp_settings) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM chat_memory WHERE user_phone='91A'"
        ).fetchone()["c"]
    assert n == 10
    # Verify the kept rows are the newest (turns 20..24)
    msgs = load_recent_turns("91A", n=5, settings=tmp_settings)
    assert [m.content for m in msgs] == [
        "u20", "a20", "u21", "a21", "u22", "a22", "u23", "a23", "u24", "a24"
    ]
