"""Unit tests for the per-room conversation history seed -- see
benchmarks/experiments.md for the bug this fixes: with no system message,
the LLM produced overly long replies and, on one turn, drifted into Hindi
(which Kokoro's English-only TTS then rendered as garbage audio).
"""

from agent.worker import SYSTEM_PROMPT, _new_history


def test_new_history_starts_with_a_system_message():
    history = _new_history()

    assert history[0]["role"] == "system"
    assert history[0]["content"] == SYSTEM_PROMPT


def test_new_history_has_no_user_or_assistant_turns_yet():
    history = _new_history()

    assert len(history) == 1


def test_system_prompt_instructs_concise_replies():
    assert "concise" in SYSTEM_PROMPT.lower() or "short" in SYSTEM_PROMPT.lower()


def test_system_prompt_instructs_english_only_replies():
    assert "english" in SYSTEM_PROMPT.lower()


def test_new_history_returns_a_fresh_list_each_call():
    a = _new_history()
    b = _new_history()
    a.append({"role": "user", "content": "hi"})

    assert b == [{"role": "system", "content": SYSTEM_PROMPT}]
