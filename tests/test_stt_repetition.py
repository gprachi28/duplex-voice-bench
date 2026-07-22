"""Unit tests for is_repetition_loop -- a cheap post-hoc filter for
Whisper's known decoder-repetition-loop failure mode. Confirmed live:
'should' repeated ~100 times, and separately 12 times, both during
barge-in on already-playing TTS audio -- see benchmarks/experiments.md.
Pure string logic, no model involved, so this runs on every platform
(unlike test_stt_backend.py, which is Apple-Silicon-only).
"""

from agent.stt import is_repetition_loop


def test_detects_a_long_run_of_the_same_word():
    text = " ".join(["should"] * 100)
    assert is_repetition_loop(text) is True


def test_detects_a_short_run_at_the_default_threshold():
    text = " ".join(["should"] * 12)
    assert is_repetition_loop(text) is True


def test_does_not_flag_a_normal_sentence():
    assert is_repetition_loop("What is the capital of Australia?") is False


def test_does_not_flag_brief_natural_repetition():
    # "no no no" is real emphatic speech, well under the loop threshold.
    assert is_repetition_loop("No no no, that's not what I meant.") is False


def test_is_case_insensitive():
    text = " ".join(["Should"] * 8)
    assert is_repetition_loop(text) is True


def test_ignores_trailing_punctuation_per_word():
    text = " ".join(["should,"] * 8)
    assert is_repetition_loop(text) is True


def test_empty_text_is_not_a_loop():
    assert is_repetition_loop("") is False


def test_respects_a_custom_threshold():
    text = " ".join(["should"] * 4)
    assert is_repetition_loop(text, min_repeats=5) is False
    assert is_repetition_loop(text, min_repeats=4) is True
