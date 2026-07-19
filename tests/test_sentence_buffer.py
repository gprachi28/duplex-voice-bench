"""Sentence buffer unit tests -- see design.md's "LLM -> TTS handoff" section.

Flush trigger: a sentence-boundary char ([. ? ! ,]) AND the segment
accumulated since the last flush is >= 20 chars (the ElevenLabs-recommended
minimum flush size). Boundaries hit before 20 chars are treated as regular
text and folded into the next segment instead of firing a premature flush.
"""

from agent.sentence_buffer import SentenceBuffer


def test_push_returns_nothing_when_under_min_length_with_no_boundary():
    buf = SentenceBuffer()
    assert buf.push("Hi") == []


def test_push_does_not_flush_on_boundary_before_min_length():
    buf = SentenceBuffer()
    assert buf.push("Hi,") == []


def test_push_flushes_on_boundary_once_min_length_reached():
    buf = SentenceBuffer()
    assert buf.push("This is a longer sentence.") == ["This is a longer sentence."]


def test_push_accumulates_across_multiple_pushes_before_flushing():
    buf = SentenceBuffer()
    assert buf.push("This is ") == []
    assert buf.push("a longer") == []
    assert buf.push(" sentence.") == ["This is a longer sentence."]


def test_push_starts_new_segment_after_a_flush():
    buf = SentenceBuffer()
    buf.push("This is a longer sentence.")
    assert buf.push(" Hi") == []


def test_push_handles_two_boundaries_past_min_length_in_one_token():
    buf = SentenceBuffer()
    result = buf.push("This is one sentence. This is another sentence.")
    assert result == ["This is one sentence.", "This is another sentence."]


def test_flush_returns_remaining_buffer_content():
    buf = SentenceBuffer()
    buf.push("too short")
    assert buf.flush() == "too short"


def test_flush_returns_none_when_buffer_empty():
    buf = SentenceBuffer()
    assert buf.flush() is None


def test_flush_returns_none_after_a_push_that_already_flushed_everything():
    buf = SentenceBuffer()
    buf.push("This is a longer sentence.")
    assert buf.flush() is None
