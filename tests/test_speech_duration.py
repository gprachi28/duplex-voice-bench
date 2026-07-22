"""Unit tests for _speech_duration_since -- see benchmarks/experiments.md:
the installed livekit-plugins-silero hardcodes VADEvent.speech_duration to
0.0 on every END_OF_SPEECH event (it zeroes pub_speech_duration right
before constructing the event -- site-packages/livekit/plugins/silero/vad.py),
so the SPEECH_END log line needs its own wall-clock measurement instead.
"""

from agent.worker import _speech_duration_since


def test_returns_elapsed_time_since_speech_started():
    assert _speech_duration_since(started_at=10.0, now=12.5) == 2.5


def test_returns_zero_when_no_speech_start_was_recorded():
    assert _speech_duration_since(started_at=None, now=12.5) == 0.0
