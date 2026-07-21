"""Unit tests for PlaybackPump -- real-time-paced audio submission with
pause/resume-with-rewind for barge-in. Fake capture_frame/clear_queue
callables, no real LiveKit/audio.

play_audio previously pushed a whole TTS segment into rtc.AudioSource.
capture_frame() in one call -- no way to pause mid-segment and resume
later, since capture_frame's queue is a one-way destructive FIFO.
PlaybackPump submits audio in small fixed-size frames instead, so "pause"
is "stop submitting more frames" and "resume" is "keep submitting from a
rewound sample offset" -- using Kokoro's own word-level alignment
(WordTiming) already computed during synthesis to choose a natural
rewind point instead of resuming mid-syllable. See
docs/superpowers/specs/2026-07-20-barge-in_heard_text.md.
"""

import asyncio

import numpy as np
import pytest

from agent.playback import PlaybackPump, PlaybackState, _compute_rewind
from agent.tts import WordTiming

SR = 24_000
FRAME_SAMPLES = SR * 20 // 1000  # 480 samples at 20ms/frame


def _fake_capture(sink: list[np.ndarray]):
    async def capture(frame: np.ndarray) -> None:
        sink.append(frame.copy())

    return capture


def _fake_clear(sink: list[str]):
    def clear() -> None:
        sink.append("clear_queue")

    return clear


def test_submit_slices_into_fixed_frames():
    captured: list[np.ndarray] = []
    pump = PlaybackPump(_fake_capture(captured), _fake_clear([]), SR)
    audio = np.ones(FRAME_SAMPLES * 3 + 100, dtype=np.float32)

    asyncio.run(pump.submit(audio, []))

    assert len(captured) == 4  # 3 full frames + 1 partial (100 samples)
    assert all(len(f) <= FRAME_SAMPLES for f in captured)
    assert sum(len(f) for f in captured) == len(audio)


def test_pause_stops_submission_and_clears_queue():
    captured: list[np.ndarray] = []
    clears: list[str] = []
    pump = PlaybackPump(_fake_capture(captured), _fake_clear(clears), SR)

    async def _run():
        await pump.submit(np.ones(FRAME_SAMPLES * 2, dtype=np.float32), [])
        paused_at = pump.pause()
        # Nothing should be submitted after pause even if more audio
        # arrives (the LLM/TTS pipeline upstream keeps running).
        await pump.submit(np.ones(FRAME_SAMPLES, dtype=np.float32), [])
        return paused_at

    paused_at = asyncio.run(_run())

    assert len(captured) == 2  # only the pre-pause frames
    assert clears == ["clear_queue"]
    assert paused_at == pytest.approx(2 * FRAME_SAMPLES / SR)


def test_rewind_computes_index_n_words_back():
    words = [
        WordTiming(f"W{i} ", start=i * 0.1, end=(i + 1) * 0.1) for i in range(5)
    ]
    # Paused mid-W4 (index 4). Rewind 2 words back -> index 2 (W2).
    rewind_s, keep_count = _compute_rewind(words, paused_at_s=0.45, n_words=2)
    assert keep_count == 2
    assert rewind_s == pytest.approx(0.2)


def test_rewind_clamps_to_zero_near_start():
    words = [WordTiming("W0 ", start=0.0, end=0.1), WordTiming("W1 ", start=0.1, end=0.2)]
    rewind_s, keep_count = _compute_rewind(words, paused_at_s=0.15, n_words=2)
    assert keep_count == 0
    assert rewind_s == 0.0


def test_rewind_with_no_words_yet_returns_zero():
    rewind_s, keep_count = _compute_rewind([], paused_at_s=0.0, n_words=2)
    assert keep_count == 0
    assert rewind_s == 0.0


def test_resume_resubmits_from_rewound_offset():
    captured: list[np.ndarray] = []
    pump = PlaybackPump(_fake_capture(captured), _fake_clear([]), SR)
    words = [WordTiming(f"W{i} ", start=i * 0.1, end=(i + 1) * 0.1) for i in range(10)]
    audio = np.arange(len(words) * int(0.1 * SR), dtype=np.float32)

    async def _run():
        await pump.submit(audio, words)
        pump.pause()
        captured.clear()  # isolate what gets (re)submitted by resume
        # Force the pump to believe only half the segment had actually
        # played, simulating a pause landing mid-segment rather than
        # after the whole (short, instantly-submitted) segment finished.
        pump._submitted_samples = int(0.5 * SR)
        keep_count = pump.resume(rewind_words=2)
        await asyncio.sleep(0.05)  # let the background drain task run
        return keep_count

    keep_count = asyncio.run(_run())

    # paused_at=0.5s -> current word index 5 (W5) -> rewind 2 -> index 3 (W3)
    assert keep_count == 3
    # Resume re-submits from word 3's start (0.3s), not from the 0.5s
    # pause point -- so the resumed submission covers more samples than
    # the 0.5s-to-end remainder (0.5s worth) would have.
    resubmitted_samples = sum(len(f) for f in captured)
    assert resubmitted_samples == pytest.approx(len(audio) - int(0.3 * SR))
    assert pump._submitted_samples == len(pump._audio)  # fully drained, uninterrupted


def test_stop_drops_retained_segments():
    captured: list[np.ndarray] = []
    clears: list[str] = []
    pump = PlaybackPump(_fake_capture(captured), _fake_clear(clears), SR)

    async def _run():
        await pump.submit(np.ones(FRAME_SAMPLES, dtype=np.float32), [])
        pump.stop()

    asyncio.run(_run())

    assert clears == ["clear_queue"]
    assert pump._audio.size == 0
    assert pump._state == PlaybackState.STOPPED


def test_reset_for_new_reply_drops_audio_and_resumes_playing_state():
    pump = PlaybackPump(_fake_capture([]), _fake_clear([]), SR)

    async def _run():
        await pump.submit(np.ones(FRAME_SAMPLES, dtype=np.float32), [])
        pump.stop()
        pump.reset_for_new_reply()

    asyncio.run(_run())

    assert pump._audio.size == 0
    assert pump._state == PlaybackState.PLAYING
