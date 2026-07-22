"""Unit tests for the TurnMetrics wiring through _dispatch_gate_result and
_synthesize_and_play -- see agent/metrics.py and design.md's "Observability"
section. Fake STT/LLM/TTS/pump, no real audio, no network.
"""

import asyncio
import json

import numpy as np
import pytest

import agent.worker as worker_module
from agent.playback import PlaybackState
from agent.tts import TTSResult, WordTiming
from agent.turn_gate import Fire, ForceFire
from agent.worker import (
    FALLBACK_REPLY,
    ActiveReply,
    _dispatch_gate_result,
    _synthesize_and_play,
)


class FakeSTT:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: np.ndarray) -> str:
        return self._text


class FakeLLM:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def stream_chat(self, messages):
        for chunk in self._chunks:
            yield chunk


class FakeTTS:
    sample_rate = 24_000

    def synthesize(self, text: str) -> TTSResult:
        return TTSResult(np.ones(len(text), dtype=np.float32), [])


class FakePump:
    def __init__(self) -> None:
        self.submitted: list[tuple[np.ndarray, list[WordTiming]]] = []
        self.state = PlaybackState.PLAYING

    async def submit(self, audio: np.ndarray, words: list[WordTiming]) -> None:
        self.submitted.append((audio, words))

    def pause(self) -> float:
        return 0.0

    def resume(self, rewind_words: int = 2) -> int:
        return 0

    def stop(self) -> None:
        self.state = PlaybackState.STOPPED

    def reset_for_new_reply(self) -> None:
        self.state = PlaybackState.PLAYING


@pytest.fixture
def metrics_path(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(worker_module, "METRICS_LOG_PATH", str(path))
    return path


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_dispatch_with_t0_writes_one_metrics_record(metrics_path):
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            FakeLLM(["Hello!"]),
            FakeTTS(),
            [],
            asyncio.Lock(),
            ActiveReply(room="room-1"),
            FakePump(),
            t0=100.0,
            smart_turn_prob=0.95,
        )
    )

    records = _read_records(metrics_path)
    assert len(records) == 1
    record = records[0]
    assert record["room"] == "room-1"
    assert record["turn_id"] == "room-1-1"
    assert record["forced"] is False
    assert record["interrupted"] is False
    assert record["smart_turn_prob"] == 0.95
    assert record["end_of_turn_s"] is not None  # t1 - t0, both stages reached
    assert record["ttfa_s"] is not None  # t5 - t0, TTS actually ran
    assert record["prompt_version"] == worker_module.SYSTEM_PROMPT_VERSION


def test_dispatch_on_stt_repetition_loop_skips_llm_and_speaks_fallback(metrics_path):
    class ExplodingLLM:
        async def stream_chat(self, messages):
            raise AssertionError("LLM must not be called on a repetition-loop transcript")
            yield  # pragma: no cover -- makes this an async generator

    pump = FakePump()
    history: list[dict[str, str]] = []
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT(" ".join(["should"] * 50)),
            ExplodingLLM(),
            FakeTTS(),
            history,
            asyncio.Lock(),
            ActiveReply(room="room-1"),
            pump,
            t0=100.0,
            smart_turn_prob=0.95,
        )
    )

    assert history == []  # neither the garbage transcript nor a fallback reply
    assert len(pump.submitted) == 1  # the fallback was spoken via the normal TTS path

    record = _read_records(metrics_path)[0]
    assert record["stt_repetition_detected"] is True
    assert record["llm_ttft_s"] is None  # LLM stage never reached


def test_dispatch_without_t0_writes_no_metrics_record(metrics_path):
    # Existing call sites (and tests) that don't pass t0 shouldn't produce
    # metrics I/O -- there's nothing meaningful to record without a start time.
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            FakeLLM(["Hello!"]),
            FakeTTS(),
            [],
            asyncio.Lock(),
            ActiveReply(),
            FakePump(),
        )
    )
    assert not metrics_path.exists()


def test_dispatch_marks_forced_from_force_fire_result(metrics_path):
    asyncio.run(
        _dispatch_gate_result(
            ForceFire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            FakeLLM(["Hello!"]),
            FakeTTS(),
            [],
            asyncio.Lock(),
            ActiveReply(room="room-1"),
            FakePump(),
            t0=0.0,
        )
    )
    assert _read_records(metrics_path)[0]["forced"] is True


def test_dispatch_marks_interrupted_when_stt_fails_mid_turn(metrics_path):
    class FailingSTT:
        def transcribe(self, audio: np.ndarray) -> str:
            raise RuntimeError("stt failed")

    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FailingSTT(),
            FakeLLM(["unused"]),
            FakeTTS(),
            [],
            asyncio.Lock(),
            ActiveReply(room="room-1"),
            FakePump(),
            t0=0.0,
        )
    )
    record = _read_records(metrics_path)[0]
    assert record["transcription_s"] is None  # never completed
    assert record["ttfa_s"] is None


def test_synthesize_and_play_stamps_t4_and_t5_once_per_turn():
    from agent.metrics import TurnMetrics

    tts = FakeTTS()
    pump = FakePump()
    active_reply = ActiveReply()
    metrics = TurnMetrics(turn_id="t", room="r", combination_id="c")

    asyncio.run(_synthesize_and_play(tts, "First.", pump, active_reply, metrics))
    first_t4, first_t5 = metrics.t4, metrics.t5
    assert first_t4 is not None
    assert first_t5 is not None

    asyncio.run(_synthesize_and_play(tts, "Second.", pump, active_reply, metrics))
    assert metrics.t4 == first_t4  # not overwritten by the second segment
    assert metrics.t5 == first_t5


def test_synthesize_and_play_without_metrics_is_a_noop_for_metrics():
    tts = FakeTTS()
    pump = FakePump()
    active_reply = ActiveReply()
    # Existing call sites that don't pass `metrics` must keep working unchanged.
    asyncio.run(_synthesize_and_play(tts, "Hello.", pump, active_reply))
    assert pump.submitted  # normal playback still happened
