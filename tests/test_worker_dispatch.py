"""Pure unit tests for _dispatch_gate_result's LLM + TTS wiring -- fake
STT/LLM/TTS backends, no real audio, no network. Exercises history
bookkeeping, the sentence-buffer-gated TTS handoff, and the history_lock
serialization documented in worker.py's docstring: TurnGate doesn't prevent
a second Fire from spawning a second dispatch task while a prior one's LLM
stream is still running, so the lock must keep concurrent turns from
interleaving in the shared history list.
"""

import asyncio

import numpy as np
import pytest

from agent.turn_gate import Continue, Fire
from agent.worker import _dispatch_gate_result


class FakeSTT:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: np.ndarray) -> str:
        return self._text


class FakeLLM:
    def __init__(self, chunks: list[str], delay_s: float = 0.0) -> None:
        self._chunks = chunks
        self._delay_s = delay_s
        self.calls: list[list[dict[str, str]]] = []

    async def stream_chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        for chunk in self._chunks:
            yield chunk


class FakeTTS:
    sample_rate = 24_000

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self.synthesized: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        if self._fail:
            raise RuntimeError("synth failed")
        self.synthesized.append(text)
        return np.ones(len(text), dtype=np.float32)


def _recording_player(sink: list[np.ndarray]):
    async def play(audio: np.ndarray) -> None:
        sink.append(audio)

    return play


def test_continue_result_skips_stt_llm_and_history():
    llm = FakeLLM(["should not be called"])
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    asyncio.run(
        _dispatch_gate_result(
            Continue(),
            FakeSTT("unused"),
            llm,
            tts,
            history,
            asyncio.Lock(),
            _recording_player(played),
        )
    )
    assert llm.calls == []
    assert history == []
    assert tts.synthesized == []
    assert played == []


def test_fire_appends_user_then_assistant_turn():
    llm = FakeLLM(["Hel", "lo!"])
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            llm,
            tts,
            history,
            asyncio.Lock(),
            _recording_player(played),
        )
    )
    assert history == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_fire_sends_full_prior_history_to_llm():
    llm = FakeLLM(["ok"])
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = [
        {"role": "user", "content": "earlier turn"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("new turn"),
            llm,
            tts,
            history,
            asyncio.Lock(),
            _recording_player(played),
        )
    )
    assert llm.calls == [
        [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "new turn"},
        ]
    ]


def test_overlapping_dispatches_serialize_and_preserve_turn_order(caplog):
    slow_llm = FakeLLM(["slow-reply"], delay_s=0.2)
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()
    play = _recording_player(played)

    async def _run_both():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("first"),
                slow_llm,
                tts,
                history,
                lock,
                play,
            )
        )
        await asyncio.sleep(0.05)
        second = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("second"),
                slow_llm,
                tts,
                history,
                lock,
                play,
            )
        )
        await asyncio.gather(first, second)

    with caplog.at_level("WARNING"):
        asyncio.run(_run_both())

    assert history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "slow-reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "slow-reply"},
    ]
    assert any("overlapping turn" in r.message for r in caplog.records)


def test_fire_flushes_tts_mid_stream_on_sentence_boundary_in_order():
    llm = FakeLLM(["This is one sentence. ", "This is another sentence."])
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            llm,
            tts,
            history,
            asyncio.Lock(),
            _recording_player(played),
        )
    )
    assert tts.synthesized == ["This is one sentence.", "This is another sentence."]
    assert len(played) == 2


def test_fire_keeps_history_when_tts_synthesis_fails(caplog):
    llm = FakeLLM(["Hello there, this is a longer reply."])
    tts = FakeTTS(fail=True)
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    with caplog.at_level("ERROR"):
        asyncio.run(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("hi there"),
                llm,
                tts,
                history,
                asyncio.Lock(),
                _recording_player(played),
            )
        )
    assert history == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello there, this is a longer reply."},
    ]
    assert played == []
