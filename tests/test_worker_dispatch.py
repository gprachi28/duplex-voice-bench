"""Pure unit tests for _dispatch_gate_result's LLM + TTS + barge-in wiring
-- fake STT/LLM/TTS backends, no real audio, no network. Exercises history
bookkeeping, the sentence-buffer-gated TTS handoff, and barge-in: a new
confirmed Fire/ForceFire cancels whatever reply is still in flight rather
than waiting for it, per docs/superpowers/specs/2026-07-19-barge-in-design.md.
"""

import asyncio

import numpy as np
import pytest

from agent.turn_gate import Continue, Fire
from agent.worker import ActiveReply, _dispatch_gate_result


class FakeSTT:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: np.ndarray) -> str:
        return self._text


class FakeLLM:
    def __init__(
        self,
        chunks: list[str],
        delay_s: float = 0.0,
        cancel_log: list[str] | None = None,
    ) -> None:
        self._chunks = chunks
        self._delay_s = delay_s
        self._cancel_log = cancel_log
        self.calls: list[list[dict[str, str]]] = []

    async def stream_chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        if self._delay_s:
            try:
                await asyncio.sleep(self._delay_s)
            except asyncio.CancelledError:
                if self._cancel_log is not None:
                    self._cancel_log.append("llm_cancelled")
                raise
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


def _noop_clear_audio() -> None:
    pass


def _recording_clear_audio(sink: list[str]):
    def clear() -> None:
        sink.append("clear_audio")

    return clear


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
            ActiveReply(),
            _recording_player(played),
            _noop_clear_audio,
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
            ActiveReply(),
            _recording_player(played),
            _noop_clear_audio,
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
            ActiveReply(),
            _recording_player(played),
            _noop_clear_audio,
        )
    )
    assert llm.calls == [
        [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "new turn"},
        ]
    ]


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
            ActiveReply(),
            _recording_player(played),
            _noop_clear_audio,
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
                ActiveReply(),
                _recording_player(played),
                _noop_clear_audio,
            )
        )
    assert history == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello there, this is a longer reply."},
    ]
    assert played == []


def test_barge_in_cancels_in_flight_reply_and_completes_new_turn(caplog):
    call_order: list[str] = []
    slow_llm = FakeLLM(["slow-reply"], delay_s=0.2, cancel_log=call_order)
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()
    active_reply = ActiveReply()
    play = _recording_player(played)
    clear_audio = _recording_clear_audio(call_order)

    async def _run_both():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("first"),
                slow_llm,
                tts,
                history,
                lock,
                active_reply,
                play,
                clear_audio,
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
                active_reply,
                play,
                clear_audio,
            )
        )
        await asyncio.gather(first, second, return_exceptions=True)

    with caplog.at_level("INFO"):
        asyncio.run(_run_both())

    # First turn's question stays (it really was asked); its reply never
    # arrives because cancellation happened before the LLM could finish.
    # Second turn completes normally, uninterrupted.
    assert history == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "slow-reply"},
    ]
    assert call_order == ["clear_audio", "llm_cancelled"]
    assert any(
        "barge-in: cancelled in-flight reply" in r.message for r in caplog.records
    )


def test_cancelling_an_already_finished_reply_is_a_noop():
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()
    active_reply = ActiveReply()
    play = _recording_player(played)
    clear_calls: list[str] = []
    clear_audio = _recording_clear_audio(clear_calls)

    async def _run_sequential():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("first"),
                FakeLLM(["first-reply"]),
                tts,
                history,
                lock,
                active_reply,
                play,
                clear_audio,
            )
        )
        await first
        second = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("second"),
                FakeLLM(["second-reply"]),
                tts,
                history,
                lock,
                active_reply,
                play,
                clear_audio,
            )
        )
        await second

    asyncio.run(_run_sequential())

    assert history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first-reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second-reply"},
    ]
    assert clear_calls == []
