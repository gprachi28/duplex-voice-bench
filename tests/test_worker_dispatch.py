"""Pure unit tests for _dispatch_gate_result's LLM + TTS + barge-in wiring
-- fake STT/LLM/TTS backends, no real audio, no network. Exercises history
bookkeeping, the sentence-buffer-gated TTS handoff, and barge-in: a new
confirmed Fire/ForceFire cooperatively interrupts whatever reply is still
in flight rather than waiting for it, per
docs/superpowers/specs/2026-07-19-barge-in-design.md's "Revision" section
-- not asyncio.Task.cancel(), which corrupts LiveKit's AudioSource if it
lands mid-capture_frame() (confirmed via live reproduction).
"""

import asyncio
import time

import numpy as np
import pytest

from agent.tts import TTSResult, WordTiming
from agent.turn_gate import Continue, Fire
from agent.worker import (
    ActiveReply,
    HeardWord,
    _dispatch_gate_result,
    _synthesize_and_play,
    heard_text,
)


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
        mid_delay_s: float = 0.0,
        mid_delay_after: int = 0,
    ) -> None:
        self._chunks = chunks
        self._delay_s = delay_s
        self._mid_delay_s = mid_delay_s
        self._mid_delay_after = mid_delay_after
        self.calls: list[list[dict[str, str]]] = []

    async def stream_chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        for i, chunk in enumerate(self._chunks):
            yield chunk
            if self._mid_delay_s and i == self._mid_delay_after:
                await asyncio.sleep(self._mid_delay_s)


class FakeTTS:
    sample_rate = 24_000

    def __init__(self, fail: bool = False, words: list[WordTiming] | None = None) -> None:
        self._fail = fail
        self._words = words or []
        self.synthesized: list[str] = []

    def synthesize(self, text: str) -> TTSResult:
        if self._fail:
            raise RuntimeError("synth failed")
        self.synthesized.append(text)
        return TTSResult(np.ones(len(text), dtype=np.float32), self._words)


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


def test_barge_in_interrupts_in_flight_reply_and_completes_new_turn(caplog):
    tts = FakeTTS()
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()
    active_reply = ActiveReply()
    play = _recording_player(played)
    clear_calls: list[str] = []
    clear_audio = _recording_clear_audio(clear_calls)

    async def _run_both():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("first"),
                FakeLLM(["first-reply"], delay_s=0.2),
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
                FakeLLM(["second-reply"]),
                tts,
                history,
                lock,
                active_reply,
                play,
                clear_audio,
            )
        )
        await asyncio.gather(first, second)

    with caplog.at_level("INFO"):
        asyncio.run(_run_both())

    # First turn's question stays (it really was asked); its reply never
    # arrives because the interruption flag was checked before it could
    # synthesize/play/append anything. Second turn completes normally.
    assert history == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second-reply"},
    ]
    assert tts.synthesized == ["second-reply"]
    assert clear_calls == ["clear_audio"]
    assert any(
        "barge-in: interrupting in-flight reply" in r.message
        for r in caplog.records
    )


def test_barge_in_discards_audio_synthesized_after_interruption_flagged():
    """TTS synthesis runs in a thread executor and can't be cancelled
    mid-call. If active_reply.interrupted flips true while a segment is
    already synthesizing, the finished audio must be discarded -- not
    played -- rather than checking the flag only before/after the whole
    synthesize+play unit (which lets one full segment play after every
    barge-in, audible as the agent talking over the user)."""
    active_reply = ActiveReply()
    played: list[np.ndarray] = []
    play = _recording_player(played)
    history: list[dict[str, str]] = []

    class InterruptingTTS:
        sample_rate = 24_000

        def synthesize(self, text: str) -> TTSResult:
            # Simulate a second turn's barge-in landing while this
            # (uncancellable) synthesis call is already in flight.
            active_reply.interrupted = True
            return TTSResult(np.ones(len(text), dtype=np.float32), [])

    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            FakeLLM(["Hello there, this is a longer reply."]),
            InterruptingTTS(),
            history,
            asyncio.Lock(),
            active_reply,
            play,
            _noop_clear_audio,
        )
    )
    assert played == []
    assert history == [{"role": "user", "content": "hi there"}]


def test_barging_in_on_an_already_finished_reply_is_a_noop():
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


def test_heard_text_returns_words_up_to_interrupt():
    timeline = [
        HeardWord("Hello ", end=0.3),
        HeardWord("there, ", end=0.6),
        HeardWord("friend.", end=1.0),
    ]
    assert heard_text(timeline, interrupted_at=0.7) == "Hello there,"


def test_heard_text_empty_timeline_returns_empty():
    assert heard_text([], interrupted_at=5.0) == ""


def test_heard_text_interrupt_before_first_word_returns_empty():
    timeline = [HeardWord("Hello ", end=0.3)]
    assert heard_text(timeline, interrupted_at=0.1) == ""


def test_heard_text_interrupt_after_all_words_returns_full():
    timeline = [HeardWord("Hello ", end=0.3), HeardWord("there.", end=0.6)]
    assert heard_text(timeline, interrupted_at=10.0) == "Hello there."


def test_heard_text_none_interrupt_returns_empty():
    timeline = [HeardWord("Hello ", end=0.3)]
    assert heard_text(timeline, interrupted_at=None) == ""


def test_synthesize_and_play_populates_heard_timeline():
    words = [
        WordTiming("Hi ", start=0.0, end=0.3),
        WordTiming("there.", start=0.3, end=0.6),
    ]
    tts = FakeTTS(words=words)
    active_reply = ActiveReply()
    played: list[np.ndarray] = []
    play = _recording_player(played)

    before = time.monotonic()
    asyncio.run(_synthesize_and_play(tts, "Hi there.", play, active_reply))
    after = time.monotonic()

    assert [w.text for w in active_reply.heard_timeline] == ["Hi ", "there."]
    first_end, second_end = (w.end for w in active_reply.heard_timeline)
    assert before + 0.3 <= first_end <= after + 0.3
    assert before + 0.6 <= second_end <= after + 0.6
    assert first_end < second_end

    expected_audio_duration = len("Hi there.") / tts.sample_rate
    assert (
        before + expected_audio_duration
        <= active_reply.playback_cursor
        <= after + expected_audio_duration
    )


def test_barge_in_mid_playback_appends_truncated_prefix():
    # sentence_buffer needs >= 20 chars before it will flush, so the first
    # chunk alone must clear that bar to get synthesized+played before the
    # interrupt lands.
    words = [
        WordTiming("This ", start=0.0, end=0.02),
        WordTiming("is ", start=0.02, end=0.04),
        WordTiming("a ", start=0.04, end=0.06),
        WordTiming("longer ", start=0.06, end=0.08),
        WordTiming("greeting.", start=0.08, end=0.10),
    ]
    tts = FakeTTS(words=words)
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()
    active_reply = ActiveReply()
    play = _recording_player(played)

    full_reply = "This is a longer greeting. This is more."

    async def _run_both():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("first"),
                FakeLLM(
                    ["This is a longer greeting. ", "This is more."],
                    mid_delay_s=0.2,
                    mid_delay_after=0,
                ),
                tts,
                history,
                lock,
                active_reply,
                play,
                _noop_clear_audio,
            )
        )
        # First's mid-delay (0.2s) is well past the point its first segment
        # ("This is a longer greeting.") has synthesized and played, but
        # well before its second chunk arrives -- this lands the barge-in
        # mid-playback.
        await asyncio.sleep(0.08)
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
                _noop_clear_audio,
            )
        )
        await asyncio.gather(first, second)

    asyncio.run(_run_both())

    assert history[0] == {"role": "user", "content": "first"}
    heard = history[1]
    assert heard["role"] == "assistant"
    assert heard["content"]
    assert full_reply.startswith(heard["content"])
    assert history[2] == {"role": "user", "content": "second"}
    assert history[3] == {"role": "assistant", "content": "second-reply"}


def test_new_reply_resets_heard_timeline_and_cursor():
    tts = FakeTTS(words=[WordTiming("Hi ", start=0.0, end=0.02)])
    played: list[np.ndarray] = []
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()
    active_reply = ActiveReply()
    play = _recording_player(played)

    async def _run_sequential():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("first"),
                FakeLLM(["Hi"]),
                tts,
                history,
                lock,
                active_reply,
                play,
                _noop_clear_audio,
            )
        )
        await first
        second = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)),
                FakeSTT("second"),
                FakeLLM(["Hi"]),
                tts,
                history,
                lock,
                active_reply,
                play,
                _noop_clear_audio,
            )
        )
        await second

    asyncio.run(_run_sequential())

    # _synthesize_and_play appends to heard_timeline; if the second
    # dispatch didn't reset it at the start, it would hold both replies'
    # words (2 entries) instead of just its own (1).
    assert len(active_reply.heard_timeline) == 1
    assert active_reply.playback_cursor > 0.0
