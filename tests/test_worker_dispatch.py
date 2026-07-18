"""Pure unit tests for _dispatch_gate_result's LLM wiring -- fake STT/LLM
backends, no real audio, no network. Exercises history bookkeeping and the
history_lock serialization documented in worker.py's docstring: TurnGate
doesn't prevent a second Fire from spawning a second dispatch task while a
prior one's LLM stream is still running, so the lock must keep concurrent
turns from interleaving in the shared history list.
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


def test_continue_result_skips_stt_llm_and_history():
    llm = FakeLLM(["should not be called"])
    history: list[dict[str, str]] = []
    asyncio.run(
        _dispatch_gate_result(Continue(), FakeSTT("unused"), llm, history, asyncio.Lock())
    )
    assert llm.calls == []
    assert history == []


def test_fire_appends_user_then_assistant_turn():
    llm = FakeLLM(["Hel", "lo!"])
    history: list[dict[str, str]] = []
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("hi there"),
            llm,
            history,
            asyncio.Lock(),
        )
    )
    assert history == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_fire_sends_full_prior_history_to_llm():
    llm = FakeLLM(["ok"])
    history: list[dict[str, str]] = [
        {"role": "user", "content": "earlier turn"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    asyncio.run(
        _dispatch_gate_result(
            Fire(np.zeros(1, dtype=np.float32)),
            FakeSTT("new turn"),
            llm,
            history,
            asyncio.Lock(),
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
    history: list[dict[str, str]] = []
    lock = asyncio.Lock()

    async def _run_both():
        first = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)), FakeSTT("first"), slow_llm, history, lock
            )
        )
        await asyncio.sleep(0.05)
        second = asyncio.create_task(
            _dispatch_gate_result(
                Fire(np.zeros(1, dtype=np.float32)), FakeSTT("second"), slow_llm, history, lock
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
