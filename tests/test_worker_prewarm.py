"""Unit tests for the LLM warmup helper used by _prewarm -- see
benchmarks/experiments.md: STT and the LLM were never warmed (unlike TTS,
which already gets a dummy synthesize() call), so the first live utterance
of a session paid a full model-load cost on top of normal inference. Only
the LLM side is unit-testable without a real backend; the STT/TTS warmup
calls in _prewarm itself require the real mlx-whisper/Kokoro models and are
verified live, same as _prewarm's existing behavior today.
"""

import asyncio

from agent.worker import _warm_llm


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def stream_chat(self, messages):
        self.calls.append(messages)
        yield "hi"
        yield " there"


def test_warm_llm_drains_the_stream_without_error():
    llm = FakeLLM()
    asyncio.run(_warm_llm(llm))  # must not raise
    assert len(llm.calls) == 1


def test_warm_llm_sends_a_minimal_user_message():
    llm = FakeLLM()
    asyncio.run(_warm_llm(llm))
    assert llm.calls[0] == [{"role": "user", "content": "Hi"}]
