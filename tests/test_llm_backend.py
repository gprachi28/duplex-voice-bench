"""OllamaBackend test against a real local Ollama server.

Requires `ollama serve` running locally with `llama3.2:3b` pulled (see
README's "Ollama LLM setup"); skipped otherwise. Assertions are structural
(nonempty joined text, list of str chunks) not content-based, matching
test_stt_backend.py's style -- LLM output is nondeterministic.
"""

import asyncio

import httpx
import pytest

from agent.llm import OLLAMA_HOST_DEFAULT, OllamaBackend


def _ollama_reachable() -> bool:
    try:
        httpx.get(OLLAMA_HOST_DEFAULT, timeout=1.0)
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama server not reachable at {OLLAMA_HOST_DEFAULT}",
)


def _collect(agen):
    async def _run():
        return [chunk async for chunk in agen]

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def backend() -> OllamaBackend:
    return OllamaBackend(OLLAMA_HOST_DEFAULT)


def test_stream_chat_yields_nonempty_text(backend):
    chunks = _collect(
        backend.stream_chat(
            [{"role": "user", "content": "Reply with exactly one word: hello."}]
        )
    )
    assert all(isinstance(c, str) for c in chunks)
    assert len("".join(chunks).strip()) > 0


def test_stream_chat_accepts_multi_message_history(backend):
    messages = [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Got it, blue."},
        {"role": "user", "content": "What color did I just mention?"},
    ]
    chunks = _collect(backend.stream_chat(messages))
    assert len("".join(chunks).strip()) > 0
