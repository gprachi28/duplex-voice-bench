"""GPT4oBackend test against the real OpenAI API.

Requires OPENAI_API_KEY to be set; skipped otherwise. Assertions are
structural (nonempty joined text, list of str chunks), matching
test_llm_backend.py's style -- LLM output is nondeterministic.
"""

import asyncio
import os

import pytest

from agent.llm import GPT4oBackend

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY, reason="OPENAI_API_KEY not set"
)


def _collect(agen):
    async def _run():
        return [chunk async for chunk in agen]

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def backend() -> GPT4oBackend:
    return GPT4oBackend(OPENAI_API_KEY)


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
