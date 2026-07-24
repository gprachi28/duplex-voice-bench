"""LLM backend interface: Ollama (local) and GPT-4o (cloud) implementations.

Consumes the conversation history worker.py accumulates per session and
streams back the assistant's reply. create_llm_backend() selects between
them via the LLM_BACKEND env var, so a benchmark combination from
design.md's "Benchmark Combinations" table can be run with no code change.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Protocol

import httpx
import openai

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST_DEFAULT = "http://localhost:11434"
GPT4O_MODEL = "gpt-4o"


class LLMBackend(Protocol):
    model: str

    def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class OllamaBackend:
    """Streams chat completions from a local Ollama server's /api/chat.

    `host` is read from OLLAMA_HOST with a fallback, not a hard-required
    env var like SMART_TURN_MODEL_PATH -- there's no sane default for a
    filesystem path to model weights, but Ollama's own localhost:11434
    convention is a reasonable default for a network location.
    """

    def __init__(self, host: str, model: str = OLLAMA_MODEL) -> None:
        self.model = model
        self._client = httpx.AsyncClient(base_url=host, timeout=30.0)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            "/api/chat",
            json={"model": self.model, "messages": messages, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    return


class GPT4oBackend:
    """Streams chat completions from OpenAI's GPT-4o."""

    def __init__(self, api_key: str, model: str = GPT4O_MODEL) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


_backend_instance: LLMBackend | None = None


def create_llm_backend() -> LLMBackend:
    """Return the process-wide LLMBackend, constructing it on first call.

    Selected via LLM_BACKEND ("ollama", the default, or "gpt4o") -- a plain
    env var, matching the benchmark suite's other per-run config (e.g.
    OLLAMA_HOST), for choosing a combination from design.md's table without
    a code change. Construction only builds an HTTP client -- no network
    call -- so this is safe to preload in _prewarm regardless of which
    backend is selected.
    """
    global _backend_instance
    if _backend_instance is None:
        backend_name = os.environ.get("LLM_BACKEND", "ollama")
        if backend_name == "ollama":
            host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST_DEFAULT)
            _backend_instance = OllamaBackend(host)
        elif backend_name == "gpt4o":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "LLM_BACKEND=gpt4o requires OPENAI_API_KEY to be set"
                )
            model = os.environ.get("OPENAI_MODEL", GPT4O_MODEL)
            _backend_instance = GPT4oBackend(api_key, model=model)
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {backend_name!r}")
    return _backend_instance
