"""LLM backend interface and the Ollama streaming implementation.

Consumes the conversation history worker.py accumulates per session and
streams back the assistant's reply. LLMBackend exists as a Protocol now
(not after a second implementation exists) for the same reason STTBackend
does -- see stt.py's docstring and design.md's "LLM -- streaming" section:
pluggable backends (Ollama local vs. GPT-4o cloud) are a stated project
goal, so a Protocol keeps callers backend-agnostic ahead of the second
implementation landing for A/B benchmarking.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST_DEFAULT = "http://localhost:11434"


class LLMBackend(Protocol):
    def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class OllamaBackend:
    """Streams chat completions from a local Ollama server's /api/chat.

    `host` is read from OLLAMA_HOST with a fallback, not a hard-required
    env var like SMART_TURN_MODEL_PATH -- there's no sane default for a
    filesystem path to model weights, but Ollama's own localhost:11434
    convention is a reasonable default for a network location.
    """

    def __init__(self, host: str, model: str = OLLAMA_MODEL) -> None:
        self._model = model
        self._client = httpx.AsyncClient(base_url=host, timeout=30.0)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            "/api/chat",
            json={"model": self._model, "messages": messages, "stream": True},
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


_backend_instance: OllamaBackend | None = None


def create_llm_backend() -> OllamaBackend:
    """Return the process-wide OllamaBackend, constructing it on first call.

    Construction only builds an httpx.AsyncClient -- no network call -- so
    this is safe to preload in _prewarm even if Ollama isn't running yet.
    """
    global _backend_instance
    if _backend_instance is None:
        host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST_DEFAULT)
        _backend_instance = OllamaBackend(host)
    return _backend_instance
