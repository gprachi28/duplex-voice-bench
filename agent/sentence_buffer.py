"""Buffers the LLM token stream and decides when to flush text to TTS.

See design.md's "LLM -> TTS handoff" section: flushing one token at a time
gives TTS no prosodic context, so accumulate until a sentence boundary
([. ? ! ,]) AND the ElevenLabs-recommended minimum flush size (20 chars)
are both satisfied.
"""

from __future__ import annotations

BOUNDARY_CHARS = ".?!,"
MIN_FLUSH_LENGTH = 20


class SentenceBuffer:
    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> list[str]:
        """Feed a token from the LLM stream; return any segments now ready
        for TTS (usually zero or one, but a token can contain more than one
        boundary)."""
        self._buf += token
        flushed: list[str] = []

        start = 0
        for i, ch in enumerate(self._buf):
            if ch in BOUNDARY_CHARS and (i + 1 - start) >= MIN_FLUSH_LENGTH:
                flushed.append(self._buf[start : i + 1].lstrip())
                start = i + 1

        self._buf = self._buf[start:]
        return flushed

    def flush(self) -> str | None:
        """Return and clear any remaining buffered text (stream end)."""
        remainder = self._buf.strip()
        self._buf = ""
        return remainder or None
