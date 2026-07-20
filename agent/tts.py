"""Text-to-speech backend interface and the Kokoro (MLX) implementation.

Consumes sentences flushed by agent/sentence_buffer.py and returns
synthesised audio for the worker to publish back into the room.
TTSBackend exists as a Protocol now (not after a second implementation
exists) for the same reason STTBackend/LLMBackend do -- see design.md's
"TTS -- pluggable" section: swapping in a cloud backend (ElevenLabs) for
A/B benchmarking is a stated project goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from mlx_audio.tts.utils import load

KOKORO_REPO = "prince-canuma/Kokoro-82M"
KOKORO_VOICE = "af_heart"


@dataclass
class WordTiming:
    """One word's audio span within a synthesized segment. `text` includes
    the token's trailing whitespace so joining WordTimings back together
    reconstructs spacing. Backend-agnostic (no misaki/Kokoro types) so a
    future ElevenLabs/Cartesia backend can supply the same shape -- see
    docs/superpowers/specs/2026-07-20-barge-in_heard_text.md."""

    text: str
    start: float
    end: float


@dataclass
class TTSResult:
    audio: np.ndarray
    words: list[WordTiming] = field(default_factory=list)


class TTSBackend(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> TTSResult: ...


class KokoroBackend:
    """mlx-audio's Kokoro-82M, MLX-accelerated. `repo` is an HF Hub repo
    id; mlx-audio resolves and caches weights itself on first use."""

    def __init__(self, repo: str = KOKORO_REPO, voice: str = KOKORO_VOICE) -> None:
        self._model = load(repo)
        self._voice = voice
        self.sample_rate = self._model.sample_rate

    def synthesize(self, text: str) -> TTSResult:
        # model.generate() wraps the pipeline but only unpacks
        # (graphemes, phonemes, audio) from each Result, discarding
        # .tokens (word-level alignment). Call the pipeline directly to
        # keep it. Result.audio is (1, N) batch-first; [0] unwraps to the
        # 1-D waveform model.generate() would otherwise have returned.
        pipeline = self._model._get_pipeline("a")
        segments: list[np.ndarray] = []
        words: list[WordTiming] = []
        offset_s = 0.0
        for result in pipeline(text, voice=self._voice):
            if result.audio is None:
                continue
            audio = np.array(result.audio[0], dtype=np.float32)
            segments.append(audio)
            for tok in result.tokens or []:
                if tok.start_ts is None or tok.end_ts is None:
                    continue
                words.append(
                    WordTiming(
                        text=tok.text + tok.whitespace,
                        start=offset_s + tok.start_ts,
                        end=offset_s + tok.end_ts,
                    )
                )
            offset_s += len(audio) / self.sample_rate
        if not segments:
            return TTSResult(np.zeros(0, dtype=np.float32), [])
        return TTSResult(np.concatenate(segments), words)


_backend_instance: KokoroBackend | None = None


def create_tts_backend() -> KokoroBackend:
    """Return the process-wide KokoroBackend, loading it on first call."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = KokoroBackend()
    return _backend_instance
