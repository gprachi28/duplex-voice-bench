"""Text-to-speech backend interface and the Kokoro (MLX) implementation.

Consumes sentences flushed by agent/sentence_buffer.py and returns
synthesised audio for the worker to publish back into the room.
TTSBackend exists as a Protocol now (not after a second implementation
exists) for the same reason STTBackend/LLMBackend do -- see design.md's
"TTS -- pluggable" section: swapping in a cloud backend (ElevenLabs) for
A/B benchmarking is a stated project goal.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from mlx_audio.tts.utils import load

KOKORO_REPO = "prince-canuma/Kokoro-82M"
KOKORO_VOICE = "af_heart"


class TTSBackend(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


class KokoroBackend:
    """mlx-audio's Kokoro-82M, MLX-accelerated. `repo` is an HF Hub repo
    id; mlx-audio resolves and caches weights itself on first use."""

    def __init__(self, repo: str = KOKORO_REPO, voice: str = KOKORO_VOICE) -> None:
        self._model = load(repo)
        self._voice = voice
        self.sample_rate = self._model.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        segments = [
            np.array(result.audio, dtype=np.float32)
            for result in self._model.generate(text, voice=self._voice)
        ]
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)


_backend_instance: KokoroBackend | None = None


def create_tts_backend() -> KokoroBackend:
    """Return the process-wide KokoroBackend, loading it on first call."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = KokoroBackend()
    return _backend_instance
