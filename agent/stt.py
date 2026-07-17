"""Speech-to-text backend interface and the mlx-whisper implementation.

Consumes the same 16 kHz mono float32 buffer agent/audio.py normalises to.
STTBackend exists as a protocol now (not after a second implementation
exists) because pluggable STT backends are a stated project goal — see
design.md's "STT — pluggable" section — enabling later A/B benchmarking
against a second backend without touching callers.
"""

from __future__ import annotations

from typing import Protocol

import mlx_whisper
import numpy as np

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


class STTBackend(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class MlxWhisperBackend:
    """mlx-whisper, Metal-accelerated. `model` is an HF Hub repo id;
    mlx-whisper resolves and caches weights itself on first use."""

    def __init__(self, model: str = WHISPER_MODEL) -> None:
        self._model = model

    def transcribe(self, audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._model)
        return result["text"].strip()


_backend_instance: MlxWhisperBackend | None = None


def create_stt_backend() -> MlxWhisperBackend:
    """Return the process-wide MlxWhisperBackend, loading it on first call."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = MlxWhisperBackend()
    return _backend_instance
