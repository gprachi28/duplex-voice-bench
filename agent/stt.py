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
        # language="en": this pipeline is English-only (SYSTEM_PROMPT
        # enforces English replies, Kokoro TTS is English-only). Without it,
        # mlx_whisper.transcribe() runs a full extra encoder pass to guess
        # the language before transcribing -- confirmed via direct-call
        # benchmark to cost ~0.56s per turn, roughly half of measured STT
        # latency.
        #
        # temperature=0.0: mlx_whisper's default is a 6-value fallback tuple
        # that reruns the full decode at higher temperatures whenever
        # compression_ratio_threshold ("too repetitive") or logprob_threshold
        # fails. is_repetition_loop() below already discards repetitive
        # output regardless of which temperature produced it, so the
        # fallback buys nothing on its primary trigger while costing up to
        # ~6x a single decode pass (confirmed via direct-call benchmark:
        # 4.04s vs 0.64s). See benchmarks/experiments.md.
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self._model, language="en", temperature=0.0
        )
        return result["text"].strip()


def is_repetition_loop(text: str, min_repeats: int = 5) -> bool:
    """True if `text` contains the same word repeated `min_repeats`+ times
    consecutively -- Whisper's known decoder-repetition-loop failure mode.
    Confirmed live: 'should' repeated ~100 times, and separately 12 times,
    both during barge-in on already-playing TTS audio (see
    benchmarks/experiments.md). A real utterance essentially never repeats
    one word this many times in a row, so this is a cheap, reliable filter
    regardless of what actually triggers the loop."""
    words = [w.strip(".,!?;:").lower() for w in text.split()]
    run = 1
    for i in range(1, len(words)):
        if words[i] and words[i] == words[i - 1]:
            run += 1
            if run >= min_repeats:
                return True
        else:
            run = 1
    return False


_backend_instance: MlxWhisperBackend | None = None


def create_stt_backend() -> MlxWhisperBackend:
    """Return the process-wide MlxWhisperBackend, loading it on first call."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = MlxWhisperBackend()
    return _backend_instance
