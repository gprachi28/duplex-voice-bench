"""Accumulates utterance audio across one or more VAD speech segments and
decides, on each Silero END_OF_SPEECH, whether Smart Turn's completion
probability says the turn is actually done.

Unlike SmartTurnObserver's fixed 8s ring buffer, this buffer grows to fit
the whole utterance so STT never sees truncated audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.audio import TARGET_SR

GATE_THRESHOLD = 0.5
MAX_UTTERANCE_S = 30.0


@dataclass(frozen=True)
class Fire:
    audio: np.ndarray


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class ForceFire:
    audio: np.ndarray


GateResult = Fire | Continue | ForceFire


class TurnGate:
    def __init__(
        self,
        threshold: float = GATE_THRESHOLD,
        max_duration_s: float = MAX_UTTERANCE_S,
        sample_rate: int = TARGET_SR,
    ) -> None:
        self._threshold = threshold
        self._max_samples = int(max_duration_s * sample_rate)
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._open = False
        self._over_budget = False

    def begin(self) -> None:
        self._open = True

    def push(self, samples: np.ndarray) -> bool:
        if not self._open or self._over_budget:
            return False
        samples = np.asarray(samples, dtype=np.float32)
        self._chunks.append(samples)
        self._total_samples += len(samples)
        if self._total_samples >= self._max_samples:
            self._over_budget = True
            return True
        return False

    def evaluate(self, smart_turn_prob: float) -> GateResult:
        audio = self._concat()
        if self._over_budget:
            self._clear()
            return ForceFire(audio)
        if smart_turn_prob >= self._threshold:
            self._clear()
            return Fire(audio)
        self._open = False
        return Continue()

    def _concat(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def _clear(self) -> None:
        self._chunks = []
        self._total_samples = 0
        self._open = False
        self._over_budget = False


def create_turn_gate() -> TurnGate:
    """Construct a TurnGate with default config. No model to preload, so
    unlike create_vad()/create_stt_backend() this is a plain constructor
    call, not a cached singleton — kept as a factory for naming consistency
    with the other agent/*.py modules."""
    return TurnGate()
