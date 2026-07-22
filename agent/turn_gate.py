"""Accumulates utterance audio across one or more VAD speech segments and
decides, on each Silero END_OF_SPEECH, whether Smart Turn's completion
probability says the turn is actually done.

Unlike SmartTurnObserver's fixed 8s ring buffer, this buffer grows to fit
the whole utterance so STT never sees truncated audio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from agent.audio import TARGET_SR

GATE_THRESHOLD = 0.5
MAX_UTTERANCE_S = 30.0
MAX_TURN_WALL_CLOCK_S = 15.0
# Below this, Whisper hallucinates canned phrases on the near-empty clip
# instead of returning nothing -- confirmed live: a 0.5s first-utterance
# segment scored smart_turn_prob=0.75 and fired straight into STT, which
# produced "Sous-titrage Société Radio-Canada" (Whisper's well-known
# near-silence hallucination). 0.3s matches Silero's own trailing-silence
# window (design.md). Only gates the probability-based Fire below -- the
# ForceFire safety valves exist to prevent hangs, not to judge signal
# quality, so they're untouched.
MIN_UTTERANCE_S = 0.3


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
    """Smart Turn's completion probability can stay persistently low across
    several short, fragmented bursts of real speech (confirmed live: a turn
    scored 0.07/0.01/0.01/0.05 across four bursts spanning 47s of wall
    clock). MAX_UTTERANCE_S alone can't catch that -- it only counts
    accumulated speech samples, and fragmented bursts rarely add up to 30s
    of real audio. max_wall_clock_s is a second, independent valve: it
    force-fires once this much time has passed since the turn's first
    begin(), regardless of how little speech has accumulated."""

    def __init__(
        self,
        threshold: float = GATE_THRESHOLD,
        max_duration_s: float = MAX_UTTERANCE_S,
        sample_rate: int = TARGET_SR,
        max_wall_clock_s: float = MAX_TURN_WALL_CLOCK_S,
        min_duration_s: float = MIN_UTTERANCE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._max_samples = int(max_duration_s * sample_rate)
        self._max_wall_clock_s = max_wall_clock_s
        self._min_samples = int(min_duration_s * sample_rate)
        self._clock = clock
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._open = False
        self._over_budget = False
        self._turn_started_at: float | None = None

    @property
    def is_open(self) -> bool:
        """True from the first begin() of a turn until it resolves via
        Fire/ForceFire (_clear()) -- spans Continue's silent gaps too, since
        the turn is still pending resolution throughout. Used to drive the
        client's "listening" visual indicator (see design.md's Known
        Issues: "No feedback during a long pause")."""
        return self._turn_started_at is not None

    def begin(self) -> None:
        self._open = True
        if self._turn_started_at is None:
            self._turn_started_at = self._clock()

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
        started_at = self._turn_started_at
        if started_at is not None and self._clock() - started_at >= self._max_wall_clock_s:
            self._clear()
            return ForceFire(audio)
        if smart_turn_prob >= self._threshold and self._total_samples >= self._min_samples:
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
        self._turn_started_at = None


def create_turn_gate() -> TurnGate:
    """Construct a TurnGate with default config. No model to preload, so
    unlike create_vad()/create_stt_backend() this is a plain constructor
    call, not a cached singleton — kept as a factory for naming consistency
    with the other agent/*.py modules."""
    return TurnGate()
