"""Integration test: TurnGate's gating decision using real Smart Turn
scores on the recorded fixtures (no LiveKit, no STT).

Requires SMART_TURN_MODEL_PATH (see tests/test_smart_turn_model.py).
"""

import os
import wave

import numpy as np
import pytest

from agent.smart_turn import SmartTurnScorer
from agent.turn_gate import Continue, Fire, TurnGate

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "smart_turn")
MODEL_PATH = os.environ.get("SMART_TURN_MODEL_PATH")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH, reason="SMART_TURN_MODEL_PATH not set"
)


def _load_wav_f32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def scorer() -> SmartTurnScorer:
    return SmartTurnScorer(MODEL_PATH)


def test_complete_utterance_gate_fires(scorer):
    audio = _load_wav_f32(os.path.join(FIXTURES_DIR, "complete.wav"))
    prob = scorer.score(audio)
    assert prob >= 0.5  # matches test_smart_turn_model.py's expectation for this fixture

    gate = TurnGate()
    gate.begin()
    gate.push(audio)
    result = gate.evaluate(prob)
    assert isinstance(result, Fire)
    assert np.array_equal(result.audio, audio.astype(np.float32))


def test_incomplete_utterance_gate_continues(scorer):
    audio = _load_wav_f32(os.path.join(FIXTURES_DIR, "incomplete.wav"))
    prob = scorer.score(audio)
    assert prob < 0.5  # matches test_smart_turn_model.py's expectation for this fixture

    gate = TurnGate()
    gate.begin()
    gate.push(audio)
    result = gate.evaluate(prob)
    assert isinstance(result, Continue)
