"""ONNX inference tests for SmartTurnScorer, verified against labeled fixtures.

Requires SMART_TURN_MODEL_PATH (model weights are gitignored, not committed).
"""

import os
import wave

import numpy as np
import pytest

from agent.smart_turn import SmartTurnScorer

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


def test_complete_utterance_scores_above_threshold(scorer):
    audio = _load_wav_f32(os.path.join(FIXTURES_DIR, "complete.wav"))
    assert scorer.score(audio) > 0.5


def test_incomplete_utterance_scores_below_threshold(scorer):
    audio = _load_wav_f32(os.path.join(FIXTURES_DIR, "incomplete.wav"))
    assert scorer.score(audio) < 0.5