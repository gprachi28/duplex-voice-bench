"""mlx-whisper backend test against the recorded Smart Turn fixtures.

Requires Apple Silicon (mlx-whisper is Metal-only); skipped otherwise.
Downloads the large-v3 model from the HF Hub on first run (a few GB) via
mlx-whisper's own cache -- not the models/ dir used for Smart Turn's
manually-downloaded ONNX weights.
"""

import os
import platform
import wave

import numpy as np
import pytest

from agent.stt import MlxWhisperBackend

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "smart_turn")

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="mlx-whisper requires Apple Silicon",
)


def _load_wav_f32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def backend() -> MlxWhisperBackend:
    return MlxWhisperBackend()


def test_transcribe_complete_utterance_returns_nonempty_text(backend):
    audio = _load_wav_f32(os.path.join(FIXTURES_DIR, "complete.wav"))
    text = backend.transcribe(audio)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_transcribe_incomplete_utterance_returns_nonempty_text(backend):
    audio = _load_wav_f32(os.path.join(FIXTURES_DIR, "incomplete.wav"))
    text = backend.transcribe(audio)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
