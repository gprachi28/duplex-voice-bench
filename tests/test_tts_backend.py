"""Kokoro TTS backend test -- mirrors test_stt_backend.py's skip/fixture
pattern for a real, weights-backed local model.

Requires Apple Silicon (mlx-audio's Kokoro model needs the MLX runtime);
skipped otherwise. Downloads the Kokoro-82M weights from the HF Hub on
first run via mlx-audio's own cache (not the models/ dir used for Smart
Turn's manually-downloaded ONNX weights).
"""

import platform

import numpy as np
import pytest

from agent.tts import KokoroBackend

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="mlx-audio's Kokoro backend requires Apple Silicon",
)


@pytest.fixture(scope="module")
def backend() -> KokoroBackend:
    return KokoroBackend()


def test_synthesize_returns_nonempty_float32_mono_audio(backend):
    audio = backend.synthesize("Hello there, this is a test.")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) > 0


def test_synthesize_produces_longer_audio_for_longer_text(backend):
    short = backend.synthesize("Hi.")
    long = backend.synthesize(
        "The quick brown fox jumps over the lazy dog near the riverbank."
    )
    assert len(long) > len(short)


def test_backend_exposes_sample_rate(backend):
    assert backend.sample_rate == 24_000
